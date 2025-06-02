"""
TDA-Aware Loss Functions for KAN_TDA Integration

This module implements specialized loss functions that incorporate topological
data analysis principles to improve model training and topological consistency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import warnings

try:
    from layers.TakensEmbedding import TakensEmbedding
    from layers.PersistentHomology import PersistentHomologyComputer
    from utils.persistence_landscapes import PersistenceLandscape, TopologicalFeatureExtractor
except ImportError:
    warnings.warn("TDA modules not found. Some loss functions may not work.")


class PersistenceLoss(nn.Module):
    """
    Loss function that penalizes unstable topological features and promotes
    persistence stability across predictions.
    """
    
    def __init__(
        self,
        embedding_dims: List[int] = [2, 3, 5],
        embedding_delays: List[int] = [1, 2, 4],
        persistence_threshold: float = 0.01,
        stability_weight: float = 1.0,
        consistency_weight: float = 0.5,
        device: torch.device = None
    ):
        """
        Initialize PersistenceLoss.
        
        Args:
            embedding_dims: Takens embedding dimensions to use
            embedding_delays: Delay parameters for embeddings
            persistence_threshold: Minimum persistence for stable features
            stability_weight: Weight for persistence stability term
            consistency_weight: Weight for cross-scale consistency term
            device: Device for computation
        """
        super(PersistenceLoss, self).__init__()
        
        self.embedding_dims = embedding_dims
        self.embedding_delays = embedding_delays
        self.persistence_threshold = persistence_threshold
        self.stability_weight = stability_weight
        self.consistency_weight = consistency_weight
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        # Initialize TDA components
        try:
            self.takens_embedding = TakensEmbedding(
                dims=embedding_dims,
                delays=embedding_delays,
                strategy='multi_scale'
            ).to(device)
            
            self.homology_computer = PersistentHomologyComputer(
                backend='ripser',
                max_dimension=1,
                distance_matrix_batch_size=1000
            )
            
            self.landscape_computer = PersistenceLandscape()
            self.feature_extractor = TopologicalFeatureExtractor()
        except:
            warnings.warn("Could not initialize TDA components. Loss will return zero.")
            self.tda_available = False
        else:
            self.tda_available = True
    
    def _compute_persistence_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute persistence features for input tensor.
        
        Args:
            x: Input tensor [B, T, C] or [B, T]
            
        Returns:
            Dictionary with persistence features
        """
        if not self.tda_available:
            return {}
        
        # Handle different input shapes
        if x.dim() == 3:
            # Multi-variate: use first channel for TDA
            x_tda = x[:, :, 0]  # [B, T]
        else:
            x_tda = x  # [B, T]
        
        batch_size = x_tda.shape[0]
        persistence_features = {}
        
        for b in range(batch_size):
            series = x_tda[b].cpu().numpy()  # [T]
            
            try:
                # Compute Takens embeddings
                embeddings = self.takens_embedding._multi_scale_embedding(
                    torch.from_numpy(series).unsqueeze(0).to(self.device)
                )
                
                batch_features = {}
                
                for (dim, delay), embedding in embeddings.items():
                    # Convert to numpy for homology computation
                    point_cloud = embedding.squeeze(0).cpu().numpy()
                    
                    # Compute persistent homology
                    diagrams = self.homology_computer.compute_persistence_diagrams(point_cloud)
                    
                    # Extract persistence values
                    if len(diagrams) > 0 and len(diagrams[0]) > 0:
                        # 0-dimensional persistence (connected components)
                        h0_persistence = []
                        for birth, death in diagrams[0]:
                            if death != np.inf:
                                h0_persistence.append(death - birth)
                        
                        # 1-dimensional persistence (loops) if available
                        h1_persistence = []
                        if len(diagrams) > 1 and len(diagrams[1]) > 0:
                            for birth, death in diagrams[1]:
                                if death != np.inf:
                                    h1_persistence.append(death - birth)
                        
                        batch_features[f'h0_persistence_{dim}_{delay}'] = torch.tensor(
                            h0_persistence, device=self.device
                        )
                        batch_features[f'h1_persistence_{dim}_{delay}'] = torch.tensor(
                            h1_persistence, device=self.device
                        )
                
                persistence_features[f'batch_{b}'] = batch_features
                
            except Exception as e:
                # Handle computation errors gracefully
                warnings.warn(f"TDA computation failed for batch {b}: {e}")
                persistence_features[f'batch_{b}'] = {}
        
        return persistence_features
    
    def _stability_loss(self, persistence_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute stability loss based on persistence values.
        
        Args:
            persistence_features: Dictionary with persistence features
            
        Returns:
            Stability loss value
        """
        if not persistence_features:
            return torch.tensor(0.0, device=self.device)
        
        stability_losses = []
        
        for batch_key, batch_features in persistence_features.items():
            if not batch_features:
                continue
                
            for feature_key, persistence_values in batch_features.items():
                if len(persistence_values) == 0:
                    continue
                
                # Penalize features with persistence below threshold
                stable_mask = persistence_values >= self.persistence_threshold
                unstable_persistence = persistence_values[~stable_mask]
                
                if len(unstable_persistence) > 0:
                    # Penalty increases as persistence decreases
                    penalty = torch.sum(
                        (self.persistence_threshold - unstable_persistence) ** 2
                    )
                    stability_losses.append(penalty)
        
        if stability_losses:
            return torch.stack(stability_losses).mean()
        else:
            return torch.tensor(0.0, device=self.device)
    
    def _consistency_loss(self, persistence_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute consistency loss across different embedding scales.
        
        Args:
            persistence_features: Dictionary with persistence features
            
        Returns:
            Consistency loss value
        """
        if not persistence_features:
            return torch.tensor(0.0, device=self.device)
        
        consistency_losses = []
        
        for batch_key, batch_features in persistence_features.items():
            if not batch_features:
                continue
            
            # Group features by homology dimension
            h0_features = []
            h1_features = []
            
            for feature_key, persistence_values in batch_features.items():
                if 'h0_persistence' in feature_key and len(persistence_values) > 0:
                    h0_features.append(persistence_values)
                elif 'h1_persistence' in feature_key and len(persistence_values) > 0:
                    h1_features.append(persistence_values)
            
            # Compute consistency within each homology dimension
            for feature_group in [h0_features, h1_features]:
                if len(feature_group) > 1:
                    # Compute pairwise consistency
                    for i in range(len(feature_group)):
                        for j in range(i + 1, len(feature_group)):
                            feat_i = feature_group[i]
                            feat_j = feature_group[j]
                            
                            # Use statistical measures for consistency
                            if len(feat_i) > 0 and len(feat_j) > 0:
                                mean_i = torch.mean(feat_i)
                                mean_j = torch.mean(feat_j)
                                
                                # Penalize large differences in mean persistence
                                consistency_loss = torch.abs(mean_i - mean_j)
                                consistency_losses.append(consistency_loss)
        
        if consistency_losses:
            return torch.stack(consistency_losses).mean()
        else:
            return torch.tensor(0.0, device=self.device)
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        base_loss: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute persistence loss.
        
        Args:
            predictions: Model predictions [B, T, C] or [B, T]
            targets: Target values [B, T, C] or [B, T]
            base_loss: Optional base loss to add to
            
        Returns:
            Dictionary with loss components
        """
        # Compute persistence features for both predictions and targets
        pred_features = self._compute_persistence_features(predictions)
        target_features = self._compute_persistence_features(targets)
        
        # Compute stability losses
        pred_stability = self._stability_loss(pred_features)
        target_stability = self._stability_loss(target_features)
        
        # Compute consistency losses
        pred_consistency = self._consistency_loss(pred_features)
        target_consistency = self._consistency_loss(target_features)
        
        # Combine losses
        stability_loss = (pred_stability + target_stability) * self.stability_weight
        consistency_loss = (pred_consistency + target_consistency) * self.consistency_weight
        
        persistence_loss = stability_loss + consistency_loss
        
        # Add to base loss if provided
        if base_loss is not None:
            total_loss = base_loss + persistence_loss
        else:
            total_loss = persistence_loss
        
        return {
            'total_loss': total_loss,
            'persistence_loss': persistence_loss,
            'stability_loss': stability_loss,
            'consistency_loss': consistency_loss,
            'base_loss': base_loss if base_loss is not None else torch.tensor(0.0)
        }


class TopologicalConsistencyLoss(nn.Module):
    """
    Loss function that enforces topological consistency between different
    frequency bands and scales in the KAN_TDA decomposition.
    """
    
    def __init__(
        self,
        consistency_weight: float = 1.0,
        cross_scale_weight: float = 0.5,
        frequency_weight: float = 0.3,
        device: torch.device = None
    ):
        """
        Initialize TopologicalConsistencyLoss.
        
        Args:
            consistency_weight: Weight for overall consistency term
            cross_scale_weight: Weight for cross-scale consistency
            frequency_weight: Weight for frequency band consistency
            device: Device for computation
        """
        super(TopologicalConsistencyLoss, self).__init__()
        
        self.consistency_weight = consistency_weight
        self.cross_scale_weight = cross_scale_weight
        self.frequency_weight = frequency_weight
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
    
    def _compute_topological_signature(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a simple topological signature for a time series.
        
        Args:
            x: Input tensor [B, T] or [B, T, C]
            
        Returns:
            Topological signature tensor
        """
        if x.dim() == 3:
            x = x.mean(dim=-1)  # Average across channels
        
        # Simple topological features
        # 1. Local extrema count (approximates 0-dimensional persistence)
        diff = torch.diff(x, dim=1)
        sign_changes = torch.diff(torch.sign(diff), dim=1)
        extrema_count = torch.sum(torch.abs(sign_changes) > 1, dim=1).float()
        
        # 2. Trend consistency (approximates global structure)
        trend = torch.mean(diff, dim=1)
        
        # 3. Variance (approximates complexity)
        variance = torch.var(x, dim=1)
        
        # 4. Autocorrelation at lag 1 (approximates periodicity)
        x_shifted = torch.roll(x, shifts=1, dims=1)
        autocorr = F.cosine_similarity(x[:, 1:], x_shifted[:, 1:], dim=1)
        
        # Combine into signature
        signature = torch.stack([
            extrema_count / x.shape[1],  # Normalized extrema density
            torch.tanh(trend),           # Bounded trend
            torch.log1p(variance),       # Log variance
            autocorr                     # Autocorrelation
        ], dim=1)
        
        return signature
    
    def _consistency_loss_between_signatures(
        self,
        sig1: torch.Tensor,
        sig2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute consistency loss between topological signatures.
        
        Args:
            sig1: First signature [B, D]
            sig2: Second signature [B, D]
            
        Returns:
            Consistency loss
        """
        # Use L2 distance between signatures
        return F.mse_loss(sig1, sig2)
    
    def forward(
        self,
        frequency_bands: List[torch.Tensor],
        original_signal: torch.Tensor,
        reconstructed_signal: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute topological consistency loss.
        
        Args:
            frequency_bands: List of frequency band tensors
            original_signal: Original input signal
            reconstructed_signal: Reconstructed signal from frequency bands
            
        Returns:
            Dictionary with loss components
        """
        losses = {}
        
        # 1. Consistency between original and reconstructed signals
        orig_signature = self._compute_topological_signature(original_signal)
        recon_signature = self._compute_topological_signature(reconstructed_signal)
        
        reconstruction_consistency = self._consistency_loss_between_signatures(
            orig_signature, recon_signature
        )
        losses['reconstruction_consistency'] = reconstruction_consistency
        
        # 2. Cross-scale consistency between frequency bands
        if len(frequency_bands) > 1:
            cross_scale_losses = []
            
            for i in range(len(frequency_bands)):
                for j in range(i + 1, len(frequency_bands)):
                    sig_i = self._compute_topological_signature(frequency_bands[i])
                    sig_j = self._compute_topological_signature(frequency_bands[j])
                    
                    # Consistency should be weighted by frequency relationship
                    weight = 1.0 / (abs(i - j) + 1)  # Closer bands should be more consistent
                    consistency = self._consistency_loss_between_signatures(sig_i, sig_j)
                    cross_scale_losses.append(weight * consistency)
            
            if cross_scale_losses:
                cross_scale_consistency = torch.stack(cross_scale_losses).mean()
                losses['cross_scale_consistency'] = cross_scale_consistency
            else:
                losses['cross_scale_consistency'] = torch.tensor(0.0, device=self.device)
        else:
            losses['cross_scale_consistency'] = torch.tensor(0.0, device=self.device)
        
        # 3. Frequency band preservation (each band should preserve some structure)
        if frequency_bands:
            frequency_preservation_losses = []
            
            for band in frequency_bands:
                band_signature = self._compute_topological_signature(band)
                
                # Penalize bands with too little structure (all features near zero)
                structure_penalty = torch.exp(-torch.norm(band_signature, dim=1)).mean()
                frequency_preservation_losses.append(structure_penalty)
            
            frequency_preservation = torch.stack(frequency_preservation_losses).mean()
            losses['frequency_preservation'] = frequency_preservation
        else:
            losses['frequency_preservation'] = torch.tensor(0.0, device=self.device)
        
        # Combine losses
        total_consistency_loss = (
            reconstruction_consistency +
            self.cross_scale_weight * losses['cross_scale_consistency'] +
            self.frequency_weight * losses['frequency_preservation']
        ) * self.consistency_weight
        
        losses['total_consistency_loss'] = total_consistency_loss
        
        return losses


class StructuralPreservationLoss(nn.Module):
    """
    Loss function that ensures the model preserves important structural
    properties of the time series during forecasting.
    """
    
    def __init__(
        self,
        preservation_weight: float = 1.0,
        trend_weight: float = 0.4,
        periodicity_weight: float = 0.3,
        complexity_weight: float = 0.3,
        device: torch.device = None
    ):
        """
        Initialize StructuralPreservationLoss.
        
        Args:
            preservation_weight: Overall weight for preservation loss
            trend_weight: Weight for trend preservation
            periodicity_weight: Weight for periodicity preservation
            complexity_weight: Weight for complexity preservation
            device: Device for computation
        """
        super(StructuralPreservationLoss, self).__init__()
        
        self.preservation_weight = preservation_weight
        self.trend_weight = trend_weight
        self.periodicity_weight = periodicity_weight
        self.complexity_weight = complexity_weight
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
    
    def _extract_trend(self, x: torch.Tensor, window_size: int = 5) -> torch.Tensor:
        """
        Extract trend component using moving average.
        
        Args:
            x: Input tensor [B, T] or [B, T, C]
            window_size: Window size for moving average
            
        Returns:
            Trend component
        """
        if x.dim() == 3:
            x = x.mean(dim=-1)  # Average across channels
        
        # Apply moving average
        kernel = torch.ones(window_size, device=x.device) / window_size
        
        # Pad the input
        padding = window_size // 2
        x_padded = F.pad(x, (padding, padding), mode='reflect')
        
        # Apply convolution for moving average
        trend = F.conv1d(
            x_padded.unsqueeze(1),
            kernel.unsqueeze(0).unsqueeze(0),
            padding=0
        ).squeeze(1)
        
        return trend
    
    def _extract_periodicity_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract periodicity features using FFT.
        
        Args:
            x: Input tensor [B, T]
            
        Returns:
            Periodicity features
        """
        # Compute FFT
        x_fft = torch.fft.fft(x, dim=1)
        power_spectrum = torch.abs(x_fft) ** 2
        
        # Extract dominant frequencies (top-k peaks)
        k = min(5, x.shape[1] // 4)  # Top 5 frequencies or quarter of sequence
        top_k_power, top_k_indices = torch.topk(power_spectrum, k, dim=1)
        
        # Normalize by total power
        total_power = torch.sum(power_spectrum, dim=1, keepdim=True)
        normalized_power = top_k_power / (total_power + 1e-8)
        
        return normalized_power
    
    def _compute_complexity_measure(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute complexity measure based on sample entropy approximation.
        
        Args:
            x: Input tensor [B, T]
            
        Returns:
            Complexity measure
        """
        # Simple complexity measure: variance of differences
        diff = torch.diff(x, dim=1)
        complexity = torch.var(diff, dim=1)
        
        # Add second-order differences for more complexity
        diff2 = torch.diff(diff, dim=1)
        complexity2 = torch.var(diff2, dim=1)
        
        # Combine measures
        total_complexity = complexity + 0.5 * complexity2
        
        return total_complexity
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        input_sequence: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute structural preservation loss.
        
        Args:
            predictions: Model predictions [B, T, C] or [B, T]
            targets: Target values [B, T, C] or [B, T]
            input_sequence: Optional input sequence for context
            
        Returns:
            Dictionary with loss components
        """
        # Handle different input shapes
        if predictions.dim() == 3:
            pred_series = predictions.mean(dim=-1)
            target_series = targets.mean(dim=-1)
        else:
            pred_series = predictions
            target_series = targets
        
        losses = {}
        
        # 1. Trend preservation
        pred_trend = self._extract_trend(pred_series)
        target_trend = self._extract_trend(target_series)
        
        trend_loss = F.mse_loss(pred_trend, target_trend)
        losses['trend_loss'] = trend_loss
        
        # 2. Periodicity preservation
        pred_periodicity = self._extract_periodicity_features(pred_series)
        target_periodicity = self._extract_periodicity_features(target_series)
        
        periodicity_loss = F.mse_loss(pred_periodicity, target_periodicity)
        losses['periodicity_loss'] = periodicity_loss
        
        # 3. Complexity preservation
        pred_complexity = self._compute_complexity_measure(pred_series)
        target_complexity = self._compute_complexity_measure(target_series)
        
        complexity_loss = F.mse_loss(pred_complexity, target_complexity)
        losses['complexity_loss'] = complexity_loss
        
        # 4. If input sequence is provided, ensure continuity
        if input_sequence is not None:
            if input_sequence.dim() == 3:
                input_series = input_sequence.mean(dim=-1)
            else:
                input_series = input_sequence
            
            # Continuity at the boundary
            last_input = input_series[:, -1]
            first_pred = pred_series[:, 0]
            
            continuity_loss = F.mse_loss(first_pred, last_input)
            losses['continuity_loss'] = continuity_loss
        else:
            losses['continuity_loss'] = torch.tensor(0.0, device=self.device)
        
        # Combine losses
        total_preservation_loss = (
            self.trend_weight * trend_loss +
            self.periodicity_weight * periodicity_loss +
            self.complexity_weight * complexity_loss +
            0.1 * losses['continuity_loss']  # Small weight for continuity
        ) * self.preservation_weight
        
        losses['total_preservation_loss'] = total_preservation_loss
        
        return losses


class AdaptiveLossWeighting(nn.Module):
    """
    Adaptive loss weighting mechanism that balances multiple loss components
    based on their relative magnitudes and training progress.
    """
    
    def __init__(
        self,
        loss_names: List[str],
        initial_weights: Optional[Dict[str, float]] = None,
        adaptation_rate: float = 0.01,
        min_weight: float = 0.01,
        max_weight: float = 10.0,
        device: torch.device = None
    ):
        """
        Initialize AdaptiveLossWeighting.
        
        Args:
            loss_names: Names of loss components to balance
            initial_weights: Initial weights for each loss component
            adaptation_rate: Rate of weight adaptation
            min_weight: Minimum allowed weight
            max_weight: Maximum allowed weight
            device: Device for computation
        """
        super(AdaptiveLossWeighting, self).__init__()
        
        self.loss_names = loss_names
        self.adaptation_rate = adaptation_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        # Initialize weights
        if initial_weights is None:
            initial_weights = {name: 1.0 for name in loss_names}
        
        # Store weights as parameters
        self.weights = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(initial_weights.get(name, 1.0), device=device))
            for name in loss_names
        })
        
        # Track loss history for adaptation
        self.loss_history = {name: [] for name in loss_names}
        self.step_count = 0
    
    def update_weights(self, loss_dict: Dict[str, torch.Tensor]):
        """
        Update loss weights based on current loss values.
        
        Args:
            loss_dict: Dictionary with current loss values
        """
        self.step_count += 1
        
        # Update loss history
        for name in self.loss_names:
            if name in loss_dict:
                loss_value = loss_dict[name].detach().item()
                self.loss_history[name].append(loss_value)
                
                # Keep only recent history
                if len(self.loss_history[name]) > 100:
                    self.loss_history[name] = self.loss_history[name][-100:]
        
        # Adapt weights based on relative loss magnitudes
        if self.step_count > 10:  # Start adaptation after some steps
            loss_magnitudes = {}
            
            for name in self.loss_names:
                if name in loss_dict and self.loss_history[name]:
                    # Use recent average magnitude
                    recent_losses = self.loss_history[name][-10:]
                    avg_magnitude = np.mean([abs(x) for x in recent_losses])
                    loss_magnitudes[name] = avg_magnitude
            
            if len(loss_magnitudes) > 1:
                # Compute relative scales
                max_magnitude = max(loss_magnitudes.values())
                
                for name in self.loss_names:
                    if name in loss_magnitudes:
                        relative_scale = loss_magnitudes[name] / (max_magnitude + 1e-8)
                        
                        # Adapt weight inversely to magnitude (balance losses)
                        target_weight = 1.0 / (relative_scale + 1e-8)
                        target_weight = np.clip(target_weight, self.min_weight, self.max_weight)
                        
                        # Smooth adaptation
                        current_weight = self.weights[name].item()
                        new_weight = (1 - self.adaptation_rate) * current_weight + \
                                   self.adaptation_rate * target_weight
                        
                        self.weights[name].data = torch.tensor(new_weight, device=self.device)
    
    def forward(self, loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute weighted combination of losses.
        
        Args:
            loss_dict: Dictionary with loss components
            
        Returns:
            Weighted total loss
        """
        # Update weights
        self.update_weights(loss_dict)
        
        # Compute weighted sum
        total_loss = torch.tensor(0.0, device=self.device)
        
        for name in self.loss_names:
            if name in loss_dict:
                weight = torch.clamp(self.weights[name], self.min_weight, self.max_weight)
                total_loss += weight * loss_dict[name]
        
        return total_loss
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current loss weights."""
        return {name: weight.item() for name, weight in self.weights.items()}
    
    def get_loss_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics about loss history."""
        stats = {}
        
        for name, history in self.loss_history.items():
            if history:
                stats[name] = {
                    'mean': np.mean(history),
                    'std': np.std(history),
                    'min': np.min(history),
                    'max': np.max(history),
                    'recent_mean': np.mean(history[-10:]) if len(history) >= 10 else np.mean(history)
                }
            else:
                stats[name] = {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'recent_mean': 0}
        
        return stats 