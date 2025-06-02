"""
Dynamic Architecture Adaptation for TDA-KAN_TDA Integration

This module provides adaptive architecture components that can dynamically adjust
the model complexity based on data characteristics and topological features.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from collections import defaultdict
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ArchitectureConfig:
    """Configuration for dynamic architecture adaptation."""
    
    # Base architecture parameters
    base_d_model: int = 16
    base_e_layers: int = 2
    base_down_sampling_layers: int = 0
    base_begin_order: int = 1
    
    # Adaptation parameters
    max_d_model: int = 128
    max_e_layers: int = 8
    max_down_sampling_layers: int = 4
    max_begin_order: int = 10
    
    # Complexity thresholds
    low_complexity_threshold: float = 0.3
    high_complexity_threshold: float = 0.7
    
    # Adaptation strategies
    adaptation_strategy: str = "gradual"  # gradual, aggressive, conservative
    enable_tda_guidance: bool = True
    enable_performance_feedback: bool = True
    
    # Resource constraints
    max_parameters: Optional[int] = None
    max_memory_mb: Optional[int] = None
    target_inference_time_ms: Optional[float] = None


class DataComplexityAnalyzer:
    """Analyzes time series data complexity to guide architecture adaptation."""
    
    def __init__(self, config: ArchitectureConfig):
        self.config = config
        self.complexity_cache = {}
        
    def analyze_complexity(self, x: torch.Tensor, tda_features: Optional[Dict] = None) -> Dict[str, float]:
        """
        Analyze the complexity of input time series data.
        
        Args:
            x: Input time series tensor [batch_size, seq_len, features]
            tda_features: Optional TDA features from TDAFeatureExtractor
            
        Returns:
            Dictionary containing various complexity metrics
        """
        # Create cache key
        cache_key = self._create_cache_key(x)
        if cache_key in self.complexity_cache:
            return self.complexity_cache[cache_key]
        
        complexity_metrics = {}
        
        # 1. Statistical complexity measures
        complexity_metrics.update(self._compute_statistical_complexity(x))
        
        # 2. Frequency domain complexity
        complexity_metrics.update(self._compute_frequency_complexity(x))
        
        # 3. Temporal dependency complexity
        complexity_metrics.update(self._compute_temporal_complexity(x))
        
        # 4. TDA-based complexity (if available)
        if tda_features is not None:
            complexity_metrics.update(self._compute_tda_complexity(tda_features))
        
        # 5. Overall complexity score
        complexity_metrics['overall_complexity'] = self._compute_overall_complexity(complexity_metrics)
        
        # Cache results
        self.complexity_cache[cache_key] = complexity_metrics
        
        return complexity_metrics
    
    def _create_cache_key(self, x: torch.Tensor) -> str:
        """Create a cache key for the input tensor."""
        return f"{x.shape}_{x.mean().item():.4f}_{x.std().item():.4f}"
    
    def _compute_statistical_complexity(self, x: torch.Tensor) -> Dict[str, float]:
        """Compute statistical complexity measures."""
        x_np = x.detach().cpu().numpy()
        
        metrics = {}
        
        # Variance-based complexity
        variance = np.var(x_np, axis=1).mean()
        metrics['variance_complexity'] = min(variance / (variance + 1.0), 1.0)
        
        # Skewness and kurtosis
        from scipy import stats
        skewness = np.abs(stats.skew(x_np, axis=1)).mean()
        kurtosis = np.abs(stats.kurtosis(x_np, axis=1)).mean()
        metrics['distribution_complexity'] = min((skewness + kurtosis) / 10.0, 1.0)
        
        # Range complexity
        ranges = np.ptp(x_np, axis=1).mean()
        metrics['range_complexity'] = min(ranges / (ranges + 1.0), 1.0)
        
        return metrics
    
    def _compute_frequency_complexity(self, x: torch.Tensor) -> Dict[str, float]:
        """Compute frequency domain complexity measures."""
        # FFT analysis
        x_fft = torch.fft.fft(x, dim=1)
        power_spectrum = torch.abs(x_fft) ** 2
        
        # Spectral entropy
        power_spectrum_norm = power_spectrum / (power_spectrum.sum(dim=1, keepdim=True) + 1e-8)
        spectral_entropy = -(power_spectrum_norm * torch.log(power_spectrum_norm + 1e-8)).sum(dim=1).mean()
        
        # Dominant frequency analysis
        dominant_freq_power = power_spectrum.max(dim=1)[0].mean()
        total_power = power_spectrum.mean()
        frequency_concentration = dominant_freq_power / (total_power + 1e-8)
        
        return {
            'spectral_entropy': min(spectral_entropy.item() / 10.0, 1.0),
            'frequency_concentration': min(frequency_concentration.item(), 1.0),
            'frequency_complexity': min(spectral_entropy.item() / 5.0, 1.0)
        }
    
    def _compute_temporal_complexity(self, x: torch.Tensor) -> Dict[str, float]:
        """Compute temporal dependency complexity measures."""
        # Autocorrelation analysis
        def autocorr(x, max_lag=20):
            x_centered = x - x.mean(dim=1, keepdim=True)
            autocorrs = []
            for lag in range(1, min(max_lag, x.size(1))):
                if lag >= x.size(1):
                    break
                corr = torch.corrcoef(torch.stack([
                    x_centered[:, :-lag].flatten(),
                    x_centered[:, lag:].flatten()
                ]))[0, 1]
                if not torch.isnan(corr):
                    autocorrs.append(corr.abs().item())
            return autocorrs
        
        autocorrs = autocorr(x)
        if autocorrs:
            temporal_persistence = np.mean(autocorrs)
            temporal_decay = np.mean(np.diff(autocorrs)) if len(autocorrs) > 1 else 0
        else:
            temporal_persistence = 0
            temporal_decay = 0
        
        # Trend analysis
        time_indices = torch.arange(x.size(1), dtype=torch.float32).unsqueeze(0).repeat(x.size(0), 1)
        trend_strength = torch.abs(torch.corrcoef(torch.stack([
            x.flatten(),
            time_indices.flatten()
        ]))[0, 1])
        
        if torch.isnan(trend_strength):
            trend_strength = torch.tensor(0.0)
        
        return {
            'temporal_persistence': min(temporal_persistence, 1.0),
            'temporal_decay': min(abs(temporal_decay), 1.0),
            'trend_strength': min(trend_strength.item(), 1.0),
            'temporal_complexity': min((temporal_persistence + abs(temporal_decay)) / 2.0, 1.0)
        }
    
    def _compute_tda_complexity(self, tda_features: Dict) -> Dict[str, float]:
        """Compute TDA-based complexity measures."""
        complexity = {}
        
        # Persistence-based complexity
        if 'persistence_diagrams' in tda_features:
            diagrams = tda_features['persistence_diagrams']
            if diagrams:
                # Count persistent features
                total_features = sum(len(d) for d in diagrams if d is not None)
                complexity['tda_feature_count'] = min(total_features / 100.0, 1.0)
                
                # Maximum persistence
                max_persistence = 0
                for d in diagrams:
                    if d is not None and len(d) > 0:
                        persistence_values = d[:, 1] - d[:, 0]  # death - birth
                        max_persistence = max(max_persistence, persistence_values.max().item())
                complexity['tda_max_persistence'] = min(max_persistence, 1.0)
        
        # Landscape-based complexity
        if 'landscape_features' in tda_features:
            landscapes = tda_features['landscape_features']
            if landscapes is not None:
                landscape_norm = torch.norm(landscapes).item()
                complexity['tda_landscape_complexity'] = min(landscape_norm / 10.0, 1.0)
        
        # Overall TDA complexity
        if complexity:
            complexity['tda_complexity'] = np.mean(list(complexity.values()))
        else:
            complexity['tda_complexity'] = 0.0
        
        return complexity
    
    def _compute_overall_complexity(self, metrics: Dict[str, float]) -> float:
        """Compute overall complexity score from individual metrics."""
        # Weight different complexity types
        weights = {
            'variance_complexity': 0.15,
            'distribution_complexity': 0.10,
            'frequency_complexity': 0.25,
            'temporal_complexity': 0.25,
            'tda_complexity': 0.25
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                weighted_sum += metrics[metric] * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0


class AdaptiveArchitecture(nn.Module):
    """
    Adaptive architecture that can dynamically adjust based on data complexity.
    """
    
    def __init__(self, config: ArchitectureConfig):
        super().__init__()
        self.config = config
        self.complexity_analyzer = DataComplexityAnalyzer(config)
        self.current_architecture = self._get_base_architecture()
        self.adaptation_history = []
        self.performance_history = []
        
    def _get_base_architecture(self) -> Dict[str, Any]:
        """Get base architecture configuration."""
        return {
            'd_model': self.config.base_d_model,
            'e_layers': self.config.base_e_layers,
            'down_sampling_layers': self.config.base_down_sampling_layers,
            'begin_order': self.config.base_begin_order
        }
    
    def determine_architecture(self, x: torch.Tensor, tda_features: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Determine optimal architecture based on data complexity.
        
        Args:
            x: Input time series tensor
            tda_features: Optional TDA features
            
        Returns:
            Dictionary containing architecture parameters
        """
        # Analyze data complexity
        complexity_metrics = self.complexity_analyzer.analyze_complexity(x, tda_features)
        overall_complexity = complexity_metrics['overall_complexity']
        
        # Determine architecture based on complexity
        architecture = self._adapt_architecture(overall_complexity, complexity_metrics)
        
        # Apply resource constraints
        architecture = self._apply_resource_constraints(architecture)
        
        # Log adaptation
        self._log_adaptation(complexity_metrics, architecture)
        
        # Update current architecture
        self.current_architecture = architecture
        self.adaptation_history.append({
            'complexity_metrics': complexity_metrics,
            'architecture': architecture.copy()
        })
        
        return architecture
    
    def _adapt_architecture(self, overall_complexity: float, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Adapt architecture based on complexity metrics."""
        architecture = self._get_base_architecture()
        
        # Determine adaptation level
        if overall_complexity < self.config.low_complexity_threshold:
            adaptation_factor = 0.5  # Reduce complexity
        elif overall_complexity > self.config.high_complexity_threshold:
            adaptation_factor = 1.5  # Increase complexity
        else:
            adaptation_factor = 1.0  # Keep base
        
        # Apply adaptation strategy
        if self.config.adaptation_strategy == "gradual":
            adaptation_factor = 1.0 + 0.3 * (adaptation_factor - 1.0)
        elif self.config.adaptation_strategy == "conservative":
            adaptation_factor = 1.0 + 0.1 * (adaptation_factor - 1.0)
        # aggressive uses full adaptation_factor
        
        # Adapt specific parameters
        
        # 1. Model dimension
        if 'frequency_complexity' in metrics and metrics['frequency_complexity'] > 0.6:
            architecture['d_model'] = min(
                int(self.config.base_d_model * adaptation_factor * 1.2),
                self.config.max_d_model
            )
        else:
            architecture['d_model'] = min(
                int(self.config.base_d_model * adaptation_factor),
                self.config.max_d_model
            )
        
        # 2. Number of layers
        if 'temporal_complexity' in metrics and metrics['temporal_complexity'] > 0.7:
            architecture['e_layers'] = min(
                int(self.config.base_e_layers * adaptation_factor * 1.3),
                self.config.max_e_layers
            )
        else:
            architecture['e_layers'] = min(
                int(self.config.base_e_layers * adaptation_factor),
                self.config.max_e_layers
            )
        
        # 3. Downsampling layers (frequency decomposition depth)
        if 'frequency_complexity' in metrics and metrics['frequency_complexity'] > 0.5:
            architecture['down_sampling_layers'] = min(
                int(self.config.base_down_sampling_layers + adaptation_factor),
                self.config.max_down_sampling_layers
            )
        
        # 4. KAN order
        if 'tda_complexity' in metrics and metrics['tda_complexity'] > 0.6:
            architecture['begin_order'] = min(
                int(self.config.base_begin_order * adaptation_factor * 1.1),
                self.config.max_begin_order
            )
        else:
            architecture['begin_order'] = min(
                int(self.config.base_begin_order * adaptation_factor),
                self.config.max_begin_order
            )
        
        return architecture
    
    def _apply_resource_constraints(self, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """Apply resource constraints to architecture."""
        # Estimate parameters
        estimated_params = self._estimate_parameters(architecture)
        
        # Apply parameter constraint
        if self.config.max_parameters and estimated_params > self.config.max_parameters:
            # Reduce architecture complexity
            reduction_factor = (self.config.max_parameters / estimated_params) ** 0.5
            architecture['d_model'] = max(int(architecture['d_model'] * reduction_factor), 8)
            architecture['e_layers'] = max(int(architecture['e_layers'] * reduction_factor), 1)
        
        # Apply memory constraint (simplified estimation)
        if self.config.max_memory_mb:
            estimated_memory = estimated_params * 4 / (1024 * 1024)  # 4 bytes per float32
            if estimated_memory > self.config.max_memory_mb:
                reduction_factor = (self.config.max_memory_mb / estimated_memory) ** 0.5
                architecture['d_model'] = max(int(architecture['d_model'] * reduction_factor), 8)
        
        return architecture
    
    def _estimate_parameters(self, architecture: Dict[str, Any]) -> int:
        """Estimate number of parameters for given architecture."""
        d_model = architecture['d_model']
        e_layers = architecture['e_layers']
        begin_order = architecture['begin_order']
        down_sampling_layers = architecture['down_sampling_layers']
        
        # Rough parameter estimation
        # Embedding layer
        params = d_model * 2  # value + temporal embedding
        
        # KAN layers (Chebyshev)
        kan_params_per_layer = d_model * d_model * (begin_order + 1)
        params += kan_params_per_layer * e_layers * (down_sampling_layers + 1)
        
        # Convolution layers
        conv_params_per_layer = d_model * 3  # depthwise conv
        params += conv_params_per_layer * e_layers * (down_sampling_layers + 1)
        
        # Output projection
        params += d_model * 1  # assuming single output
        
        return params
    
    def _log_adaptation(self, complexity_metrics: Dict[str, float], architecture: Dict[str, Any]):
        """Log architecture adaptation details."""
        logger.info(f"Architecture Adaptation:")
        logger.info(f"  Overall Complexity: {complexity_metrics['overall_complexity']:.3f}")
        logger.info(f"  Architecture: {architecture}")
        logger.info(f"  Estimated Parameters: {self._estimate_parameters(architecture):,}")
    
    def update_performance_feedback(self, performance_metrics: Dict[str, float]):
        """Update performance feedback for future adaptations."""
        if self.config.enable_performance_feedback:
            self.performance_history.append({
                'architecture': self.current_architecture.copy(),
                'performance': performance_metrics.copy()
            })
            
            # Keep only recent history
            if len(self.performance_history) > 10:
                self.performance_history = self.performance_history[-10:]
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get statistics about architecture adaptations."""
        if not self.adaptation_history:
            return {}
        
        stats = {
            'total_adaptations': len(self.adaptation_history),
            'complexity_range': {
                'min': min(h['complexity_metrics']['overall_complexity'] for h in self.adaptation_history),
                'max': max(h['complexity_metrics']['overall_complexity'] for h in self.adaptation_history),
                'mean': np.mean([h['complexity_metrics']['overall_complexity'] for h in self.adaptation_history])
            },
            'architecture_range': {
                'd_model': {
                    'min': min(h['architecture']['d_model'] for h in self.adaptation_history),
                    'max': max(h['architecture']['d_model'] for h in self.adaptation_history)
                },
                'e_layers': {
                    'min': min(h['architecture']['e_layers'] for h in self.adaptation_history),
                    'max': max(h['architecture']['e_layers'] for h in self.adaptation_history)
                }
            }
        }
        
        return stats


class ArchitectureOptimizer:
    """
    Optimizer for finding optimal architecture configurations.
    """
    
    def __init__(self, config: ArchitectureConfig):
        self.config = config
        self.optimization_history = []
        
    def optimize_architecture(self, 
                            data_samples: List[torch.Tensor],
                            tda_features_list: List[Dict],
                            validation_fn: callable,
                            max_iterations: int = 20) -> Dict[str, Any]:
        """
        Optimize architecture using validation feedback.
        
        Args:
            data_samples: List of data samples for testing
            tda_features_list: List of TDA features for each sample
            validation_fn: Function that returns performance metrics
            max_iterations: Maximum optimization iterations
            
        Returns:
            Optimal architecture configuration
        """
        adaptive_arch = AdaptiveArchitecture(self.config)
        best_architecture = None
        best_performance = float('-inf')
        
        for iteration in range(max_iterations):
            # Test current architecture on all samples
            total_performance = 0.0
            architecture_votes = defaultdict(list)
            
            for data, tda_features in zip(data_samples, tda_features_list):
                # Get architecture for this sample
                architecture = adaptive_arch.determine_architecture(data, tda_features)
                
                # Evaluate performance
                performance = validation_fn(architecture, data)
                total_performance += performance
                
                # Collect architecture votes
                for key, value in architecture.items():
                    architecture_votes[key].append(value)
            
            # Average performance
            avg_performance = total_performance / len(data_samples)
            
            # Create consensus architecture
            consensus_architecture = {}
            for key, values in architecture_votes.items():
                if isinstance(values[0], int):
                    consensus_architecture[key] = int(np.median(values))
                else:
                    consensus_architecture[key] = np.median(values)
            
            # Update best if improved
            if avg_performance > best_performance:
                best_performance = avg_performance
                best_architecture = consensus_architecture.copy()
            
            # Log progress
            logger.info(f"Optimization Iteration {iteration + 1}/{max_iterations}")
            logger.info(f"  Average Performance: {avg_performance:.4f}")
            logger.info(f"  Best Performance: {best_performance:.4f}")
            
            # Store history
            self.optimization_history.append({
                'iteration': iteration,
                'architecture': consensus_architecture,
                'performance': avg_performance
            })
            
            # Early stopping if performance plateaus
            if len(self.optimization_history) >= 5:
                recent_performances = [h['performance'] for h in self.optimization_history[-5:]]
                if max(recent_performances) - min(recent_performances) < 0.001:
                    logger.info("Performance plateaued, stopping optimization early")
                    break
        
        return best_architecture
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate optimization report."""
        if not self.optimization_history:
            return {}
        
        performances = [h['performance'] for h in self.optimization_history]
        
        return {
            'total_iterations': len(self.optimization_history),
            'best_performance': max(performances),
            'performance_improvement': max(performances) - performances[0] if len(performances) > 1 else 0,
            'convergence_iteration': np.argmax(performances) + 1,
            'final_architecture': self.optimization_history[-1]['architecture']
        }


# Convenience functions
def create_adaptive_architecture(complexity_threshold_low: float = 0.3,
                                complexity_threshold_high: float = 0.7,
                                adaptation_strategy: str = "gradual") -> AdaptiveArchitecture:
    """Create an adaptive architecture with common settings."""
    config = ArchitectureConfig(
        low_complexity_threshold=complexity_threshold_low,
        high_complexity_threshold=complexity_threshold_high,
        adaptation_strategy=adaptation_strategy
    )
    return AdaptiveArchitecture(config)


def analyze_data_complexity(x: torch.Tensor, tda_features: Optional[Dict] = None) -> Dict[str, float]:
    """Convenience function to analyze data complexity."""
    config = ArchitectureConfig()
    analyzer = DataComplexityAnalyzer(config)
    return analyzer.analyze_complexity(x, tda_features)


def optimize_architecture_for_dataset(data_samples: List[torch.Tensor],
                                     tda_features_list: List[Dict],
                                     validation_fn: callable) -> Dict[str, Any]:
    """Convenience function to optimize architecture for a dataset."""
    config = ArchitectureConfig()
    optimizer = ArchitectureOptimizer(config)
    return optimizer.optimize_architecture(data_samples, tda_features_list, validation_fn) 