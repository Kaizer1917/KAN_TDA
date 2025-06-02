"""
Hierarchical TDA Implementation for Multi-Resolution Analysis

This module implements hierarchical topological data analysis (TDA) that processes
time series data at multiple resolutions simultaneously, capturing patterns at
different temporal scales and providing cross-scale feature fusion.

Key Components:
- HierarchicalTDAProcessor: Main class for multi-resolution TDA
- AdaptiveResolutionSelector: Automatic resolution selection
- CrossScaleFusion: Feature fusion across resolutions
- ResolutionValidator: Validation and benchmarking tools

Mathematical Foundation:
The hierarchical approach applies TDA at multiple temporal resolutions:
- Fine resolution: Captures short-term patterns and noise
- Medium resolution: Captures periodic and oscillatory patterns  
- Coarse resolution: Captures long-term trends and structural changes

Author: TDA-KAN Integration Team
Date: Current Session
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from dataclasses import dataclass, field
import logging
from scipy import signal
import matplotlib.pyplot as plt
from collections import defaultdict
import time
from itertools import combinations
import torch.nn.functional as F

# Import TDA components
try:
    from .TDAFeatureExtractor import TDAFeatureExtractor, TDAConfig
    from .SpectralTDA import SpectralTDAProcessor, SpectralTDAConfig
    from .PersistentHomology import PersistentHomologyComputer
    from ..utils.persistence_landscapes import PersistenceLandscape, TopologicalFeatureExtractor
except ImportError:
    # Fallback for testing
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from layers.TDAFeatureExtractor import TDAFeatureExtractor, TDAConfig
    from layers.SpectralTDA import SpectralTDAProcessor, SpectralTDAConfig
    from layers.PersistentHomology import PersistentHomologyComputer
    from utils.persistence_landscapes import PersistenceLandscape, TopologicalFeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HierarchicalTDAConfig:
    """Configuration for hierarchical multi-resolution TDA analysis."""
    
    # Resolution parameters
    resolution_levels: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    min_resolution: int = 1
    max_resolution: int = 32
    adaptive_resolution: bool = True
    
    # Downsampling strategy
    downsampling_method: str = 'average'  # 'average', 'max', 'min', 'decimation'
    overlap_ratio: float = 0.5
    window_type: str = 'hann'
    
    # TDA parameters per resolution
    embedding_dims: List[int] = field(default_factory=lambda: [2, 3, 5])
    embedding_delays: List[int] = field(default_factory=lambda: [1, 2, 4])
    max_homology_dim: int = 2
    persistence_threshold: float = 0.01
    
    # Cross-scale analysis
    enable_cross_scale_fusion: bool = True
    fusion_strategy: str = 'attention'  # 'concat', 'attention', 'weighted'
    cross_scale_attention_heads: int = 4
    
    # Feature extraction
    extract_landscapes: bool = True
    landscape_resolution: int = 100
    extract_persistence_stats: bool = True
    extract_stability_features: bool = True
    
    # Performance options
    parallel_processing: bool = True
    enable_caching: bool = True
    gpu_acceleration: bool = True
    
    # Validation parameters
    information_criterion: str = 'aic'  # 'aic', 'bic', 'cross_validation'
    validation_split: float = 0.2
    complexity_penalty: float = 0.1
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Validate resolution levels
        if not self.resolution_levels:
            self.resolution_levels = [1, 2, 4, 8, 16]
        
        # Sort resolution levels
        self.resolution_levels = sorted(set(self.resolution_levels))
        
        # Validate downsampling method
        valid_methods = ['average', 'max', 'min', 'decimation']
        if self.downsampling_method not in valid_methods:
            raise ValueError(f"downsampling_method must be one of {valid_methods}")
        
        # Validate fusion strategy
        valid_strategies = ['concat', 'attention', 'weighted']
        if self.fusion_strategy not in valid_strategies:
            raise ValueError(f"fusion_strategy must be one of {valid_strategies}")
        
        # Ensure minimum resolution constraints
        self.min_resolution = max(1, self.min_resolution)
        self.max_resolution = max(self.min_resolution, self.max_resolution)


class HierarchicalTDAProcessor(nn.Module):
    """
    Hierarchical TDA processor for multi-resolution analysis.
    
    This class performs TDA analysis at multiple temporal resolutions,
    capturing patterns at different scales and providing unified
    cross-scale feature representations.
    """
    
    def __init__(self, config: Optional[HierarchicalTDAConfig] = None):
        super().__init__()
        self.config = config or HierarchicalTDAConfig()
        
        # Initialize TDA extractors for each resolution
        self.tda_extractors = nn.ModuleDict()
        self.spectral_processors = nn.ModuleDict()
        
        for resolution in self.config.resolution_levels:
            # Create TDA config for this resolution
            tda_config = TDAConfig(
                embedding_dims=self.config.embedding_dims,
                embedding_delays=[d * resolution for d in self.config.embedding_delays],
                max_homology_dim=self.config.max_homology_dim,
                enable_caching=self.config.enable_caching
            )
            
            # Create spectral TDA config
            spectral_config = SpectralTDAConfig(
                window_size=max(32, 64 // resolution),
                hop_length=max(16, 32 // resolution),
                enable_caching=self.config.enable_caching
            )
            
            self.tda_extractors[f'res_{resolution}'] = TDAFeatureExtractor(tda_config)
            self.spectral_processors[f'res_{resolution}'] = SpectralTDAProcessor(spectral_config)
        
        # Initialize cross-scale fusion components
        if self.config.enable_cross_scale_fusion:
            self._initialize_fusion_components()
        
        # Initialize adaptive resolution selector
        if self.config.adaptive_resolution:
            self.resolution_selector = AdaptiveResolutionSelector(self.config)
        
        # Performance tracking
        self.computation_stats = {
            'total_computations': 0,
            'resolution_usage': defaultdict(int),
            'average_computation_time': defaultdict(float),
            'cross_scale_fusion_time': 0.0,
            'cache_hits': defaultdict(int)
        }
        
        # Cache for computed results
        self.cache = {} if self.config.enable_caching else None
        
        logger.info(f"HierarchicalTDAProcessor initialized with {len(self.config.resolution_levels)} resolution levels")
    
    def _initialize_fusion_components(self):
        """Initialize cross-scale feature fusion components."""
        if self.config.fusion_strategy == 'attention':
            # Estimate feature dimensions for attention
            estimated_dim = len(self.config.embedding_dims) * len(self.config.embedding_delays) * 50
            
            self.cross_scale_attention = nn.MultiheadAttention(
                embed_dim=estimated_dim,
                num_heads=self.config.cross_scale_attention_heads,
                batch_first=True
            )
            
            # Projection layers for different resolutions
            self.resolution_projections = nn.ModuleDict()
            for resolution in self.config.resolution_levels:
                self.resolution_projections[f'res_{resolution}'] = nn.Linear(
                    estimated_dim, estimated_dim
                )
        
        elif self.config.fusion_strategy == 'weighted':
            # Learnable weights for each resolution
            self.resolution_weights = nn.Parameter(
                torch.ones(len(self.config.resolution_levels)) / len(self.config.resolution_levels)
            )
    
    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Perform hierarchical multi-resolution TDA analysis.
        
        Args:
            x: Input time series tensor [batch_size, seq_len, features]
            
        Returns:
            Dictionary containing multi-resolution TDA results
        """
        start_time = time.time()
        
        # Input validation
        if x.dim() < 2:
            x = x.unsqueeze(0)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        batch_size, seq_len, n_features = x.shape
        
        # Check cache
        cache_key = self._generate_cache_key(x)
        if self.cache is not None and cache_key in self.cache:
            self.computation_stats['cache_hits']['total'] += 1
            return self.cache[cache_key]
        
        # Adaptive resolution selection if enabled
        if self.config.adaptive_resolution:
            selected_resolutions = self.resolution_selector.select_resolutions(x)
        else:
            selected_resolutions = self.config.resolution_levels
        
        # Process at each resolution level
        resolution_results = {}
        
        for resolution in selected_resolutions:
            resolution_start = time.time()
            
            # Downsample input for this resolution
            downsampled_x = self._downsample_input(x, resolution)
            
            # Extract TDA features at this resolution
            tda_features = self.tda_extractors[f'res_{resolution}'](downsampled_x)
            
            # Extract spectral TDA features
            spectral_features = self.spectral_processors[f'res_{resolution}'].extract_features(
                downsampled_x.squeeze(-1) if downsampled_x.shape[-1] == 1 else downsampled_x[:, :, 0]
            )
            
            # Combine features for this resolution
            resolution_results[f'resolution_{resolution}'] = {
                'downsampled_input': downsampled_x,
                'tda_features': tda_features,
                'spectral_features': spectral_features,
                'resolution_level': resolution,
                'effective_length': downsampled_x.shape[1]
            }
            
            # Update statistics
            resolution_time = time.time() - resolution_start
            self.computation_stats['resolution_usage'][resolution] += 1
            self.computation_stats['average_computation_time'][resolution] = (
                (self.computation_stats['average_computation_time'][resolution] * 
                 (self.computation_stats['resolution_usage'][resolution] - 1) + resolution_time) /
                self.computation_stats['resolution_usage'][resolution]
            )
        
        # Cross-scale feature fusion
        if self.config.enable_cross_scale_fusion and len(resolution_results) > 1:
            fusion_start = time.time()
            fused_features = self._perform_cross_scale_fusion(resolution_results)
            resolution_results['cross_scale_fusion'] = fused_features
            self.computation_stats['cross_scale_fusion_time'] += time.time() - fusion_start
        
        # Extract hierarchical features
        hierarchical_features = self._extract_hierarchical_features(resolution_results)
        
        # Compile final results
        results = {
            'resolution_results': resolution_results,
            'hierarchical_features': hierarchical_features,
            'selected_resolutions': selected_resolutions,
            'computation_stats': {
                'total_time': time.time() - start_time,
                'num_resolutions': len(selected_resolutions),
                'input_shape': x.shape
            }
        }
        
        # Cache results
        if self.cache is not None:
            self.cache[cache_key] = results
        
        # Update global statistics
        self.computation_stats['total_computations'] += 1
        
        return results 

    def _downsample_input(self, x: torch.Tensor, resolution: int) -> torch.Tensor:
        """
        Downsample input time series for specified resolution level.
        
        Args:
            x: Input tensor [batch_size, seq_len, features]
            resolution: Downsampling factor
            
        Returns:
            Downsampled tensor
        """
        if resolution == 1:
            return x
        
        batch_size, seq_len, n_features = x.shape
        
        if self.config.downsampling_method == 'average':
            # Average pooling with overlap
            kernel_size = resolution
            stride = max(1, int(resolution * (1 - self.config.overlap_ratio)))
            
            # Reshape for pooling: [batch_size * features, 1, seq_len]
            x_reshaped = x.permute(0, 2, 1).contiguous().view(-1, 1, seq_len)
            
            # Apply average pooling
            pooled = F.avg_pool1d(x_reshaped, kernel_size=kernel_size, stride=stride)
            
            # Reshape back: [batch_size, new_seq_len, features]
            new_seq_len = pooled.shape[-1]
            downsampled = pooled.view(batch_size, n_features, new_seq_len).permute(0, 2, 1)
            
        elif self.config.downsampling_method == 'max':
            # Max pooling
            kernel_size = resolution
            stride = max(1, int(resolution * (1 - self.config.overlap_ratio)))
            
            x_reshaped = x.permute(0, 2, 1).contiguous().view(-1, 1, seq_len)
            pooled = F.max_pool1d(x_reshaped, kernel_size=kernel_size, stride=stride)
            
            new_seq_len = pooled.shape[-1]
            downsampled = pooled.view(batch_size, n_features, new_seq_len).permute(0, 2, 1)
            
        elif self.config.downsampling_method == 'min':
            # Min pooling (negative max pooling)
            kernel_size = resolution
            stride = max(1, int(resolution * (1 - self.config.overlap_ratio)))
            
            x_reshaped = x.permute(0, 2, 1).contiguous().view(-1, 1, seq_len)
            pooled = -F.max_pool1d(-x_reshaped, kernel_size=kernel_size, stride=stride)
            
            new_seq_len = pooled.shape[-1]
            downsampled = pooled.view(batch_size, n_features, new_seq_len).permute(0, 2, 1)
            
        elif self.config.downsampling_method == 'decimation':
            # Simple decimation (every nth sample)
            downsampled = x[:, ::resolution, :]
            
        else:
            raise ValueError(f"Unknown downsampling method: {self.config.downsampling_method}")
        
        return downsampled
    
    def _perform_cross_scale_fusion(self, resolution_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform cross-scale feature fusion across different resolutions.
        
        Args:
            resolution_results: Results from different resolution levels
            
        Returns:
            Fused features and fusion metadata
        """
        # Collect features from all resolutions
        features_list = self._collect_features_for_fusion(resolution_results)
        
        if not features_list:
            return {
                'fusion_method': self.config.fusion_strategy,
                'fused_features': torch.empty(0),
                'num_resolutions': 0
            }
        
        # Apply fusion strategy
        if self.config.fusion_strategy == 'concat':
            fused_features = self._fuse_features_concat(features_list)
        elif self.config.fusion_strategy == 'attention':
            fused_features = self._fuse_features_attention(features_list)
        elif self.config.fusion_strategy == 'weighted':
            fused_features = self._fuse_features_weighted(features_list)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.config.fusion_strategy}")
        
        return {
            'fusion_method': self.config.fusion_strategy,
            'fused_features': fused_features,
            'num_resolutions': len(features_list),
            'feature_shapes': [f.shape for f in features_list]
        }
    
    def _collect_features_for_fusion(self, resolution_results: Dict[str, Any]) -> List[torch.Tensor]:
        """Collect features from all resolutions for fusion."""
        features_list = []
        
        for res_key, res_data in resolution_results.items():
            if res_key.startswith('resolution_'):
                # Get TDA and spectral features
                tda_features = res_data.get('tda_features', torch.empty(0))
                spectral_features = res_data.get('spectral_features', torch.empty(0))
                
                # Ensure features are tensors
                if not isinstance(tda_features, torch.Tensor):
                    tda_features = torch.tensor(tda_features, dtype=torch.float32)
                if not isinstance(spectral_features, torch.Tensor):
                    spectral_features = torch.tensor(spectral_features, dtype=torch.float32)
                
                # Combine TDA and spectral features
                if tda_features.numel() > 0 and spectral_features.numel() > 0:
                    # Ensure both have batch dimension
                    if tda_features.dim() == 1:
                        tda_features = tda_features.unsqueeze(0)
                    if spectral_features.dim() == 1:
                        spectral_features = spectral_features.unsqueeze(0)
                    
                    # Flatten to 2D
                    tda_flat = tda_features.view(tda_features.shape[0], -1)
                    spectral_flat = spectral_features.view(spectral_features.shape[0], -1)
                    
                    # Concatenate TDA and spectral features
                    combined = torch.cat([tda_flat, spectral_flat], dim=1)
                    features_list.append(combined)
                elif tda_features.numel() > 0:
                    if tda_features.dim() == 1:
                        tda_features = tda_features.unsqueeze(0)
                    features_list.append(tda_features.view(tda_features.shape[0], -1))
                elif spectral_features.numel() > 0:
                    if spectral_features.dim() == 1:
                        spectral_features = spectral_features.unsqueeze(0)
                    features_list.append(spectral_features.view(spectral_features.shape[0], -1))
        
        return features_list
    
    def _fuse_features_attention(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        """Fuse features using attention mechanism."""
        if len(features_list) <= 1:
            return features_list[0] if features_list else torch.empty(0)
        
        # Ensure all features have the same batch size
        batch_size = features_list[0].shape[0]
        
        # Flatten all features to 2D and pad to same size
        flattened_features = []
        max_features = max(f.numel() // batch_size for f in features_list)
        
        for features in features_list:
            # Flatten to [batch_size, feature_dim]
            flat = features.view(batch_size, -1)
            
            # Pad to max_features if needed
            if flat.shape[1] < max_features:
                padding = torch.zeros(batch_size, max_features - flat.shape[1], 
                                    device=flat.device, dtype=flat.dtype)
                flat = torch.cat([flat, padding], dim=1)
            
            flattened_features.append(flat)
        
        # Stack features: [batch_size, num_resolutions, feature_dim]
        stacked = torch.stack(flattened_features, dim=1)
        
        # Ensure embed_dim is divisible by num_heads
        embed_dim = stacked.shape[2]
        num_heads = min(8, embed_dim)  # Start with 8 heads
        
        # Adjust num_heads to be a divisor of embed_dim
        while embed_dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        
        if num_heads < 1:
            num_heads = 1
        
        try:
            # Create attention layer
            attention = torch.nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                batch_first=True
            ).to(stacked.device)
            
            # Apply self-attention
            attended, _ = attention(stacked, stacked, stacked)
            
            # Aggregate across resolutions (mean pooling)
            fused = torch.mean(attended, dim=1)
            
        except Exception as e:
            # Fallback to simple concatenation if attention fails
            logger.warning(f"Attention fusion failed: {e}, falling back to concatenation")
            fused = torch.cat(flattened_features, dim=1)
        
        return fused
    
    def _fuse_features_weighted(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        """Fuse features using learned weights."""
        if len(features_list) <= 1:
            return features_list[0] if features_list else torch.empty(0)
        
        batch_size = features_list[0].shape[0]
        
        # Flatten all features to 2D
        flattened_features = []
        for features in features_list:
            flat = features.view(batch_size, -1)
            flattened_features.append(flat)
        
        # Simple weighted average (equal weights for now)
        weights = torch.ones(len(flattened_features), device=flattened_features[0].device)
        weights = weights / weights.sum()
        
        # Ensure all features have the same size by padding
        max_features = max(f.shape[1] for f in flattened_features)
        padded_features = []
        
        for features in flattened_features:
            if features.shape[1] < max_features:
                padding = torch.zeros(batch_size, max_features - features.shape[1], 
                                    device=features.device, dtype=features.dtype)
                features = torch.cat([features, padding], dim=1)
            padded_features.append(features)
        
        # Weighted sum
        fused = torch.zeros_like(padded_features[0])
        for weight, features in zip(weights, padded_features):
            fused += weight * features
        
        return fused
    
    def _fuse_features_concat(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        """Fuse features using concatenation."""
        if len(features_list) <= 1:
            return features_list[0] if features_list else torch.empty(0)
        
        batch_size = features_list[0].shape[0]
        
        # Flatten all features to 2D
        flattened_features = []
        for features in features_list:
            flat = features.view(batch_size, -1)
            flattened_features.append(flat)
        
        # Concatenate along feature dimension
        fused = torch.cat(flattened_features, dim=1)
        
        return fused
    
    def _extract_hierarchical_features(self, resolution_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features that capture hierarchical relationships across scales.
        
        Args:
            resolution_results: Results from all resolution levels
            
        Returns:
            Hierarchical features dictionary
        """
        hierarchical_features = {}
        
        # Extract resolution-specific statistics
        resolution_stats = {}
        for res_key, res_data in resolution_results.items():
            if res_key.startswith('resolution_'):
                resolution = res_data['resolution_level']
                
                # TDA feature statistics
                tda_feat = res_data['tda_features']
                if tda_feat.numel() > 0:
                    resolution_stats[resolution] = {
                        'tda_mean': torch.mean(tda_feat).item(),
                        'tda_std': torch.std(tda_feat).item(),
                        'tda_max': torch.max(tda_feat).item(),
                        'tda_min': torch.min(tda_feat).item(),
                        'effective_length': res_data['effective_length']
                    }
        
        # Cross-resolution consistency features
        if len(resolution_stats) >= 2:
            resolutions = sorted(resolution_stats.keys())
            
            # Compute consistency metrics
            tda_means = [resolution_stats[r]['tda_mean'] for r in resolutions]
            tda_stds = [resolution_stats[r]['tda_std'] for r in resolutions]
            
            hierarchical_features.update({
                'cross_resolution_mean_consistency': float(1.0 / (np.std(tda_means) + 1e-8)),
                'cross_resolution_std_consistency': float(1.0 / (np.std(tda_stds) + 1e-8)),
                'resolution_trend_slope': self._compute_trend_slope(resolutions, tda_means),
                'resolution_complexity_ratio': float(max(tda_stds) / (min(tda_stds) + 1e-8))
            })
            
            # Scale-dependent features
            hierarchical_features.update(self._extract_scale_dependent_features(resolution_stats))
        
        # Multi-scale persistence features
        if 'cross_scale_fusion' in resolution_results:
            fusion_data = resolution_results['cross_scale_fusion']
            if 'fused_features' in fusion_data:
                fused_feat = fusion_data['fused_features']
                hierarchical_features.update({
                    'fused_feature_magnitude': torch.norm(fused_feat).item(),
                    'fused_feature_sparsity': (fused_feat == 0).float().mean().item(),
                    'fusion_method': fusion_data['fusion_method']
                })
        
        return hierarchical_features
    
    def _compute_trend_slope(self, resolutions: List[int], values: List[float]) -> float:
        """Compute trend slope across resolutions."""
        if len(resolutions) < 2:
            return 0.0
        
        # Linear regression slope
        x = np.array(resolutions, dtype=float)
        y = np.array(values, dtype=float)
        
        if np.std(x) > 0 and np.std(y) > 0:
            correlation = np.corrcoef(x, y)[0, 1]
            slope = correlation * (np.std(y) / np.std(x))
            return float(slope) if not np.isnan(slope) else 0.0
        
        return 0.0
    
    def _extract_scale_dependent_features(self, resolution_stats: Dict[int, Dict]) -> Dict[str, float]:
        """Extract features that depend on scale relationships."""
        features = {}
        
        resolutions = sorted(resolution_stats.keys())
        
        # Scale-dependent complexity
        complexities = []
        for resolution in resolutions:
            stats = resolution_stats[resolution]
            # Use coefficient of variation as complexity measure
            complexity = stats['tda_std'] / (abs(stats['tda_mean']) + 1e-8)
            complexities.append(complexity)
        
        if complexities:
            features.update({
                'max_scale_complexity': float(max(complexities)),
                'min_scale_complexity': float(min(complexities)),
                'complexity_range': float(max(complexities) - min(complexities)),
                'complexity_trend': self._compute_trend_slope(resolutions, complexities)
            })
        
        # Scale-dependent information content
        information_contents = []
        for resolution in resolutions:
            stats = resolution_stats[resolution]
            # Use range as proxy for information content
            info_content = stats['tda_max'] - stats['tda_min']
            information_contents.append(info_content)
        
        if information_contents:
            features.update({
                'max_information_content': float(max(information_contents)),
                'information_decay_rate': self._compute_trend_slope(resolutions, information_contents),
                'information_concentration': float(max(information_contents) / (sum(information_contents) + 1e-8))
            })
        
        return features
    
    def _generate_cache_key(self, x: torch.Tensor) -> str:
        """Generate cache key for input tensor."""
        tensor_hash = hash(x.cpu().numpy().tobytes())
        return f"hierarchical_{tensor_hash}_{x.shape}"
    
    def get_computation_stats(self) -> Dict[str, Any]:
        """Get detailed computation statistics."""
        stats = dict(self.computation_stats)
        
        # Add derived statistics
        if stats['total_computations'] > 0:
            stats['average_resolutions_per_computation'] = (
                sum(stats['resolution_usage'].values()) / stats['total_computations']
            )
        
        # Cache statistics
        if self.cache is not None:
            stats['cache_size'] = len(self.cache)
            total_cache_requests = sum(stats['cache_hits'].values())
            if stats['total_computations'] > 0:
                stats['cache_hit_rate'] = total_cache_requests / stats['total_computations']
        
        return stats
    
    def clear_cache(self):
        """Clear computation cache."""
        if self.cache is not None:
            self.cache.clear()
            logger.info("Hierarchical TDA cache cleared")
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract hierarchical TDA features as tensor for ML integration.
        
        Args:
            x: Input time series tensor
            
        Returns:
            Feature tensor [batch_size, n_features]
        """
        results = self.forward(x)
        
        # Extract hierarchical features
        hierarchical_features = results['hierarchical_features']
        
        # Convert to tensor
        feature_values = []
        for key, value in hierarchical_features.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                feature_values.append(float(value))
        
        # Add cross-scale fusion features if available
        if 'cross_scale_fusion' in results['resolution_results']:
            fusion_data = results['resolution_results']['cross_scale_fusion']
            if 'fused_features' in fusion_data:
                fused_feat = fusion_data['fused_features']
                if fused_feat.numel() > 0:
                    # Add summary statistics of fused features
                    feature_values.extend([
                        torch.mean(fused_feat).item(),
                        torch.std(fused_feat).item(),
                        torch.max(fused_feat).item(),
                        torch.min(fused_feat).item(),
                        torch.norm(fused_feat).item()
                    ])
        
        # Ensure we have features
        if not feature_values:
            # Create default features
            batch_size = x.shape[0] if x.dim() >= 2 else 1
            feature_values = [0.0] * 20  # Default 20 features
        
        # Convert to tensor with proper batch dimension
        feature_tensor = torch.tensor(feature_values, dtype=torch.float32)
        batch_size = x.shape[0] if x.dim() >= 2 else 1
        
        if feature_tensor.dim() == 1:
            feature_tensor = feature_tensor.unsqueeze(0).expand(batch_size, -1)
        
        return feature_tensor 


class AdaptiveResolutionSelector:
    """
    Adaptive resolution selector for optimal multi-resolution TDA analysis.
    
    This class automatically selects the optimal set of resolution levels
    based on data characteristics, information content, and computational
    constraints.
    """
    
    def __init__(self, config: HierarchicalTDAConfig):
        self.config = config
        
        # Information-theoretic measures
        self.information_measures = {
            'entropy': self._compute_entropy,
            'mutual_information': self._compute_mutual_information,
            'complexity': self._compute_complexity
        }
        
        # Resolution selection history for learning
        self.selection_history = []
        self.performance_history = []
        
        logger.info("AdaptiveResolutionSelector initialized")
    
    def select_resolutions(self, x: torch.Tensor) -> List[int]:
        """
        Select optimal resolution levels for the given input.
        
        Args:
            x: Input time series tensor [batch_size, seq_len, features]
            
        Returns:
            List of selected resolution levels
        """
        # Analyze data characteristics
        data_characteristics = self._analyze_data_characteristics(x)
        
        # Generate candidate resolution sets
        candidate_sets = self._generate_candidate_resolution_sets(x.shape[1])
        
        # Evaluate each candidate set
        best_resolutions = self._evaluate_resolution_sets(
            candidate_sets, data_characteristics, x
        )
        
        # Store selection for learning
        self.selection_history.append({
            'input_shape': x.shape,
            'selected_resolutions': best_resolutions,
            'data_characteristics': data_characteristics
        })
        
        return best_resolutions
    
    def _analyze_data_characteristics(self, x: torch.Tensor) -> Dict[str, float]:
        """Analyze characteristics of input data for resolution selection."""
        characteristics = {}
        
        # Convert to numpy for analysis
        if x.dim() == 3:
            # Use first feature for analysis
            data = x[:, :, 0].cpu().numpy()
        else:
            data = x.cpu().numpy()
        
        # Temporal characteristics
        for i in range(data.shape[0]):
            signal = data[i]
            
            # Autocorrelation analysis
            autocorr = self._compute_autocorrelation(signal)
            characteristics[f'autocorr_decay_rate_{i}'] = self._compute_decay_rate(autocorr)
            characteristics[f'autocorr_periodicity_{i}'] = self._detect_periodicity(autocorr)
            
            # Frequency domain analysis
            fft_vals = np.abs(np.fft.fft(signal))
            characteristics[f'spectral_entropy_{i}'] = self._compute_spectral_entropy(fft_vals)
            characteristics[f'spectral_centroid_{i}'] = self._compute_spectral_centroid(fft_vals)
            
            # Complexity measures
            characteristics[f'sample_entropy_{i}'] = self._compute_sample_entropy(signal)
            characteristics[f'variance_{i}'] = float(np.var(signal))
            
            # Trend analysis
            characteristics[f'trend_strength_{i}'] = self._compute_trend_strength(signal)
        
        # Aggregate characteristics across batch
        aggregated = {}
        for key in characteristics:
            if not key.endswith('_0'):  # Skip first sample to avoid duplication
                continue
            base_key = key.replace('_0', '')
            values = [characteristics[k] for k in characteristics if k.startswith(base_key)]
            aggregated[f'{base_key}_mean'] = float(np.mean(values))
            aggregated[f'{base_key}_std'] = float(np.std(values))
        
        return aggregated
    
    def _generate_candidate_resolution_sets(self, seq_len: int) -> List[List[int]]:
        """Generate candidate resolution sets based on sequence length."""
        # Base resolution sets
        candidates = [
            [1, 2, 4],           # Fine-grained
            [1, 2, 4, 8],        # Balanced
            [1, 4, 8, 16],       # Coarse-grained
            [2, 4, 8],           # Medium resolution
            [1, 2, 4, 8, 16],    # Full spectrum
        ]
        
        # Adaptive candidates based on sequence length
        max_resolution = min(seq_len // 4, self.config.max_resolution)
        
        # Logarithmic spacing
        log_resolutions = []
        current = 1
        while current <= max_resolution:
            log_resolutions.append(current)
            current *= 2
        
        if len(log_resolutions) >= 3:
            candidates.append(log_resolutions)
        
        # Linear spacing for short sequences
        if seq_len < 100:
            linear_resolutions = list(range(1, min(seq_len // 8, 8) + 1))
            if len(linear_resolutions) >= 2:
                candidates.append(linear_resolutions)
        
        # Filter candidates to ensure valid resolutions
        valid_candidates = []
        for candidate in candidates:
            valid_candidate = [r for r in candidate if r <= max_resolution and r >= self.config.min_resolution]
            if len(valid_candidate) >= 2:  # Need at least 2 resolutions
                valid_candidates.append(sorted(set(valid_candidate)))
        
        return valid_candidates
    
    def _evaluate_resolution_sets(self, candidate_sets: List[List[int]], 
                                 data_characteristics: Dict[str, float],
                                 x: torch.Tensor) -> List[int]:
        """Evaluate candidate resolution sets and select the best one."""
        if not candidate_sets:
            return self.config.resolution_levels
        
        best_score = float('-inf')
        best_resolutions = candidate_sets[0]
        
        for resolutions in candidate_sets:
            score = self._score_resolution_set(resolutions, data_characteristics, x)
            
            if score > best_score:
                best_score = score
                best_resolutions = resolutions
        
        return best_resolutions
    
    def _score_resolution_set(self, resolutions: List[int], 
                             data_characteristics: Dict[str, float],
                             x: torch.Tensor) -> float:
        """Score a resolution set based on information content and efficiency."""
        # Information coverage score
        info_score = self._compute_information_coverage_score(resolutions, data_characteristics)
        
        # Computational efficiency score
        efficiency_score = self._compute_efficiency_score(resolutions, x.shape[1])
        
        # Resolution diversity score
        diversity_score = self._compute_diversity_score(resolutions)
        
        # Combine scores with weights
        total_score = (
            0.5 * info_score +
            0.3 * efficiency_score +
            0.2 * diversity_score
        )
        
        return total_score
    
    def _compute_information_coverage_score(self, resolutions: List[int], 
                                          data_characteristics: Dict[str, float]) -> float:
        """Compute how well the resolution set covers information content."""
        # Extract relevant characteristics
        autocorr_decay = data_characteristics.get('autocorr_decay_rate_mean', 0.5)
        periodicity = data_characteristics.get('autocorr_periodicity_mean', 0.0)
        spectral_entropy = data_characteristics.get('spectral_entropy_mean', 1.0)
        
        # Score based on resolution coverage
        min_res, max_res = min(resolutions), max(resolutions)
        
        # Coverage of temporal scales
        temporal_coverage = min(max_res / 16.0, 1.0)  # Prefer higher max resolution
        
        # Fine-scale coverage (important for high-frequency patterns)
        fine_coverage = 1.0 / min_res  # Prefer lower min resolution
        
        # Periodicity matching
        if periodicity > 0:
            # Try to match detected periodicity with resolution levels
            period_match = max([1.0 / abs(r - periodicity) for r in resolutions if abs(r - periodicity) > 0] + [0.1])
        else:
            period_match = 0.5
        
        # Entropy-based weighting
        entropy_weight = min(spectral_entropy, 1.0)
        
        score = entropy_weight * (0.4 * temporal_coverage + 0.4 * fine_coverage + 0.2 * period_match)
        return min(score, 1.0)
    
    def _compute_efficiency_score(self, resolutions: List[int], seq_len: int) -> float:
        """Compute computational efficiency score for resolution set."""
        # Estimate computational cost
        total_cost = 0
        for resolution in resolutions:
            downsampled_length = seq_len // resolution
            # Cost roughly proportional to length squared (for TDA computation)
            cost = downsampled_length ** 1.5
            total_cost += cost
        
        # Normalize by single resolution cost
        single_res_cost = seq_len ** 1.5
        normalized_cost = total_cost / single_res_cost
        
        # Efficiency score (lower cost = higher score)
        efficiency = 1.0 / (1.0 + normalized_cost)
        
        # Penalty for too many resolutions
        num_resolutions = len(resolutions)
        if num_resolutions > 5:
            efficiency *= 0.8 ** (num_resolutions - 5)
        
        return efficiency
    
    def _compute_diversity_score(self, resolutions: List[int]) -> float:
        """Compute diversity score for resolution set."""
        if len(resolutions) < 2:
            return 0.0
        
        # Geometric spacing score (prefer exponential spacing)
        ratios = []
        for i in range(1, len(resolutions)):
            ratio = resolutions[i] / resolutions[i-1]
            ratios.append(ratio)
        
        # Prefer consistent ratios around 2-4
        target_ratio = 2.0
        ratio_consistency = 1.0 / (1.0 + np.std(ratios))
        ratio_optimality = 1.0 / (1.0 + abs(np.mean(ratios) - target_ratio))
        
        # Range coverage
        range_coverage = (max(resolutions) - min(resolutions)) / max(resolutions)
        
        diversity = 0.4 * ratio_consistency + 0.3 * ratio_optimality + 0.3 * range_coverage
        return min(diversity, 1.0)
    
    def _compute_autocorrelation(self, signal: np.ndarray, max_lag: int = None) -> np.ndarray:
        """Compute autocorrelation function."""
        if max_lag is None:
            max_lag = min(len(signal) // 4, 50)
        
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr[:max_lag] / autocorr[0]  # Normalize
        
        return autocorr
    
    def _compute_decay_rate(self, autocorr: np.ndarray) -> float:
        """Compute decay rate of autocorrelation."""
        if len(autocorr) < 2:
            return 0.0
        
        # Find where autocorrelation drops to 1/e
        threshold = 1.0 / np.e
        decay_idx = np.where(autocorr < threshold)[0]
        
        if len(decay_idx) > 0:
            return float(1.0 / (decay_idx[0] + 1))
        else:
            return 0.1  # Very slow decay
    
    def _detect_periodicity(self, autocorr: np.ndarray) -> float:
        """Detect dominant periodicity from autocorrelation."""
        if len(autocorr) < 4:
            return 0.0
        
        # Find peaks in autocorrelation (excluding lag 0)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(autocorr[1:], height=0.1)
        
        if len(peaks) > 0:
            # Return the lag of the highest peak
            peak_heights = autocorr[peaks + 1]
            dominant_peak = peaks[np.argmax(peak_heights)] + 1
            return float(dominant_peak)
        
        return 0.0
    
    def _compute_spectral_entropy(self, fft_vals: np.ndarray) -> float:
        """Compute spectral entropy."""
        # Use only positive frequencies
        power = fft_vals[:len(fft_vals)//2] ** 2
        power = power / np.sum(power)  # Normalize
        
        # Compute entropy
        entropy = -np.sum(power * np.log(power + 1e-10))
        return float(entropy / np.log(len(power)))  # Normalize by max entropy
    
    def _compute_spectral_centroid(self, fft_vals: np.ndarray) -> float:
        """Compute spectral centroid (center of mass of spectrum)."""
        power = fft_vals[:len(fft_vals)//2] ** 2
        freqs = np.arange(len(power))
        
        if np.sum(power) > 0:
            centroid = np.sum(freqs * power) / np.sum(power)
            return float(centroid / len(power))  # Normalize
        
        return 0.5
    
    def _compute_sample_entropy(self, signal: np.ndarray, m: int = 2, r: float = None) -> float:
        """Compute sample entropy as complexity measure."""
        if r is None:
            r = 0.2 * np.std(signal)
        
        if len(signal) < m + 1:
            return 0.0
        
        def _maxdist(xi, xj, m):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            patterns = np.array([signal[i:i+m] for i in range(len(signal) - m + 1)])
            C = np.zeros(len(patterns))
            
            for i in range(len(patterns)):
                template = patterns[i]
                for j in range(len(patterns)):
                    if _maxdist(template, patterns[j], m) <= r:
                        C[i] += 1.0
            
            phi = np.mean(np.log(C / len(patterns)))
            return phi
        
        try:
            return float(_phi(m) - _phi(m + 1))
        except:
            return 0.0
    
    def _compute_trend_strength(self, signal: np.ndarray) -> float:
        """Compute trend strength using linear regression."""
        if len(signal) < 3:
            return 0.0
        
        x = np.arange(len(signal))
        
        # Linear regression
        correlation = np.corrcoef(x, signal)[0, 1]
        
        return float(abs(correlation)) if not np.isnan(correlation) else 0.0
    
    def update_performance_feedback(self, resolutions: List[int], performance_metric: float):
        """Update performance feedback for learning."""
        self.performance_history.append({
            'resolutions': resolutions,
            'performance': performance_metric,
            'timestamp': time.time()
        })
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
    
    def get_selection_statistics(self) -> Dict[str, Any]:
        """Get statistics about resolution selection."""
        if not self.selection_history:
            return {}
        
        # Resolution usage statistics
        all_resolutions = []
        for selection in self.selection_history:
            all_resolutions.extend(selection['selected_resolutions'])
        
        from collections import Counter
        resolution_counts = Counter(all_resolutions)
        
        stats = {
            'total_selections': len(self.selection_history),
            'resolution_usage': dict(resolution_counts),
            'average_num_resolutions': np.mean([len(s['selected_resolutions']) for s in self.selection_history]),
            'most_common_resolutions': resolution_counts.most_common(5)
        }
        
        # Performance statistics if available
        if self.performance_history:
            performances = [p['performance'] for p in self.performance_history]
            stats.update({
                'average_performance': np.mean(performances),
                'performance_std': np.std(performances),
                'best_performance': max(performances),
                'performance_trend': self._compute_trend_slope(
                    list(range(len(performances))), performances
                )
            })
        
        return stats 

    def _compute_entropy(self, data: np.ndarray) -> float:
        """Compute Shannon entropy of data."""
        # Discretize data into bins
        hist, _ = np.histogram(data, bins=20, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        
        if len(hist) == 0:
            return 0.0
        
        # Normalize to probabilities
        probs = hist / np.sum(hist)
        
        # Compute entropy
        entropy = -np.sum(probs * np.log2(probs))
        return float(entropy)
    
    def _compute_mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute mutual information between two signals."""
        # Simple implementation using histograms
        try:
            hist_xy, _, _ = np.histogram2d(x, y, bins=10)
            hist_x, _ = np.histogram(x, bins=10)
            hist_y, _ = np.histogram(y, bins=10)
            
            # Normalize to probabilities
            p_xy = hist_xy / np.sum(hist_xy)
            p_x = hist_x / np.sum(hist_x)
            p_y = hist_y / np.sum(hist_y)
            
            # Compute mutual information
            mi = 0.0
            for i in range(len(p_x)):
                for j in range(len(p_y)):
                    if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                        mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
            
            return float(mi)
        except:
            return 0.0
    
    def _compute_complexity(self, data: np.ndarray) -> float:
        """Compute complexity measure of data."""
        # Use Lempel-Ziv complexity approximation
        # Convert to binary string for complexity analysis
        median_val = np.median(data)
        binary_string = ''.join(['1' if x > median_val else '0' for x in data])
        
        # Count unique substrings (approximation of LZ complexity)
        substrings = set()
        for i in range(len(binary_string)):
            for j in range(i+1, min(i+10, len(binary_string)+1)):  # Limit substring length
                substrings.add(binary_string[i:j])
        
        # Normalize by maximum possible complexity
        max_complexity = len(binary_string) * (len(binary_string) + 1) // 2
        complexity = len(substrings) / max_complexity if max_complexity > 0 else 0.0
        
        return float(complexity)
    
    def _compute_trend_slope(self, x_values: List[float], y_values: List[float]) -> float:
        """Compute trend slope using linear regression."""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0
        
        x = np.array(x_values, dtype=float)
        y = np.array(y_values, dtype=float)
        
        if np.std(x) > 0 and np.std(y) > 0:
            correlation = np.corrcoef(x, y)[0, 1]
            slope = correlation * (np.std(y) / np.std(x))
            return float(slope) if not np.isnan(slope) else 0.0
        
        return 0.0


class ResolutionValidator:
    """
    Validator for multi-resolution TDA analysis.
    
    This class provides validation and benchmarking tools for evaluating
    the effectiveness of different resolution selection strategies.
    """
    
    def __init__(self, config: HierarchicalTDAConfig):
        self.config = config
        self.validation_results = []
        
        logger.info("ResolutionValidator initialized")
    
    def validate_resolution_selection(self, processor: HierarchicalTDAProcessor,
                                    test_data: torch.Tensor,
                                    ground_truth: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Validate resolution selection effectiveness.
        
        Args:
            processor: HierarchicalTDAProcessor to validate
            test_data: Test time series data
            ground_truth: Optional ground truth for supervised validation
            
        Returns:
            Validation results dictionary
        """
        validation_results = {}
        
        # Test different resolution strategies
        strategies = [
            ('fixed_fine', [1, 2, 4]),
            ('fixed_coarse', [4, 8, 16]),
            ('fixed_balanced', [1, 4, 8, 16]),
            ('adaptive', None)  # Use adaptive selection
        ]
        
        for strategy_name, fixed_resolutions in strategies:
            # Temporarily override resolution selection
            original_adaptive = processor.config.adaptive_resolution
            original_resolutions = processor.config.resolution_levels
            
            if fixed_resolutions is not None:
                processor.config.adaptive_resolution = False
                processor.config.resolution_levels = fixed_resolutions
            else:
                processor.config.adaptive_resolution = True
            
            # Run analysis
            start_time = time.time()
            results = processor.forward(test_data)
            computation_time = time.time() - start_time
            
            # Extract features for evaluation
            features = processor.extract_features(test_data)
            
            # Evaluate results
            strategy_results = {
                'computation_time': computation_time,
                'num_resolutions': len(results['selected_resolutions']),
                'selected_resolutions': results['selected_resolutions'],
                'feature_dimensionality': features.shape[1] if len(features.shape) > 1 else features.numel(),
                'hierarchical_features': results['hierarchical_features']
            }
            
            # Information content analysis
            strategy_results['information_metrics'] = self._compute_information_metrics(features)
            
            # Computational efficiency
            strategy_results['efficiency_metrics'] = self._compute_efficiency_metrics(
                results, test_data.shape[1]
            )
            
            validation_results[strategy_name] = strategy_results
            
            # Restore original settings
            processor.config.adaptive_resolution = original_adaptive
            processor.config.resolution_levels = original_resolutions
        
        # Compare strategies
        comparison = self._compare_strategies(validation_results)
        validation_results['strategy_comparison'] = comparison
        
        # Store results
        self.validation_results.append({
            'timestamp': time.time(),
            'test_data_shape': test_data.shape,
            'results': validation_results
        })
        
        return validation_results
    
    def _compute_information_metrics(self, features: torch.Tensor) -> Dict[str, float]:
        """Compute information content metrics for features."""
        if features.numel() == 0:
            return {'entropy': 0.0, 'variance': 0.0, 'range': 0.0}
        
        features_np = features.cpu().numpy().flatten()
        
        metrics = {
            'entropy': self._compute_entropy_from_features(features_np),
            'variance': float(np.var(features_np)),
            'range': float(np.max(features_np) - np.min(features_np)),
            'mean_absolute_value': float(np.mean(np.abs(features_np))),
            'sparsity': float(np.mean(features_np == 0)),
            'kurtosis': float(self._compute_kurtosis(features_np))
        }
        
        return metrics
    
    def _compute_efficiency_metrics(self, results: Dict[str, Any], seq_len: int) -> Dict[str, float]:
        """Compute computational efficiency metrics."""
        computation_stats = results.get('computation_stats', {})
        
        metrics = {
            'total_time': computation_stats.get('total_time', 0.0),
            'time_per_resolution': 0.0,
            'time_per_sample': 0.0,
            'memory_efficiency': 1.0  # Placeholder
        }
        
        num_resolutions = computation_stats.get('num_resolutions', 1)
        if num_resolutions > 0:
            metrics['time_per_resolution'] = metrics['total_time'] / num_resolutions
        
        if seq_len > 0:
            metrics['time_per_sample'] = metrics['total_time'] / seq_len
        
        return metrics
    
    def _compare_strategies(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare different resolution selection strategies."""
        comparison = {
            'best_information_content': None,
            'best_efficiency': None,
            'best_overall': None,
            'rankings': {}
        }
        
        strategies = [k for k in validation_results.keys() if k != 'strategy_comparison']
        
        if not strategies:
            return comparison
        
        # Rank by information content
        info_scores = {}
        for strategy in strategies:
            info_metrics = validation_results[strategy]['information_metrics']
            # Combine entropy and variance as information score
            info_score = info_metrics['entropy'] + 0.5 * info_metrics['variance']
            info_scores[strategy] = info_score
        
        best_info = max(info_scores, key=info_scores.get)
        comparison['best_information_content'] = best_info
        
        # Rank by efficiency
        efficiency_scores = {}
        for strategy in strategies:
            eff_metrics = validation_results[strategy]['efficiency_metrics']
            # Lower time is better
            efficiency_score = 1.0 / (eff_metrics['total_time'] + 1e-6)
            efficiency_scores[strategy] = efficiency_score
        
        best_efficiency = max(efficiency_scores, key=efficiency_scores.get)
        comparison['best_efficiency'] = best_efficiency
        
        # Overall ranking (weighted combination)
        overall_scores = {}
        for strategy in strategies:
            info_score = info_scores[strategy]
            eff_score = efficiency_scores[strategy]
            # Normalize scores
            norm_info = info_score / max(info_scores.values()) if max(info_scores.values()) > 0 else 0
            norm_eff = eff_score / max(efficiency_scores.values()) if max(efficiency_scores.values()) > 0 else 0
            
            overall_score = 0.6 * norm_info + 0.4 * norm_eff
            overall_scores[strategy] = overall_score
        
        best_overall = max(overall_scores, key=overall_scores.get)
        comparison['best_overall'] = best_overall
        
        # Detailed rankings
        comparison['rankings'] = {
            'information_content': sorted(strategies, key=lambda s: info_scores[s], reverse=True),
            'efficiency': sorted(strategies, key=lambda s: efficiency_scores[s], reverse=True),
            'overall': sorted(strategies, key=lambda s: overall_scores[s], reverse=True)
        }
        
        return comparison
    
    def _compute_entropy_from_features(self, features: np.ndarray) -> float:
        """Compute entropy from feature vector."""
        if len(features) == 0:
            return 0.0
        
        # Discretize features
        hist, _ = np.histogram(features, bins=min(20, len(features)), density=True)
        hist = hist[hist > 0]
        
        if len(hist) == 0:
            return 0.0
        
        # Normalize and compute entropy
        probs = hist / np.sum(hist)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        return float(entropy)
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis (measure of tail heaviness)."""
        if len(data) < 4:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        # Fourth moment
        fourth_moment = np.mean(((data - mean) / std) ** 4)
        kurtosis = fourth_moment - 3  # Excess kurtosis
        
        return float(kurtosis)
    
    def benchmark_performance(self, processor: HierarchicalTDAProcessor,
                            benchmark_datasets: List[torch.Tensor]) -> Dict[str, Any]:
        """
        Benchmark performance across multiple datasets.
        
        Args:
            processor: HierarchicalTDAProcessor to benchmark
            benchmark_datasets: List of test datasets
            
        Returns:
            Benchmark results
        """
        benchmark_results = {
            'dataset_results': [],
            'aggregate_metrics': {},
            'scalability_analysis': {}
        }
        
        for i, dataset in enumerate(benchmark_datasets):
            dataset_result = self.validate_resolution_selection(processor, dataset)
            dataset_result['dataset_id'] = i
            dataset_result['dataset_shape'] = dataset.shape
            
            benchmark_results['dataset_results'].append(dataset_result)
        
        # Aggregate metrics across datasets
        benchmark_results['aggregate_metrics'] = self._aggregate_benchmark_metrics(
            benchmark_results['dataset_results']
        )
        
        # Scalability analysis
        benchmark_results['scalability_analysis'] = self._analyze_scalability(
            benchmark_results['dataset_results']
        )
        
        return benchmark_results
    
    def _aggregate_benchmark_metrics(self, dataset_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate metrics across multiple datasets."""
        if not dataset_results:
            return {}
        
        # Collect metrics from all datasets and strategies
        all_metrics = defaultdict(list)
        
        for dataset_result in dataset_results:
            for strategy, strategy_result in dataset_result.items():
                if strategy == 'strategy_comparison':
                    continue
                
                # Information metrics
                info_metrics = strategy_result.get('information_metrics', {})
                for metric, value in info_metrics.items():
                    all_metrics[f'{strategy}_{metric}'].append(value)
                
                # Efficiency metrics
                eff_metrics = strategy_result.get('efficiency_metrics', {})
                for metric, value in eff_metrics.items():
                    all_metrics[f'{strategy}_{metric}'].append(value)
        
        # Compute aggregate statistics
        aggregated = {}
        for metric_name, values in all_metrics.items():
            if values:
                aggregated[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values))
                }
        
        return aggregated
    
    def _analyze_scalability(self, dataset_results: List[Dict]) -> Dict[str, Any]:
        """Analyze scalability with respect to dataset size."""
        scalability = {
            'time_complexity': {},
            'memory_complexity': {},
            'feature_scaling': {}
        }
        
        # Extract dataset sizes and corresponding metrics
        sizes = []
        times = defaultdict(list)
        feature_dims = defaultdict(list)
        
        for result in dataset_results:
            dataset_shape = result.get('dataset_shape', (0, 0, 0))
            seq_len = dataset_shape[1] if len(dataset_shape) > 1 else 0
            sizes.append(seq_len)
            
            for strategy, strategy_result in result.items():
                if strategy == 'strategy_comparison':
                    continue
                
                time_val = strategy_result.get('computation_time', 0.0)
                feature_dim = strategy_result.get('feature_dimensionality', 0)
                
                times[strategy].append(time_val)
                feature_dims[strategy].append(feature_dim)
        
        # Analyze time complexity
        for strategy in times:
            if len(sizes) > 1 and len(times[strategy]) > 1:
                # Fit power law: time = a * size^b
                log_sizes = np.log(np.array(sizes) + 1)
                log_times = np.log(np.array(times[strategy]) + 1e-6)
                
                if np.std(log_sizes) > 0:
                    correlation = np.corrcoef(log_sizes, log_times)[0, 1]
                    slope = correlation * (np.std(log_times) / np.std(log_sizes))
                    
                    scalability['time_complexity'][strategy] = {
                        'exponent': float(slope),
                        'correlation': float(correlation)
                    }
        
        # Analyze feature scaling
        for strategy in feature_dims:
            if len(sizes) > 1 and len(feature_dims[strategy]) > 1:
                correlation = np.corrcoef(sizes, feature_dims[strategy])[0, 1]
                scalability['feature_scaling'][strategy] = {
                    'correlation_with_size': float(correlation) if not np.isnan(correlation) else 0.0
                }
        
        return scalability
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation results."""
        if not self.validation_results:
            return {}
        
        summary = {
            'total_validations': len(self.validation_results),
            'strategies_tested': set(),
            'average_performance': {},
            'best_configurations': {}
        }
        
        # Collect all strategies tested
        for validation in self.validation_results:
            for strategy in validation['results']:
                if strategy != 'strategy_comparison':
                    summary['strategies_tested'].add(strategy)
        
        summary['strategies_tested'] = list(summary['strategies_tested'])
        
        return summary


# Utility functions
def create_hierarchical_tda_pipeline(config: Optional[HierarchicalTDAConfig] = None) -> HierarchicalTDAProcessor:
    """
    Factory function to create hierarchical TDA pipeline.
    
    Args:
        config: Configuration for hierarchical TDA analysis
        
    Returns:
        HierarchicalTDAProcessor instance
    """
    return HierarchicalTDAProcessor(config or HierarchicalTDAConfig())


def validate_hierarchical_tda(processor: HierarchicalTDAProcessor,
                             test_patterns: List[str] = None) -> Dict[str, Any]:
    """
    Validate hierarchical TDA processor on known patterns.
    
    Args:
        processor: HierarchicalTDAProcessor to validate
        test_patterns: List of test pattern types
        
    Returns:
        Validation results
    """
    if test_patterns is None:
        test_patterns = ['sine_multi_scale', 'trend_with_noise', 'periodic_burst']
    
    validation_results = {}
    
    for pattern_type in test_patterns:
        # Generate test pattern
        test_data = _generate_test_pattern(pattern_type)
        
        # Run hierarchical analysis
        start_time = time.time()
        results = processor.forward(test_data)
        analysis_time = time.time() - start_time
        
        # Extract features
        features = processor.extract_features(test_data)
        
        # Validate results
        pattern_results = {
            'pattern_type': pattern_type,
            'analysis_time': analysis_time,
            'selected_resolutions': results['selected_resolutions'],
            'feature_count': features.shape[1] if len(features.shape) > 1 else features.numel(),
            'hierarchical_features': results['hierarchical_features']
        }
        
        # Pattern-specific validation
        if pattern_type == 'sine_multi_scale':
            pattern_results['expected_behavior'] = 'Should detect multiple frequency components'
        elif pattern_type == 'trend_with_noise':
            pattern_results['expected_behavior'] = 'Should separate trend from noise at different scales'
        elif pattern_type == 'periodic_burst':
            pattern_results['expected_behavior'] = 'Should detect periodic structure with bursts'
        
        validation_results[pattern_type] = pattern_results
    
    return validation_results


def _generate_test_pattern(pattern_type: str, length: int = 200) -> torch.Tensor:
    """Generate test patterns for validation."""
    t = np.linspace(0, 4*np.pi, length)
    
    if pattern_type == 'sine_multi_scale':
        # Multiple sine waves at different scales
        signal = (np.sin(t) + 0.5*np.sin(4*t) + 0.25*np.sin(16*t) + 
                 0.1*np.random.randn(length))
    
    elif pattern_type == 'trend_with_noise':
        # Linear trend with additive noise
        trend = 0.5 * t
        noise = 0.2 * np.random.randn(length)
        signal = trend + noise
    
    elif pattern_type == 'periodic_burst':
        # Periodic bursts
        base_signal = 0.1 * np.sin(t)
        bursts = np.zeros_like(t)
        burst_indices = np.arange(20, length, 40)  # Every 40 samples
        for idx in burst_indices:
            if idx < length - 10:
                bursts[idx:idx+10] = np.sin(8*t[idx:idx+10])
        signal = base_signal + bursts + 0.05*np.random.randn(length)
    
    else:
        # Default: simple sine wave
        signal = np.sin(t) + 0.1*np.random.randn(length)
    
    return torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)


# Export main classes and functions
__all__ = [
    'HierarchicalTDAConfig',
    'HierarchicalTDAProcessor',
    'AdaptiveResolutionSelector', 
    'ResolutionValidator',
    'create_hierarchical_tda_pipeline',
    'validate_hierarchical_tda'
] 