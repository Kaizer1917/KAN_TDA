"""
TDA Feature Extractor - Integrated Pipeline for Topological Data Analysis

This module provides a comprehensive pipeline for extracting topological features
from time series data, integrating Takens embedding, persistent homology, and
persistence landscapes into a unified framework optimized for KAN_TDA integration.

Author: TDA-KAN_TDA Integration Team
Date: 2024
"""

import torch
import torch.nn as nn
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import pickle
import hashlib
import time
import logging
from collections import defaultdict

# Import TDA components
from .TakensEmbedding import TakensEmbedding
from .PersistentHomology import PersistentHomologyComputer
from utils.persistence_landscapes import (
    PersistenceLandscape, 
    TopologicalFeatureExtractor as LandscapeFeatureExtractor,
    PersistenceLandscapeVisualizer
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TDAConfig:
    """Configuration class for TDA feature extraction pipeline."""
    
    # Takens Embedding Parameters
    embedding_dims: List[int] = field(default_factory=lambda: [2, 3, 5, 10])
    embedding_delays: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    embedding_strategy: str = "multi_scale"  # single, multi_scale, adaptive
    optimize_parameters: bool = True
    
    # Persistent Homology Parameters
    max_homology_dim: int = 2
    persistence_threshold: float = 0.01
    homology_backend: str = "ripser"  # ripser, gudhi, giotto
    distance_metric: str = "euclidean"
    
    # Persistence Landscape Parameters
    landscape_resolution: int = 500
    max_landscapes: int = 5
    landscape_orders: List[int] = field(default_factory=lambda: [1, 2, 3])
    
    # Feature Selection Parameters
    feature_selection_method: str = "persistence_based"  # all, persistence_based, variance_based
    min_persistence_ratio: float = 0.05
    max_features: Optional[int] = None
    
    # Performance Parameters
    enable_caching: bool = True
    cache_dir: str = "./tda_cache"
    parallel_processing: bool = True
    gpu_acceleration: bool = True
    batch_size: int = 32
    
    # Integration Parameters
    output_format: str = "tensor"  # tensor, numpy, dict
    normalize_features: bool = True
    feature_scaling: str = "standard"  # standard, minmax, robust
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.embedding_strategy not in ["single", "multi_scale", "adaptive"]:
            raise ValueError(f"Invalid embedding_strategy: {self.embedding_strategy}")
        
        if self.homology_backend not in ["ripser", "gudhi", "giotto"]:
            raise ValueError(f"Invalid homology_backend: {self.homology_backend}")
        
        if self.feature_selection_method not in ["all", "persistence_based", "variance_based"]:
            raise ValueError(f"Invalid feature_selection_method: {self.feature_selection_method}")
        
        if self.max_homology_dim < 0 or self.max_homology_dim > 3:
            warnings.warn("max_homology_dim outside typical range [0, 3]")
        
        if self.persistence_threshold < 0 or self.persistence_threshold > 1:
            warnings.warn("persistence_threshold should typically be in [0, 1]")


class TDAFeatureExtractor(nn.Module):
    """
    Integrated TDA Feature Extraction Pipeline
    
    This class provides a complete pipeline for extracting topological features
    from time series data, designed for seamless integration with KAN_TDA.
    
    Features:
    - Multi-scale Takens embedding
    - Persistent homology computation
    - Persistence landscape generation
    - Feature selection and optimization
    - Caching and performance optimization
    - GPU acceleration support
    """
    
    def __init__(self, config: Optional[TDAConfig] = None):
        """
        Initialize TDA Feature Extractor.
        
        Args:
            config: TDA configuration object. If None, uses default configuration.
        """
        super(TDAFeatureExtractor, self).__init__()
        
        # Configuration
        self.config = config if config is not None else TDAConfig()
        
        # Initialize components
        self._initialize_components()
        
        # Performance tracking
        self.performance_stats = defaultdict(list)
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Setup caching
        if self.config.enable_caching:
            self._setup_cache_directory()
        
        logger.info(f"TDAFeatureExtractor initialized with config: {self.config}")
    
    def _initialize_components(self):
        """Initialize TDA computation components."""
        # Takens Embedding
        self.takens_embedding = TakensEmbedding(
            dims=self.config.embedding_dims,
            delays=self.config.embedding_delays,
            strategy=self.config.embedding_strategy
        )
        
        # Persistent Homology Computer
        self.homology_computer = PersistentHomologyComputer(
            backend=self.config.homology_backend,
            max_dimension=self.config.max_homology_dim,
            metric=self.config.distance_metric
        )
        
        # Landscape Feature Extractor
        self.landscape_extractor = LandscapeFeatureExtractor()
        
        # Feature normalizer (will be initialized after first feature extraction)
        self.feature_normalizer = None
        self._feature_stats = None
        
        logger.info("TDA components initialized successfully")
    
    def _setup_cache_directory(self):
        """Setup caching directory structure."""
        cache_path = Path(self.config.cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different cache types
        (cache_path / "embeddings").mkdir(exist_ok=True)
        (cache_path / "homology").mkdir(exist_ok=True)
        (cache_path / "landscapes").mkdir(exist_ok=True)
        (cache_path / "features").mkdir(exist_ok=True)
        
        logger.info(f"Cache directory setup at: {cache_path}")
    
    def _generate_cache_key(self, data: torch.Tensor, operation: str, **kwargs) -> str:
        """Generate unique cache key for data and operation."""
        # Create hash from data shape, dtype, and first/last few values
        data_info = {
            'shape': data.shape,
            'dtype': str(data.dtype),
            'first_values': data.flatten()[:10].tolist() if data.numel() > 0 else [],
            'last_values': data.flatten()[-10:].tolist() if data.numel() > 10 else [],
            'operation': operation,
            'kwargs': sorted(kwargs.items())
        }
        
        # Generate hash
        data_str = str(data_info)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str, cache_type: str) -> Optional[Any]:
        """Load data from cache if available."""
        if not self.config.enable_caching:
            return None
        
        cache_file = Path(self.config.cache_dir) / cache_type / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                self._cache_hits += 1
                logger.debug(f"Cache hit for {cache_type}: {cache_key}")
                return data
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_key}: {e}")
                self._cache_misses += 1
                return None
        else:
            self._cache_misses += 1
            return None
    
    def _save_to_cache(self, data: Any, cache_key: str, cache_type: str):
        """Save data to cache."""
        if not self.config.enable_caching:
            return
        
        cache_file = Path(self.config.cache_dir) / cache_type / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            logger.debug(f"Cached {cache_type}: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_key}: {e}")
    
    def _validate_input(self, x: torch.Tensor) -> torch.Tensor:
        """Validate and preprocess input tensor."""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
        elif x.dim() > 2:
            raise ValueError(f"Input tensor must be 1D or 2D, got {x.dim()}D")
        
        if x.shape[1] < max(self.config.embedding_dims) + max(self.config.embedding_delays):
            warnings.warn(f"Input sequence length {x.shape[1]} may be too short for embedding")
        
        return x.float()
    
    def _track_performance(self, operation: str, duration: float, **kwargs):
        """Track performance statistics."""
        self.performance_stats[operation].append({
            'duration': duration,
            'timestamp': time.time(),
            **kwargs
        })
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get caching performance statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics."""
        stats = {}
        
        for operation, timings in self.performance_stats.items():
            if timings:
                durations = [t['duration'] for t in timings]
                stats[operation] = {
                    'count': len(durations),
                    'mean_duration': np.mean(durations),
                    'std_duration': np.std(durations),
                    'min_duration': np.min(durations),
                    'max_duration': np.max(durations),
                    'total_duration': np.sum(durations)
                }
        
        # Add cache statistics
        stats['caching'] = self.get_cache_stats()
        
        return stats
    
    def clear_cache(self):
        """Clear all cached data."""
        if self.config.enable_caching:
            cache_path = Path(self.config.cache_dir)
            for cache_type in ['embeddings', 'homology', 'landscapes', 'features']:
                cache_dir = cache_path / cache_type
                if cache_dir.exists():
                    for cache_file in cache_dir.glob("*.pkl"):
                        cache_file.unlink()
            
            self._cache_hits = 0
            self._cache_misses = 0
            logger.info("Cache cleared successfully")
    
    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Normalize features using configured scaling method."""
        if not self.config.normalize_features:
            return features
        
        # Always apply normalization based on current batch statistics
        # This ensures consistency across different batch sizes
        if self.config.feature_scaling == "standard":
            mean = features.mean(dim=0, keepdim=True)
            std = features.std(dim=0, keepdim=True)
            # Avoid division by zero
            std = torch.where(std < 1e-8, torch.ones_like(std), std)
            normalized = (features - mean) / std
        elif self.config.feature_scaling == "minmax":
            min_vals = features.min(dim=0, keepdim=True)[0]
            max_vals = features.max(dim=0, keepdim=True)[0]
            # Avoid division by zero
            range_vals = max_vals - min_vals
            range_vals = torch.where(range_vals < 1e-8, torch.ones_like(range_vals), range_vals)
            normalized = (features - min_vals) / range_vals
        elif self.config.feature_scaling == "robust":
            median = features.median(dim=0, keepdim=True)[0]
            mad = torch.median(torch.abs(features - median), dim=0, keepdim=True)[0]
            # Avoid division by zero
            mad = torch.where(mad < 1e-8, torch.ones_like(mad), mad)
            normalized = (features - median) / mad
        else:
            normalized = features
        
        # Ensure no NaN values
        normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        
        return normalized
    
    def _compute_takens_embeddings(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute Takens embeddings for input time series.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            
        Returns:
            Dictionary mapping embedding keys to point clouds
        """
        start_time = time.time()
        
        # Generate cache key
        cache_key = self._generate_cache_key(
            x, "takens_embedding",
            dims=self.config.embedding_dims,
            delays=self.config.embedding_delays,
            strategy=self.config.embedding_strategy
        )
        
        # Try to load from cache
        cached_embeddings = self._load_from_cache(cache_key, "embeddings")
        if cached_embeddings is not None:
            self._track_performance("takens_embedding", time.time() - start_time, cached=True)
            return cached_embeddings
        
        # Compute embeddings
        embeddings = {}
        
        for i, x_single in enumerate(x):
            # Compute embedding for single time series
            embedding_result = self.takens_embedding(x_single.unsqueeze(0))
            
            # Handle different embedding output formats
            if embedding_result.dim() == 4:
                # Multi-scale embedding: [batch, n_combinations, n_points, dim]
                batch_size, n_combinations, n_points, embedding_dim = embedding_result.shape
                
                for j in range(n_combinations):
                    # Extract 2D point cloud for each combination
                    point_cloud = embedding_result[0, j, :, :]  # [n_points, dim]
                    
                    # Remove zero-padded points
                    non_zero_mask = point_cloud.abs().sum(dim=1) > 1e-8
                    if non_zero_mask.any():
                        point_cloud = point_cloud[non_zero_mask]
                        
                        # Create descriptive key
                        dim_val = self.config.embedding_dims[j % len(self.config.embedding_dims)]
                        delay_val = self.config.embedding_delays[j // len(self.config.embedding_dims)]
                        key = f"series_{i}_dim_{dim_val}_delay_{delay_val}"
                        embeddings[key] = point_cloud
            
            elif embedding_result.dim() == 3:
                # Single embedding: [batch, n_points, dim]
                point_cloud = embedding_result[0]  # [n_points, dim]
                embeddings[f"series_{i}_default"] = point_cloud
            
            else:
                logger.warning(f"Unexpected embedding shape: {embedding_result.shape}")
                # Fallback: flatten to 2D
                point_cloud = embedding_result.view(-1, embedding_result.shape[-1])
                embeddings[f"series_{i}_fallback"] = point_cloud
        
        # Cache results
        self._save_to_cache(embeddings, cache_key, "embeddings")
        
        duration = time.time() - start_time
        self._track_performance("takens_embedding", duration, 
                               num_series=x.shape[0], cached=False)
        
        return embeddings
    
    def _compute_persistent_homology(self, embeddings: Dict[str, torch.Tensor]) -> Dict[str, List]:
        """
        Compute persistent homology for all embeddings.
        
        Args:
            embeddings: Dictionary of point clouds from Takens embedding
            
        Returns:
            Dictionary mapping embedding keys to persistence diagrams
        """
        start_time = time.time()
        
        # Generate cache key based on embeddings
        embeddings_hash = hashlib.md5(
            str([(k, v.shape, v.flatten()[:5].tolist()) for k, v in embeddings.items()]).encode()
        ).hexdigest()
        
        cache_key = f"homology_{embeddings_hash}_{self.config.max_homology_dim}_{self.config.homology_backend}"
        
        # Try to load from cache
        cached_diagrams = self._load_from_cache(cache_key, "homology")
        if cached_diagrams is not None:
            self._track_performance("persistent_homology", time.time() - start_time, cached=True)
            return cached_diagrams
        
        # Compute persistence diagrams
        diagrams = {}
        
        for key, point_cloud in embeddings.items():
            try:
                # Convert to numpy for homology computation
                if isinstance(point_cloud, torch.Tensor):
                    point_cloud_np = point_cloud.detach().cpu().numpy()
                else:
                    point_cloud_np = np.array(point_cloud)
                
                # Compute persistence diagram
                diagram_obj = self.homology_computer.compute_diagrams(point_cloud_np)
                diagram = diagram_obj.diagrams  # Extract the list of numpy arrays
                diagrams[key] = diagram
                
            except Exception as e:
                logger.warning(f"Failed to compute homology for {key}: {e}")
                # Create empty diagram as fallback
                diagrams[key] = [np.array([]).reshape(0, 2) for _ in range(self.config.max_homology_dim + 1)]
        
        # Cache results
        self._save_to_cache(diagrams, cache_key, "homology")
        
        duration = time.time() - start_time
        self._track_performance("persistent_homology", duration, 
                               num_embeddings=len(embeddings), cached=False)
        
        return diagrams
    
    def _compute_persistence_landscapes(self, diagrams: Dict[str, List]) -> Dict[str, torch.Tensor]:
        """
        Compute persistence landscapes from persistence diagrams.
        
        Args:
            diagrams: Dictionary mapping keys to persistence diagrams
            
        Returns:
            Dictionary mapping keys to landscape tensors
        """
        start_time = time.time()
        
        # Generate cache key
        diagrams_hash = hashlib.md5(str(diagrams).encode()).hexdigest()
        cache_key = f"landscapes_{diagrams_hash}_{self.config.landscape_resolution}_{self.config.max_landscapes}"
        
        # Try to load from cache
        cached_landscapes = self._load_from_cache(cache_key, "landscapes")
        if cached_landscapes is not None:
            self._track_performance("persistence_landscapes", time.time() - start_time, cached=True)
            return cached_landscapes
        
        # Compute landscapes
        landscapes = {}
        
        for key, diagram_list in diagrams.items():
            try:
                key_landscapes = []
                
                # Process each homology dimension
                for dim, diagram in enumerate(diagram_list):
                    if len(diagram) > 0:
                        # Create persistence landscape
                        landscape = PersistenceLandscape(
                            diagram, 
                            resolution=self.config.landscape_resolution
                        )
                        
                        # Compute landscape functions up to max_landscapes
                        for k in range(1, min(self.config.max_landscapes + 1, len(diagram) + 1)):
                            try:
                                landscape_func = landscape.get_landscape(k)
                                key_landscapes.append(landscape_func)
                            except Exception as e:
                                logger.debug(f"Could not compute landscape {k} for {key}: {e}")
                                # Add zero landscape as fallback
                                zero_landscape = np.zeros(self.config.landscape_resolution)
                                key_landscapes.append(zero_landscape)
                    else:
                        # Empty diagram - create zero landscape
                        zero_landscape = np.zeros(self.config.landscape_resolution)
                        key_landscapes.append(zero_landscape)
                
                # Convert to tensor
                if key_landscapes:
                    landscapes[key] = torch.tensor(np.array(key_landscapes), dtype=torch.float32)
                else:
                    # Fallback for completely empty diagrams
                    landscapes[key] = torch.zeros((1, self.config.landscape_resolution), dtype=torch.float32)
                    
            except Exception as e:
                logger.warning(f"Failed to compute landscapes for {key}: {e}")
                # Create zero landscape as fallback
                landscapes[key] = torch.zeros((1, self.config.landscape_resolution), dtype=torch.float32)
        
        # Cache results
        self._save_to_cache(landscapes, cache_key, "landscapes")
        
        duration = time.time() - start_time
        self._track_performance("persistence_landscapes", duration, 
                               num_diagrams=len(diagrams), cached=False)
        
        return landscapes
    
    def _extract_topological_features(self, landscapes: Dict[str, torch.Tensor], 
                                    diagrams: Dict[str, List]) -> torch.Tensor:
        """
        Extract comprehensive topological features from landscapes and diagrams.
        
        Args:
            landscapes: Dictionary of persistence landscapes
            diagrams: Dictionary of persistence diagrams
            
        Returns:
            Feature tensor of shape (batch_size, num_features)
        """
        start_time = time.time()
        
        all_features = []
        
        # Group by series (assuming keys like "series_0_dim_2_delay_1")
        series_groups = defaultdict(list)
        for key in landscapes.keys():
            if key.startswith("series_"):
                series_id = key.split("_")[1]
                series_groups[series_id].append(key)
        
        # Extract features for each series
        for series_id in sorted(series_groups.keys()):
            series_features = []
            series_keys = series_groups[series_id]
            
            for key in series_keys:
                landscape = landscapes[key]
                diagram_list = diagrams[key]
                
                # Landscape-based features
                landscape_features = self._extract_landscape_features(landscape)
                series_features.extend(landscape_features)
                
                # Diagram-based features
                diagram_features = self._extract_diagram_features(diagram_list)
                series_features.extend(diagram_features)
            
            all_features.append(series_features)
        
        # Convert to tensor
        if all_features:
            # Pad sequences to same length if needed
            max_len = max(len(features) for features in all_features)
            padded_features = []
            
            for features in all_features:
                if len(features) < max_len:
                    features.extend([0.0] * (max_len - len(features)))
                padded_features.append(features)
            
            feature_tensor = torch.tensor(padded_features, dtype=torch.float32)
        else:
            # Fallback for empty features
            feature_tensor = torch.zeros((1, 10), dtype=torch.float32)
        
        duration = time.time() - start_time
        self._track_performance("feature_extraction", duration, 
                               num_series=len(series_groups), cached=False)
        
        return feature_tensor
    
    def _extract_landscape_features(self, landscape: torch.Tensor) -> List[float]:
        """Extract statistical features from persistence landscapes."""
        features = []
        
        for i, landscape_func in enumerate(landscape):
            # Ensure no NaN or inf values
            landscape_func = torch.nan_to_num(landscape_func, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Basic statistics with safe computation
            integral = float(landscape_func.sum())
            maximum = float(landscape_func.max())
            mean_val = float(landscape_func.mean())
            
            # Safe standard deviation computation
            if landscape_func.numel() > 1:
                std_val = float(landscape_func.std())
            else:
                std_val = 0.0
            
            l2_norm = float((landscape_func ** 2).sum())
            
            features.extend([integral, maximum, mean_val, std_val, l2_norm])
            
            # Support analysis with safe computation
            non_zero = landscape_func > 1e-8
            if non_zero.any():
                non_zero_indices = non_zero.nonzero().flatten()
                support_start = float(non_zero_indices[0])
                support_end = float(non_zero_indices[-1])
                support_length = support_end - support_start
            else:
                support_start = support_end = support_length = 0.0
            
            features.extend([support_start, support_end, support_length])
        
        # Ensure all features are finite
        features = [f if np.isfinite(f) else 0.0 for f in features]
        
        return features
    
    def _extract_diagram_features(self, diagram_list: List[np.ndarray]) -> List[float]:
        """Extract statistical features from persistence diagrams."""
        features = []
        
        for dim, diagram in enumerate(diagram_list):
            if len(diagram) > 0:
                # Persistence values with safe computation
                persistence = diagram[:, 1] - diagram[:, 0]
                
                # Remove any infinite or NaN persistence values
                finite_mask = np.isfinite(persistence)
                if finite_mask.any():
                    persistence = persistence[finite_mask]
                    diagram_finite = diagram[finite_mask]
                else:
                    # All values are infinite/NaN, use empty arrays
                    persistence = np.array([])
                    diagram_finite = np.array([]).reshape(0, 2)
                
                if len(persistence) > 0:
                    # Basic statistics
                    features.extend([
                        len(persistence),                    # Number of points
                        float(np.max(persistence)),          # Maximum persistence
                        float(np.mean(persistence)),         # Mean persistence
                        float(np.std(persistence)) if len(persistence) > 1 else 0.0,  # Std persistence
                        float(np.sum(persistence)),          # Total persistence
                    ])
                    
                    # Birth/death statistics
                    births = diagram_finite[:, 0]
                    deaths = diagram_finite[:, 1]
                    
                    # Remove infinite death times for statistics
                    finite_births = births[np.isfinite(births)]
                    finite_deaths = deaths[np.isfinite(deaths)]
                    
                    features.extend([
                        float(np.mean(finite_births)) if len(finite_births) > 0 else 0.0,  # Mean birth time
                        float(np.mean(finite_deaths)) if len(finite_deaths) > 0 else 0.0,  # Mean death time
                        float(np.std(finite_births)) if len(finite_births) > 1 else 0.0,   # Std birth time
                        float(np.std(finite_deaths)) if len(finite_deaths) > 1 else 0.0,   # Std death time
                    ])
                else:
                    # No finite persistence values
                    features.extend([0.0] * 9)
            else:
                # Empty diagram
                features.extend([0.0] * 9)
        
        # Ensure all features are finite
        features = [f if np.isfinite(f) else 0.0 for f in features]
        
        return features
    
    def _select_features(self, features: torch.Tensor, 
                        diagrams: Dict[str, List]) -> torch.Tensor:
        """
        Select most important features based on configured method.
        
        Args:
            features: Full feature tensor
            diagrams: Persistence diagrams for persistence-based selection
            
        Returns:
            Selected feature tensor
        """
        if self.config.feature_selection_method == "all":
            return features
        
        if self.config.feature_selection_method == "persistence_based":
            return self._persistence_based_selection(features, diagrams)
        
        elif self.config.feature_selection_method == "variance_based":
            return self._variance_based_selection(features)
        
        return features
    
    def _persistence_based_selection(self, features: torch.Tensor, 
                                   diagrams: Dict[str, List]) -> torch.Tensor:
        """Select features based on persistence importance."""
        # Calculate persistence importance for each feature group
        importance_scores = []
        
        for key, diagram_list in diagrams.items():
            for diagram in diagram_list:
                if len(diagram) > 0:
                    persistence = diagram[:, 1] - diagram[:, 0]
                    # Remove infinite values for importance calculation
                    finite_persistence = persistence[np.isfinite(persistence)]
                    if len(finite_persistence) > 0:
                        max_persistence = float(finite_persistence.max())
                        total_persistence = float(finite_persistence.sum())
                        importance_scores.extend([max_persistence, total_persistence])
                    else:
                        importance_scores.extend([0.0, 0.0])
                else:
                    importance_scores.extend([0.0, 0.0])
        
        # Ensure we have enough importance scores for all features
        # Repeat the pattern if we have fewer scores than features
        while len(importance_scores) < features.shape[1]:
            if len(importance_scores) > 0:
                importance_scores.extend(importance_scores[:min(len(importance_scores), 
                                                              features.shape[1] - len(importance_scores))])
            else:
                importance_scores.extend([1.0] * (features.shape[1] - len(importance_scores)))
        
        # Truncate if we have too many scores
        importance_scores = importance_scores[:features.shape[1]]
        
        # Select features above threshold
        importance_tensor = torch.tensor(importance_scores)
        if importance_tensor.max() > 0:
            threshold = importance_tensor.max() * self.config.min_persistence_ratio
            selected_indices = (importance_tensor >= threshold).nonzero().flatten()
        else:
            # If all importance scores are zero, select first few features
            selected_indices = torch.arange(min(10, features.shape[1]))
        
        if len(selected_indices) == 0:
            # Fallback: select top features
            _, selected_indices = importance_tensor.topk(min(10, features.shape[1]))
        
        if self.config.max_features is not None:
            selected_indices = selected_indices[:self.config.max_features]
        
        return features[:, selected_indices]
    
    def _variance_based_selection(self, features: torch.Tensor) -> torch.Tensor:
        """Select features based on variance."""
        if features.shape[0] < 2:
            return features  # Need at least 2 samples for variance
        
        # Calculate variance across batch dimension
        feature_variance = features.var(dim=0)
        
        # Select features with highest variance
        if self.config.max_features is not None:
            _, selected_indices = feature_variance.topk(
                min(self.config.max_features, features.shape[1])
            )
        else:
            # Select features above mean variance
            threshold = feature_variance.mean()
            selected_indices = (feature_variance >= threshold).nonzero().flatten()
            
            if len(selected_indices) == 0:
                selected_indices = torch.arange(features.shape[1])
        
        return features[:, selected_indices]
    
    def forward(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Main forward pass for TDA feature extraction.
        
        Args:
            x: Input time series tensor of shape (batch_size, seq_len) or (seq_len,)
            
        Returns:
            Extracted TDA features tensor of shape (batch_size, num_features)
        """
        start_time = time.time()
        
        # Validate and preprocess input
        x = self._validate_input(x)
        
        # Generate cache key for complete pipeline
        cache_key = self._generate_cache_key(x, "complete_pipeline", config=str(self.config))
        
        # Try to load complete result from cache
        cached_features = self._load_from_cache(cache_key, "features")
        if cached_features is not None:
            self._track_performance("complete_pipeline", time.time() - start_time, cached=True)
            return cached_features
        
        try:
            # Step 1: Compute Takens embeddings
            logger.info("Computing Takens embeddings...")
            embeddings = self._compute_takens_embeddings(x)
            
            # Step 2: Compute persistent homology
            logger.info("Computing persistent homology...")
            diagrams = self._compute_persistent_homology(embeddings)
            
            # Step 3: Compute persistence landscapes
            logger.info("Computing persistence landscapes...")
            landscapes = self._compute_persistence_landscapes(diagrams)
            
            # Step 4: Extract topological features
            logger.info("Extracting topological features...")
            features = self._extract_topological_features(landscapes, diagrams)
            
            # Step 5: Feature selection
            logger.info("Selecting features...")
            features = self._select_features(features, diagrams)
            
            # Step 6: Normalize features
            if self.config.normalize_features:
                features = self._normalize_features(features)
            
            # Cache complete result
            self._save_to_cache(features, cache_key, "features")
            
            duration = time.time() - start_time
            self._track_performance("complete_pipeline", duration, 
                                   input_shape=x.shape, output_shape=features.shape, cached=False)
            
            logger.info(f"TDA feature extraction completed in {duration:.3f}s. "
                       f"Output shape: {features.shape}")
            
            return features
            
        except Exception as e:
            logger.error(f"TDA feature extraction failed: {e}")
            # Return fallback features
            fallback_features = torch.zeros((x.shape[0], 10), dtype=torch.float32)
            return fallback_features
    
    def extract_features_batch(self, x_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Extract features from a batch of time series with potentially different lengths.
        
        Args:
            x_list: List of time series tensors
            
        Returns:
            Batch feature tensor
        """
        if not x_list:
            return torch.zeros((0, 10), dtype=torch.float32)
        
        # Process each time series individually
        feature_list = []
        
        for i, x in enumerate(x_list):
            try:
                features = self.forward(x.unsqueeze(0) if x.dim() == 1 else x)
                feature_list.append(features)
            except Exception as e:
                logger.warning(f"Failed to extract features for series {i}: {e}")
                # Add fallback features
                fallback = torch.zeros((1, 10), dtype=torch.float32)
                feature_list.append(fallback)
        
        # Concatenate all features
        if feature_list:
            # Ensure all feature tensors have the same number of features
            max_features = max(f.shape[1] for f in feature_list)
            
            padded_features = []
            for features in feature_list:
                if features.shape[1] < max_features:
                    padding = torch.zeros((features.shape[0], max_features - features.shape[1]))
                    features = torch.cat([features, padding], dim=1)
                padded_features.append(features)
            
            return torch.cat(padded_features, dim=0)
        else:
            return torch.zeros((len(x_list), 10), dtype=torch.float32)
    
    def get_feature_names(self) -> List[str]:
        """Get descriptive names for extracted features."""
        feature_names = []
        
        # Generate names based on configuration
        for dim in self.config.embedding_dims:
            for delay in self.config.embedding_delays:
                for landscape_k in range(1, self.config.max_landscapes + 1):
                    # Landscape features
                    base_name = f"dim_{dim}_delay_{delay}_landscape_{landscape_k}"
                    feature_names.extend([
                        f"{base_name}_integral",
                        f"{base_name}_maximum",
                        f"{base_name}_mean",
                        f"{base_name}_std",
                        f"{base_name}_l2_norm",
                        f"{base_name}_support_start",
                        f"{base_name}_support_end",
                        f"{base_name}_support_length"
                    ])
                
                # Diagram features for each homology dimension
                for hom_dim in range(self.config.max_homology_dim + 1):
                    base_name = f"dim_{dim}_delay_{delay}_homology_{hom_dim}"
                    feature_names.extend([
                        f"{base_name}_num_points",
                        f"{base_name}_max_persistence",
                        f"{base_name}_mean_persistence",
                        f"{base_name}_std_persistence",
                        f"{base_name}_total_persistence",
                        f"{base_name}_mean_birth",
                        f"{base_name}_mean_death",
                        f"{base_name}_std_birth",
                        f"{base_name}_std_death"
                    ])
        
        return feature_names
    
    def get_feature_importance(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Compute feature importance scores.
        
        Args:
            x: Input time series for importance calculation
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        # Extract features
        features = self.forward(x)
        feature_names = self.get_feature_names()
        
        # Compute importance based on variance and magnitude
        if features.shape[0] > 1:
            variance_scores = features.var(dim=0)
            magnitude_scores = features.abs().mean(dim=0)
            importance_scores = variance_scores * magnitude_scores
        else:
            importance_scores = features.abs().flatten()
        
        # Normalize scores
        if importance_scores.sum() > 0:
            importance_scores = importance_scores / importance_scores.sum()
        
        # Create importance dictionary
        importance_dict = {}
        for i, name in enumerate(feature_names[:len(importance_scores)]):
            importance_dict[name] = float(importance_scores[i])
        
        return importance_dict
    
    def optimize_parameters(self, x: torch.Tensor, 
                          optimization_method: str = "grid_search") -> TDAConfig:
        """
        Optimize TDA parameters for given time series.
        
        Args:
            x: Sample time series for optimization
            optimization_method: Method for optimization ("grid_search", "random_search")
            
        Returns:
            Optimized TDAConfig
        """
        logger.info(f"Optimizing TDA parameters using {optimization_method}")
        
        if optimization_method == "grid_search":
            return self._grid_search_optimization(x)
        elif optimization_method == "random_search":
            return self._random_search_optimization(x)
        else:
            logger.warning(f"Unknown optimization method: {optimization_method}")
            return self.config
    
    def _grid_search_optimization(self, x: torch.Tensor) -> TDAConfig:
        """Optimize parameters using grid search."""
        best_config = self.config
        best_score = 0.0
        
        # Define parameter grids
        dim_options = [[2, 3], [2, 3, 5], [2, 3, 5, 10]]
        delay_options = [[1, 2], [1, 2, 4], [1, 2, 4, 8]]
        threshold_options = [0.01, 0.05, 0.1]
        
        for dims in dim_options:
            for delays in delay_options:
                for threshold in threshold_options:
                    # Create test configuration
                    test_config = TDAConfig(
                        embedding_dims=dims,
                        embedding_delays=delays,
                        persistence_threshold=threshold,
                        enable_caching=False  # Disable caching for optimization
                    )
                    
                    try:
                        # Test configuration
                        test_extractor = TDAFeatureExtractor(test_config)
                        features = test_extractor.forward(x[:1])  # Use single sample
                        
                        # Score based on feature diversity and magnitude
                        score = self._score_features(features)
                        
                        if score > best_score:
                            best_score = score
                            best_config = test_config
                            
                    except Exception as e:
                        logger.warning(f"Failed to test config {dims}, {delays}, {threshold}: {e}")
                        continue
        
        logger.info(f"Best configuration found with score: {best_score}")
        return best_config
    
    def _random_search_optimization(self, x: torch.Tensor, n_trials: int = 20) -> TDAConfig:
        """Optimize parameters using random search."""
        best_config = self.config
        best_score = 0.0
        
        import random
        
        for trial in range(n_trials):
            # Random parameter selection
            dims = random.choice([[2, 3], [2, 3, 5], [2, 3, 5, 10]])
            delays = random.choice([[1, 2], [1, 2, 4], [1, 2, 4, 8]])
            threshold = random.uniform(0.01, 0.1)
            resolution = random.choice([100, 250, 500])
            
            test_config = TDAConfig(
                embedding_dims=dims,
                embedding_delays=delays,
                persistence_threshold=threshold,
                landscape_resolution=resolution,
                enable_caching=False
            )
            
            try:
                test_extractor = TDAFeatureExtractor(test_config)
                features = test_extractor.forward(x[:1])
                score = self._score_features(features)
                
                if score > best_score:
                    best_score = score
                    best_config = test_config
                    
            except Exception as e:
                logger.warning(f"Failed trial {trial}: {e}")
                continue
        
        logger.info(f"Random search completed. Best score: {best_score}")
        return best_config
    
    def _score_features(self, features: torch.Tensor) -> float:
        """Score feature quality for parameter optimization."""
        if features.numel() == 0:
            return 0.0
        
        # Combine multiple quality metrics
        non_zero_ratio = (features != 0).float().mean()
        magnitude_score = features.abs().mean()
        diversity_score = features.std()
        
        # Weighted combination
        score = (0.4 * non_zero_ratio + 0.3 * magnitude_score + 0.3 * diversity_score)
        return float(score)
    
    def visualize_features(self, x: torch.Tensor, save_path: Optional[str] = None):
        """
        Visualize extracted TDA features.
        
        Args:
            x: Input time series
            save_path: Optional path to save visualization
        """
        try:
            import matplotlib.pyplot as plt
            
            # Extract features and get importance
            features = self.forward(x)
            importance = self.get_feature_importance(x)
            
            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Feature values heatmap
            if features.shape[0] > 1:
                im1 = axes[0, 0].imshow(features.T, aspect='auto', cmap='viridis')
                axes[0, 0].set_title('TDA Features Heatmap')
                axes[0, 0].set_xlabel('Time Series Index')
                axes[0, 0].set_ylabel('Feature Index')
                plt.colorbar(im1, ax=axes[0, 0])
            else:
                axes[0, 0].bar(range(features.shape[1]), features[0])
                axes[0, 0].set_title('TDA Feature Values')
                axes[0, 0].set_xlabel('Feature Index')
                axes[0, 0].set_ylabel('Feature Value')
            
            # Plot 2: Feature importance
            if importance:
                top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
                names, scores = zip(*top_features)
                axes[0, 1].barh(range(len(names)), scores)
                axes[0, 1].set_yticks(range(len(names)))
                axes[0, 1].set_yticklabels([n.split('_')[-1] for n in names])
                axes[0, 1].set_title('Top 20 Feature Importance')
                axes[0, 1].set_xlabel('Importance Score')
            
            # Plot 3: Performance statistics
            stats = self.get_performance_stats()
            if stats:
                operations = list(stats.keys())
                durations = [stats[op].get('mean_duration', 0) for op in operations if op != 'caching']
                axes[1, 0].bar(operations, durations)
                axes[1, 0].set_title('Operation Performance')
                axes[1, 0].set_ylabel('Mean Duration (s)')
                axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Plot 4: Cache statistics
            cache_stats = self.get_cache_stats()
            if cache_stats['total_requests'] > 0:
                labels = ['Cache Hits', 'Cache Misses']
                sizes = [cache_stats['cache_hits'], cache_stats['cache_misses']]
                axes[1, 1].pie(sizes, labels=labels, autopct='%1.1f%%')
                axes[1, 1].set_title(f"Cache Performance (Hit Rate: {cache_stats['hit_rate']:.2%})")
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Visualization saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("Matplotlib not available for visualization")
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
    
    def export_features(self, x: torch.Tensor, filepath: str, format: str = "csv"):
        """
        Export extracted features to file.
        
        Args:
            x: Input time series
            filepath: Output file path
            format: Export format ("csv", "json", "pickle")
        """
        features = self.forward(x)
        feature_names = self.get_feature_names()
        
        if format == "csv":
            import pandas as pd
            df = pd.DataFrame(features.numpy(), columns=feature_names[:features.shape[1]])
            df.to_csv(filepath, index=False)
            
        elif format == "json":
            import json
            data = {
                'features': features.tolist(),
                'feature_names': feature_names[:features.shape[1]],
                'config': str(self.config)
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
                
        elif format == "pickle":
            data = {
                'features': features,
                'feature_names': feature_names[:features.shape[1]],
                'config': self.config
            }
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        
        logger.info(f"Features exported to {filepath} in {format} format")
    
    def __repr__(self) -> str:
        """String representation of the TDA extractor."""
        return (f"TDAFeatureExtractor(\n"
                f"  embedding_dims={self.config.embedding_dims},\n"
                f"  embedding_delays={self.config.embedding_delays},\n"
                f"  max_homology_dim={self.config.max_homology_dim},\n"
                f"  landscape_resolution={self.config.landscape_resolution},\n"
                f"  caching_enabled={self.config.enable_caching}\n"
                f")")


# Convenience functions for easy usage
def extract_tda_features(x: Union[torch.Tensor, np.ndarray], 
                        config: Optional[TDAConfig] = None) -> torch.Tensor:
    """
    Convenience function to extract TDA features from time series.
    
    Args:
        x: Input time series
        config: Optional TDA configuration
        
    Returns:
        Extracted TDA features
    """
    extractor = TDAFeatureExtractor(config)
    return extractor.forward(x)


def create_tda_config(**kwargs) -> TDAConfig:
    """
    Convenience function to create TDA configuration.
    
    Args:
        **kwargs: Configuration parameters
        
    Returns:
        TDAConfig object
    """
    return TDAConfig(**kwargs)


def optimize_tda_config(x: torch.Tensor, method: str = "grid_search") -> TDAConfig:
    """
    Convenience function to optimize TDA configuration.
    
    Args:
        x: Sample time series
        method: Optimization method
        
    Returns:
        Optimized TDAConfig
    """
    extractor = TDAFeatureExtractor()
    return extractor.optimize_parameters(x, method) 