"""
Comprehensive Configuration System for TDA-KAN_TDA Integration

This module provides a complete configuration management system for TDA-aware
parameters, validation, and optimization settings.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import warnings
import json
import yaml
from pathlib import Path


class TDAStrategy(Enum):
    """Enumeration of TDA computation strategies."""
    SINGLE = "single"
    MULTI_SCALE = "multi_scale"
    ADAPTIVE = "adaptive"
    HIERARCHICAL = "hierarchical"


class FusionStrategy(Enum):
    """Enumeration of fusion strategies."""
    EARLY = "early"
    MID = "mid"
    LATE = "late"
    ATTENTION = "attention"
    GATE = "gate"
    ADAPTIVE = "adaptive"


class HomologyBackend(Enum):
    """Enumeration of persistent homology backends."""
    RIPSER = "ripser"
    GUDHI = "gudhi"
    GIOTTO = "giotto"


@dataclass
class TakensEmbeddingConfig:
    """Configuration for Takens embedding parameters."""
    
    # Embedding dimensions to use
    dims: List[int] = field(default_factory=lambda: [2, 3, 5, 10])
    
    # Delay parameters
    delays: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    
    # Strategy for embedding computation
    strategy: TDAStrategy = TDAStrategy.MULTI_SCALE
    
    # Automatic parameter optimization
    auto_optimize: bool = True
    
    # Optimization method for automatic parameter selection
    optimization_method: str = "mutual_info"  # "mutual_info", "fnn", "autocorr"
    
    # Maximum embedding dimension for auto-optimization
    max_auto_dim: int = 15
    
    # Maximum delay for auto-optimization
    max_auto_delay: int = 20
    
    # Quality threshold for embedding selection
    quality_threshold: float = 0.7
    
    def validate(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []
        
        if not self.dims or any(d < 2 for d in self.dims):
            errors.append("Embedding dimensions must be >= 2")
        
        if not self.delays or any(d < 1 for d in self.delays):
            errors.append("Delays must be >= 1")
        
        if self.dims and self.max_auto_dim < max(self.dims):
            errors.append("max_auto_dim should be >= max(dims)")
        
        if self.delays and self.max_auto_delay < max(self.delays):
            errors.append("max_auto_delay should be >= max(delays)")
        
        if not 0 < self.quality_threshold <= 1:
            errors.append("quality_threshold must be in (0, 1]")
        
        return errors


@dataclass
class PersistentHomologyConfig:
    """Configuration for persistent homology computation."""
    
    # Backend for homology computation
    backend: HomologyBackend = HomologyBackend.RIPSER
    
    # Maximum homology dimension to compute
    max_dimension: int = 2
    
    # Minimum persistence threshold for stable features
    persistence_threshold: float = 0.01
    
    # Distance matrix computation batch size
    distance_batch_size: int = 1000
    
    # Maximum number of points for homology computation
    max_points: int = 2000
    
    # Subsampling strategy if too many points
    subsampling_strategy: str = "random"  # "random", "farthest", "grid"
    
    # Metric for distance computation
    distance_metric: str = "euclidean"  # "euclidean", "manhattan", "chebyshev"
    
    # Enable parallel computation
    parallel: bool = True
    
    # Number of parallel workers
    n_workers: int = 4
    
    def validate(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []
        
        if self.max_dimension < 0:
            errors.append("max_dimension must be >= 0")
        
        if self.persistence_threshold <= 0:
            errors.append("persistence_threshold must be > 0")
        
        if self.distance_batch_size <= 0:
            errors.append("distance_batch_size must be > 0")
        
        if self.max_points <= 0:
            errors.append("max_points must be > 0")
        
        if self.n_workers <= 0:
            errors.append("n_workers must be > 0")
        
        return errors


@dataclass
class PersistenceLandscapeConfig:
    """Configuration for persistence landscape computation."""
    
    # Resolution for landscape computation
    resolution: int = 500
    
    # Number of landscapes to compute
    num_landscapes: int = 5
    
    # Integration orders for landscape statistics
    integration_orders: List[int] = field(default_factory=lambda: [1, 2, 3])
    
    # Enable landscape resampling for consistency
    enable_resampling: bool = True
    
    # Resampling resolution
    resampling_resolution: int = 100
    
    # Statistical features to extract
    extract_statistics: List[str] = field(default_factory=lambda: [
        "integral", "maximum", "norm", "support", "moments"
    ])
    
    def validate(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []
        
        if self.resolution <= 0:
            errors.append("resolution must be > 0")
        
        if self.num_landscapes <= 0:
            errors.append("num_landscapes must be > 0")
        
        if not self.integration_orders or any(o <= 0 for o in self.integration_orders):
            errors.append("integration_orders must be positive")
        
        if self.resampling_resolution <= 0:
            errors.append("resampling_resolution must be > 0")
        
        return errors


@dataclass
class FeatureFusionConfig:
    """Configuration for TDA-frequency feature fusion."""
    
    # Primary fusion strategy
    fusion_strategy: FusionStrategy = FusionStrategy.ADAPTIVE
    
    # Available strategies for adaptive fusion
    available_strategies: List[FusionStrategy] = field(default_factory=lambda: [
        FusionStrategy.EARLY, FusionStrategy.ATTENTION, 
        FusionStrategy.GATE, FusionStrategy.LATE
    ])
    
    # Cross-modal attention configuration
    attention_heads: int = 4
    attention_hidden_dim: int = 64
    attention_dropout: float = 0.1
    attention_temperature: float = 1.0
    
    # TDA feature importance weight
    tda_weight: float = 0.3
    
    # Frequency feature importance weight
    freq_weight: float = 0.7
    
    # Enable adaptive weight learning
    adaptive_weighting: bool = True
    
    # Weight adaptation rate
    weight_adaptation_rate: float = 0.01
    
    def validate(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []
        
        if self.attention_heads <= 0:
            errors.append("attention_heads must be > 0")
        
        if self.attention_hidden_dim <= 0:
            errors.append("attention_hidden_dim must be > 0")
        
        if not 0 <= self.attention_dropout <= 1:
            errors.append("attention_dropout must be in [0, 1]")
        
        if self.attention_temperature <= 0:
            errors.append("attention_temperature must be > 0")
        
        if not 0 <= self.tda_weight <= 1:
            errors.append("tda_weight must be in [0, 1]")
        
        if not 0 <= self.freq_weight <= 1:
            errors.append("freq_weight must be in [0, 1]")
        
        if abs(self.tda_weight + self.freq_weight - 1.0) > 1e-6:
            errors.append("tda_weight + freq_weight should equal 1.0")
        
        if not 0 < self.weight_adaptation_rate <= 1:
            errors.append("weight_adaptation_rate must be in (0, 1]")
        
        return errors


@dataclass
class TDALossConfig:
    """Configuration for TDA-aware loss functions."""
    
    # Enable TDA-aware losses
    enable_tda_losses: bool = True
    
    # Persistence loss configuration
    persistence_loss_weight: float = 0.1
    persistence_stability_weight: float = 1.0
    persistence_consistency_weight: float = 0.5
    
    # Topological consistency loss configuration
    consistency_loss_weight: float = 0.05
    cross_scale_consistency_weight: float = 0.5
    frequency_consistency_weight: float = 0.3
    
    # Structural preservation loss configuration
    preservation_loss_weight: float = 0.03
    trend_preservation_weight: float = 0.4
    periodicity_preservation_weight: float = 0.3
    complexity_preservation_weight: float = 0.3
    
    # Adaptive loss weighting
    adaptive_loss_weighting: bool = True
    loss_adaptation_rate: float = 0.01
    min_loss_weight: float = 0.001
    max_loss_weight: float = 10.0
    
    def validate(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []
        
        weights = [
            self.persistence_loss_weight, self.consistency_loss_weight,
            self.preservation_loss_weight, self.persistence_stability_weight,
            self.persistence_consistency_weight, self.cross_scale_consistency_weight,
            self.frequency_consistency_weight, self.trend_preservation_weight,
            self.periodicity_preservation_weight, self.complexity_preservation_weight
        ]
        
        if any(w < 0 for w in weights):
            errors.append("All loss weights must be >= 0")
        
        if not 0 < self.loss_adaptation_rate <= 1:
            errors.append("loss_adaptation_rate must be in (0, 1]")
        
        if self.min_loss_weight <= 0:
            errors.append("min_loss_weight must be > 0")
        
        if self.max_loss_weight <= self.min_loss_weight:
            errors.append("max_loss_weight must be > min_loss_weight")
        
        return errors


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    
    # Enable GPU acceleration
    use_gpu: bool = True
    
    # GPU device ID
    gpu_device: int = 0
    
    # Enable mixed precision training
    mixed_precision: bool = False
    
    # Enable gradient checkpointing
    gradient_checkpointing: bool = False
    
    # Batch size for TDA computations
    tda_batch_size: int = 8
    
    # Enable TDA computation caching
    enable_caching: bool = True
    
    # Cache size limit (MB)
    cache_size_mb: int = 1024
    
    # Enable parallel TDA processing
    parallel_tda: bool = True
    
    # Number of TDA workers
    tda_workers: int = 4
    
    # Memory optimization level (0=none, 1=basic, 2=aggressive)
    memory_optimization: int = 1
    
    # Chunk size for memory-efficient attention
    attention_chunk_size: int = 512
    
    def validate(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []
        
        if self.gpu_device < 0:
            errors.append("gpu_device must be >= 0")
        
        if self.tda_batch_size <= 0:
            errors.append("tda_batch_size must be > 0")
        
        if self.cache_size_mb <= 0:
            errors.append("cache_size_mb must be > 0")
        
        if self.tda_workers <= 0:
            errors.append("tda_workers must be > 0")
        
        if self.memory_optimization not in [0, 1, 2]:
            errors.append("memory_optimization must be 0, 1, or 2")
        
        if self.attention_chunk_size <= 0:
            errors.append("attention_chunk_size must be > 0")
        
        return errors


@dataclass
class TDAConfig:
    """Comprehensive TDA configuration."""
    
    # Sub-configurations
    takens_embedding: TakensEmbeddingConfig = field(default_factory=TakensEmbeddingConfig)
    persistent_homology: PersistentHomologyConfig = field(default_factory=PersistentHomologyConfig)
    persistence_landscapes: PersistenceLandscapeConfig = field(default_factory=PersistenceLandscapeConfig)
    feature_fusion: FeatureFusionConfig = field(default_factory=FeatureFusionConfig)
    tda_losses: TDALossConfig = field(default_factory=TDALossConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Global TDA settings
    enable_tda: bool = True
    tda_feature_selection: bool = True
    feature_selection_method: str = "persistence"  # "persistence", "variance", "importance"
    max_tda_features: int = 50
    
    # Integration settings
    integration_points: List[str] = field(default_factory=lambda: [
        "frequency_decomposition", "m_kan_blocks", "frequency_mixing"
    ])
    
    # Debugging and monitoring
    debug_mode: bool = False
    save_intermediate_results: bool = False
    log_tda_performance: bool = True
    
    def validate(self) -> Dict[str, List[str]]:
        """Validate all configuration components."""
        validation_errors = {}
        
        # Validate sub-configurations
        validation_errors['takens_embedding'] = self.takens_embedding.validate()
        validation_errors['persistent_homology'] = self.persistent_homology.validate()
        validation_errors['persistence_landscapes'] = self.persistence_landscapes.validate()
        validation_errors['feature_fusion'] = self.feature_fusion.validate()
        validation_errors['tda_losses'] = self.tda_losses.validate()
        validation_errors['performance'] = self.performance.validate()
        
        # Validate global settings
        global_errors = []
        
        if self.max_tda_features <= 0:
            global_errors.append("max_tda_features must be > 0")
        
        if self.feature_selection_method not in ["persistence", "variance", "importance"]:
            global_errors.append("feature_selection_method must be 'persistence', 'variance', or 'importance'")
        
        valid_integration_points = [
            "frequency_decomposition", "m_kan_blocks", "frequency_mixing", "loss_functions"
        ]
        invalid_points = [p for p in self.integration_points if p not in valid_integration_points]
        if invalid_points:
            global_errors.append(f"Invalid integration points: {invalid_points}")
        
        validation_errors['global'] = global_errors
        
        # Remove empty error lists
        validation_errors = {k: v for k, v in validation_errors.items() if v}
        
        return validation_errors
    
    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        return {
            'tda_enabled': self.enable_tda,
            'embedding_dims': self.takens_embedding.dims,
            'embedding_delays': self.takens_embedding.delays,
            'homology_backend': self.persistent_homology.backend.value,
            'max_homology_dim': self.persistent_homology.max_dimension,
            'fusion_strategy': self.feature_fusion.fusion_strategy.value,
            'tda_losses_enabled': self.tda_losses.enable_tda_losses,
            'gpu_enabled': self.performance.use_gpu,
            'caching_enabled': self.performance.enable_caching,
            'integration_points': self.integration_points,
            'max_features': self.max_tda_features
        }


class TDAConfigManager:
    """Manager for TDA configuration with templates and optimization."""
    
    def __init__(self):
        self.templates = self._create_default_templates()
    
    def _create_default_templates(self) -> Dict[str, TDAConfig]:
        """Create default configuration templates."""
        templates = {}
        
        # Fast template (minimal TDA for speed)
        fast_config = TDAConfig()
        fast_config.takens_embedding.dims = [2, 3]
        fast_config.takens_embedding.delays = [1, 2]
        fast_config.persistent_homology.max_dimension = 1
        fast_config.persistence_landscapes.num_landscapes = 3
        fast_config.feature_fusion.fusion_strategy = FusionStrategy.GATE
        fast_config.tda_losses.enable_tda_losses = False
        fast_config.performance.memory_optimization = 2
        templates['fast'] = fast_config
        
        # Balanced template (good performance/accuracy trade-off)
        balanced_config = TDAConfig()
        # Uses default values which are already balanced
        templates['balanced'] = balanced_config
        
        # Accurate template (maximum TDA features for best accuracy)
        accurate_config = TDAConfig()
        accurate_config.takens_embedding.dims = [2, 3, 5, 7, 10]
        accurate_config.takens_embedding.delays = [1, 2, 3, 4, 6, 8]
        accurate_config.persistent_homology.max_dimension = 2
        accurate_config.persistence_landscapes.num_landscapes = 7
        accurate_config.feature_fusion.fusion_strategy = FusionStrategy.ADAPTIVE
        accurate_config.tda_losses.enable_tda_losses = True
        accurate_config.max_tda_features = 100
        templates['accurate'] = accurate_config
        
        # Memory-efficient template
        memory_config = TDAConfig()
        memory_config.takens_embedding.dims = [2, 3]
        memory_config.takens_embedding.delays = [1, 2]
        memory_config.persistent_homology.max_points = 1000
        memory_config.persistence_landscapes.resolution = 200
        memory_config.performance.memory_optimization = 2
        memory_config.performance.tda_batch_size = 4
        memory_config.performance.attention_chunk_size = 256
        templates['memory_efficient'] = memory_config
        
        return templates
    
    def get_template(self, template_name: str) -> TDAConfig:
        """Get a configuration template."""
        if template_name not in self.templates:
            available = list(self.templates.keys())
            raise ValueError(f"Template '{template_name}' not found. Available: {available}")
        
        return self.templates[template_name]
    
    def create_config_for_dataset(
        self,
        dataset_characteristics: Dict[str, Any]
    ) -> TDAConfig:
        """
        Create optimized configuration based on dataset characteristics.
        
        Args:
            dataset_characteristics: Dictionary with dataset info
                - seq_len: Sequence length
                - n_features: Number of features
                - n_samples: Number of samples
                - complexity: Estimated complexity ('low', 'medium', 'high')
                - memory_constraint: Memory constraint ('low', 'medium', 'high')
        
        Returns:
            Optimized TDA configuration
        """
        seq_len = dataset_characteristics.get('seq_len', 96)
        n_features = dataset_characteristics.get('n_features', 7)
        n_samples = dataset_characteristics.get('n_samples', 1000)
        complexity = dataset_characteristics.get('complexity', 'medium')
        memory_constraint = dataset_characteristics.get('memory_constraint', 'medium')
        
        # Start with balanced template
        config = self.get_template('balanced')
        
        # Adjust based on sequence length
        if seq_len < 50:
            # Short sequences: use smaller embeddings
            config.takens_embedding.dims = [2, 3]
            config.takens_embedding.delays = [1, 2]
        elif seq_len > 200:
            # Long sequences: can use larger embeddings
            config.takens_embedding.dims = [2, 3, 5, 7, 10]
            config.takens_embedding.delays = [1, 2, 4, 8, 12]
        
        # Adjust based on complexity
        if complexity == 'low':
            config.persistent_homology.max_dimension = 1
            config.persistence_landscapes.num_landscapes = 3
            config.feature_fusion.fusion_strategy = FusionStrategy.EARLY
        elif complexity == 'high':
            config.persistent_homology.max_dimension = 2
            config.persistence_landscapes.num_landscapes = 7
            config.feature_fusion.fusion_strategy = FusionStrategy.ADAPTIVE
            config.tda_losses.enable_tda_losses = True
        
        # Adjust based on memory constraints
        if memory_constraint == 'low':
            config.persistent_homology.max_points = 500
            config.persistence_landscapes.resolution = 200
            config.performance.memory_optimization = 2
            config.performance.tda_batch_size = 2
        elif memory_constraint == 'high':
            config.persistent_homology.max_points = 3000
            config.persistence_landscapes.resolution = 1000
            config.performance.memory_optimization = 0
        
        # Adjust based on dataset size
        if n_samples < 500:
            # Small dataset: reduce caching, enable more TDA features
            config.performance.cache_size_mb = 256
            config.max_tda_features = 30
        elif n_samples > 10000:
            # Large dataset: increase caching, optimize for speed
            config.performance.cache_size_mb = 2048
            config.performance.parallel_tda = True
            config.performance.tda_workers = 8
        
        return config
    
    def optimize_config_for_hardware(
        self,
        config: TDAConfig,
        available_memory_gb: float,
        gpu_available: bool = True,
        cpu_cores: int = 4
    ) -> TDAConfig:
        """
        Optimize configuration for available hardware.
        
        Args:
            config: Base configuration to optimize
            available_memory_gb: Available memory in GB
            gpu_available: Whether GPU is available
            cpu_cores: Number of CPU cores
            
        Returns:
            Hardware-optimized configuration
        """
        optimized_config = config
        
        # GPU optimization
        optimized_config.performance.use_gpu = gpu_available
        if gpu_available:
            optimized_config.performance.mixed_precision = True
            optimized_config.performance.gradient_checkpointing = True
        
        # Memory optimization
        if available_memory_gb < 4:
            # Low memory: aggressive optimization
            optimized_config.performance.memory_optimization = 2
            optimized_config.performance.tda_batch_size = 2
            optimized_config.performance.attention_chunk_size = 128
            optimized_config.persistent_homology.max_points = 500
        elif available_memory_gb > 16:
            # High memory: can use larger batches and more features
            optimized_config.performance.memory_optimization = 0
            optimized_config.performance.tda_batch_size = 16
            optimized_config.performance.attention_chunk_size = 1024
            optimized_config.persistent_homology.max_points = 5000
        
        # CPU optimization
        optimized_config.performance.tda_workers = min(cpu_cores, 8)
        optimized_config.persistent_homology.n_workers = min(cpu_cores // 2, 4)
        
        return optimized_config
    
    def save_config(self, config: TDAConfig, filepath: Union[str, Path]):
        """Save configuration to file."""
        filepath = Path(filepath)
        
        # Convert to dictionary
        config_dict = self._config_to_dict(config)
        
        # Save based on file extension
        if filepath.suffix.lower() == '.json':
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif filepath.suffix.lower() in ['.yml', '.yaml']:
            with open(filepath, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    def load_config(self, filepath: Union[str, Path]) -> TDAConfig:
        """Load configuration from file."""
        filepath = Path(filepath)
        
        # Load based on file extension
        if filepath.suffix.lower() == '.json':
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
        elif filepath.suffix.lower() in ['.yml', '.yaml']:
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        return self._dict_to_config(config_dict)
    
    def _config_to_dict(self, config: TDAConfig) -> Dict[str, Any]:
        """Convert TDAConfig to dictionary."""
        # This is a simplified version - in practice, you'd want more sophisticated serialization
        return {
            'takens_embedding': {
                'dims': config.takens_embedding.dims,
                'delays': config.takens_embedding.delays,
                'strategy': config.takens_embedding.strategy.value,
                'auto_optimize': config.takens_embedding.auto_optimize,
                'optimization_method': config.takens_embedding.optimization_method,
                'max_auto_dim': config.takens_embedding.max_auto_dim,
                'max_auto_delay': config.takens_embedding.max_auto_delay,
                'quality_threshold': config.takens_embedding.quality_threshold
            },
            'persistent_homology': {
                'backend': config.persistent_homology.backend.value,
                'max_dimension': config.persistent_homology.max_dimension,
                'persistence_threshold': config.persistent_homology.persistence_threshold,
                'distance_batch_size': config.persistent_homology.distance_batch_size,
                'max_points': config.persistent_homology.max_points,
                'subsampling_strategy': config.persistent_homology.subsampling_strategy,
                'distance_metric': config.persistent_homology.distance_metric,
                'parallel': config.persistent_homology.parallel,
                'n_workers': config.persistent_homology.n_workers
            },
            'persistence_landscapes': {
                'resolution': config.persistence_landscapes.resolution,
                'num_landscapes': config.persistence_landscapes.num_landscapes,
                'integration_orders': config.persistence_landscapes.integration_orders,
                'enable_resampling': config.persistence_landscapes.enable_resampling,
                'resampling_resolution': config.persistence_landscapes.resampling_resolution,
                'extract_statistics': config.persistence_landscapes.extract_statistics
            },
            'feature_fusion': {
                'fusion_strategy': config.feature_fusion.fusion_strategy.value,
                'available_strategies': [s.value for s in config.feature_fusion.available_strategies],
                'attention_heads': config.feature_fusion.attention_heads,
                'attention_hidden_dim': config.feature_fusion.attention_hidden_dim,
                'attention_dropout': config.feature_fusion.attention_dropout,
                'attention_temperature': config.feature_fusion.attention_temperature,
                'tda_weight': config.feature_fusion.tda_weight,
                'freq_weight': config.feature_fusion.freq_weight,
                'adaptive_weighting': config.feature_fusion.adaptive_weighting,
                'weight_adaptation_rate': config.feature_fusion.weight_adaptation_rate
            },
            'tda_losses': {
                'enable_tda_losses': config.tda_losses.enable_tda_losses,
                'persistence_loss_weight': config.tda_losses.persistence_loss_weight,
                'persistence_stability_weight': config.tda_losses.persistence_stability_weight,
                'persistence_consistency_weight': config.tda_losses.persistence_consistency_weight,
                'consistency_loss_weight': config.tda_losses.consistency_loss_weight,
                'cross_scale_consistency_weight': config.tda_losses.cross_scale_consistency_weight,
                'frequency_consistency_weight': config.tda_losses.frequency_consistency_weight,
                'preservation_loss_weight': config.tda_losses.preservation_loss_weight,
                'trend_preservation_weight': config.tda_losses.trend_preservation_weight,
                'periodicity_preservation_weight': config.tda_losses.periodicity_preservation_weight,
                'complexity_preservation_weight': config.tda_losses.complexity_preservation_weight,
                'adaptive_loss_weighting': config.tda_losses.adaptive_loss_weighting,
                'loss_adaptation_rate': config.tda_losses.loss_adaptation_rate,
                'min_loss_weight': config.tda_losses.min_loss_weight,
                'max_loss_weight': config.tda_losses.max_loss_weight
            },
            'performance': {
                'use_gpu': config.performance.use_gpu,
                'gpu_device': config.performance.gpu_device,
                'mixed_precision': config.performance.mixed_precision,
                'gradient_checkpointing': config.performance.gradient_checkpointing,
                'tda_batch_size': config.performance.tda_batch_size,
                'enable_caching': config.performance.enable_caching,
                'cache_size_mb': config.performance.cache_size_mb,
                'parallel_tda': config.performance.parallel_tda,
                'tda_workers': config.performance.tda_workers,
                'memory_optimization': config.performance.memory_optimization,
                'attention_chunk_size': config.performance.attention_chunk_size
            },
            'global': {
                'enable_tda': config.enable_tda,
                'tda_feature_selection': config.tda_feature_selection,
                'feature_selection_method': config.feature_selection_method,
                'max_tda_features': config.max_tda_features,
                'integration_points': config.integration_points,
                'debug_mode': config.debug_mode,
                'save_intermediate_results': config.save_intermediate_results,
                'log_tda_performance': config.log_tda_performance
            }
        }
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> TDAConfig:
        """Convert dictionary to TDAConfig."""
        # This is a simplified version - in practice, you'd want more sophisticated deserialization
        config = TDAConfig()
        
        # Update takens embedding config
        if 'takens_embedding' in config_dict:
            te_dict = config_dict['takens_embedding']
            config.takens_embedding.dims = te_dict.get('dims', config.takens_embedding.dims)
            config.takens_embedding.delays = te_dict.get('delays', config.takens_embedding.delays)
            config.takens_embedding.strategy = TDAStrategy(te_dict.get('strategy', config.takens_embedding.strategy.value))
            config.takens_embedding.auto_optimize = te_dict.get('auto_optimize', config.takens_embedding.auto_optimize)
            config.takens_embedding.optimization_method = te_dict.get('optimization_method', config.takens_embedding.optimization_method)
            config.takens_embedding.max_auto_dim = te_dict.get('max_auto_dim', config.takens_embedding.max_auto_dim)
            config.takens_embedding.max_auto_delay = te_dict.get('max_auto_delay', config.takens_embedding.max_auto_delay)
            config.takens_embedding.quality_threshold = te_dict.get('quality_threshold', config.takens_embedding.quality_threshold)
        
        # Update other configs similarly...
        # (Implementation would continue for all config sections)
        
        return config 