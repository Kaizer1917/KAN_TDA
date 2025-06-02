"""
Parameter Validation and Optimization for TDA-KAN_TDA Integration

This module provides comprehensive parameter validation, optimization, and
automatic tuning capabilities for TDA-aware parameters.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
import warnings
from scipy.optimize import minimize, differential_evolution
from sklearn.model_selection import ParameterGrid
import time
import logging
from dataclasses import dataclass
from enum import Enum

try:
    from configs.tda_config import TDAConfig, TDAConfigManager
    from layers.TakensEmbedding import TakensEmbedding
    from layers.PersistentHomology import PersistentHomologyComputer
    from utils.persistence_landscapes import TopologicalFeatureExtractor
except ImportError:
    warnings.warn("TDA modules not found. Some optimization features may not work.")


class OptimizationStrategy(Enum):
    """Enumeration of optimization strategies."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    EVOLUTIONARY = "evolutionary"
    GRADIENT_BASED = "gradient_based"


@dataclass
class ValidationResult:
    """Result of parameter validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    
    def __str__(self) -> str:
        result = f"Validation: {'PASSED' if self.is_valid else 'FAILED'}\n"
        if self.errors:
            result += f"Errors: {', '.join(self.errors)}\n"
        if self.warnings:
            result += f"Warnings: {', '.join(self.warnings)}\n"
        if self.suggestions:
            result += f"Suggestions: {', '.join(self.suggestions)}\n"
        return result


@dataclass
class OptimizationResult:
    """Result of parameter optimization."""
    best_params: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    total_time: float
    n_evaluations: int
    convergence_info: Dict[str, Any]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get optimization summary."""
        return {
            'best_score': self.best_score,
            'total_time': self.total_time,
            'n_evaluations': self.n_evaluations,
            'improvement': self.optimization_history[-1]['score'] - self.optimization_history[0]['score'] if self.optimization_history else 0,
            'converged': self.convergence_info.get('converged', False)
        }


class TDAParameterValidator:
    """Comprehensive parameter validator for TDA configurations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_config(self, config: TDAConfig) -> ValidationResult:
        """
        Comprehensive validation of TDA configuration.
        
        Args:
            config: TDA configuration to validate
            
        Returns:
            Validation result with errors, warnings, and suggestions
        """
        errors = []
        warnings = []
        suggestions = []
        
        # Basic configuration validation
        config_errors = config.validate()
        for section, section_errors in config_errors.items():
            errors.extend([f"{section}: {error}" for error in section_errors])
        
        # Cross-component validation
        cross_errors, cross_warnings, cross_suggestions = self._validate_cross_component_consistency(config)
        errors.extend(cross_errors)
        warnings.extend(cross_warnings)
        suggestions.extend(cross_suggestions)
        
        # Performance validation
        perf_warnings, perf_suggestions = self._validate_performance_settings(config)
        warnings.extend(perf_warnings)
        suggestions.extend(perf_suggestions)
        
        # Resource validation
        resource_warnings, resource_suggestions = self._validate_resource_requirements(config)
        warnings.extend(resource_warnings)
        suggestions.extend(resource_suggestions)
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def _validate_cross_component_consistency(
        self, 
        config: TDAConfig
    ) -> Tuple[List[str], List[str], List[str]]:
        """Validate consistency across different components."""
        errors = []
        warnings = []
        suggestions = []
        
        # Check embedding dimensions vs homology computation
        if config.takens_embedding.dims:
            max_embedding_dim = max(config.takens_embedding.dims)
            if max_embedding_dim > 10 and config.persistent_homology.max_points < 1000:
                warnings.append(
                    f"High embedding dimension ({max_embedding_dim}) with low max_points "
                    f"({config.persistent_homology.max_points}) may cause sparse point clouds"
                )
                suggestions.append("Consider increasing max_points or reducing embedding dimensions")
        
        # Check landscape resolution vs embedding complexity
        total_embeddings = len(config.takens_embedding.dims) * len(config.takens_embedding.delays)
        if total_embeddings > 20 and config.persistence_landscapes.resolution > 500:
            warnings.append(
                f"Many embeddings ({total_embeddings}) with high resolution "
                f"({config.persistence_landscapes.resolution}) may be computationally expensive"
            )
            suggestions.append("Consider reducing resolution or number of embeddings")
        
        # Check fusion strategy vs TDA feature count
        if config.feature_fusion.fusion_strategy.value == "attention" and config.max_tda_features > 100:
            warnings.append(
                f"Attention fusion with many TDA features ({config.max_tda_features}) "
                "may require large attention hidden dimensions"
            )
            suggestions.append("Consider increasing attention_hidden_dim or reducing max_tda_features")
        
        # Check loss weights consistency
        if config.tda_losses.enable_tda_losses:
            total_loss_weight = (
                config.tda_losses.persistence_loss_weight +
                config.tda_losses.consistency_loss_weight +
                config.tda_losses.preservation_loss_weight
            )
            if total_loss_weight > 1.0:
                warnings.append(
                    f"Total TDA loss weight ({total_loss_weight:.2f}) is high and may dominate training"
                )
                suggestions.append("Consider reducing TDA loss weights or using adaptive weighting")
        
        # Check memory settings consistency
        if config.performance.memory_optimization == 0 and config.performance.tda_batch_size > 16:
            warnings.append(
                "No memory optimization with large TDA batch size may cause OOM errors"
            )
            suggestions.append("Enable memory optimization or reduce tda_batch_size")
        
        return errors, warnings, suggestions
    
    def _validate_performance_settings(
        self, 
        config: TDAConfig
    ) -> Tuple[List[str], List[str]]:
        """Validate performance-related settings."""
        warnings = []
        suggestions = []
        
        # Check GPU settings
        if config.performance.use_gpu and not torch.cuda.is_available():
            warnings.append("GPU requested but CUDA not available")
            suggestions.append("Set use_gpu=False or install CUDA")
        
        # Check parallel settings
        if config.performance.parallel_tda and config.performance.tda_workers > 8:
            warnings.append(
                f"Many TDA workers ({config.performance.tda_workers}) may cause overhead"
            )
            suggestions.append("Consider reducing tda_workers to 4-8")
        
        # Check caching settings
        if config.performance.enable_caching and config.performance.cache_size_mb < 256:
            warnings.append(
                f"Small cache size ({config.performance.cache_size_mb}MB) may reduce effectiveness"
            )
            suggestions.append("Consider increasing cache_size_mb to at least 512MB")
        
        # Check mixed precision with TDA
        if config.performance.mixed_precision and config.persistent_homology.backend.value == "gudhi":
            warnings.append("Mixed precision may cause numerical issues with GUDHI backend")
            suggestions.append("Consider using Ripser backend with mixed precision")
        
        return warnings, suggestions
    
    def _validate_resource_requirements(
        self, 
        config: TDAConfig
    ) -> Tuple[List[str], List[str]]:
        """Validate resource requirements and constraints."""
        warnings = []
        suggestions = []
        
        # Estimate memory requirements
        estimated_memory = self._estimate_memory_usage(config)
        
        if estimated_memory > 8.0:  # GB
            warnings.append(
                f"Estimated memory usage ({estimated_memory:.1f}GB) is high"
            )
            suggestions.append("Consider enabling aggressive memory optimization")
        
        # Estimate computation time
        estimated_time = self._estimate_computation_time(config)
        
        if estimated_time > 60:  # seconds per batch
            warnings.append(
                f"Estimated computation time ({estimated_time:.1f}s/batch) is high"
            )
            suggestions.append("Consider reducing TDA complexity or enabling parallelization")
        
        return warnings, suggestions
    
    def _estimate_memory_usage(self, config: TDAConfig) -> float:
        """Estimate memory usage in GB."""
        # Simplified estimation
        base_memory = 0.5  # Base KAN_TDA memory
        
        # TDA feature memory
        n_embeddings = len(config.takens_embedding.dims) * len(config.takens_embedding.delays)
        tda_memory = n_embeddings * config.max_tda_features * 4e-9  # 4 bytes per float
        
        # Attention memory (if using attention fusion)
        if config.feature_fusion.fusion_strategy.value == "attention":
            seq_len = 96  # Assume default
            attention_memory = (
                config.performance.tda_batch_size * 
                config.feature_fusion.attention_heads * 
                seq_len * seq_len * 4e-9
            )
        else:
            attention_memory = 0
        
        # Cache memory
        cache_memory = config.performance.cache_size_mb / 1024 if config.performance.enable_caching else 0
        
        total_memory = base_memory + tda_memory + attention_memory + cache_memory
        
        return total_memory
    
    def _estimate_computation_time(self, config: TDAConfig) -> float:
        """Estimate computation time in seconds per batch."""
        # Simplified estimation based on complexity
        base_time = 0.1  # Base KAN_TDA time
        
        # TDA computation time
        n_embeddings = len(config.takens_embedding.dims) * len(config.takens_embedding.delays)
        max_points = config.persistent_homology.max_points
        
        # Homology computation is roughly O(n^3)
        homology_time = n_embeddings * (max_points / 1000) ** 2 * 0.01
        
        # Landscape computation
        landscape_time = (
            n_embeddings * 
            config.persistence_landscapes.num_landscapes * 
            config.persistence_landscapes.resolution / 100000
        )
        
        # Fusion time
        if config.feature_fusion.fusion_strategy.value == "attention":
            fusion_time = 0.05
        else:
            fusion_time = 0.01
        
        total_time = base_time + homology_time + landscape_time + fusion_time
        
        # Apply parallelization speedup
        if config.performance.parallel_tda:
            speedup = min(config.performance.tda_workers, 4)  # Diminishing returns
            total_time /= speedup
        
        return total_time


class TDAParameterOptimizer:
    """Automatic parameter optimizer for TDA configurations."""
    
    def __init__(
        self,
        strategy: OptimizationStrategy = OptimizationStrategy.BAYESIAN,
        n_trials: int = 50,
        timeout: Optional[float] = None,
        random_state: int = 42
    ):
        """
        Initialize parameter optimizer.
        
        Args:
            strategy: Optimization strategy to use
            n_trials: Maximum number of optimization trials
            timeout: Maximum optimization time in seconds
            random_state: Random seed for reproducibility
        """
        self.strategy = strategy
        self.n_trials = n_trials
        self.timeout = timeout
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        
        # Set random seeds
        np.random.seed(random_state)
        torch.manual_seed(random_state)
    
    def optimize_config(
        self,
        base_config: TDAConfig,
        objective_function: Callable[[TDAConfig], float],
        parameter_space: Dict[str, Any],
        maximize: bool = False
    ) -> OptimizationResult:
        """
        Optimize TDA configuration parameters.
        
        Args:
            base_config: Base configuration to optimize
            objective_function: Function that evaluates configuration quality
            parameter_space: Dictionary defining parameter search space
            maximize: Whether to maximize (True) or minimize (False) objective
            
        Returns:
            Optimization result with best parameters and history
        """
        start_time = time.time()
        optimization_history = []
        
        self.logger.info(f"Starting parameter optimization with {self.strategy.value}")
        
        if self.strategy == OptimizationStrategy.GRID_SEARCH:
            result = self._grid_search_optimization(
                base_config, objective_function, parameter_space, maximize
            )
        elif self.strategy == OptimizationStrategy.RANDOM_SEARCH:
            result = self._random_search_optimization(
                base_config, objective_function, parameter_space, maximize
            )
        elif self.strategy == OptimizationStrategy.EVOLUTIONARY:
            result = self._evolutionary_optimization(
                base_config, objective_function, parameter_space, maximize
            )
        else:
            raise ValueError(f"Optimization strategy {self.strategy} not implemented")
        
        total_time = time.time() - start_time
        
        return OptimizationResult(
            best_params=result['best_params'],
            best_score=result['best_score'],
            optimization_history=result['history'],
            total_time=total_time,
            n_evaluations=len(result['history']),
            convergence_info=result.get('convergence_info', {})
        )
    
    def _grid_search_optimization(
        self,
        base_config: TDAConfig,
        objective_function: Callable[[TDAConfig], float],
        parameter_space: Dict[str, Any],
        maximize: bool
    ) -> Dict[str, Any]:
        """Grid search optimization."""
        param_grid = ParameterGrid(parameter_space)
        
        best_score = float('-inf') if maximize else float('inf')
        best_params = None
        history = []
        
        for i, params in enumerate(param_grid):
            if self.timeout and time.time() - self.start_time > self.timeout:
                break
            
            # Update configuration with current parameters
            config = self._update_config_with_params(base_config, params)
            
            # Evaluate configuration
            try:
                score = objective_function(config)
                
                # Track best
                if (maximize and score > best_score) or (not maximize and score < best_score):
                    best_score = score
                    best_params = params.copy()
                
                history.append({
                    'iteration': i,
                    'params': params.copy(),
                    'score': score
                })
                
                self.logger.debug(f"Trial {i}: score={score:.4f}, params={params}")
                
            except Exception as e:
                self.logger.warning(f"Trial {i} failed: {e}")
                history.append({
                    'iteration': i,
                    'params': params.copy(),
                    'score': float('nan'),
                    'error': str(e)
                })
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'history': history,
            'convergence_info': {'converged': True}
        }
    
    def _random_search_optimization(
        self,
        base_config: TDAConfig,
        objective_function: Callable[[TDAConfig], float],
        parameter_space: Dict[str, Any],
        maximize: bool
    ) -> Dict[str, Any]:
        """Random search optimization."""
        best_score = float('-inf') if maximize else float('inf')
        best_params = None
        history = []
        
        for i in range(self.n_trials):
            if self.timeout and time.time() - self.start_time > self.timeout:
                break
            
            # Sample random parameters
            params = self._sample_random_params(parameter_space)
            
            # Update configuration
            config = self._update_config_with_params(base_config, params)
            
            # Evaluate configuration
            try:
                score = objective_function(config)
                
                # Track best
                if (maximize and score > best_score) or (not maximize and score < best_score):
                    best_score = score
                    best_params = params.copy()
                
                history.append({
                    'iteration': i,
                    'params': params.copy(),
                    'score': score
                })
                
                self.logger.debug(f"Trial {i}: score={score:.4f}")
                
            except Exception as e:
                self.logger.warning(f"Trial {i} failed: {e}")
                history.append({
                    'iteration': i,
                    'params': params.copy(),
                    'score': float('nan'),
                    'error': str(e)
                })
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'history': history,
            'convergence_info': {'converged': False}
        }
    
    def _evolutionary_optimization(
        self,
        base_config: TDAConfig,
        objective_function: Callable[[TDAConfig], float],
        parameter_space: Dict[str, Any],
        maximize: bool
    ) -> Dict[str, Any]:
        """Evolutionary optimization using differential evolution."""
        # Convert parameter space to bounds for differential evolution
        bounds = []
        param_names = []
        
        for param_name, param_range in parameter_space.items():
            if isinstance(param_range, (list, tuple)) and len(param_range) == 2:
                bounds.append(param_range)
                param_names.append(param_name)
            elif isinstance(param_range, list):
                # Discrete values - use indices
                bounds.append((0, len(param_range) - 1))
                param_names.append(param_name)
        
        history = []
        
        def objective_wrapper(x):
            """Wrapper for differential evolution."""
            # Convert array to parameter dictionary
            params = {}
            for i, param_name in enumerate(param_names):
                param_range = parameter_space[param_name]
                if isinstance(param_range, list) and not (len(param_range) == 2 and isinstance(param_range[0], (int, float))):
                    # Discrete values
                    idx = int(round(x[i]))
                    idx = max(0, min(idx, len(param_range) - 1))
                    params[param_name] = param_range[idx]
                else:
                    params[param_name] = x[i]
            
            # Update configuration
            config = self._update_config_with_params(base_config, params)
            
            try:
                score = objective_function(config)
                
                history.append({
                    'iteration': len(history),
                    'params': params.copy(),
                    'score': score
                })
                
                return -score if maximize else score
                
            except Exception as e:
                self.logger.warning(f"Evaluation failed: {e}")
                return float('inf')
        
        # Run differential evolution
        result = differential_evolution(
            objective_wrapper,
            bounds,
            maxiter=self.n_trials // 10,  # Adjust iterations for population-based method
            seed=self.random_state,
            disp=False
        )
        
        # Find best from history
        if maximize:
            best_idx = max(range(len(history)), key=lambda i: history[i]['score'])
        else:
            best_idx = min(range(len(history)), key=lambda i: history[i]['score'])
        
        best_params = history[best_idx]['params']
        best_score = history[best_idx]['score']
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'history': history,
            'convergence_info': {
                'converged': result.success,
                'message': result.message,
                'n_iterations': result.nit
            }
        }
    
    def _sample_random_params(self, parameter_space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample random parameters from parameter space."""
        params = {}
        
        for param_name, param_range in parameter_space.items():
            if isinstance(param_range, list):
                if len(param_range) == 2 and isinstance(param_range[0], (int, float)) and isinstance(param_range[1], (int, float)):
                    # Continuous range
                    if isinstance(param_range[0], int):
                        params[param_name] = np.random.randint(param_range[0], param_range[1] + 1)
                    else:
                        params[param_name] = np.random.uniform(param_range[0], param_range[1])
                else:
                    # Discrete choices - handle nested lists properly
                    if all(isinstance(item, list) for item in param_range):
                        # List of lists - choose one list
                        params[param_name] = param_range[np.random.randint(len(param_range))]
                    else:
                        # Simple list of values
                        params[param_name] = param_range[np.random.randint(len(param_range))]
            elif isinstance(param_range, tuple) and len(param_range) == 2:
                # Continuous range
                params[param_name] = np.random.uniform(param_range[0], param_range[1])
        
        return params
    
    def _update_config_with_params(self, base_config: TDAConfig, params: Dict[str, Any]) -> TDAConfig:
        """Update configuration with parameter values."""
        # Create a copy of the base configuration
        import copy
        config = copy.deepcopy(base_config)
        
        # Update parameters using dot notation
        for param_path, value in params.items():
            self._set_nested_param(config, param_path, value)
        
        return config
    
    def _set_nested_param(self, config: TDAConfig, param_path: str, value: Any):
        """Set nested parameter using dot notation (e.g., 'takens_embedding.dims')."""
        parts = param_path.split('.')
        obj = config
        
        # Navigate to the parent object
        for part in parts[:-1]:
            obj = getattr(obj, part)
        
        # Set the final parameter
        setattr(obj, parts[-1], value)


class TDAParameterTuner:
    """High-level interface for TDA parameter tuning."""
    
    def __init__(self):
        self.validator = TDAParameterValidator()
        self.config_manager = TDAConfigManager()
        self.logger = logging.getLogger(__name__)
    
    def auto_tune_for_dataset(
        self,
        dataset_characteristics: Dict[str, Any],
        validation_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        optimization_budget: int = 50,
        hardware_constraints: Optional[Dict[str, Any]] = None
    ) -> TDAConfig:
        """
        Automatically tune TDA configuration for a specific dataset.
        
        Args:
            dataset_characteristics: Dataset properties
            validation_data: Optional validation data for optimization
            optimization_budget: Number of optimization trials
            hardware_constraints: Hardware limitations
            
        Returns:
            Optimized TDA configuration
        """
        self.logger.info("Starting automatic TDA parameter tuning")
        
        # Create initial configuration based on dataset
        base_config = self.config_manager.create_config_for_dataset(dataset_characteristics)
        
        # Apply hardware constraints if provided
        if hardware_constraints:
            base_config = self.config_manager.optimize_config_for_hardware(
                base_config, **hardware_constraints
            )
        
        # Validate initial configuration
        validation_result = self.validator.validate_config(base_config)
        if not validation_result.is_valid:
            self.logger.warning(f"Initial configuration has issues: {validation_result}")
            # Apply suggestions to fix issues
            base_config = self._apply_validation_suggestions(base_config, validation_result)
        
        # If no validation data provided, return the base configuration
        if validation_data is None:
            self.logger.info("No validation data provided, returning base configuration")
            return base_config
        
        # Define optimization objective
        def objective_function(config: TDAConfig) -> float:
            return self._evaluate_config_performance(config, validation_data)
        
        # Define parameter search space
        parameter_space = self._create_parameter_search_space(dataset_characteristics)
        
        # Run optimization
        optimizer = TDAParameterOptimizer(
            strategy=OptimizationStrategy.RANDOM_SEARCH,
            n_trials=optimization_budget
        )
        
        optimization_result = optimizer.optimize_config(
            base_config, objective_function, parameter_space, maximize=True
        )
        
        # Update configuration with best parameters
        optimized_config = optimizer._update_config_with_params(
            base_config, optimization_result.best_params
        )
        
        self.logger.info(
            f"Optimization completed. Best score: {optimization_result.best_score:.4f}, "
            f"Evaluations: {optimization_result.n_evaluations}, "
            f"Time: {optimization_result.total_time:.1f}s"
        )
        
        return optimized_config
    
    def _apply_validation_suggestions(
        self, 
        config: TDAConfig, 
        validation_result: ValidationResult
    ) -> TDAConfig:
        """Apply validation suggestions to fix configuration issues."""
        # This is a simplified implementation
        # In practice, you'd parse suggestions and apply fixes automatically
        
        for suggestion in validation_result.suggestions:
            if "reduce embedding dimensions" in suggestion.lower():
                config.takens_embedding.dims = [2, 3, 5]
            elif "increase max_points" in suggestion.lower():
                config.persistent_homology.max_points = min(
                    config.persistent_homology.max_points * 2, 5000
                )
            elif "enable memory optimization" in suggestion.lower():
                config.performance.memory_optimization = 2
        
        return config
    
    def _evaluate_config_performance(
        self, 
        config: TDAConfig, 
        validation_data: Tuple[torch.Tensor, torch.Tensor]
    ) -> float:
        """Evaluate configuration performance on validation data."""
        # This is a placeholder implementation
        # In practice, you'd train a model with the configuration and evaluate performance
        
        try:
            # Simulate performance evaluation
            # Higher scores for balanced configurations
            score = 0.0
            
            # Reward moderate complexity
            n_embeddings = len(config.takens_embedding.dims) * len(config.takens_embedding.delays)
            if 5 <= n_embeddings <= 15:
                score += 0.3
            
            # Reward appropriate homology dimension
            if config.persistent_homology.max_dimension == 1:
                score += 0.2
            elif config.persistent_homology.max_dimension == 2:
                score += 0.1
            
            # Reward efficient fusion strategy
            if config.feature_fusion.fusion_strategy.value in ['gate', 'early']:
                score += 0.2
            
            # Reward memory optimization
            if config.performance.memory_optimization > 0:
                score += 0.1
            
            # Add some randomness to simulate real performance variation
            score += np.random.normal(0, 0.05)
            
            return max(0, min(1, score))  # Clamp to [0, 1]
            
        except Exception as e:
            self.logger.warning(f"Performance evaluation failed: {e}")
            return 0.0
    
    def _create_parameter_search_space(
        self, 
        dataset_characteristics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create parameter search space based on dataset characteristics."""
        seq_len = dataset_characteristics.get('seq_len', 96)
        complexity = dataset_characteristics.get('complexity', 'medium')
        
        # Base parameter space
        parameter_space = {
            'takens_embedding.dims': [[2, 3], [2, 3, 5], [2, 3, 5, 7], [3, 5, 7, 10]],
            'takens_embedding.delays': [[1, 2], [1, 2, 4], [1, 2, 4, 8], [2, 4, 6, 8]],
            'persistent_homology.max_dimension': [1, 2],
            'persistence_landscapes.num_landscapes': [3, 5, 7],
            'feature_fusion.fusion_strategy': ['early', 'gate', 'attention', 'adaptive'],
            'feature_fusion.tda_weight': [0.1, 0.2, 0.3, 0.4, 0.5],
            'max_tda_features': [20, 30, 50, 70, 100]
        }
        
        # Adjust based on sequence length
        if seq_len < 50:
            parameter_space['takens_embedding.dims'] = [[2, 3], [2, 3, 5]]
            parameter_space['takens_embedding.delays'] = [[1, 2], [1, 2, 3]]
        elif seq_len > 200:
            parameter_space['takens_embedding.delays'].append([1, 3, 6, 12])
        
        # Adjust based on complexity
        if complexity == 'low':
            parameter_space['persistent_homology.max_dimension'] = [1]
            parameter_space['persistence_landscapes.num_landscapes'] = [3, 5]
        elif complexity == 'high':
            parameter_space['persistence_landscapes.num_landscapes'] = [5, 7, 10]
            parameter_space['max_tda_features'] = [50, 70, 100, 150]
        
        return parameter_space 