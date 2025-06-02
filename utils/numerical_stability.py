"""
Numerical Stability Verification for TDA-KAN_TDA Integration

This module provides comprehensive numerical stability monitoring, verification,
and automatic correction mechanisms for TDA-aware components.
"""

import torch
import torch.nn as nn
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
import logging
from dataclasses import dataclass
from enum import Enum
import time
from contextlib import contextmanager

try:
    from configs.tda_config import TDAConfig
except ImportError:
    warnings.warn("TDA config not found. Some stability features may not work.")


class StabilityLevel(Enum):
    """Enumeration of numerical stability levels."""
    STABLE = "stable"
    WARNING = "warning"
    UNSTABLE = "unstable"
    CRITICAL = "critical"


@dataclass
class StabilityReport:
    """Report of numerical stability analysis."""
    level: StabilityLevel
    issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    metrics: Dict[str, float]
    timestamp: float
    
    def __str__(self) -> str:
        report = f"Stability Level: {self.level.value.upper()}\n"
        report += f"Timestamp: {time.ctime(self.timestamp)}\n"
        
        if self.issues:
            report += f"\nISSUES:\n" + "\n".join(f"- {issue}" for issue in self.issues)
        
        if self.warnings:
            report += f"\nWARNINGS:\n" + "\n".join(f"- {warning}" for warning in self.warnings)
        
        if self.recommendations:
            report += f"\nRECOMMENDATIONS:\n" + "\n".join(f"- {rec}" for rec in self.recommendations)
        
        if self.metrics:
            report += f"\nMETRICS:\n" + "\n".join(f"- {k}: {v:.6e}" for k, v in self.metrics.items())
        
        return report


class NumericalStabilityMonitor:
    """Monitor for tracking numerical stability during computation."""
    
    def __init__(
        self,
        device: torch.device,
        check_frequency: int = 10,
        log_level: int = logging.INFO
    ):
        """
        Initialize stability monitor.
        
        Args:
            device: Device to monitor
            check_frequency: How often to perform stability checks
            log_level: Logging level for stability messages
        """
        self.device = device
        self.check_frequency = check_frequency
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        # Stability tracking
        self.step_count = 0
        self.stability_history = []
        self.current_issues = []
        
        # Thresholds for stability detection
        self.thresholds = {
            'gradient_norm_max': 1e6,
            'gradient_norm_min': 1e-10,
            'weight_norm_max': 1e6,
            'activation_norm_max': 1e6,
            'loss_value_max': 1e6,
            'nan_tolerance': 0,
            'inf_tolerance': 0,
            'condition_number_max': 1e12
        }
        
        # Automatic correction settings
        self.auto_correct = True
        self.correction_history = []
    
    def check_tensor_stability(
        self, 
        tensor: torch.Tensor, 
        name: str = "tensor",
        check_gradients: bool = True
    ) -> StabilityReport:
        """
        Check numerical stability of a tensor.
        
        Args:
            tensor: Tensor to check
            name: Name for logging
            check_gradients: Whether to check gradients if available
            
        Returns:
            Stability report
        """
        issues = []
        warnings = []
        recommendations = []
        metrics = {}
        
        # Basic tensor checks
        if torch.isnan(tensor).any():
            issues.append(f"NaN values detected in {name}")
            recommendations.append(f"Check computation leading to {name}")
        
        if torch.isinf(tensor).any():
            issues.append(f"Infinite values detected in {name}")
            recommendations.append(f"Check for division by zero or overflow in {name}")
        
        # Magnitude checks
        tensor_norm = torch.norm(tensor).item()
        metrics[f'{name}_norm'] = tensor_norm
        
        if tensor_norm > self.thresholds['activation_norm_max']:
            warnings.append(f"Large tensor norm in {name}: {tensor_norm:.2e}")
            recommendations.append(f"Consider gradient clipping or normalization for {name}")
        
        if tensor_norm < 1e-10 and tensor.numel() > 1:
            warnings.append(f"Very small tensor norm in {name}: {tensor_norm:.2e}")
            recommendations.append(f"Check for vanishing values in {name}")
        
        # Range checks
        tensor_min = tensor.min().item()
        tensor_max = tensor.max().item()
        metrics[f'{name}_min'] = tensor_min
        metrics[f'{name}_max'] = tensor_max
        metrics[f'{name}_range'] = tensor_max - tensor_min
        
        # Gradient checks
        if check_gradients and tensor.grad is not None:
            grad_norm = torch.norm(tensor.grad).item()
            metrics[f'{name}_grad_norm'] = grad_norm
            
            if torch.isnan(tensor.grad).any():
                issues.append(f"NaN gradients in {name}")
                recommendations.append(f"Check backward computation for {name}")
            
            if torch.isinf(tensor.grad).any():
                issues.append(f"Infinite gradients in {name}")
                recommendations.append(f"Apply gradient clipping for {name}")
            
            if grad_norm > self.thresholds['gradient_norm_max']:
                warnings.append(f"Large gradient norm in {name}: {grad_norm:.2e}")
                recommendations.append(f"Apply gradient clipping for {name}")
            
            if grad_norm < self.thresholds['gradient_norm_min']:
                warnings.append(f"Very small gradient norm in {name}: {grad_norm:.2e}")
                recommendations.append(f"Check for vanishing gradients in {name}")
        
        # Determine stability level
        if issues:
            level = StabilityLevel.CRITICAL
        elif len(warnings) > 2:
            level = StabilityLevel.UNSTABLE
        elif warnings:
            level = StabilityLevel.WARNING
        else:
            level = StabilityLevel.STABLE
        
        return StabilityReport(
            level=level,
            issues=issues,
            warnings=warnings,
            recommendations=recommendations,
            metrics=metrics,
            timestamp=time.time()
        )
    
    def check_model_stability(
        self, 
        model: nn.Module, 
        input_data: Optional[torch.Tensor] = None
    ) -> StabilityReport:
        """
        Check numerical stability of a model.
        
        Args:
            model: Model to check
            input_data: Optional input data for forward pass analysis
            
        Returns:
            Comprehensive stability report
        """
        all_issues = []
        all_warnings = []
        all_recommendations = []
        all_metrics = {}
        
        # Check model parameters
        for name, param in model.named_parameters():
            if param is not None:
                param_report = self.check_tensor_stability(param, f"param_{name}")
                all_issues.extend(param_report.issues)
                all_warnings.extend(param_report.warnings)
                all_recommendations.extend(param_report.recommendations)
                all_metrics.update(param_report.metrics)
        
        # Check model buffers
        for name, buffer in model.named_buffers():
            if buffer is not None:
                buffer_report = self.check_tensor_stability(buffer, f"buffer_{name}", check_gradients=False)
                all_issues.extend(buffer_report.issues)
                all_warnings.extend(buffer_report.warnings)
                all_recommendations.extend(buffer_report.recommendations)
                all_metrics.update(buffer_report.metrics)
        
        # Forward pass analysis if input provided
        if input_data is not None:
            try:
                with torch.no_grad():
                    output = model(input_data)
                    output_report = self.check_tensor_stability(output, "model_output", check_gradients=False)
                    all_issues.extend(output_report.issues)
                    all_warnings.extend(output_report.warnings)
                    all_recommendations.extend(output_report.recommendations)
                    all_metrics.update(output_report.metrics)
            except Exception as e:
                all_issues.append(f"Forward pass failed: {str(e)}")
                all_recommendations.append("Check model architecture and input compatibility")
        
        # Model-specific checks
        total_params = sum(p.numel() for p in model.parameters())
        all_metrics['total_parameters'] = total_params
        
        # Check for parameter initialization issues
        zero_params = sum((p == 0).sum().item() for p in model.parameters())
        if zero_params > total_params * 0.5:
            all_warnings.append(f"Many zero parameters: {zero_params}/{total_params}")
            all_recommendations.append("Check parameter initialization")
        
        # Determine overall stability level
        if all_issues:
            level = StabilityLevel.CRITICAL
        elif len(all_warnings) > 5:
            level = StabilityLevel.UNSTABLE
        elif all_warnings:
            level = StabilityLevel.WARNING
        else:
            level = StabilityLevel.STABLE
        
        return StabilityReport(
            level=level,
            issues=all_issues,
            warnings=all_warnings,
            recommendations=all_recommendations,
            metrics=all_metrics,
            timestamp=time.time()
        )
    
    def check_loss_stability(
        self, 
        loss_values: Dict[str, torch.Tensor],
        step: int
    ) -> StabilityReport:
        """
        Check stability of loss values during training.
        
        Args:
            loss_values: Dictionary of loss components
            step: Current training step
            
        Returns:
            Loss stability report
        """
        issues = []
        warnings = []
        recommendations = []
        metrics = {}
        
        for loss_name, loss_value in loss_values.items():
            # Check for NaN/Inf
            if torch.isnan(loss_value).any():
                issues.append(f"NaN loss in {loss_name}")
                recommendations.append(f"Check computation of {loss_name}")
            
            if torch.isinf(loss_value).any():
                issues.append(f"Infinite loss in {loss_name}")
                recommendations.append(f"Check for division by zero in {loss_name}")
            
            # Check magnitude
            loss_val = loss_value.item() if loss_value.numel() == 1 else torch.norm(loss_value).item()
            metrics[f'{loss_name}_value'] = loss_val
            
            if loss_val > self.thresholds['loss_value_max']:
                warnings.append(f"Very large loss in {loss_name}: {loss_val:.2e}")
                recommendations.append(f"Consider loss scaling for {loss_name}")
            
            if loss_val < 0 and 'loss' in loss_name.lower():
                warnings.append(f"Negative loss in {loss_name}: {loss_val:.2e}")
                recommendations.append(f"Check loss computation for {loss_name}")
        
        # Check loss progression if we have history
        if len(self.stability_history) > 5:
            recent_losses = [report.metrics.get('total_loss', 0) for report in self.stability_history[-5:]]
            if all(loss > recent_losses[0] * 2 for loss in recent_losses[1:]):
                warnings.append("Loss consistently increasing")
                recommendations.append("Check learning rate or model architecture")
        
        # Determine stability level
        if issues:
            level = StabilityLevel.CRITICAL
        elif len(warnings) > 2:
            level = StabilityLevel.UNSTABLE
        elif warnings:
            level = StabilityLevel.WARNING
        else:
            level = StabilityLevel.STABLE
        
        return StabilityReport(
            level=level,
            issues=issues,
            warnings=warnings,
            recommendations=recommendations,
            metrics=metrics,
            timestamp=time.time()
        )
    
    def apply_automatic_corrections(
        self, 
        model: nn.Module, 
        report: StabilityReport
    ) -> List[str]:
        """
        Apply automatic corrections based on stability report.
        
        Args:
            model: Model to correct
            report: Stability report with issues
            
        Returns:
            List of corrections applied
        """
        if not self.auto_correct:
            return []
        
        corrections = []
        
        # Fix NaN parameters
        for name, param in model.named_parameters():
            if param is not None and torch.isnan(param).any():
                # Replace NaN with small random values
                nan_mask = torch.isnan(param)
                param.data[nan_mask] = torch.randn_like(param.data[nan_mask]) * 0.01
                corrections.append(f"Replaced NaN values in {name}")
        
        # Fix infinite parameters
        for name, param in model.named_parameters():
            if param is not None and torch.isinf(param).any():
                # Clamp infinite values
                param.data = torch.clamp(param.data, -1e6, 1e6)
                corrections.append(f"Clamped infinite values in {name}")
        
        # Fix very large parameters
        for name, param in model.named_parameters():
            if param is not None:
                param_norm = torch.norm(param).item()
                if param_norm > self.thresholds['weight_norm_max']:
                    # Normalize large parameters
                    param.data = param.data / (param_norm / 1e3)
                    corrections.append(f"Normalized large parameters in {name}")
        
        # Fix zero gradients (if in training mode)
        if model.training:
            for name, param in model.named_parameters():
                if param.grad is not None and torch.norm(param.grad).item() < 1e-12:
                    # Add small noise to zero gradients
                    param.grad.data += torch.randn_like(param.grad.data) * 1e-8
                    corrections.append(f"Added noise to zero gradients in {name}")
        
        self.correction_history.extend(corrections)
        return corrections
    
    @contextmanager
    def stability_context(self, model: nn.Module, check_interval: int = 1):
        """
        Context manager for automatic stability monitoring.
        
        Args:
            model: Model to monitor
            check_interval: Steps between stability checks
        """
        try:
            yield self
        finally:
            if self.step_count % check_interval == 0:
                report = self.check_model_stability(model)
                self.stability_history.append(report)
                
                if report.level in [StabilityLevel.UNSTABLE, StabilityLevel.CRITICAL]:
                    self.logger.warning(f"Stability issue detected: {report.level.value}")
                    corrections = self.apply_automatic_corrections(model, report)
                    if corrections:
                        self.logger.info(f"Applied corrections: {corrections}")
            
            self.step_count += 1


class TDAStabilityVerifier:
    """Specialized stability verifier for TDA components."""
    
    def __init__(self, device: torch.device):
        """Initialize TDA stability verifier."""
        self.device = device
        self.monitor = NumericalStabilityMonitor(device)
        self.logger = logging.getLogger(__name__)
    
    def verify_takens_embedding_stability(
        self, 
        time_series: torch.Tensor,
        embedding_dim: int,
        delay: int
    ) -> StabilityReport:
        """
        Verify stability of Takens embedding computation.
        
        Args:
            time_series: Input time series
            embedding_dim: Embedding dimension
            delay: Time delay
            
        Returns:
            Stability report for Takens embedding
        """
        issues = []
        warnings = []
        recommendations = []
        metrics = {}
        
        # Check input time series
        input_report = self.monitor.check_tensor_stability(time_series, "time_series", check_gradients=False)
        issues.extend(input_report.issues)
        warnings.extend(input_report.warnings)
        
        # Check embedding parameters
        if embedding_dim < 2:
            issues.append(f"Embedding dimension too small: {embedding_dim}")
            recommendations.append("Use embedding dimension >= 2")
        
        if embedding_dim > time_series.shape[-1] // 2:
            warnings.append(f"Embedding dimension large relative to series length")
            recommendations.append("Consider reducing embedding dimension")
        
        if delay < 1:
            issues.append(f"Invalid delay parameter: {delay}")
            recommendations.append("Use positive delay values")
        
        # Check for sufficient data points
        required_length = (embedding_dim - 1) * delay + 1
        if time_series.shape[-1] < required_length:
            issues.append(f"Insufficient data for embedding: need {required_length}, got {time_series.shape[-1]}")
            recommendations.append("Increase time series length or reduce embedding parameters")
        
        # Simulate embedding computation stability
        try:
            # Create embedding indices
            indices = torch.arange(0, required_length, delay, device=self.device)
            if len(indices) != embedding_dim:
                issues.append("Embedding index computation failed")
        except Exception as e:
            issues.append(f"Embedding computation error: {str(e)}")
        
        # Check for numerical conditioning
        series_std = torch.std(time_series).item()
        series_mean = torch.mean(time_series).item()
        metrics['series_std'] = series_std
        metrics['series_mean'] = series_mean
        
        if series_std < 1e-10:
            warnings.append("Very low variance in time series")
            recommendations.append("Check for constant or near-constant values")
        
        # Determine stability level
        if issues:
            level = StabilityLevel.CRITICAL
        elif len(warnings) > 2:
            level = StabilityLevel.UNSTABLE
        elif warnings:
            level = StabilityLevel.WARNING
        else:
            level = StabilityLevel.STABLE
        
        return StabilityReport(
            level=level,
            issues=issues,
            warnings=warnings,
            recommendations=recommendations,
            metrics=metrics,
            timestamp=time.time()
        )
    
    def verify_persistence_computation_stability(
        self, 
        point_cloud: torch.Tensor,
        max_dimension: int = 1
    ) -> StabilityReport:
        """
        Verify stability of persistent homology computation.
        
        Args:
            point_cloud: Point cloud data
            max_dimension: Maximum homology dimension
            
        Returns:
            Stability report for persistence computation
        """
        issues = []
        warnings = []
        recommendations = []
        metrics = {}
        
        # Check point cloud
        cloud_report = self.monitor.check_tensor_stability(point_cloud, "point_cloud", check_gradients=False)
        issues.extend(cloud_report.issues)
        warnings.extend(cloud_report.warnings)
        
        # Check point cloud properties
        n_points, dim = point_cloud.shape[-2:]
        metrics['n_points'] = n_points
        metrics['point_dim'] = dim
        
        if n_points < 3:
            issues.append(f"Too few points for homology: {n_points}")
            recommendations.append("Need at least 3 points for meaningful homology")
        
        if n_points > 5000:
            warnings.append(f"Large point cloud may be slow: {n_points} points")
            recommendations.append("Consider subsampling for efficiency")
        
        if dim > 10:
            warnings.append(f"High dimensional point cloud: {dim}D")
            recommendations.append("High dimensions may cause numerical issues")
        
        # Check for degenerate configurations
        # Compute pairwise distances
        try:
            dists = torch.cdist(point_cloud, point_cloud)
            min_dist = torch.min(dists[dists > 0]).item()
            max_dist = torch.max(dists).item()
            
            metrics['min_distance'] = min_dist
            metrics['max_distance'] = max_dist
            metrics['distance_ratio'] = max_dist / min_dist if min_dist > 0 else float('inf')
            
            if min_dist < 1e-10:
                warnings.append("Very close or duplicate points detected")
                recommendations.append("Remove duplicate points or add noise")
            
            if metrics['distance_ratio'] > 1e6:
                warnings.append("Large distance ratio may cause numerical issues")
                recommendations.append("Consider rescaling point cloud")
            
        except Exception as e:
            issues.append(f"Distance computation failed: {str(e)}")
        
        # Check homology dimension parameter
        if max_dimension < 0:
            issues.append(f"Invalid homology dimension: {max_dimension}")
            recommendations.append("Use non-negative homology dimensions")
        
        if max_dimension > dim - 1:
            warnings.append(f"Homology dimension {max_dimension} >= point dimension {dim}")
            recommendations.append("Homology dimension should be < point dimension")
        
        # Determine stability level
        if issues:
            level = StabilityLevel.CRITICAL
        elif len(warnings) > 2:
            level = StabilityLevel.UNSTABLE
        elif warnings:
            level = StabilityLevel.WARNING
        else:
            level = StabilityLevel.STABLE
        
        return StabilityReport(
            level=level,
            issues=issues,
            warnings=warnings,
            recommendations=recommendations,
            metrics=metrics,
            timestamp=time.time()
        )
    
    def verify_fusion_stability(
        self, 
        tda_features: torch.Tensor,
        freq_features: torch.Tensor,
        fusion_weights: Optional[torch.Tensor] = None
    ) -> StabilityReport:
        """
        Verify stability of TDA-frequency feature fusion.
        
        Args:
            tda_features: TDA feature tensor
            freq_features: Frequency feature tensor
            fusion_weights: Optional fusion weights
            
        Returns:
            Stability report for fusion operation
        """
        issues = []
        warnings = []
        recommendations = []
        metrics = {}
        
        # Check input features
        tda_report = self.monitor.check_tensor_stability(tda_features, "tda_features")
        freq_report = self.monitor.check_tensor_stability(freq_features, "freq_features")
        
        issues.extend(tda_report.issues)
        issues.extend(freq_report.issues)
        warnings.extend(tda_report.warnings)
        warnings.extend(freq_report.warnings)
        
        # Check feature compatibility
        if tda_features.shape[0] != freq_features.shape[0]:
            issues.append("Batch size mismatch between TDA and frequency features")
            recommendations.append("Ensure consistent batch sizes")
        
        # Check feature magnitudes
        tda_norm = torch.norm(tda_features).item()
        freq_norm = torch.norm(freq_features).item()
        
        metrics['tda_feature_norm'] = tda_norm
        metrics['freq_feature_norm'] = freq_norm
        
        if tda_norm > 0 and freq_norm > 0:
            magnitude_ratio = max(tda_norm, freq_norm) / min(tda_norm, freq_norm)
            metrics['magnitude_ratio'] = magnitude_ratio
            
            if magnitude_ratio > 100:
                warnings.append(f"Large magnitude difference between features: {magnitude_ratio:.1f}")
                recommendations.append("Consider feature normalization")
        
        # Check fusion weights if provided
        if fusion_weights is not None:
            weight_report = self.monitor.check_tensor_stability(fusion_weights, "fusion_weights")
            issues.extend(weight_report.issues)
            warnings.extend(weight_report.warnings)
            
            # Check weight properties
            if torch.any(fusion_weights < 0):
                warnings.append("Negative fusion weights detected")
                recommendations.append("Consider using non-negative weights")
            
            weight_sum = torch.sum(fusion_weights).item()
            if abs(weight_sum - 1.0) > 0.1:
                warnings.append(f"Fusion weights don't sum to 1: {weight_sum:.3f}")
                recommendations.append("Consider normalizing fusion weights")
        
        # Check for potential numerical issues in fusion
        if tda_features.dtype != freq_features.dtype:
            warnings.append("Different dtypes in fusion inputs")
            recommendations.append("Ensure consistent data types")
        
        # Determine stability level
        if issues:
            level = StabilityLevel.CRITICAL
        elif len(warnings) > 2:
            level = StabilityLevel.UNSTABLE
        elif warnings:
            level = StabilityLevel.WARNING
        else:
            level = StabilityLevel.STABLE
        
        return StabilityReport(
            level=level,
            issues=issues,
            warnings=warnings,
            recommendations=recommendations,
            metrics=metrics,
            timestamp=time.time()
        )
    
    def verify_complete_pipeline_stability(
        self, 
        model: nn.Module,
        sample_input: torch.Tensor,
        config: Optional['TDAConfig'] = None
    ) -> StabilityReport:
        """
        Verify stability of complete TDA-KAN_TDA pipeline.
        
        Args:
            model: Complete TDA-KAN_TDA model
            sample_input: Sample input for testing
            config: Optional TDA configuration
            
        Returns:
            Comprehensive pipeline stability report
        """
        all_issues = []
        all_warnings = []
        all_recommendations = []
        all_metrics = {}
        
        # Check model stability
        model_report = self.monitor.check_model_stability(model, sample_input)
        all_issues.extend(model_report.issues)
        all_warnings.extend(model_report.warnings)
        all_recommendations.extend(model_report.recommendations)
        all_metrics.update(model_report.metrics)
        
        # Check input stability
        input_report = self.monitor.check_tensor_stability(sample_input, "pipeline_input", check_gradients=False)
        all_issues.extend(input_report.issues)
        all_warnings.extend(input_report.warnings)
        
        # Configuration-specific checks
        if config is not None:
            # Check if configuration parameters are numerically sound
            if hasattr(config, 'takens_embedding'):
                if max(config.takens_embedding.dims) > 20:
                    all_warnings.append("Very high embedding dimensions may cause instability")
                    all_recommendations.append("Consider reducing embedding dimensions")
            
            if hasattr(config, 'persistent_homology'):
                if config.persistent_homology.max_points > 10000:
                    all_warnings.append("Large point clouds may cause memory/numerical issues")
                    all_recommendations.append("Consider reducing max_points")
        
        # Test forward pass stability
        try:
            with torch.no_grad():
                output = model(sample_input)
                output_report = self.monitor.check_tensor_stability(output, "pipeline_output", check_gradients=False)
                all_issues.extend(output_report.issues)
                all_warnings.extend(output_report.warnings)
                all_metrics.update(output_report.metrics)
        except Exception as e:
            all_issues.append(f"Pipeline forward pass failed: {str(e)}")
            all_recommendations.append("Check model architecture and input compatibility")
        
        # Test gradient flow if model is in training mode
        if model.training:
            try:
                model.zero_grad()
                output = model(sample_input)
                loss = output.sum()
                loss.backward()
                
                # Check gradients
                grad_norms = []
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = torch.norm(param.grad).item()
                        grad_norms.append(grad_norm)
                        if torch.isnan(param.grad).any():
                            all_issues.append(f"NaN gradients in {name}")
                        if torch.isinf(param.grad).any():
                            all_issues.append(f"Infinite gradients in {name}")
                
                if grad_norms:
                    all_metrics['mean_grad_norm'] = np.mean(grad_norms)
                    all_metrics['max_grad_norm'] = np.max(grad_norms)
                    all_metrics['min_grad_norm'] = np.min(grad_norms)
                    
                    if np.max(grad_norms) > 1e6:
                        all_warnings.append("Very large gradients detected")
                        all_recommendations.append("Apply gradient clipping")
                    
                    if np.min(grad_norms) < 1e-10:
                        all_warnings.append("Very small gradients detected")
                        all_recommendations.append("Check for vanishing gradients")
                
            except Exception as e:
                all_issues.append(f"Gradient computation failed: {str(e)}")
                all_recommendations.append("Check backward pass implementation")
        
        # Determine overall stability level
        if all_issues:
            level = StabilityLevel.CRITICAL
        elif len(all_warnings) > 5:
            level = StabilityLevel.UNSTABLE
        elif all_warnings:
            level = StabilityLevel.WARNING
        else:
            level = StabilityLevel.STABLE
        
        return StabilityReport(
            level=level,
            issues=all_issues,
            warnings=all_warnings,
            recommendations=all_recommendations,
            metrics=all_metrics,
            timestamp=time.time()
        )


def create_stability_test_suite(device: torch.device) -> Dict[str, Callable]:
    """
    Create a comprehensive stability test suite.
    
    Args:
        device: Device for testing
        
    Returns:
        Dictionary of stability test functions
    """
    verifier = TDAStabilityVerifier(device)
    
    def test_basic_tensor_operations():
        """Test basic tensor operation stability."""
        # Test various tensor operations
        x = torch.randn(100, 10, device=device)
        
        tests = {
            'addition': lambda: x + x,
            'multiplication': lambda: x * x,
            'division': lambda: x / (x + 1e-8),
            'exponential': lambda: torch.exp(torch.clamp(x, -10, 10)),
            'logarithm': lambda: torch.log(torch.abs(x) + 1e-8),
            'matrix_multiply': lambda: torch.mm(x, x.T),
            'svd': lambda: torch.svd(x),
            'eigenvalues': lambda: torch.eig(torch.mm(x, x.T))
        }
        
        results = {}
        for name, operation in tests.items():
            try:
                result = operation()
                report = verifier.monitor.check_tensor_stability(result, name, check_gradients=False)
                results[name] = report.level
            except Exception as e:
                results[name] = StabilityLevel.CRITICAL
        
        return results
    
    def test_gradient_flow():
        """Test gradient flow stability."""
        x = torch.randn(10, 5, device=device, requires_grad=True)
        
        # Test various functions
        functions = {
            'linear': lambda x: torch.sum(x),
            'quadratic': lambda x: torch.sum(x ** 2),
            'exponential': lambda x: torch.sum(torch.exp(torch.clamp(x, -10, 10))),
            'trigonometric': lambda x: torch.sum(torch.sin(x)),
            'complex': lambda x: torch.sum(torch.tanh(x) * torch.exp(-x**2))
        }
        
        results = {}
        for name, func in functions.items():
            try:
                x.grad = None
                y = func(x)
                y.backward()
                
                if x.grad is not None:
                    report = verifier.monitor.check_tensor_stability(x.grad, f"{name}_grad", check_gradients=False)
                    results[name] = report.level
                else:
                    results[name] = StabilityLevel.CRITICAL
            except Exception as e:
                results[name] = StabilityLevel.CRITICAL
        
        return results
    
    def test_numerical_precision():
        """Test numerical precision limits."""
        results = {}
        
        # Test different data types
        dtypes = [torch.float16, torch.float32, torch.float64]
        
        for dtype in dtypes:
            try:
                x = torch.randn(100, device=device, dtype=dtype)
                
                # Test precision-sensitive operations
                small_val = torch.tensor(1e-7, device=device, dtype=dtype)
                large_val = torch.tensor(1e7, device=device, dtype=dtype)
                
                # Division by small number
                div_result = x / small_val
                
                # Multiplication by large number
                mul_result = x * large_val
                
                # Check stability
                div_report = verifier.monitor.check_tensor_stability(div_result, f"div_{dtype}", check_gradients=False)
                mul_report = verifier.monitor.check_tensor_stability(mul_result, f"mul_{dtype}", check_gradients=False)
                
                if div_report.level == StabilityLevel.STABLE and mul_report.level == StabilityLevel.STABLE:
                    results[str(dtype)] = StabilityLevel.STABLE
                else:
                    results[str(dtype)] = StabilityLevel.WARNING
                    
            except Exception as e:
                results[str(dtype)] = StabilityLevel.CRITICAL
        
        return results
    
    return {
        'basic_operations': test_basic_tensor_operations,
        'gradient_flow': test_gradient_flow,
        'numerical_precision': test_numerical_precision
    }


def run_comprehensive_stability_verification(
    model: nn.Module,
    sample_input: torch.Tensor,
    config: Optional['TDAConfig'] = None,
    device: Optional[torch.device] = None
) -> Dict[str, StabilityReport]:
    """
    Run comprehensive stability verification on TDA-KAN_TDA system.
    
    Args:
        model: TDA-KAN_TDA model to verify
        sample_input: Sample input for testing
        config: Optional TDA configuration
        device: Device for computation
        
    Returns:
        Dictionary of stability reports for different components
    """
    if device is None:
        device = next(model.parameters()).device
    
    verifier = TDAStabilityVerifier(device)
    reports = {}
    
    # Overall pipeline stability
    reports['pipeline'] = verifier.verify_complete_pipeline_stability(model, sample_input, config)
    
    # Component-specific stability tests
    if hasattr(model, 'tda_components'):
        # Test TDA-specific components if available
        try:
            # Simulate Takens embedding
            time_series = sample_input[:, :, 0]  # Use first feature
            reports['takens_embedding'] = verifier.verify_takens_embedding_stability(
                time_series, embedding_dim=3, delay=1
            )
            
            # Simulate point cloud for persistence
            point_cloud = torch.randn(sample_input.shape[0], 100, 3, device=device)
            reports['persistence'] = verifier.verify_persistence_computation_stability(point_cloud)
            
        except Exception as e:
            reports['tda_components'] = StabilityReport(
                level=StabilityLevel.CRITICAL,
                issues=[f"TDA component testing failed: {str(e)}"],
                warnings=[],
                recommendations=["Check TDA component implementation"],
                metrics={},
                timestamp=time.time()
            )
    
    # Test fusion stability if fusion components exist
    try:
        tda_features = torch.randn(sample_input.shape[0], 10, device=device)
        freq_features = torch.randn(sample_input.shape[0], sample_input.shape[1], 16, device=device)
        reports['fusion'] = verifier.verify_fusion_stability(tda_features, freq_features)
    except Exception as e:
        reports['fusion'] = StabilityReport(
            level=StabilityLevel.WARNING,
            issues=[],
            warnings=[f"Fusion testing failed: {str(e)}"],
            recommendations=["Check fusion component implementation"],
            metrics={},
            timestamp=time.time()
        )
    
    # Run basic stability tests
    test_suite = create_stability_test_suite(device)
    for test_name, test_func in test_suite.items():
        try:
            test_results = test_func()
            # Convert test results to stability report
            issues = [f"{op} failed" for op, level in test_results.items() if level == StabilityLevel.CRITICAL]
            warnings = [f"{op} unstable" for op, level in test_results.items() if level in [StabilityLevel.UNSTABLE, StabilityLevel.WARNING]]
            
            if issues:
                level = StabilityLevel.CRITICAL
            elif warnings:
                level = StabilityLevel.WARNING
            else:
                level = StabilityLevel.STABLE
            
            reports[test_name] = StabilityReport(
                level=level,
                issues=issues,
                warnings=warnings,
                recommendations=[],
                metrics={op: level.value for op, level in test_results.items()},
                timestamp=time.time()
            )
        except Exception as e:
            reports[test_name] = StabilityReport(
                level=StabilityLevel.CRITICAL,
                issues=[f"Test {test_name} failed: {str(e)}"],
                warnings=[],
                recommendations=[],
                metrics={},
                timestamp=time.time()
            )
    
    return reports 