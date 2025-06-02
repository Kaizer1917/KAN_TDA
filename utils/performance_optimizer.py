"""
Performance Optimization and Analysis Tools for TDA-KAN Integration

This module provides comprehensive performance analysis, bottleneck detection,
and optimization recommendations for the TDA-KAN system.
"""

import time
import psutil
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
from contextlib import contextmanager
import threading
import gc

@dataclass
class PerformanceConfig:
    """Configuration for performance analysis."""
    enable_memory_tracking: bool = True
    enable_gpu_tracking: bool = True
    enable_timing_analysis: bool = True
    enable_bottleneck_detection: bool = True
    sampling_interval: float = 0.1  # seconds
    memory_threshold_mb: float = 1000.0
    time_threshold_ms: float = 100.0
    gpu_memory_threshold_mb: float = 500.0
    profile_gradients: bool = True
    profile_activations: bool = True
    save_detailed_logs: bool = True
    output_dir: str = "performance_logs"

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    cpu_utilization: float = 0.0
    gpu_utilization: float = 0.0
    peak_memory_mb: float = 0.0
    memory_efficiency: float = 0.0
    throughput: float = 0.0
    bottlenecks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

class PerformanceProfiler:
    """Advanced performance profiler for TDA-KAN components."""
    
    def __init__(self, config: PerformanceConfig = None):
        self.config = config or PerformanceConfig()
        self.metrics_history = defaultdict(list)
        self.timing_data = defaultdict(list)
        self.memory_data = defaultdict(list)
        self.gpu_data = defaultdict(list)
        self.bottlenecks = defaultdict(list)
        self.active_timers = {}
        self.monitoring_thread = None
        self.monitoring_active = False
        
        # Create output directory
        Path(self.config.output_dir).mkdir(exist_ok=True)
    
    @contextmanager
    def profile_section(self, section_name: str):
        """Context manager for profiling code sections."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_gpu_memory = self._get_gpu_memory()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            end_gpu_memory = self._get_gpu_memory()
            
            execution_time = (end_time - start_time) * 1000  # ms
            memory_delta = end_memory - start_memory
            gpu_memory_delta = end_gpu_memory - start_gpu_memory
            
            self.timing_data[section_name].append(execution_time)
            self.memory_data[section_name].append(memory_delta)
            self.gpu_data[section_name].append(gpu_memory_delta)
            
            # Check for bottlenecks
            if execution_time > self.config.time_threshold_ms:
                self.bottlenecks[section_name].append(f"Slow execution: {execution_time:.2f}ms")
            
            if abs(memory_delta) > self.config.memory_threshold_mb:
                self.bottlenecks[section_name].append(f"High memory usage: {memory_delta:.2f}MB")
    
    def start_monitoring(self):
        """Start continuous system monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitor_system)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop continuous system monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
    
    def _monitor_system(self):
        """Continuous system monitoring loop."""
        while self.monitoring_active:
            timestamp = time.time()
            
            # System metrics
            cpu_percent = psutil.cpu_percent()
            memory_info = psutil.virtual_memory()
            
            self.metrics_history['timestamp'].append(timestamp)
            self.metrics_history['cpu_percent'].append(cpu_percent)
            self.metrics_history['memory_percent'].append(memory_info.percent)
            self.metrics_history['memory_used_mb'].append(memory_info.used / 1024 / 1024)
            
            # GPU metrics
            if torch.cuda.is_available() and self.config.enable_gpu_tracking:
                gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_utilization = self._get_gpu_utilization()
                self.metrics_history['gpu_memory_mb'].append(gpu_memory)
                self.metrics_history['gpu_utilization'].append(gpu_utilization)
            
            time.sleep(self.config.sampling_interval)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if not self.config.enable_memory_tracking:
            return 0.0
        
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def _get_gpu_memory(self) -> float:
        """Get current GPU memory usage in MB."""
        if not torch.cuda.is_available() or not self.config.enable_gpu_tracking:
            return 0.0
        
        return torch.cuda.memory_allocated() / 1024 / 1024
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu
        except:
            return 0.0
    
    def analyze_performance(self) -> PerformanceMetrics:
        """Analyze collected performance data."""
        metrics = PerformanceMetrics()
        
        if not self.timing_data:
            return metrics
        
        # Aggregate timing data
        total_time = sum(sum(times) for times in self.timing_data.values())
        metrics.execution_time = total_time
        
        # Memory analysis
        if self.memory_data:
            memory_values = [val for vals in self.memory_data.values() for val in vals]
            metrics.memory_usage_mb = sum(memory_values)
            metrics.peak_memory_mb = max(memory_values) if memory_values else 0.0
        
        # GPU analysis
        if self.gpu_data:
            gpu_values = [val for vals in self.gpu_data.values() for val in vals]
            metrics.gpu_memory_mb = sum(gpu_values)
        
        # System metrics
        if self.metrics_history['cpu_percent']:
            metrics.cpu_utilization = np.mean(self.metrics_history['cpu_percent'])
        
        if self.metrics_history.get('gpu_utilization'):
            metrics.gpu_utilization = np.mean(self.metrics_history['gpu_utilization'])
        
        # Calculate efficiency
        if metrics.peak_memory_mb > 0:
            metrics.memory_efficiency = metrics.memory_usage_mb / metrics.peak_memory_mb
        
        # Collect bottlenecks
        for section, issues in self.bottlenecks.items():
            metrics.bottlenecks.extend([f"{section}: {issue}" for issue in issues])
        
        # Generate recommendations
        metrics.recommendations = self._generate_recommendations(metrics)
        
        return metrics
    
    def _generate_recommendations(self, metrics: PerformanceMetrics) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Memory recommendations
        if metrics.memory_usage_mb > self.config.memory_threshold_mb:
            recommendations.append("Consider reducing batch size or using gradient checkpointing")
            recommendations.append("Implement memory-efficient attention mechanisms")
        
        if metrics.memory_efficiency < 0.7:
            recommendations.append("Optimize memory allocation patterns")
            recommendations.append("Use in-place operations where possible")
        
        # GPU recommendations
        if metrics.gpu_memory_mb > self.config.gpu_memory_threshold_mb:
            recommendations.append("Consider mixed precision training")
            recommendations.append("Implement gradient accumulation")
        
        if metrics.gpu_utilization < 70:
            recommendations.append("Increase batch size to improve GPU utilization")
            recommendations.append("Consider data loading optimizations")
        
        # CPU recommendations
        if metrics.cpu_utilization > 90:
            recommendations.append("Optimize data preprocessing pipeline")
            recommendations.append("Consider multiprocessing for data loading")
        
        # Timing recommendations
        slow_sections = [section for section, times in self.timing_data.items() 
                        if np.mean(times) > self.config.time_threshold_ms]
        
        for section in slow_sections:
            recommendations.append(f"Optimize {section} - consider caching or algorithmic improvements")
        
        return recommendations
    
    def generate_report(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        metrics = self.analyze_performance()
        
        report = {
            'summary': {
                'total_execution_time_ms': metrics.execution_time,
                'peak_memory_mb': metrics.peak_memory_mb,
                'gpu_memory_mb': metrics.gpu_memory_mb,
                'cpu_utilization': metrics.cpu_utilization,
                'gpu_utilization': metrics.gpu_utilization,
                'memory_efficiency': metrics.memory_efficiency
            },
            'timing_breakdown': {
                section: {
                    'mean_ms': np.mean(times),
                    'std_ms': np.std(times),
                    'min_ms': np.min(times),
                    'max_ms': np.max(times),
                    'total_ms': np.sum(times),
                    'count': len(times)
                }
                for section, times in self.timing_data.items()
            },
            'memory_analysis': {
                section: {
                    'mean_mb': np.mean(memory),
                    'std_mb': np.std(memory),
                    'peak_mb': np.max(memory),
                    'total_mb': np.sum(memory)
                }
                for section, memory in self.memory_data.items()
            },
            'bottlenecks': metrics.bottlenecks,
            'recommendations': metrics.recommendations,
            'system_metrics': {
                'cpu_history': self.metrics_history.get('cpu_percent', []),
                'memory_history': self.metrics_history.get('memory_used_mb', []),
                'gpu_memory_history': self.metrics_history.get('gpu_memory_mb', []),
                'timestamps': self.metrics_history.get('timestamp', [])
            }
        }
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        return report
    
    def visualize_performance(self, save_dir: Optional[str] = None):
        """Create performance visualization plots."""
        if save_dir:
            Path(save_dir).mkdir(exist_ok=True)
        
        # Timing breakdown
        if self.timing_data:
            plt.figure(figsize=(12, 6))
            sections = list(self.timing_data.keys())
            mean_times = [np.mean(self.timing_data[section]) for section in sections]
            
            plt.bar(sections, mean_times)
            plt.title('Average Execution Time by Section')
            plt.xlabel('Section')
            plt.ylabel('Time (ms)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(f"{save_dir}/timing_breakdown.png")
            plt.show()
        
        # Memory usage over time
        if self.metrics_history.get('memory_used_mb'):
            plt.figure(figsize=(12, 6))
            timestamps = self.metrics_history['timestamp']
            memory_usage = self.metrics_history['memory_used_mb']
            
            plt.plot(timestamps, memory_usage)
            plt.title('Memory Usage Over Time')
            plt.xlabel('Time')
            plt.ylabel('Memory (MB)')
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(f"{save_dir}/memory_timeline.png")
            plt.show()
        
        # System utilization
        if self.metrics_history.get('cpu_percent'):
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # CPU utilization
            axes[0].plot(self.metrics_history['timestamp'], 
                        self.metrics_history['cpu_percent'])
            axes[0].set_title('CPU Utilization')
            axes[0].set_ylabel('CPU %')
            
            # GPU utilization (if available)
            if self.metrics_history.get('gpu_utilization'):
                axes[1].plot(self.metrics_history['timestamp'], 
                            self.metrics_history['gpu_utilization'])
                axes[1].set_title('GPU Utilization')
                axes[1].set_ylabel('GPU %')
            
            axes[-1].set_xlabel('Time')
            plt.tight_layout()
            
            if save_dir:
                plt.savefig(f"{save_dir}/system_utilization.png")
            plt.show()

class MemoryOptimizer:
    """Memory optimization utilities for TDA-KAN models."""
    
    def __init__(self):
        self.memory_snapshots = []
        self.optimization_history = []
    
    def analyze_memory_usage(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Analyze memory usage of model components."""
        analysis = {
            'parameter_memory': {},
            'buffer_memory': {},
            'total_parameters': 0,
            'total_memory_mb': 0.0
        }
        
        for name, param in model.named_parameters():
            param_memory = param.numel() * param.element_size()
            analysis['parameter_memory'][name] = {
                'shape': list(param.shape),
                'dtype': str(param.dtype),
                'memory_bytes': param_memory,
                'memory_mb': param_memory / 1024 / 1024,
                'requires_grad': param.requires_grad
            }
            analysis['total_parameters'] += param.numel()
            analysis['total_memory_mb'] += param_memory / 1024 / 1024
        
        for name, buffer in model.named_buffers():
            buffer_memory = buffer.numel() * buffer.element_size()
            analysis['buffer_memory'][name] = {
                'shape': list(buffer.shape),
                'dtype': str(buffer.dtype),
                'memory_bytes': buffer_memory,
                'memory_mb': buffer_memory / 1024 / 1024
            }
            analysis['total_memory_mb'] += buffer_memory / 1024 / 1024
        
        return analysis
    
    def suggest_optimizations(self, model: torch.nn.Module) -> List[str]:
        """Suggest memory optimizations for the model."""
        suggestions = []
        analysis = self.analyze_memory_usage(model)
        
        # Check for large parameters
        large_params = [
            name for name, info in analysis['parameter_memory'].items()
            if info['memory_mb'] > 100
        ]
        
        if large_params:
            suggestions.append(f"Consider parameter sharing or low-rank approximations for: {large_params}")
        
        # Check for float64 parameters
        float64_params = [
            name for name, info in analysis['parameter_memory'].items()
            if 'float64' in info['dtype']
        ]
        
        if float64_params:
            suggestions.append(f"Convert to float32 for memory savings: {float64_params}")
        
        # Check total memory
        if analysis['total_memory_mb'] > 1000:
            suggestions.append("Consider model pruning or quantization")
            suggestions.append("Implement gradient checkpointing")
        
        return suggestions
    
    def optimize_model_memory(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply memory optimizations to the model."""
        optimized_model = model
        
        # Convert float64 to float32
        for param in optimized_model.parameters():
            if param.dtype == torch.float64:
                param.data = param.data.float()
        
        # Enable gradient checkpointing if available
        if hasattr(optimized_model, 'gradient_checkpointing_enable'):
            optimized_model.gradient_checkpointing_enable()
        
        return optimized_model

# Convenience functions
def profile_tda_kan_training(model, data_loader, optimizer, num_epochs=1, 
                           config: PerformanceConfig = None):
    """Profile TDA-KAN training performance."""
    profiler = PerformanceProfiler(config)
    profiler.start_monitoring()
    
    try:
        for epoch in range(num_epochs):
            with profiler.profile_section(f"epoch_{epoch}"):
                for batch_idx, (data, target) in enumerate(data_loader):
                    with profiler.profile_section("forward_pass"):
                        output = model(data)
                    
                    with profiler.profile_section("loss_computation"):
                        loss = torch.nn.functional.mse_loss(output, target)
                    
                    with profiler.profile_section("backward_pass"):
                        optimizer.zero_grad()
                        loss.backward()
                    
                    with profiler.profile_section("optimizer_step"):
                        optimizer.step()
                    
                    if batch_idx >= 10:  # Limit for profiling
                        break
    
    finally:
        profiler.stop_monitoring()
    
    return profiler.generate_report()

def analyze_tda_component_performance(tda_component, input_data, 
                                    config: PerformanceConfig = None):
    """Analyze performance of specific TDA components."""
    profiler = PerformanceProfiler(config)
    
    with profiler.profile_section("tda_component_forward"):
        output = tda_component(input_data)
    
    return profiler.analyze_performance() 