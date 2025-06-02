"""
Memory Efficiency Analysis and Optimization for TDA-KAN Integration

This module provides comprehensive memory analysis, leak detection, and optimization
strategies specifically designed for the TDA-KAN system's complex memory patterns.
"""

import torch
import numpy as np
import psutil
import gc
import tracemalloc
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
import threading
import weakref
from contextlib import contextmanager
import warnings

@dataclass
class MemoryConfig:
    """Configuration for memory analysis."""
    enable_leak_detection: bool = True
    enable_fragmentation_analysis: bool = True
    enable_gradient_memory_tracking: bool = True
    enable_activation_memory_tracking: bool = True
    memory_snapshot_interval: float = 1.0  # seconds
    leak_threshold_mb: float = 50.0
    fragmentation_threshold: float = 0.3
    max_snapshots: int = 1000
    track_tensor_lifecycle: bool = True
    analyze_persistence_memory: bool = True
    output_dir: str = "memory_analysis"

@dataclass
class MemorySnapshot:
    """Single memory snapshot data."""
    timestamp: float
    total_memory_mb: float
    gpu_memory_mb: float
    tensor_count: int
    largest_tensor_mb: float
    fragmentation_ratio: float
    gc_objects: int
    active_tensors: Dict[str, int] = field(default_factory=dict)
    memory_by_component: Dict[str, float] = field(default_factory=dict)

@dataclass
class MemoryAnalysisReport:
    """Comprehensive memory analysis report."""
    peak_memory_mb: float = 0.0
    average_memory_mb: float = 0.0
    memory_efficiency: float = 0.0
    fragmentation_score: float = 0.0
    potential_leaks: List[str] = field(default_factory=list)
    optimization_opportunities: List[str] = field(default_factory=list)
    component_breakdown: Dict[str, float] = field(default_factory=dict)
    tensor_lifecycle_analysis: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

class TensorTracker:
    """Track tensor lifecycle and memory usage."""
    
    def __init__(self):
        self.tensor_registry = weakref.WeakSet()
        self.tensor_history = defaultdict(list)
        self.creation_stacks = {}
        self.size_history = defaultdict(list)
    
    def register_tensor(self, tensor: torch.Tensor, name: str = None):
        """Register a tensor for tracking."""
        if tensor not in self.tensor_registry:
            self.tensor_registry.add(tensor)
            tensor_id = id(tensor)
            
            # Record creation info
            self.tensor_history[tensor_id].append({
                'action': 'created',
                'timestamp': time.time(),
                'size_mb': tensor.numel() * tensor.element_size() / 1024 / 1024,
                'shape': list(tensor.shape),
                'dtype': str(tensor.dtype),
                'device': str(tensor.device),
                'name': name or f"tensor_{tensor_id}"
            })
            
            # Store creation stack
            self.creation_stacks[tensor_id] = tracemalloc.get_traceback()
    
    def track_tensor_operation(self, tensor: torch.Tensor, operation: str):
        """Track an operation on a tensor."""
        tensor_id = id(tensor)
        self.tensor_history[tensor_id].append({
            'action': operation,
            'timestamp': time.time(),
            'size_mb': tensor.numel() * tensor.element_size() / 1024 / 1024
        })
    
    def get_tensor_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tensor statistics."""
        active_tensors = len(self.tensor_registry)
        total_memory = sum(
            t.numel() * t.element_size() / 1024 / 1024 
            for t in self.tensor_registry if t.is_cuda or not t.is_cuda
        )
        
        # Analyze tensor sizes
        sizes = [t.numel() * t.element_size() / 1024 / 1024 for t in self.tensor_registry]
        
        return {
            'active_tensors': active_tensors,
            'total_memory_mb': total_memory,
            'average_tensor_size_mb': np.mean(sizes) if sizes else 0,
            'largest_tensor_mb': max(sizes) if sizes else 0,
            'tensor_size_distribution': {
                'small_tensors_mb': sum(s for s in sizes if s < 1),
                'medium_tensors_mb': sum(s for s in sizes if 1 <= s < 100),
                'large_tensors_mb': sum(s for s in sizes if s >= 100)
            }
        }

class MemoryLeakDetector:
    """Detect and analyze memory leaks in TDA-KAN components."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.baseline_memory = None
        self.memory_growth_history = deque(maxlen=100)
        self.suspected_leaks = []
        self.component_memory_tracking = defaultdict(list)
    
    def set_baseline(self):
        """Set memory baseline for leak detection."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.baseline_memory = {
            'system': psutil.virtual_memory().used / 1024 / 1024,
            'gpu': torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        }
    
    def check_for_leaks(self, component_name: str = "unknown") -> List[str]:
        """Check for potential memory leaks."""
        if not self.baseline_memory:
            self.set_baseline()
            return []
        
        current_memory = {
            'system': psutil.virtual_memory().used / 1024 / 1024,
            'gpu': torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        }
        
        # Calculate growth
        system_growth = current_memory['system'] - self.baseline_memory['system']
        gpu_growth = current_memory['gpu'] - self.baseline_memory['gpu']
        
        self.memory_growth_history.append({
            'timestamp': time.time(),
            'system_growth': system_growth,
            'gpu_growth': gpu_growth,
            'component': component_name
        })
        
        # Detect leaks
        leaks = []
        if system_growth > self.config.leak_threshold_mb:
            leaks.append(f"System memory leak detected: +{system_growth:.2f}MB in {component_name}")
        
        if gpu_growth > self.config.leak_threshold_mb:
            leaks.append(f"GPU memory leak detected: +{gpu_growth:.2f}MB in {component_name}")
        
        # Check for consistent growth pattern
        if len(self.memory_growth_history) >= 10:
            recent_growth = [entry['system_growth'] for entry in list(self.memory_growth_history)[-10:]]
            if all(growth > 0 for growth in recent_growth):
                avg_growth = np.mean(recent_growth)
                if avg_growth > self.config.leak_threshold_mb / 10:
                    leaks.append(f"Consistent memory growth pattern detected: {avg_growth:.2f}MB/iteration")
        
        self.suspected_leaks.extend(leaks)
        return leaks

class MemoryFragmentationAnalyzer:
    """Analyze memory fragmentation patterns."""
    
    def __init__(self):
        self.allocation_history = []
        self.fragmentation_scores = []
    
    def analyze_fragmentation(self) -> Dict[str, float]:
        """Analyze current memory fragmentation."""
        if not torch.cuda.is_available():
            return {'fragmentation_ratio': 0.0, 'largest_free_block': 0.0}
        
        # Get GPU memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        allocated_memory = torch.cuda.memory_allocated() / 1024 / 1024
        cached_memory = torch.cuda.memory_reserved() / 1024 / 1024
        
        # Calculate fragmentation metrics
        free_memory = total_memory - allocated_memory
        fragmentation_ratio = (cached_memory - allocated_memory) / total_memory if total_memory > 0 else 0
        
        # Try to allocate largest possible block to measure fragmentation
        largest_free_block = 0
        try:
            # Binary search for largest allocatable block
            low, high = 0, int(free_memory * 1024 * 1024)  # Convert to bytes
            
            while low < high:
                mid = (low + high + 1) // 2
                try:
                    test_tensor = torch.empty(mid // 4, dtype=torch.float32, device='cuda')
                    del test_tensor
                    torch.cuda.empty_cache()
                    low = mid
                except RuntimeError:
                    high = mid - 1
            
            largest_free_block = low / 1024 / 1024  # Convert back to MB
            
        except Exception:
            largest_free_block = free_memory * 0.5  # Conservative estimate
        
        return {
            'fragmentation_ratio': fragmentation_ratio,
            'largest_free_block_mb': largest_free_block,
            'free_memory_mb': free_memory,
            'cached_memory_mb': cached_memory,
            'allocated_memory_mb': allocated_memory,
            'total_memory_mb': total_memory
        }

class TDAMemoryAnalyzer:
    """Specialized memory analyzer for TDA components."""
    
    def __init__(self, config: MemoryConfig = None):
        self.config = config or MemoryConfig()
        self.tensor_tracker = TensorTracker()
        self.leak_detector = MemoryLeakDetector(self.config)
        self.fragmentation_analyzer = MemoryFragmentationAnalyzer()
        self.snapshots = deque(maxlen=self.config.max_snapshots)
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Component-specific tracking
        self.component_memory = defaultdict(float)
        self.persistence_memory_usage = []
        self.attention_memory_usage = []
        self.kan_layer_memory_usage = []
        
        # Create output directory
        Path(self.config.output_dir).mkdir(exist_ok=True)
    
    def start_monitoring(self):
        """Start continuous memory monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.leak_detector.set_baseline()
        
        if self.config.memory_snapshot_interval > 0:
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
    
    def _monitoring_loop(self):
        """Continuous monitoring loop."""
        while self.monitoring_active:
            self._take_snapshot()
            time.sleep(self.config.memory_snapshot_interval)
    
    def _take_snapshot(self):
        """Take a memory snapshot."""
        # System memory
        memory_info = psutil.virtual_memory()
        total_memory_mb = memory_info.used / 1024 / 1024
        
        # GPU memory
        gpu_memory_mb = 0
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
        
        # Tensor statistics
        tensor_stats = self.tensor_tracker.get_tensor_statistics()
        
        # Fragmentation analysis
        fragmentation_info = self.fragmentation_analyzer.analyze_fragmentation()
        
        # GC statistics
        gc_objects = len(gc.get_objects())
        
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            total_memory_mb=total_memory_mb,
            gpu_memory_mb=gpu_memory_mb,
            tensor_count=tensor_stats['active_tensors'],
            largest_tensor_mb=tensor_stats['largest_tensor_mb'],
            fragmentation_ratio=fragmentation_info['fragmentation_ratio'],
            gc_objects=gc_objects,
            memory_by_component=dict(self.component_memory)
        )
        
        self.snapshots.append(snapshot)
    
    @contextmanager
    def track_component(self, component_name: str):
        """Context manager to track memory usage of specific components."""
        # Record initial state
        initial_memory = self._get_current_memory()
        initial_gpu = self._get_gpu_memory()
        
        try:
            yield
        finally:
            # Record final state
            final_memory = self._get_current_memory()
            final_gpu = self._get_gpu_memory()
            
            # Calculate usage
            memory_delta = final_memory - initial_memory
            gpu_delta = final_gpu - initial_gpu
            
            self.component_memory[component_name] += memory_delta + gpu_delta
            
            # Check for leaks
            leaks = self.leak_detector.check_for_leaks(component_name)
            if leaks:
                print(f"Memory leaks detected in {component_name}: {leaks}")
    
    def _get_current_memory(self) -> float:
        """Get current system memory usage in MB."""
        return psutil.virtual_memory().used / 1024 / 1024
    
    def _get_gpu_memory(self) -> float:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0
    
    def analyze_persistence_memory(self, persistence_diagrams: List[torch.Tensor]):
        """Analyze memory usage of persistence diagrams."""
        total_memory = 0
        for i, diagram in enumerate(persistence_diagrams):
            if diagram is not None:
                memory_mb = diagram.numel() * diagram.element_size() / 1024 / 1024
                total_memory += memory_mb
                self.tensor_tracker.register_tensor(diagram, f"persistence_diagram_{i}")
        
        self.persistence_memory_usage.append({
            'timestamp': time.time(),
            'total_memory_mb': total_memory,
            'num_diagrams': len(persistence_diagrams),
            'average_diagram_size_mb': total_memory / len(persistence_diagrams) if persistence_diagrams else 0
        })
        
        return total_memory
    
    def analyze_attention_memory(self, attention_weights: torch.Tensor, 
                               attention_maps: Optional[torch.Tensor] = None):
        """Analyze memory usage of attention mechanisms."""
        total_memory = 0
        
        # Attention weights
        if attention_weights is not None:
            weights_memory = attention_weights.numel() * attention_weights.element_size() / 1024 / 1024
            total_memory += weights_memory
            self.tensor_tracker.register_tensor(attention_weights, "attention_weights")
        
        # Attention maps
        if attention_maps is not None:
            maps_memory = attention_maps.numel() * attention_maps.element_size() / 1024 / 1024
            total_memory += maps_memory
            self.tensor_tracker.register_tensor(attention_maps, "attention_maps")
        
        self.attention_memory_usage.append({
            'timestamp': time.time(),
            'total_memory_mb': total_memory,
            'weights_memory_mb': weights_memory if attention_weights is not None else 0,
            'maps_memory_mb': maps_memory if attention_maps is not None else 0
        })
        
        return total_memory
    
    def analyze_kan_layer_memory(self, kan_weights: torch.Tensor, 
                               spline_coefficients: Optional[torch.Tensor] = None):
        """Analyze memory usage of KAN layers."""
        total_memory = 0
        
        # KAN weights
        if kan_weights is not None:
            weights_memory = kan_weights.numel() * kan_weights.element_size() / 1024 / 1024
            total_memory += weights_memory
            self.tensor_tracker.register_tensor(kan_weights, "kan_weights")
        
        # Spline coefficients
        if spline_coefficients is not None:
            spline_memory = spline_coefficients.numel() * spline_coefficients.element_size() / 1024 / 1024
            total_memory += spline_memory
            self.tensor_tracker.register_tensor(spline_coefficients, "spline_coefficients")
        
        self.kan_layer_memory_usage.append({
            'timestamp': time.time(),
            'total_memory_mb': total_memory,
            'weights_memory_mb': weights_memory if kan_weights is not None else 0,
            'spline_memory_mb': spline_memory if spline_coefficients is not None else 0
        })
        
        return total_memory
    
    def generate_analysis_report(self) -> MemoryAnalysisReport:
        """Generate comprehensive memory analysis report."""
        if not self.snapshots:
            return MemoryAnalysisReport()
        
        # Basic statistics
        memory_values = [s.total_memory_mb + s.gpu_memory_mb for s in self.snapshots]
        peak_memory = max(memory_values)
        average_memory = np.mean(memory_values)
        
        # Fragmentation analysis
        fragmentation_scores = [s.fragmentation_ratio for s in self.snapshots]
        avg_fragmentation = np.mean(fragmentation_scores)
        
        # Memory efficiency (how much of peak memory is actually used on average)
        memory_efficiency = average_memory / peak_memory if peak_memory > 0 else 0
        
        # Component breakdown
        component_breakdown = dict(self.component_memory)
        
        # Generate recommendations
        recommendations = self._generate_memory_recommendations(
            peak_memory, avg_fragmentation, component_breakdown
        )
        
        # Tensor lifecycle analysis
        tensor_stats = self.tensor_tracker.get_tensor_statistics()
        
        report = MemoryAnalysisReport(
            peak_memory_mb=peak_memory,
            average_memory_mb=average_memory,
            memory_efficiency=memory_efficiency,
            fragmentation_score=avg_fragmentation,
            potential_leaks=self.leak_detector.suspected_leaks,
            component_breakdown=component_breakdown,
            tensor_lifecycle_analysis=tensor_stats,
            recommendations=recommendations
        )
        
        return report
    
    def _generate_memory_recommendations(self, peak_memory: float, 
                                       fragmentation: float,
                                       component_breakdown: Dict[str, float]) -> List[str]:
        """Generate memory optimization recommendations."""
        recommendations = []
        
        # Peak memory recommendations
        if peak_memory > 8000:  # 8GB
            recommendations.append("Consider reducing batch size or using gradient checkpointing")
            recommendations.append("Implement mixed precision training to reduce memory usage")
        
        # Fragmentation recommendations
        if fragmentation > 0.3:
            recommendations.append("High memory fragmentation detected - consider memory pooling")
            recommendations.append("Use torch.cuda.empty_cache() more frequently")
        
        # Component-specific recommendations
        if component_breakdown:
            largest_component = max(component_breakdown.items(), key=lambda x: x[1])
            if largest_component[1] > 1000:  # 1GB
                recommendations.append(f"Optimize {largest_component[0]} - largest memory consumer ({largest_component[1]:.2f}MB)")
        
        # TDA-specific recommendations
        if self.persistence_memory_usage:
            avg_persistence_memory = np.mean([p['total_memory_mb'] for p in self.persistence_memory_usage])
            if avg_persistence_memory > 500:
                recommendations.append("Consider persistence diagram compression or pruning")
        
        if self.attention_memory_usage:
            avg_attention_memory = np.mean([a['total_memory_mb'] for a in self.attention_memory_usage])
            if avg_attention_memory > 1000:
                recommendations.append("Implement memory-efficient attention mechanisms")
        
        # Leak recommendations
        if self.leak_detector.suspected_leaks:
            recommendations.append("Memory leaks detected - review tensor lifecycle management")
            recommendations.append("Ensure proper cleanup of intermediate tensors")
        
        return recommendations
    
    def visualize_memory_usage(self, save_dir: Optional[str] = None):
        """Create memory usage visualizations."""
        if not self.snapshots:
            print("No memory snapshots available for visualization")
            return
        
        if save_dir:
            Path(save_dir).mkdir(exist_ok=True)
        
        # Memory usage over time
        timestamps = [s.timestamp for s in self.snapshots]
        total_memory = [s.total_memory_mb + s.gpu_memory_mb for s in self.snapshots]
        gpu_memory = [s.gpu_memory_mb for s in self.snapshots]
        
        plt.figure(figsize=(15, 10))
        
        # Memory timeline
        plt.subplot(2, 2, 1)
        plt.plot(timestamps, total_memory, label='Total Memory', linewidth=2)
        plt.plot(timestamps, gpu_memory, label='GPU Memory', linewidth=2)
        plt.title('Memory Usage Over Time')
        plt.xlabel('Time')
        plt.ylabel('Memory (MB)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Fragmentation over time
        plt.subplot(2, 2, 2)
        fragmentation = [s.fragmentation_ratio for s in self.snapshots]
        plt.plot(timestamps, fragmentation, color='red', linewidth=2)
        plt.title('Memory Fragmentation Over Time')
        plt.xlabel('Time')
        plt.ylabel('Fragmentation Ratio')
        plt.grid(True, alpha=0.3)
        
        # Component breakdown
        plt.subplot(2, 2, 3)
        if self.component_memory:
            components = list(self.component_memory.keys())
            memory_usage = list(self.component_memory.values())
            plt.pie(memory_usage, labels=components, autopct='%1.1f%%')
            plt.title('Memory Usage by Component')
        
        # Tensor count over time
        plt.subplot(2, 2, 4)
        tensor_counts = [s.tensor_count for s in self.snapshots]
        plt.plot(timestamps, tensor_counts, color='green', linewidth=2)
        plt.title('Active Tensor Count Over Time')
        plt.xlabel('Time')
        plt.ylabel('Tensor Count')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(f"{save_dir}/memory_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_report(self, filepath: str):
        """Save analysis report to file."""
        report = self.generate_analysis_report()
        
        # Convert to serializable format
        report_dict = {
            'peak_memory_mb': report.peak_memory_mb,
            'average_memory_mb': report.average_memory_mb,
            'memory_efficiency': report.memory_efficiency,
            'fragmentation_score': report.fragmentation_score,
            'potential_leaks': report.potential_leaks,
            'optimization_opportunities': report.optimization_opportunities,
            'component_breakdown': report.component_breakdown,
            'tensor_lifecycle_analysis': report.tensor_lifecycle_analysis,
            'recommendations': report.recommendations,
            'snapshots': [
                {
                    'timestamp': s.timestamp,
                    'total_memory_mb': s.total_memory_mb,
                    'gpu_memory_mb': s.gpu_memory_mb,
                    'tensor_count': s.tensor_count,
                    'fragmentation_ratio': s.fragmentation_ratio
                }
                for s in self.snapshots
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)

# Convenience functions
def analyze_tda_kan_memory(model, sample_input, config: MemoryConfig = None):
    """Comprehensive memory analysis for TDA-KAN model."""
    analyzer = TDAMemoryAnalyzer(config)
    analyzer.start_monitoring()
    
    try:
        with analyzer.track_component("model_forward"):
            output = model(sample_input)
        
        with analyzer.track_component("model_backward"):
            if hasattr(output, 'sum'):
                loss = output.sum()
                loss.backward()
        
        time.sleep(2)  # Allow monitoring to collect data
        
    finally:
        analyzer.stop_monitoring()
    
    return analyzer.generate_analysis_report()

def optimize_tda_kan_memory(model, config: MemoryConfig = None):
    """Apply memory optimizations to TDA-KAN model."""
    recommendations = []
    
    # Convert to half precision if possible
    try:
        model.half()
        recommendations.append("Converted model to half precision")
    except:
        recommendations.append("Half precision conversion failed - model may have incompatible layers")
    
    # Enable gradient checkpointing
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        recommendations.append("Enabled gradient checkpointing")
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        recommendations.append("Cleared GPU cache")
    
    return model, recommendations 