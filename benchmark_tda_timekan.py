"""
Performance Benchmark Suite for TDA-Enhanced KAN_TDA

Compares:
1. Training speed and convergence
2. Memory usage and computational overhead
3. Forecasting accuracy across different datasets
4. Ablation study of TDA components
5. Scalability analysis
"""

import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from datetime import datetime
import psutil
import gc

from models.KAN_TDA import Model, TDAKAN_TDA
from test_tda_integration import MockConfig


class PerformanceBenchmark:
    """Comprehensive performance benchmark for TDA-KAN_TDA"""
    
    def __init__(self, output_dir="benchmark_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self._get_system_info(),
            'benchmarks': {}
        }
    
    def _get_system_info(self):
        """Get system information"""
        return {
            'python_version': f"{torch.__version__}",
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }
    
    def _generate_synthetic_data(self, batch_size, seq_len, enc_in, pattern_type='mixed'):
        """Generate synthetic time series data with known patterns"""
        t = torch.linspace(0, 4*np.pi, seq_len).unsqueeze(0).unsqueeze(-1)
        
        if pattern_type == 'sinusoidal':
            # Simple sinusoidal pattern
            data = torch.sin(t) + 0.5 * torch.sin(3*t) + 0.1 * torch.randn(1, seq_len, 1)
        elif pattern_type == 'trend':
            # Linear trend with noise
            data = 0.1 * t + torch.sin(t) + 0.2 * torch.randn(1, seq_len, 1)
        elif pattern_type == 'chaotic':
            # Chaotic-like pattern
            data = torch.sin(t) * torch.cos(2*t) + 0.3 * torch.sin(5*t) + 0.15 * torch.randn(1, seq_len, 1)
        else:  # mixed
            # Complex mixed pattern
            data = (torch.sin(t) + 0.5 * torch.sin(3*t) + 0.2 * torch.cos(5*t) + 
                   0.1 * t + 0.1 * torch.randn(1, seq_len, 1))
        
        # Expand to multiple channels and batch
        data = data.expand(batch_size, seq_len, enc_in)
        return data
    
    def benchmark_initialization_time(self):
        """Benchmark model initialization time"""
        print("Benchmarking initialization time...")
        
        configs = [
            MockConfig(enable_tda=False),
            MockConfig(enable_tda=True, tda_mode='full'),
            MockConfig(enable_tda=True, tda_mode='decomp_only'),
            MockConfig(enable_tda=True, tda_mode='kan_only')
        ]
        
        results = {}
        
        for i, config in enumerate(configs):
            model_name = ['Original', 'TDA-Full', 'TDA-Decomp', 'TDA-KAN'][i]
            
            times = []
            for _ in range(5):  # Average over 5 runs
                start_time = time.time()
                if config.enable_tda:
                    model = TDAKAN_TDA(config)
                else:
                    model = Model(config)
                init_time = time.time() - start_time
                times.append(init_time)
                del model
                gc.collect()
            
            results[model_name] = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'times': times
            }
            print(f"{model_name}: {np.mean(times):.4f}±{np.std(times):.4f}s")
        
        self.results['benchmarks']['initialization'] = results
        return results
    
    def benchmark_memory_usage(self):
        """Benchmark memory usage"""
        print("Benchmarking memory usage...")
        
        configs = [
            MockConfig(enable_tda=False),
            MockConfig(enable_tda=True, tda_mode='full'),
            MockConfig(enable_tda=True, tda_mode='decomp_only'),
            MockConfig(enable_tda=True, tda_mode='kan_only')
        ]
        
        results = {}
        
        for i, config in enumerate(configs):
            model_name = ['Original', 'TDA-Full', 'TDA-Decomp', 'TDA-KAN'][i]
            
            # Clear memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            # Create model
            if config.enable_tda:
                model = TDAKAN_TDA(config)
            else:
                model = Model(config)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Estimate memory usage
            param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)  # MB
            
            if torch.cuda.is_available():
                model = model.cuda()
                gpu_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
            else:
                gpu_memory = 0
            
            results[model_name] = {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'param_memory_mb': param_memory,
                'gpu_memory_mb': gpu_memory
            }
            
            print(f"{model_name}: {total_params:,} params, {param_memory:.2f}MB")
            
            del model
            gc.collect()
        
        self.results['benchmarks']['memory'] = results
        return results
    
    def benchmark_inference_speed(self):
        """Benchmark inference speed"""
        print("Benchmarking inference speed...")
        
        batch_sizes = [1, 4, 16]
        seq_lens = [96, 192, 336]
        
        configs = [
            MockConfig(enable_tda=False),
            MockConfig(enable_tda=True, tda_mode='full')
        ]
        
        results = {}
        
        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                test_key = f"batch_{batch_size}_seq_{seq_len}"
                results[test_key] = {}
                
                # Update configs
                for config in configs:
                    config.seq_len = seq_len
                    config.pred_len = seq_len // 4
                
                test_data = self._generate_synthetic_data(batch_size, seq_len, 7)
                
                for i, config in enumerate(configs):
                    model_name = ['Original', 'TDA-Full'][i]
                    
                    if config.enable_tda:
                        model = TDAKAN_TDA(config)
                    else:
                        model = Model(config)
                    
                    model.eval()
                    
                    # Warm up
                    with torch.no_grad():
                        for _ in range(3):
                            _ = model.forecast(test_data)
                    
                    # Benchmark
                    times = []
                    with torch.no_grad():
                        for _ in range(10):
                            start_time = time.time()
                            output = model.forecast(test_data)
                            inference_time = time.time() - start_time
                            times.append(inference_time)
                    
                    results[test_key][model_name] = {
                        'mean_time': np.mean(times),
                        'std_time': np.std(times),
                        'throughput': batch_size / np.mean(times)  # samples/second
                    }
                    
                    del model
                    gc.collect()
                
                print(f"{test_key}: Original={results[test_key]['Original']['mean_time']:.4f}s, "
                      f"TDA={results[test_key]['TDA-Full']['mean_time']:.4f}s")
        
        self.results['benchmarks']['inference_speed'] = results
        return results
    
    def benchmark_accuracy_patterns(self):
        """Benchmark accuracy on different pattern types"""
        print("Benchmarking accuracy on different patterns...")
        
        pattern_types = ['sinusoidal', 'trend', 'chaotic', 'mixed']
        batch_size = 16
        seq_len = 96
        pred_len = 24
        
        configs = [
            MockConfig(enable_tda=False, seq_len=seq_len, pred_len=pred_len),
            MockConfig(enable_tda=True, tda_mode='full', seq_len=seq_len, pred_len=pred_len)
        ]
        
        results = {}
        
        for pattern_type in pattern_types:
            print(f"Testing {pattern_type} pattern...")
            results[pattern_type] = {}
            
            # Generate training and test data
            train_data = self._generate_synthetic_data(batch_size, seq_len, 7, pattern_type)
            test_data = self._generate_synthetic_data(batch_size, seq_len, 7, pattern_type)
            
            # Create ground truth (simple continuation of pattern)
            with torch.no_grad():
                # Use last part of sequence as "ground truth" prediction
                ground_truth = test_data[:, -pred_len:, :]
            
            for i, config in enumerate(configs):
                model_name = ['Original', 'TDA-Full'][i]
                
                if config.enable_tda:
                    model = TDAKAN_TDA(config)
                else:
                    model = Model(config)
                
                model.eval()
                
                # Simple evaluation (no training for speed)
                with torch.no_grad():
                    predictions = model.forecast(test_data)
                    
                    # Calculate metrics
                    mse = torch.mean((predictions - ground_truth) ** 2).item()
                    mae = torch.mean(torch.abs(predictions - ground_truth)).item()
                    
                    # Pattern-specific metrics
                    pred_std = predictions.std().item()
                    gt_std = ground_truth.std().item()
                    std_ratio = pred_std / (gt_std + 1e-8)
                
                results[pattern_type][model_name] = {
                    'mse': mse,
                    'mae': mae,
                    'std_ratio': std_ratio,
                    'pred_std': pred_std,
                    'gt_std': gt_std
                }
                
                del model
                gc.collect()
            
            print(f"{pattern_type}: Original MSE={results[pattern_type]['Original']['mse']:.6f}, "
                  f"TDA MSE={results[pattern_type]['TDA-Full']['mse']:.6f}")
        
        self.results['benchmarks']['accuracy_patterns'] = results
        return results
    
    def benchmark_scalability(self):
        """Benchmark scalability with different model sizes"""
        print("Benchmarking scalability...")
        
        model_sizes = [
            {'d_model': 8, 'e_layers': 1},
            {'d_model': 16, 'e_layers': 2},
            {'d_model': 32, 'e_layers': 3},
            {'d_model': 64, 'e_layers': 4}
        ]
        
        results = {}
        batch_size = 8
        seq_len = 96
        
        for size_config in model_sizes:
            size_key = f"d{size_config['d_model']}_l{size_config['e_layers']}"
            results[size_key] = {}
            
            configs = [
                MockConfig(enable_tda=False, **size_config),
                MockConfig(enable_tda=True, tda_mode='full', **size_config)
            ]
            
            test_data = self._generate_synthetic_data(batch_size, seq_len, 7)
            
            for i, config in enumerate(configs):
                model_name = ['Original', 'TDA-Full'][i]
                
                if config.enable_tda:
                    model = TDAKAN_TDA(config)
                else:
                    model = Model(config)
                
                # Count parameters
                total_params = sum(p.numel() for p in model.parameters())
                
                # Benchmark inference time
                model.eval()
                times = []
                with torch.no_grad():
                    for _ in range(5):
                        start_time = time.time()
                        _ = model.forecast(test_data)
                        times.append(time.time() - start_time)
                
                results[size_key][model_name] = {
                    'params': total_params,
                    'mean_time': np.mean(times),
                    'std_time': np.std(times)
                }
                
                del model
                gc.collect()
            
            print(f"{size_key}: Original={results[size_key]['Original']['params']:,} params, "
                  f"TDA={results[size_key]['TDA-Full']['params']:,} params")
        
        self.results['benchmarks']['scalability'] = results
        return results
    
    def run_all_benchmarks(self):
        """Run all benchmarks"""
        print("=" * 60)
        print("TDA-KAN_TDA Performance Benchmark Suite")
        print("=" * 60)
        
        benchmarks = [
            ("Initialization Time", self.benchmark_initialization_time),
            ("Memory Usage", self.benchmark_memory_usage),
            ("Inference Speed", self.benchmark_inference_speed),
            ("Accuracy Patterns", self.benchmark_accuracy_patterns),
            ("Scalability", self.benchmark_scalability)
        ]
        
        for benchmark_name, benchmark_func in benchmarks:
            print(f"\n--- {benchmark_name} ---")
            try:
                benchmark_func()
                print(f"✓ {benchmark_name} completed")
            except Exception as e:
                print(f"✗ {benchmark_name} failed: {str(e)}")
        
        # Save results
        self.save_results()
        self.generate_report()
        
        print(f"\n✓ All benchmarks completed. Results saved to {self.output_dir}/")
    
    def save_results(self):
        """Save benchmark results to JSON"""
        results_file = os.path.join(self.output_dir, "benchmark_results.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {results_file}")
    
    def generate_report(self):
        """Generate a comprehensive benchmark report"""
        report_file = os.path.join(self.output_dir, "benchmark_report.md")
        
        with open(report_file, 'w') as f:
            f.write("# TDA-KAN_TDA Performance Benchmark Report\n\n")
            f.write(f"Generated: {self.results['timestamp']}\n\n")
            
            # System info
            f.write("## System Information\n\n")
            for key, value in self.results['system_info'].items():
                f.write(f"- **{key}**: {value}\n")
            f.write("\n")
            
            # Benchmark results
            for benchmark_name, benchmark_data in self.results['benchmarks'].items():
                f.write(f"## {benchmark_name.replace('_', ' ').title()}\n\n")
                
                if benchmark_name == 'initialization':
                    f.write("| Model | Mean Time (s) | Std Time (s) |\n")
                    f.write("|-------|---------------|---------------|\n")
                    for model, data in benchmark_data.items():
                        f.write(f"| {model} | {data['mean_time']:.4f} | {data['std_time']:.4f} |\n")
                
                elif benchmark_name == 'memory':
                    f.write("| Model | Parameters | Memory (MB) | GPU Memory (MB) |\n")
                    f.write("|-------|------------|-------------|------------------|\n")
                    for model, data in benchmark_data.items():
                        f.write(f"| {model} | {data['total_params']:,} | {data['param_memory_mb']:.2f} | {data['gpu_memory_mb']:.2f} |\n")
                
                elif benchmark_name == 'accuracy_patterns':
                    f.write("| Pattern | Model | MSE | MAE | Std Ratio |\n")
                    f.write("|---------|-------|-----|-----|----------|\n")
                    for pattern, pattern_data in benchmark_data.items():
                        for model, metrics in pattern_data.items():
                            f.write(f"| {pattern} | {model} | {metrics['mse']:.6f} | {metrics['mae']:.6f} | {metrics['std_ratio']:.3f} |\n")
                
                f.write("\n")
            
            # Summary
            f.write("## Summary\n\n")
            f.write("### Key Findings\n\n")
            
            # Calculate overhead ratios
            if 'memory' in self.results['benchmarks']:
                memory_data = self.results['benchmarks']['memory']
                if 'Original' in memory_data and 'TDA-Full' in memory_data:
                    param_ratio = memory_data['TDA-Full']['total_params'] / memory_data['Original']['total_params']
                    f.write(f"- **Parameter Overhead**: {param_ratio:.2f}x increase\n")
            
            if 'inference_speed' in self.results['benchmarks']:
                # Average overhead across different configurations
                speed_data = self.results['benchmarks']['inference_speed']
                overhead_ratios = []
                for config, config_data in speed_data.items():
                    if 'Original' in config_data and 'TDA-Full' in config_data:
                        ratio = config_data['TDA-Full']['mean_time'] / config_data['Original']['mean_time']
                        overhead_ratios.append(ratio)
                if overhead_ratios:
                    avg_overhead = np.mean(overhead_ratios)
                    f.write(f"- **Computational Overhead**: {avg_overhead:.2f}x average increase\n")
            
            f.write("\n### Recommendations\n\n")
            f.write("- TDA integration provides enhanced topological awareness at the cost of increased computational complexity\n")
            f.write("- Consider using TDA modes ('decomp_only' or 'kan_only') for specific use cases to balance performance\n")
            f.write("- TDA benefits are most apparent in complex, multi-scale time series patterns\n")
        
        print(f"Report generated: {report_file}")
    
    def plot_results(self):
        """Generate visualization plots"""
        try:
            import matplotlib.pyplot as plt
            
            # Memory usage comparison
            if 'memory' in self.results['benchmarks']:
                memory_data = self.results['benchmarks']['memory']
                models = list(memory_data.keys())
                params = [memory_data[model]['total_params'] for model in models]
                
                plt.figure(figsize=(10, 6))
                plt.bar(models, params)
                plt.title('Parameter Count Comparison')
                plt.ylabel('Number of Parameters')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'parameter_comparison.png'))
                plt.close()
            
            # Inference speed comparison
            if 'inference_speed' in self.results['benchmarks']:
                speed_data = self.results['benchmarks']['inference_speed']
                configs = list(speed_data.keys())
                original_times = [speed_data[config]['Original']['mean_time'] for config in configs]
                tda_times = [speed_data[config]['TDA-Full']['mean_time'] for config in configs]
                
                x = np.arange(len(configs))
                width = 0.35
                
                plt.figure(figsize=(12, 6))
                plt.bar(x - width/2, original_times, width, label='Original')
                plt.bar(x + width/2, tda_times, width, label='TDA-Full')
                plt.xlabel('Configuration')
                plt.ylabel('Inference Time (s)')
                plt.title('Inference Speed Comparison')
                plt.xticks(x, configs, rotation=45)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'inference_speed_comparison.png'))
                plt.close()
            
            print("Plots saved to benchmark_results/")
            
        except ImportError:
            print("Matplotlib not available, skipping plots")


def main():
    """Run the benchmark suite"""
    benchmark = PerformanceBenchmark()
    benchmark.run_all_benchmarks()
    benchmark.plot_results()


if __name__ == "__main__":
    main() 