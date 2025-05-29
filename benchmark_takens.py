#!/usr/bin/env python3
"""
Performance Benchmarking Suite for TakensEmbedding Module

This script provides comprehensive performance analysis including:
- CPU vs GPU comparison
- PyTorch vs NumPy implementation comparison  
- Memory usage profiling
- Scalability analysis with different input sizes
- Different embedding parameter configurations
"""

import torch
import numpy as np
import time
import psutil
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import pandas as pd
from layers.TakensEmbedding import TakensEmbedding, compute_takens_embedding


class TakensEmbeddingBenchmark:
    """Comprehensive benchmarking suite for TakensEmbedding"""
    
    def __init__(self):
        self.results = {
            'cpu_vs_gpu': {},
            'pytorch_vs_numpy': {},
            'memory_usage': {},
            'scalability': {},
            'parameter_impact': {}
        }
        
        # Check device availability
        self.has_cuda = torch.cuda.is_available()
        if self.has_cuda:
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA not available, GPU tests will be skipped")
    
    def generate_test_data(self, length: int, n_features: int = 1) -> torch.Tensor:
        """Generate synthetic test data"""
        t = torch.linspace(0, 4*np.pi, length)
        
        if n_features == 1:
            # Lorenz-like chaotic signal
            x = torch.sin(t) + 0.5 * torch.sin(3*t) + 0.1 * torch.randn(length)
            return x.unsqueeze(0)  # [1, length]
        else:
            # Multivariate coupled oscillators
            signals = []
            for i in range(n_features):
                phase = i * np.pi / n_features
                signal = torch.sin(t + phase) + 0.3 * torch.sin(2*t + phase)
                signals.append(signal)
            
            return torch.stack(signals, dim=-1).unsqueeze(0)  # [1, length, n_features]
    
    def benchmark_cpu_vs_gpu(self, 
                           sequence_lengths: List[int] = [500, 1000, 2000, 5000],
                           dims: List[int] = [2, 3, 5],
                           delays: List[int] = [1, 2],
                           n_runs: int = 5) -> Dict:
        """Benchmark CPU vs GPU performance"""
        
        if not self.has_cuda:
            print("Skipping GPU benchmark - CUDA not available")
            return {}
        
        print("\n=== CPU vs GPU Benchmark ===")
        results = {'sequence_length': [], 'cpu_time': [], 'gpu_time': [], 'speedup': []}
        
        for seq_len in sequence_lengths:
            print(f"Testing sequence length: {seq_len}")
            
            # Generate test data
            x = self.generate_test_data(seq_len)
            
            # CPU benchmark
            embedder_cpu = TakensEmbedding(dims=dims, delays=delays, device='cpu')
            cpu_times = []
            
            for _ in range(n_runs):
                start_time = time.time()
                _ = embedder_cpu(x.cpu())
                cpu_times.append(time.time() - start_time)
            
            avg_cpu_time = np.mean(cpu_times)
            
            # GPU benchmark
            embedder_gpu = TakensEmbedding(dims=dims, delays=delays, device='cuda')
            gpu_times = []
            
            for _ in range(n_runs):
                torch.cuda.synchronize()
                start_time = time.time()
                _ = embedder_gpu(x.cuda())
                torch.cuda.synchronize()
                gpu_times.append(time.time() - start_time)
            
            avg_gpu_time = np.mean(gpu_times)
            speedup = avg_cpu_time / avg_gpu_time
            
            results['sequence_length'].append(seq_len)
            results['cpu_time'].append(avg_cpu_time)
            results['gpu_time'].append(avg_gpu_time)
            results['speedup'].append(speedup)
            
            print(f"  CPU: {avg_cpu_time:.4f}s, GPU: {avg_gpu_time:.4f}s, Speedup: {speedup:.2f}x")
        
        self.results['cpu_vs_gpu'] = results
        return results
    
    def benchmark_pytorch_vs_numpy(self,
                                  sequence_lengths: List[int] = [500, 1000, 2000],
                                  dim: int = 3,
                                  tau: int = 1,
                                  n_runs: int = 10) -> Dict:
        """Benchmark PyTorch implementation vs NumPy reference"""
        
        print("\n=== PyTorch vs NumPy Benchmark ===")
        
        def numpy_takens_embedding(x: np.ndarray, dim: int, tau: int) -> np.ndarray:
            """Reference NumPy implementation"""
            n_points = len(x) - (dim - 1) * tau
            embedding = np.zeros((n_points, dim))
            
            for i in range(n_points):
                for j in range(dim):
                    embedding[i, j] = x[i + j * tau]
            
            return embedding
        
        results = {'sequence_length': [], 'numpy_time': [], 'pytorch_time': [], 'speedup': []}
        
        for seq_len in sequence_lengths:
            print(f"Testing sequence length: {seq_len}")
            
            # Generate test data
            x_torch = self.generate_test_data(seq_len)
            x_numpy = x_torch[0].numpy()
            
            # NumPy benchmark
            numpy_times = []
            for _ in range(n_runs):
                start_time = time.time()
                _ = numpy_takens_embedding(x_numpy, dim, tau)
                numpy_times.append(time.time() - start_time)
            
            avg_numpy_time = np.mean(numpy_times)
            
            # PyTorch benchmark
            embedder = TakensEmbedding(dims=[dim], delays=[tau], strategy='single')
            pytorch_times = []
            
            for _ in range(n_runs):
                start_time = time.time()
                _ = embedder(x_torch)
                pytorch_times.append(time.time() - start_time)
            
            avg_pytorch_time = np.mean(pytorch_times)
            speedup = avg_numpy_time / avg_pytorch_time
            
            results['sequence_length'].append(seq_len)
            results['numpy_time'].append(avg_numpy_time)
            results['pytorch_time'].append(avg_pytorch_time)
            results['speedup'].append(speedup)
            
            print(f"  NumPy: {avg_numpy_time:.4f}s, PyTorch: {avg_pytorch_time:.4f}s, Speedup: {speedup:.2f}x")
        
        self.results['pytorch_vs_numpy'] = results
        return results
    
    def benchmark_memory_usage(self,
                              sequence_lengths: List[int] = [1000, 5000, 10000, 20000],
                              dims: List[int] = [2, 3, 5]) -> Dict:
        """Benchmark memory usage patterns"""
        
        print("\n=== Memory Usage Benchmark ===")
        results = {'sequence_length': [], 'peak_memory_mb': [], 'embedding_size_mb': []}
        
        for seq_len in sequence_lengths:
            print(f"Testing sequence length: {seq_len}")
            
            # Generate test data
            x = self.generate_test_data(seq_len)
            
            # Clear memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                initial_gpu_memory = torch.cuda.memory_allocated() / 1024**2
            
            initial_cpu_memory = psutil.Process().memory_info().rss / 1024**2
            
            # Create embedder and compute embedding
            embedder = TakensEmbedding(dims=dims, delays=[1, 2])
            embedding = embedder(x)
            
            # Measure memory after computation
            final_cpu_memory = psutil.Process().memory_info().rss / 1024**2
            cpu_memory_used = final_cpu_memory - initial_cpu_memory
            
            if torch.cuda.is_available():
                final_gpu_memory = torch.cuda.memory_allocated() / 1024**2
                gpu_memory_used = final_gpu_memory - initial_gpu_memory
                peak_memory = max(cpu_memory_used, gpu_memory_used)
            else:
                peak_memory = cpu_memory_used
            
            # Calculate embedding size
            embedding_size_mb = embedding.numel() * embedding.element_size() / 1024**2
            
            results['sequence_length'].append(seq_len)
            results['peak_memory_mb'].append(peak_memory)
            results['embedding_size_mb'].append(embedding_size_mb)
            
            print(f"  Peak memory: {peak_memory:.2f} MB, Embedding size: {embedding_size_mb:.2f} MB")
        
        self.results['memory_usage'] = results
        return results
    
    def benchmark_scalability(self,
                             batch_sizes: List[int] = [1, 5, 10, 20],
                             sequence_length: int = 2000) -> Dict:
        """Benchmark scalability with different batch sizes"""
        
        print("\n=== Scalability Benchmark ===")
        results = {'batch_size': [], 'time_per_sample': [], 'total_time': []}
        
        for batch_size in batch_sizes:
            print(f"Testing batch size: {batch_size}")
            
            # Generate batch data
            batch_data = torch.stack([
                self.generate_test_data(sequence_length)[0] 
                for _ in range(batch_size)
            ])
            
            # Benchmark
            embedder = TakensEmbedding(dims=[2, 3], delays=[1, 2])
            
            start_time = time.time()
            _ = embedder(batch_data)
            total_time = time.time() - start_time
            
            time_per_sample = total_time / batch_size
            
            results['batch_size'].append(batch_size)
            results['time_per_sample'].append(time_per_sample)
            results['total_time'].append(total_time)
            
            print(f"  Total time: {total_time:.4f}s, Time per sample: {time_per_sample:.4f}s")
        
        self.results['scalability'] = results
        return results
    
    def benchmark_parameter_impact(self,
                                  sequence_length: int = 2000,
                                  max_dim: int = 10,
                                  max_delay: int = 8) -> Dict:
        """Benchmark impact of different embedding parameters"""
        
        print("\n=== Parameter Impact Benchmark ===")
        results = {'dims': [], 'delays': [], 'computation_time': [], 'output_size': []}
        
        x = self.generate_test_data(sequence_length)
        
        # Test different dimension configurations
        for dim in range(2, max_dim + 1):
            for delay in range(1, max_delay + 1):
                print(f"Testing dim={dim}, delay={delay}")
                
                try:
                    embedder = TakensEmbedding(dims=[dim], delays=[delay], strategy='single')
                    
                    start_time = time.time()
                    embedding = embedder(x)
                    computation_time = time.time() - start_time
                    
                    output_size = embedding.numel()
                    
                    results['dims'].append(dim)
                    results['delays'].append(delay)
                    results['computation_time'].append(computation_time)
                    results['output_size'].append(output_size)
                    
                except ValueError as e:
                    # Skip invalid parameter combinations
                    print(f"  Skipped: {str(e)}")
                    continue
        
        self.results['parameter_impact'] = results
        return results
    
    def plot_results(self, save_plots: bool = True):
        """Generate visualization plots for benchmark results"""
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('TakensEmbedding Performance Benchmark Results', fontsize=16)
        
        # Plot 1: CPU vs GPU Performance
        if self.results['cpu_vs_gpu']:
            ax = axes[0, 0]
            data = self.results['cpu_vs_gpu']
            ax.plot(data['sequence_length'], data['cpu_time'], 'b-o', label='CPU')
            ax.plot(data['sequence_length'], data['gpu_time'], 'r-o', label='GPU')
            ax.set_xlabel('Sequence Length')
            ax.set_ylabel('Time (seconds)')
            ax.set_title('CPU vs GPU Performance')
            ax.legend()
            ax.set_yscale('log')
        
        # Plot 2: PyTorch vs NumPy
        if self.results['pytorch_vs_numpy']:
            ax = axes[0, 1]
            data = self.results['pytorch_vs_numpy']
            ax.plot(data['sequence_length'], data['numpy_time'], 'g-o', label='NumPy')
            ax.plot(data['sequence_length'], data['pytorch_time'], 'b-o', label='PyTorch')
            ax.set_xlabel('Sequence Length')
            ax.set_ylabel('Time (seconds)')
            ax.set_title('PyTorch vs NumPy Performance')
            ax.legend()
        
        # Plot 3: Memory Usage
        if self.results['memory_usage']:
            ax = axes[0, 2]
            data = self.results['memory_usage']
            ax.plot(data['sequence_length'], data['peak_memory_mb'], 'r-o', label='Peak Memory')
            ax.plot(data['sequence_length'], data['embedding_size_mb'], 'b-o', label='Embedding Size')
            ax.set_xlabel('Sequence Length')
            ax.set_ylabel('Memory (MB)')
            ax.set_title('Memory Usage')
            ax.legend()
        
        # Plot 4: Scalability
        if self.results['scalability']:
            ax = axes[1, 0]
            data = self.results['scalability']
            ax.plot(data['batch_size'], data['time_per_sample'], 'g-o')
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Time per Sample (seconds)')
            ax.set_title('Scalability (Time per Sample)')
        
        # Plot 5: Parameter Impact (Heatmap)
        if self.results['parameter_impact']:
            ax = axes[1, 1]
            data = self.results['parameter_impact']
            df = pd.DataFrame(data)
            pivot_table = df.pivot(index='dims', columns='delays', values='computation_time')
            sns.heatmap(pivot_table, annot=True, fmt='.4f', ax=ax, cmap='viridis')
            ax.set_title('Computation Time by Parameters')
        
        # Plot 6: Speedup Analysis
        if self.results['cpu_vs_gpu'] and self.results['pytorch_vs_numpy']:
            ax = axes[1, 2]
            
            # GPU speedup
            gpu_data = self.results['cpu_vs_gpu']
            ax.plot(gpu_data['sequence_length'], gpu_data['speedup'], 'r-o', label='GPU vs CPU')
            
            # PyTorch speedup
            torch_data = self.results['pytorch_vs_numpy']
            ax.plot(torch_data['sequence_length'], torch_data['speedup'], 'b-o', label='PyTorch vs NumPy')
            
            ax.set_xlabel('Sequence Length')
            ax.set_ylabel('Speedup Factor')
            ax.set_title('Speedup Analysis')
            ax.legend()
            ax.axhline(y=1, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('takens_embedding_benchmark_results.png', dpi=300, bbox_inches='tight')
            print("Plots saved to 'takens_embedding_benchmark_results.png'")
        
        plt.show()
    
    def generate_report(self) -> str:
        """Generate comprehensive performance report"""
        
        report = []
        report.append("=" * 60)
        report.append("TAKENS EMBEDDING PERFORMANCE BENCHMARK REPORT")
        report.append("=" * 60)
        report.append("")
        
        # System information
        report.append("SYSTEM INFORMATION:")
        report.append(f"- CPU: {psutil.cpu_count()} cores")
        report.append(f"- Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")
        if self.has_cuda:
            report.append(f"- GPU: {torch.cuda.get_device_name(0)}")
            report.append(f"- CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            report.append("- GPU: Not available")
        report.append("")
        
        # CPU vs GPU results
        if self.results['cpu_vs_gpu']:
            report.append("CPU vs GPU PERFORMANCE:")
            data = self.results['cpu_vs_gpu']
            max_speedup = max(data['speedup'])
            avg_speedup = np.mean(data['speedup'])
            report.append(f"- Maximum speedup: {max_speedup:.2f}x")
            report.append(f"- Average speedup: {avg_speedup:.2f}x")
            report.append(f"- Speedup range: {min(data['speedup']):.2f}x - {max_speedup:.2f}x")
            report.append("")
        
        # PyTorch vs NumPy results
        if self.results['pytorch_vs_numpy']:
            report.append("PYTORCH vs NUMPY PERFORMANCE:")
            data = self.results['pytorch_vs_numpy']
            max_speedup = max(data['speedup'])
            avg_speedup = np.mean(data['speedup'])
            report.append(f"- Maximum speedup: {max_speedup:.2f}x")
            report.append(f"- Average speedup: {avg_speedup:.2f}x")
            if avg_speedup > 1:
                report.append("- PyTorch implementation is faster on average")
            else:
                report.append("- NumPy implementation is faster on average")
            report.append("")
        
        # Memory usage analysis
        if self.results['memory_usage']:
            report.append("MEMORY USAGE ANALYSIS:")
            data = self.results['memory_usage']
            max_memory = max(data['peak_memory_mb'])
            max_embedding = max(data['embedding_size_mb'])
            report.append(f"- Maximum peak memory: {max_memory:.2f} MB")
            report.append(f"- Maximum embedding size: {max_embedding:.2f} MB")
            
            # Memory efficiency
            memory_efficiency = max_embedding / max_memory * 100
            report.append(f"- Memory efficiency: {memory_efficiency:.1f}%")
            report.append("")
        
        # Scalability analysis
        if self.results['scalability']:
            report.append("SCALABILITY ANALYSIS:")
            data = self.results['scalability']
            time_variation = np.std(data['time_per_sample']) / np.mean(data['time_per_sample']) * 100
            report.append(f"- Time per sample variation: {time_variation:.1f}%")
            if time_variation < 20:
                report.append("- Good scalability (low variation)")
            else:
                report.append("- Poor scalability (high variation)")
            report.append("")
        
        # Parameter impact summary
        if self.results['parameter_impact']:
            report.append("PARAMETER IMPACT SUMMARY:")
            data = self.results['parameter_impact']
            df = pd.DataFrame(data)
            
            # Find optimal parameters (fastest computation)
            fastest_idx = df['computation_time'].idxmin()
            optimal_dim = df.loc[fastest_idx, 'dims']
            optimal_delay = df.loc[fastest_idx, 'delays']
            fastest_time = df.loc[fastest_idx, 'computation_time']
            
            report.append(f"- Fastest configuration: dim={optimal_dim}, delay={optimal_delay}")
            report.append(f"- Fastest time: {fastest_time:.4f}s")
            
            # Performance range
            slowest_time = df['computation_time'].max()
            performance_range = slowest_time / fastest_time
            report.append(f"- Performance range: {performance_range:.2f}x")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        
        if self.has_cuda and self.results['cpu_vs_gpu']:
            gpu_data = self.results['cpu_vs_gpu']
            if np.mean(gpu_data['speedup']) > 1.5:
                report.append("- Use GPU acceleration for significant speedup")
            else:
                report.append("- GPU acceleration provides minimal benefit for this workload")
        
        if self.results['parameter_impact']:
            report.append(f"- For optimal performance, use dim={optimal_dim}, delay={optimal_delay}")
        
        if self.results['memory_usage']:
            data = self.results['memory_usage']
            if max(data['peak_memory_mb']) > 1000:  # > 1GB
                report.append("- Consider using smaller batch sizes for large sequences")
        
        report.append("")
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        
        # Save report to file
        with open('takens_embedding_benchmark_report.txt', 'w') as f:
            f.write(report_text)
        
        print("Report saved to 'takens_embedding_benchmark_report.txt'")
        return report_text
    
    def run_full_benchmark(self, quick_mode: bool = False):
        """Run complete benchmark suite"""
        
        print("Starting TakensEmbedding Performance Benchmark Suite")
        print("=" * 60)
        
        if quick_mode:
            # Reduced parameters for quick testing
            seq_lengths = [500, 1000]
            dims = [2, 3]
            delays = [1, 2]
            batch_sizes = [1, 5]
            max_dim, max_delay = 5, 4
        else:
            # Full benchmark parameters
            seq_lengths = [500, 1000, 2000, 5000]
            dims = [2, 3, 5]
            delays = [1, 2, 4]
            batch_sizes = [1, 5, 10, 20]
            max_dim, max_delay = 10, 8
        
        # Run all benchmarks
        self.benchmark_pytorch_vs_numpy(seq_lengths[:3], n_runs=5)
        
        if self.has_cuda:
            self.benchmark_cpu_vs_gpu(seq_lengths, dims, delays[:2], n_runs=3)
        
        self.benchmark_memory_usage(seq_lengths, dims[:2])
        self.benchmark_scalability(batch_sizes[:3] if quick_mode else batch_sizes)
        self.benchmark_parameter_impact(seq_lengths[1], max_dim, max_delay)
        
        # Generate report and plots
        print("\nGenerating report and visualizations...")
        self.plot_results()
        report = self.generate_report()
        
        print("\n" + report)
        
        return self.results


def main():
    """Main benchmark execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='TakensEmbedding Performance Benchmark')
    parser.add_argument('--quick', action='store_true', help='Run quick benchmark with reduced parameters')
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')
    
    args = parser.parse_args()
    
    benchmark = TakensEmbeddingBenchmark()
    results = benchmark.run_full_benchmark(quick_mode=args.quick)
    
    if not args.no_plots:
        benchmark.plot_results()


if __name__ == "__main__":
    main() 