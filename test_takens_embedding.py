import torch
import numpy as np
import pytest
import warnings
import time
import matplotlib.pyplot as plt
from layers.TakensEmbedding import TakensEmbedding, compute_takens_embedding, estimate_embedding_parameters


class TestTakensEmbedding:
    """Comprehensive test suite for TakensEmbedding module"""
    
    def setup_method(self):
        """Setup test data and common configurations"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Generate synthetic test data
        self.t = torch.linspace(0, 4 * np.pi, 1000)
        self.sine_wave = torch.sin(self.t)
        self.cosine_wave = torch.cos(self.t)
        self.noise = torch.randn(1000) * 0.1
        self.noisy_sine = self.sine_wave + self.noise
        
        # Lorenz attractor (for chaotic dynamics testing)
        self.lorenz = self._generate_lorenz_attractor(1000)
        
        # Random walk
        self.random_walk = torch.cumsum(torch.randn(1000), dim=0)
        
        # Multivariate data
        self.multivariate = torch.stack([self.sine_wave, self.cosine_wave], dim=-1)
        
        # Default parameters
        self.default_dims = [2, 3, 5]
        self.default_delays = [1, 2, 4]
    
    def _generate_lorenz_attractor(self, n_points, dt=0.01):
        """Generate Lorenz attractor time series"""
        # Lorenz parameters
        sigma, rho, beta = 10.0, 28.0, 8.0/3.0
        
        # Initial conditions
        x, y, z = 1.0, 1.0, 1.0
        
        # Generate trajectory
        trajectory = []
        for _ in range(n_points):
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            
            x += dx * dt
            y += dy * dt
            z += dz * dt
            
            trajectory.append(x)
        
        return torch.tensor(trajectory, dtype=torch.float32)
    
    def test_01_initialization_valid_parameters(self):
        """Test 1: Valid initialization parameters"""
        embedder = TakensEmbedding(
            dims=[2, 3, 5],
            delays=[1, 2, 4],
            strategy='multi_scale',
            device=self.device
        )
        
        assert embedder.dims == [2, 3, 5]
        assert embedder.delays == [1, 2, 4]
        assert embedder.strategy == 'multi_scale'
        assert len(embedder.param_combinations) == 9  # 3 * 3 combinations
    
    def test_02_initialization_invalid_parameters(self):
        """Test 2: Invalid initialization parameters should raise errors"""
        
        # Invalid dims
        with pytest.raises(ValueError):
            TakensEmbedding(dims=[1])  # dim must be >= 2
        
        with pytest.raises(ValueError):
            TakensEmbedding(dims=[])  # empty dims
        
        # Invalid delays
        with pytest.raises(ValueError):
            TakensEmbedding(delays=[0])  # delay must be >= 1
        
        with pytest.raises(ValueError):
            TakensEmbedding(delays=[])  # empty delays
        
        # Invalid strategy
        with pytest.raises(ValueError):
            TakensEmbedding(strategy='invalid')
        
        # Invalid optimization method
        with pytest.raises(ValueError):
            TakensEmbedding(optimization_method='invalid')
    
    def test_03_single_embedding_computation(self):
        """Test 3: Single embedding computation with known parameters"""
        embedder = TakensEmbedding(
            dims=[3], delays=[1], strategy='single', device=self.device
        )
        
        # Test with 1D input
        x_1d = self.sine_wave.unsqueeze(0)  # [1, seq_len]
        embedding = embedder(x_1d)
        
        # Check output shape
        batch_size, n_embeddings, n_points, dim = embedding.shape
        assert batch_size == 1
        assert n_embeddings == 1  # single strategy
        assert n_points == len(self.sine_wave) - 2  # (3-1)*1 = 2
        assert dim == 3
        
        # Check embedding values make sense
        emb_sample = embedding[0, 0]  # [n_points, 3]
        assert torch.allclose(emb_sample[:, 0], self.sine_wave[:-2])
        assert torch.allclose(emb_sample[:, 1], self.sine_wave[1:-1])
        assert torch.allclose(emb_sample[:, 2], self.sine_wave[2:])
    
    def test_04_multi_scale_embedding(self):
        """Test 4: Multi-scale embedding with multiple parameter combinations"""
        embedder = TakensEmbedding(
            dims=[2, 3], delays=[1, 2], strategy='multi_scale', device=self.device
        )
        
        x = self.sine_wave.unsqueeze(0)
        embedding = embedder(x)
        
        # Check output shape
        batch_size, n_embeddings, n_points, max_dim = embedding.shape
        assert batch_size == 1
        assert n_embeddings == 4  # 2 dims * 2 delays
        assert n_points > 0
        assert max_dim >= 3  # maximum dimension should be at least 3
    
    def test_05_adaptive_embedding(self):
        """Test 5: Adaptive parameter optimization"""
        embedder = TakensEmbedding(
            dims=[2, 3, 5], delays=[1, 2, 4], 
            strategy='adaptive', optimization_method='mutual_info', 
            device=self.device
        )
        
        x = self.sine_wave.unsqueeze(0)
        embedding = embedder(x)
        
        # Check that embedding was computed
        assert embedding.shape[0] == 1  # batch size
        assert embedding.shape[1] == 1  # single feature (univariate input)
        assert embedding.shape[2] > 0  # has points
        assert embedding.shape[3] > 0  # has dimensions
        
        # Check that optimal parameters were cached
        assert len(embedder.optimal_params_cache) > 0
    
    def test_06_multivariate_input(self):
        """Test 6: Multivariate time series input"""
        embedder = TakensEmbedding(
            dims=[2, 3], delays=[1], strategy='multi_scale', device=self.device
        )
        
        # Multivariate input [batch, seq_len, features]
        x = self.multivariate.unsqueeze(0)  # [1, seq_len, 2]
        embedding = embedder(x)
        
        # For multivariate input, features should be concatenated
        batch_size, n_embeddings, n_points, total_dim = embedding.shape
        assert batch_size == 1
        assert n_embeddings == 2  # 2 dims * 1 delay
        # total_dim should be larger due to feature concatenation
        assert total_dim >= 4  # 2 features * 2 dims minimum
    
    def test_07_insufficient_data_handling(self):
        """Test 7: Handling of insufficient data scenarios"""
        embedder = TakensEmbedding(
            dims=[10], delays=[5], strategy='single', device=self.device
        )
        
        # Very short time series
        short_series = torch.randn(1, 20)  # Only 20 points
        
        # This should raise an error or warning
        with pytest.raises(ValueError):
            embedder(short_series)
    
    def test_08_gpu_acceleration(self):
        """Test 8: GPU acceleration (if available)"""
        if not torch.cuda.is_available():
            print("SKIPPED - CUDA not available")
            return
        
        embedder_gpu = TakensEmbedding(
            dims=[2, 3], delays=[1, 2], device='cuda'
        )
        embedder_cpu = TakensEmbedding(
            dims=[2, 3], delays=[1, 2], device='cpu'
        )
        
        x = self.sine_wave.unsqueeze(0)
        
        # Time GPU computation
        start_time = time.time()
        embedding_gpu = embedder_gpu(x.cuda())
        gpu_time = time.time() - start_time
        
        # Time CPU computation
        start_time = time.time()
        embedding_cpu = embedder_cpu(x.cpu())
        cpu_time = time.time() - start_time
        
        # Check results are similar (allowing for floating point differences)
        assert torch.allclose(embedding_gpu.cpu(), embedding_cpu, atol=1e-5)
        
        print(f"GPU time: {gpu_time:.4f}s, CPU time: {cpu_time:.4f}s")
    
    def test_09_caching_functionality(self):
        """Test 9: Caching system functionality"""
        embedder = TakensEmbedding(
            dims=[2, 3], delays=[1], cache_embeddings=True, device=self.device
        )
        
        x = self.sine_wave.unsqueeze(0)
        
        # First computation
        embedding1 = embedder(x)
        stats1 = embedder.get_statistics()
        
        # Second computation (should use cache)
        embedding2 = embedder(x)
        stats2 = embedder.get_statistics()
        
        # Check cache was used
        assert stats2['cache_hits'] > stats1['cache_hits']
        assert torch.allclose(embedding1, embedding2)
    
    def test_10_parameter_optimization_methods(self):
        """Test 10: Different parameter optimization methods"""
        methods = ['mutual_info', 'fnn', 'autocorr']
        
        for method in methods:
            embedder = TakensEmbedding(
                dims=[2, 3, 5], delays=[1, 2, 4],
                strategy='adaptive', optimization_method=method,
                device=self.device
            )
            
            # Test with periodic signal
            x = self.sine_wave.unsqueeze(0)
            embedding = embedder(x)
            
            # Should produce valid embedding
            assert embedding.shape[0] == 1
            assert embedding.shape[2] > 0
            assert embedding.shape[3] > 0
    
    def test_11_embedding_quality_assessment(self):
        """Test 11: Embedding quality metrics computation"""
        embedder = TakensEmbedding(
            dims=[3], delays=[1], strategy='single', device=self.device
        )
        
        # Test with Lorenz attractor (known to have fractal structure)
        x = self.lorenz.unsqueeze(0)
        embedding = embedder(x)
        
        # Compute quality metrics
        quality = embedder._compute_embedding_quality(x.unsqueeze(-1), embedding.squeeze(1))
        
        # Check that metrics are computed
        assert 'correlation_dimension' in quality
        assert 'lyapunov_exponent' in quality
        assert 'embedding_density' in quality
        assert 'reconstruction_error' in quality
        assert 'determinism' in quality
    
    def test_12_memory_efficiency(self):
        """Test 12: Memory usage for large embeddings"""
        embedder = TakensEmbedding(
            dims=[2], delays=[1], strategy='single', device=self.device
        )
        
        # Large time series
        large_series = torch.randn(1, 10000)
        
        # Monitor memory before computation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated()
        
        embedding = embedder(large_series)
        
        if torch.cuda.is_available():
            memory_after = torch.cuda.memory_allocated()
            memory_used = (memory_after - memory_before) / 1024**2  # MB
            print(f"Memory used for large embedding: {memory_used:.2f} MB")
        
        # Check embedding was computed successfully
        assert embedding.shape[0] == 1
        assert embedding.shape[2] == 10000 - 1  # n_points for dim=2, tau=1
    
    def test_13_utility_functions(self):
        """Test 13: Standalone utility functions"""
        
        # Test compute_takens_embedding function
        embedding = compute_takens_embedding(self.sine_wave, dim=3, tau=1)
        
        # Check output shape
        assert embedding.shape[0] == 1  # batch dimension added
        assert embedding.shape[1] == len(self.sine_wave) - 2  # (3-1)*1
        assert embedding.shape[2] == 3
        
        # Test estimate_embedding_parameters function
        optimal_dim, optimal_tau = estimate_embedding_parameters(
            self.sine_wave, method='autocorr'
        )
        
        assert isinstance(optimal_dim, int)
        assert isinstance(optimal_tau, int)
        assert optimal_dim >= 2
        assert optimal_tau >= 1
    
    def test_14_numerical_stability(self):
        """Test 14: Numerical stability with extreme values"""
        embedder = TakensEmbedding(
            dims=[2, 3], delays=[1], strategy='multi_scale', device=self.device
        )
        
        # Test with very large values
        large_values = torch.randn(1, 1000) * 1e6
        embedding_large = embedder(large_values)
        assert torch.isfinite(embedding_large).all()
        
        # Test with very small values
        small_values = torch.randn(1, 1000) * 1e-6
        embedding_small = embedder(small_values)
        assert torch.isfinite(embedding_small).all()
        
        # Test with mixed scales
        mixed_values = torch.cat([large_values[:, :500], small_values[:, :500]], dim=1)
        embedding_mixed = embedder(mixed_values)
        assert torch.isfinite(embedding_mixed).all()
    
    def test_15_error_handling(self):
        """Test 15: Proper error handling for edge cases"""
        embedder = TakensEmbedding(device=self.device)
        
        # Test with wrong input dimensions
        with pytest.raises(ValueError):
            invalid_input = torch.randn(1000)  # Wrong number of dimensions
            embedder(invalid_input.unsqueeze(0).unsqueeze(0).unsqueeze(0))  # 4D tensor
        
        # Test with empty tensor
        with pytest.raises(ValueError):
            empty_tensor = torch.empty(1, 0)
            embedder(empty_tensor)
        
        # Test with NaN values
        nan_values = torch.full((1, 1000), float('nan'))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            embedding = embedder(nan_values)
            # Should handle gracefully, might produce NaN but shouldn't crash
    
    def test_16_performance_benchmark(self):
        """Test 16: Performance benchmarking against numpy implementation"""
        
        def numpy_takens_embedding(x, dim, tau):
            """Reference numpy implementation"""
            n_points = len(x) - (dim - 1) * tau
            embedding = np.zeros((n_points, dim))
            
            for i in range(n_points):
                for j in range(dim):
                    embedding[i, j] = x[i + j * tau]
            
            return embedding
        
        # Test data
        x_np = self.sine_wave.numpy()
        x_torch = self.sine_wave.unsqueeze(0)
        
        dim, tau = 3, 1
        
        # Numpy computation
        start_time = time.time()
        embedding_np = numpy_takens_embedding(x_np, dim, tau)
        numpy_time = time.time() - start_time
        
        # PyTorch computation
        embedder = TakensEmbedding(dims=[dim], delays=[tau], strategy='single', device=self.device)
        start_time = time.time()
        embedding_torch = embedder(x_torch)
        torch_time = time.time() - start_time
        
        # Compare results
        embedding_torch_np = embedding_torch[0, 0].cpu().numpy()
        assert np.allclose(embedding_np, embedding_torch_np, atol=1e-6)
        
        print(f"Numpy time: {numpy_time:.6f}s, PyTorch time: {torch_time:.6f}s")
        print(f"Speedup: {numpy_time / torch_time:.2f}x")
    
    def test_17_batch_processing(self):
        """Test 17: Batch processing with multiple time series"""
        embedder = TakensEmbedding(
            dims=[2, 3], delays=[1], strategy='multi_scale', device=self.device
        )
        
        # Create batch of different time series
        batch_size = 5
        seq_len = 500
        
        batch_data = torch.stack([
            torch.sin(torch.linspace(0, 4*np.pi, seq_len) + i)
            for i in range(batch_size)
        ])
        
        # Process batch
        embedding = embedder(batch_data)
        
        # Check output shape
        assert embedding.shape[0] == batch_size
        assert embedding.shape[1] == 2  # 2 dims * 1 delay
        assert embedding.shape[2] > 0
        assert embedding.shape[3] > 0
        
        # Check that different samples produce different embeddings
        assert not torch.allclose(embedding[0], embedding[1])


def run_comprehensive_tests():
    """Run all tests and provide summary report"""
    test_instance = TestTakensEmbedding()
    test_instance.setup_method()
    
    test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
    
    print("Running TakensEmbedding Comprehensive Test Suite")
    print("=" * 50)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_method in test_methods:
        try:
            print(f"Running {test_method}... ", end="")
            getattr(test_instance, test_method)()
            print("PASSED")
            passed += 1
        except Exception as e:
            if "SKIPPED" in str(e) or "skip" in str(e).lower():
                print("SKIPPED")
                skipped += 1
            else:
                print(f"FAILED: {str(e)}")
                failed += 1
    
    print("=" * 50)
    print(f"Test Summary: {passed} passed, {failed} failed, {skipped} skipped")
    if passed + failed > 0:
        print(f"Success Rate: {passed / (passed + failed) * 100:.1f}%")
    
    return passed, failed, skipped


if __name__ == "__main__":
    run_comprehensive_tests() 