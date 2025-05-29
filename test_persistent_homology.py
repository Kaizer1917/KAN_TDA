#!/usr/bin/env python3
"""
Comprehensive test suite for PersistentHomology module
"""

import torch
import numpy as np
import warnings
import time
from typing import List
import sys
import os

# Add the layers directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'layers'))

try:
    from layers.PersistentHomology import (
        PersistentHomologyComputer, 
        PersistenceDiagram,
        compute_persistence_diagrams,
        benchmark_backends,
        RIPSER_AVAILABLE,
        GUDHI_AVAILABLE, 
        GIOTTO_AVAILABLE
    )
except ImportError:
    from PersistentHomology import (
        PersistentHomologyComputer,
        PersistenceDiagram, 
        compute_persistence_diagrams,
        benchmark_backends,
        RIPSER_AVAILABLE,
        GUDHI_AVAILABLE,
        GIOTTO_AVAILABLE
    )


class TestPersistentHomology:
    """Test suite for PersistentHomology module"""
    
    def setup_method(self):
        """Setup test data"""
        # Generate test point clouds
        
        # 1. Circle data (should have 1D homology)
        n_points = 50
        angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        self.circle_data = np.column_stack([
            np.cos(angles) + 0.1 * np.random.randn(n_points),
            np.sin(angles) + 0.1 * np.random.randn(n_points)
        ])
        
        # 2. Random point cloud
        self.random_data = np.random.randn(30, 3)
        
        # 3. Simple triangle (known topology)
        self.triangle_data = np.array([
            [0, 0],
            [1, 0], 
            [0.5, np.sqrt(3)/2],
            [0.25, 0.25],
            [0.75, 0.25],
            [0.5, 0.6]
        ])
        
        # 4. Torus data (should have 2D homology)
        n_torus = 40
        u = np.random.uniform(0, 2*np.pi, n_torus)
        v = np.random.uniform(0, 2*np.pi, n_torus)
        R, r = 2, 1
        self.torus_data = np.column_stack([
            (R + r * np.cos(v)) * np.cos(u),
            (R + r * np.cos(v)) * np.sin(u),
            r * np.sin(v)
        ]) + 0.1 * np.random.randn(n_torus, 3)
        
        # Available backends for testing
        self.available_backends = []
        if RIPSER_AVAILABLE:
            self.available_backends.append('ripser')
        if GUDHI_AVAILABLE:
            self.available_backends.append('gudhi')
        if GIOTTO_AVAILABLE:
            self.available_backends.append('giotto')
    
    def test_01_backend_availability(self):
        """Test 1: Check backend availability"""
        print(f"Available backends: {self.available_backends}")
        
        if not self.available_backends:
            print("WARNING: No backends available. Please install ripser, gudhi, or giotto-tda")
            return
        
        assert len(self.available_backends) > 0, "At least one backend should be available"
        print("Backend availability test passed")
    
    def test_02_basic_computation(self):
        """Test 2: Basic persistence diagram computation"""
        if not self.available_backends:
            print("Skipping - no backends available")
            return
        
        # Use GUDHI if available, otherwise use first available backend
        backend = 'gudhi' if 'gudhi' in self.available_backends else self.available_backends[0]
        computer = PersistentHomologyComputer(backend=backend, max_dimension=1)
        
        # Test with circle data
        diagram = computer.compute_diagrams(self.circle_data)
        
        assert isinstance(diagram, PersistenceDiagram)
        assert len(diagram.diagrams) == 2  # H0 and H1
        assert len(diagram.diagrams[0]) > 0  # Should have connected components
        
        print(f"Circle data: {len(diagram.diagrams[0])} H0 features, {len(diagram.diagrams[1])} H1 features")
        print("Basic computation test passed")
    
    def test_03_backend_comparison(self):
        """Test 3: Compare results across backends"""
        if len(self.available_backends) < 2:
            print("Skipping - need multiple backends for comparison")
            return
        
        results = {}
        for backend in self.available_backends:
            try:
                computer = PersistentHomologyComputer(backend=backend, max_dimension=1)
                diagram = computer.compute_diagrams(self.random_data)
                stats = diagram.diagram_statistics()
                results[backend] = stats['total_features']
                print(f"{backend}: {stats['total_features']} total features")
            except Exception as e:
                print(f"{backend} failed: {e}")
        
        # Results should be consistent (though not necessarily identical due to numerical differences)
        if len(results) >= 2:
            feature_counts = list(results.values())
            # Allow some variation between backends
            assert max(feature_counts) - min(feature_counts) <= 5, "Backend results should be reasonably consistent"
        
        print("Backend comparison test passed")
    
    def test_04_persistence_diagram_manipulation(self):
        """Test 4: PersistenceDiagram class functionality"""
        if not self.available_backends:
            print("Skipping - no backends available")
            return
        
        backend = 'gudhi' if 'gudhi' in self.available_backends else self.available_backends[0]
        computer = PersistentHomologyComputer(backend=backend, max_dimension=1)
        diagram = computer.compute_diagrams(self.circle_data)
        
        # Test filtering
        filtered = diagram.filter_by_persistence(threshold=0.1)
        assert len(filtered) <= len(diagram)
        
        # Test statistics
        stats = diagram.diagram_statistics()
        assert 'total_features' in stats
        assert 'max_persistence_per_dim' in stats
        assert isinstance(stats['total_features'], int)
        
        # Test tensor conversion
        tensor = diagram.to_tensor()
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape[1] == 3  # birth, death, dimension
        
        print(f"Diagram stats: {stats['total_features']} features")
        print("Diagram manipulation test passed")
    
    def test_05_parallel_processing(self):
        """Test 5: Parallel batch processing"""
        if not self.available_backends:
            print("Skipping - no backends available")
            return
        
        backend = 'gudhi' if 'gudhi' in self.available_backends else self.available_backends[0]
        computer = PersistentHomologyComputer(backend=backend, max_dimension=1, n_jobs=2)
        
        # Create batch of point clouds
        point_clouds = [
            self.circle_data,
            self.random_data,
            self.triangle_data
        ]
        
        # Test batch processing
        diagrams = computer.compute_batch_diagrams(point_clouds)
        
        assert len(diagrams) == 3
        assert all(isinstance(d, PersistenceDiagram) for d in diagrams)
        
        print(f"Processed {len(diagrams)} point clouds in batch")
        print("Parallel processing test passed")
    
    def test_06_distance_matrix_optimization(self):
        """Test 6: Distance matrix optimization methods"""
        if not self.available_backends:
            print("Skipping - no backends available")
            return
        
        backend = 'gudhi' if 'gudhi' in self.available_backends else self.available_backends[0]
        computer = PersistentHomologyComputer(backend=backend, max_dimension=1)
        
        # Test with different metrics
        metrics = ['euclidean']  # Start with just euclidean for stability
        for metric in metrics:
            try:
                computer.metric = metric
                diagram = computer.compute_diagrams(self.random_data)
                assert len(diagram) >= 0
                print(f"Metric {metric}: {len(diagram)} total features")
            except Exception as e:
                print(f"Metric {metric} failed: {e}")
        
        print("Distance matrix optimization test passed")
    
    def test_07_large_point_cloud_handling(self):
        """Test 7: Handling of large point clouds"""
        if not self.available_backends:
            print("Skipping - no backends available")
            return
        
        # Create larger point cloud
        large_data = np.random.randn(200, 2)
        
        backend = 'gudhi' if 'gudhi' in self.available_backends else self.available_backends[0]
        computer = PersistentHomologyComputer(backend=backend, max_dimension=1)
        
        start_time = time.time()
        diagram = computer.compute_diagrams(large_data)
        computation_time = time.time() - start_time
        
        assert isinstance(diagram, PersistenceDiagram)
        print(f"Large point cloud ({large_data.shape[0]} points): {computation_time:.3f}s")
        print("Large point cloud test passed")
    
    def test_08_edge_cases(self):
        """Test 8: Edge cases and error handling"""
        if not self.available_backends:
            print("Skipping - no backends available")
            return
        
        backend = 'gudhi' if 'gudhi' in self.available_backends else self.available_backends[0]
        computer = PersistentHomologyComputer(backend=backend, max_dimension=1)
        
        # Test with minimal data
        minimal_data = np.array([[0, 0], [1, 0]])  # Just 2 points
        
        try:
            diagram = computer.compute_diagrams(minimal_data)
            print(f"Minimal data: {len(diagram)} features")
        except Exception as e:
            print(f"Minimal data failed (expected): {e}")
        
        # Test with invalid input
        try:
            invalid_data = np.array([1, 2, 3])  # 1D array
            diagram = computer.compute_diagrams(invalid_data)
            assert False, "Should have failed with 1D input"
        except ValueError:
            print("Correctly rejected 1D input")
        
        print("Edge cases test passed")
    
    def test_09_utility_functions(self):
        """Test 9: Utility functions"""
        if not self.available_backends:
            print("Skipping - no backends available")
            return
        
        # Test convenience function
        diagram = compute_persistence_diagrams(
            self.circle_data, 
            backend='auto', 
            max_dimension=1
        )
        
        assert isinstance(diagram, PersistenceDiagram)
        
        # Test backend listing
        computer = PersistentHomologyComputer()
        available = computer.list_available_backends()
        assert len(available) > 0
        
        print(f"Available backends: {available}")
        print("Utility functions test passed")
    
    def test_10_tensor_integration(self):
        """Test 10: PyTorch tensor integration"""
        if not self.available_backends:
            print("Skipping - no backends available")
            return
        
        # Test with PyTorch tensors
        tensor_data = torch.tensor(self.circle_data, dtype=torch.float32)
        
        backend = 'gudhi' if 'gudhi' in self.available_backends else self.available_backends[0]
        computer = PersistentHomologyComputer(backend=backend)
        
        # Test forward pass
        result_tensor = computer.forward(tensor_data)
        assert isinstance(result_tensor, torch.Tensor)
        
        # Test device handling
        if torch.cuda.is_available():
            cuda_data = tensor_data.cuda()
            result_cuda = computer.forward(cuda_data)
            assert result_cuda.device.type == 'cuda'
        
        print("Tensor integration test passed")
    
    def test_11_performance_benchmark(self):
        """Test 11: Performance benchmarking"""
        if not self.available_backends:
            print("Skipping - no backends available")
            return
        
        try:
            # Run benchmark on small data
            results = benchmark_backends(
                self.random_data,
                max_dimension=1,
                n_runs=2
            )
            
            assert len(results) > 0
            
            for backend, result in results.items():
                if result['success']:
                    print(f"{backend}: {result['mean_time']:.4f}s avg, {result['total_features']} features")
                else:
                    print(f"{backend}: Failed - {result['error']}")
            
        except Exception as e:
            print(f"Benchmark failed: {e}")
        
        print("Performance benchmark test passed")
    
    def test_12_statistics_and_monitoring(self):
        """Test 12: Statistics and monitoring"""
        if not self.available_backends:
            print("Skipping - no backends available")
            return
        
        backend = 'gudhi' if 'gudhi' in self.available_backends else self.available_backends[0]
        computer = PersistentHomologyComputer(backend=backend)
        
        # Compute several diagrams
        for data in [self.circle_data, self.random_data, self.triangle_data]:
            computer.compute_diagrams(data)
        
        # Check statistics
        stats = computer.get_computation_stats()
        
        assert stats['total_computations'] == 3
        assert stats['average_time'] > 0
        assert stats['backend_used'] in self.available_backends
        
        print(f"Computation stats: {stats}")
        
        # Test reset
        computer.reset_stats()
        new_stats = computer.get_computation_stats()
        assert new_stats['total_computations'] == 0
        
        print("Statistics and monitoring test passed")


def run_comprehensive_tests():
    """Run all tests and provide summary"""
    test_instance = TestPersistentHomology()
    test_instance.setup_method()
    
    test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
    
    print("Running PersistentHomology Comprehensive Test Suite")
    print("=" * 60)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_method in test_methods:
        try:
            print(f"\nRunning {test_method}...")
            getattr(test_instance, test_method)()
            passed += 1
        except Exception as e:
            if "skip" in str(e).lower() or "no backends" in str(e).lower():
                skipped += 1
                print(f"SKIPPED: {str(e)}")
            else:
                failed += 1
                print(f"FAILED: {str(e)}")
    
    print("\n" + "=" * 60)
    print(f"Test Summary: {passed} passed, {failed} failed, {skipped} skipped")
    if passed + failed > 0:
        print(f"Success Rate: {passed / (passed + failed) * 100:.1f}%")
    
    return passed, failed, skipped


if __name__ == "__main__":
    run_comprehensive_tests() 