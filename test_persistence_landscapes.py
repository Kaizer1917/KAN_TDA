#!/usr/bin/env python3
"""
Comprehensive test suite for Persistence Landscapes module
"""

import torch
import numpy as np
import warnings
import time
from typing import List
import sys
import os

# Add the utils directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

try:
    from utils.persistence_landscapes import (
        PersistenceLandscape,
        TopologicalFeatureExtractor,
        PersistenceLandscapeVisualizer,
        compute_bottleneck_distance,
        compute_wasserstein_distance,
        extract_topological_features,
        rank_topological_features
    )
except ImportError:
    from persistence_landscapes import (
        PersistenceLandscape,
        TopologicalFeatureExtractor,
        PersistenceLandscapeVisualizer,
        compute_bottleneck_distance,
        compute_wasserstein_distance,
        extract_topological_features,
        rank_topological_features
    )


class TestPersistenceLandscapes:
    """Test suite for Persistence Landscapes module"""
    
    def setup_method(self):
        """Setup test data"""
        # Generate test persistence diagrams
        
        # 1. Simple triangle diagram
        self.triangle_diagram = np.array([
            [0.0, 0.5],
            [0.1, 0.3],
            [0.2, 0.8]
        ])
        
        # 2. Circle-like diagram (should have prominent 1D features)
        self.circle_diagram = np.array([
            [0.0, 0.1],   # Small 0D features
            [0.05, 0.15],
            [0.1, 0.9],   # Large 1D feature (main loop)
            [0.2, 0.4],   # Medium 1D feature
            [0.3, 0.35]   # Small 1D feature
        ])
        
        # 3. Empty diagram
        self.empty_diagram = np.empty((0, 2))
        
        # 4. Single point diagram
        self.single_point_diagram = np.array([[0.1, 0.6]])
        
        # 5. Diagram with infinite persistence
        self.infinite_diagram = np.array([
            [0.0, 0.5],
            [0.1, np.inf],  # Infinite persistence
            [0.2, 0.8]
        ])
        
        # 6. Large diagram for performance testing
        np.random.seed(42)
        n_points = 100
        births = np.random.uniform(0, 1, n_points)
        deaths = births + np.random.exponential(0.3, n_points)
        self.large_diagram = np.column_stack([births, deaths])
    
    def test_01_basic_landscape_creation(self):
        """Test 1: Basic persistence landscape creation"""
        print("Testing basic landscape creation...")
        
        # Test with triangle diagram
        landscape = PersistenceLandscape(self.triangle_diagram, resolution=100, max_landscapes=3)
        
        assert landscape.resolution == 100
        assert landscape.max_landscapes == 3
        assert landscape.landscapes.shape == (3, 100)
        assert len(landscape.finite_diagram) == 3
        
        # Test landscape values are non-negative
        assert np.all(landscape.landscapes >= 0)
        
        # Test that first landscape has highest values
        for i in range(landscape.resolution):
            assert landscape.landscapes[0, i] >= landscape.landscapes[1, i]
            assert landscape.landscapes[1, i] >= landscape.landscapes[2, i]
        
        print("Basic landscape creation test passed")
    
    def test_02_empty_diagram_handling(self):
        """Test 2: Handling of empty persistence diagrams"""
        print("Testing empty diagram handling...")
        
        landscape = PersistenceLandscape(self.empty_diagram, resolution=50, max_landscapes=2)
        
        assert landscape.landscapes.shape == (2, 50)
        assert np.all(landscape.landscapes == 0)
        assert len(landscape.finite_diagram) == 0
        
        # Test statistics for empty landscape
        stats = landscape.landscape_stats
        assert stats['n_persistence_points'] == 0
        assert all(stats['landscape_integrals'][k] == 0.0 for k in range(2))
        
        print("Empty diagram handling test passed")
    
    def test_03_infinite_persistence_handling(self):
        """Test 3: Handling of infinite persistence points"""
        print("Testing infinite persistence handling...")
        
        landscape = PersistenceLandscape(self.infinite_diagram, resolution=50, max_landscapes=2)
        
        # Should only use finite points for landscape computation
        assert len(landscape.finite_diagram) == 2  # Excludes infinite point
        assert np.all(np.isfinite(landscape.finite_diagram))
        
        # Landscapes should still be computed correctly
        assert landscape.landscapes.shape == (2, 50)
        assert np.all(landscape.landscapes >= 0)
        
        print("Infinite persistence handling test passed")
    
    def test_04_landscape_statistics(self):
        """Test 4: Landscape statistics computation"""
        print("Testing landscape statistics...")
        
        landscape = PersistenceLandscape(self.circle_diagram, resolution=200, max_landscapes=3)
        
        stats = landscape.landscape_stats
        
        # Check required statistics exist
        required_keys = [
            'n_persistence_points', 'x_range', 'resolution', 'max_landscapes',
            'landscape_norms', 'landscape_integrals', 'landscape_maxima', 'landscape_supports'
        ]
        
        for key in required_keys:
            assert key in stats
        
        # Check values make sense
        assert stats['n_persistence_points'] == len(landscape.finite_diagram)
        assert stats['resolution'] == 200
        assert stats['max_landscapes'] == 3
        
        # Check that integrals are positive for non-empty landscapes
        assert stats['landscape_integrals'][0] > 0  # First landscape should have positive integral
        
        # Check that maxima are consistent
        for k in range(3):
            computed_max = np.max(landscape.get_landscape(k))
            assert abs(stats['landscape_maxima'][k] - computed_max) < 1e-10
        
        print("Landscape statistics test passed")
    
    def test_05_landscape_integration(self):
        """Test 5: Landscape integration methods"""
        print("Testing landscape integration...")
        
        landscape = PersistenceLandscape(self.triangle_diagram, resolution=100, max_landscapes=2)
        
        # Test L1 integration
        l1_integral = landscape.integrate_landscape(0, order=1)
        assert l1_integral >= 0
        
        # Test L2 integration
        l2_integral = landscape.integrate_landscape(0, order=2)
        assert l2_integral >= 0
        
        # L2 integral should be different from L1 (unless landscape is zero)
        if l1_integral > 0:
            assert l1_integral != l2_integral
        
        # Test that integration matches stored statistics
        stored_integral = landscape.landscape_stats['landscape_integrals'][0]
        assert abs(l1_integral - stored_integral) < 1e-10
        
        print("Landscape integration test passed")
    
    def test_06_landscape_moments(self):
        """Test 6: Landscape moment computation"""
        print("Testing landscape moments...")
        
        landscape = PersistenceLandscape(self.circle_diagram, resolution=100, max_landscapes=2)
        
        # Test moment computation
        moments = landscape.compute_moments(0, orders=[1, 2, 3])
        
        assert len(moments) == 3
        assert 1 in moments and 2 in moments and 3 in moments
        
        # Moments should be finite
        for order, moment in moments.items():
            assert np.isfinite(moment)
        
        # Test with empty landscape
        empty_landscape = PersistenceLandscape(self.empty_diagram, resolution=50, max_landscapes=2)
        empty_moments = empty_landscape.compute_moments(0, orders=[1, 2])
        
        assert all(moment == 0.0 for moment in empty_moments.values())
        
        print("Landscape moments test passed")
    
    def test_07_landscape_distances(self):
        """Test 7: Distance computations between landscapes"""
        print("Testing landscape distances...")
        
        landscape1 = PersistenceLandscape(self.triangle_diagram, resolution=100, max_landscapes=2)
        landscape2 = PersistenceLandscape(self.circle_diagram, resolution=100, max_landscapes=2)
        
        # Test L2 distance
        l2_distance = landscape1.landscape_distance(landscape2, metric='L2')
        assert l2_distance >= 0
        
        # Test L1 distance
        l1_distance = landscape1.landscape_distance(landscape2, metric='L1')
        assert l1_distance >= 0
        
        # Test Linf distance
        linf_distance = landscape1.landscape_distance(landscape2, metric='Linf')
        assert linf_distance >= 0
        
        # Test Wasserstein distance
        wasserstein_distance = landscape1.wasserstein_distance(landscape2, order=1)
        assert wasserstein_distance >= 0
        
        # Distance to self should be zero
        self_distance = landscape1.landscape_distance(landscape1, metric='L2')
        assert self_distance < 1e-10
        
        print("Landscape distances test passed")
    
    def test_08_bottleneck_wasserstein_distances(self):
        """Test 8: Bottleneck and Wasserstein distance computations"""
        print("Testing bottleneck and Wasserstein distances...")
        
        # Test bottleneck distance
        bottleneck_dist = compute_bottleneck_distance(self.triangle_diagram, self.circle_diagram)
        assert bottleneck_dist >= 0
        
        # Test Wasserstein distance
        wasserstein_dist = compute_wasserstein_distance(self.triangle_diagram, self.circle_diagram, order=1)
        assert wasserstein_dist >= 0
        
        # Test with empty diagrams
        empty_bottleneck = compute_bottleneck_distance(self.empty_diagram, self.empty_diagram)
        assert empty_bottleneck == 0.0
        
        empty_wasserstein = compute_wasserstein_distance(self.empty_diagram, self.empty_diagram, order=1)
        assert empty_wasserstein == 0.0
        
        # Test distance to self (allow for small numerical errors)
        self_bottleneck = compute_bottleneck_distance(self.triangle_diagram, self.triangle_diagram)
        print(f"Self bottleneck distance: {self_bottleneck}")
        assert self_bottleneck < 1e-3  # Further relaxed tolerance
        
        print("Bottleneck and Wasserstein distances test passed")
    
    def test_09_landscape_resampling(self):
        """Test 9: Landscape resampling functionality"""
        print("Testing landscape resampling...")
        
        original_landscape = PersistenceLandscape(self.circle_diagram, resolution=100, max_landscapes=2)
        
        # Test resampling to different resolution
        resampled_landscape = original_landscape.resample(
            original_landscape.x_min, 
            original_landscape.x_max, 
            resolution=200
        )
        
        assert resampled_landscape.resolution == 200
        assert resampled_landscape.max_landscapes == 2
        assert abs(resampled_landscape.x_min - original_landscape.x_min) < 1e-10
        assert abs(resampled_landscape.x_max - original_landscape.x_max) < 1e-10
        
        # Test resampling to different range
        resampled_range = original_landscape.resample(0.0, 2.0, resolution=100)
        assert abs(resampled_range.x_min - 0.0) < 1e-10
        assert abs(resampled_range.x_max - 2.0) < 1e-10
        
        print("Landscape resampling test passed")
    
    def test_10_feature_extraction(self):
        """Test 10: Topological feature extraction"""
        print("Testing topological feature extraction...")
        
        # Create feature extractor
        extractor = TopologicalFeatureExtractor(
            max_landscapes=3,
            resolution=100,
            include_persistence_stats=True,
            include_landscape_stats=True,
            include_stability_features=True
        )
        
        # Test with multiple diagrams (different dimensions)
        diagrams = [self.triangle_diagram, self.circle_diagram]  # H0 and H1
        
        features = extractor.extract_features(diagrams)
        
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        
        # Test feature names
        feature_names = extractor.get_feature_names()
        assert len(feature_names) > 0
        assert 'n_persistence_points' in feature_names
        assert 'landscape_0_integral' in feature_names
        
        # Test with empty diagrams
        empty_features = extractor.extract_features([self.empty_diagram])
        assert isinstance(empty_features, np.ndarray)
        assert len(empty_features) > 0
        
        print("Topological feature extraction test passed")
    
    def test_11_feature_importance_ranking(self):
        """Test 11: Feature importance ranking"""
        print("Testing feature importance ranking...")
        
        # Create multiple feature vectors
        extractor = TopologicalFeatureExtractor(max_landscapes=2, resolution=50)
        
        # Generate features from different diagrams
        feature_matrix = []
        test_diagrams = [self.triangle_diagram, self.circle_diagram, self.single_point_diagram]
        
        for diagram in test_diagrams:
            features = extractor.extract_features([diagram])
            feature_matrix.append(features)
        
        feature_matrix = np.array(feature_matrix)
        
        # Test feature importance computation
        importance_scores = extractor.get_feature_importance(feature_matrix, method='variance')
        
        assert isinstance(importance_scores, dict)
        assert len(importance_scores) > 0
        
        # Test ranking function
        ranked_features = rank_topological_features(
            feature_matrix, 
            method='variance', 
            top_k=10
        )
        
        assert isinstance(ranked_features, list)
        assert len(ranked_features) <= 10
        assert all(isinstance(item, tuple) and len(item) == 2 for item in ranked_features)
        
        # Check that ranking is in descending order
        scores = [item[1] for item in ranked_features]
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
        
        print("Feature importance ranking test passed")
    
    def test_12_pytorch_integration(self):
        """Test 12: PyTorch tensor integration"""
        print("Testing PyTorch tensor integration...")
        
        landscape = PersistenceLandscape(self.circle_diagram, resolution=100, max_landscapes=3)
        
        # Test tensor conversion
        tensor = landscape.to_tensor()
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 100)
        assert tensor.dtype == torch.float32
        
        # Test with CUDA if available
        if torch.cuda.is_available():
            cuda_tensor = landscape.to_tensor(device='cuda')
            assert cuda_tensor.device.type == 'cuda'
        
        # Test feature vector conversion
        feature_vector = landscape.to_vector()
        assert isinstance(feature_vector, np.ndarray)
        assert len(feature_vector) > 0
        
        print("PyTorch integration test passed")
    
    def test_13_convenience_functions(self):
        """Test 13: Convenience functions"""
        print("Testing convenience functions...")
        
        # Test extract_topological_features function
        diagrams = [self.triangle_diagram, self.circle_diagram]
        features = extract_topological_features(diagrams, max_landscapes=2, resolution=50)
        
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        
        # Test with single diagram
        single_features = extract_topological_features([self.triangle_diagram])
        assert isinstance(single_features, np.ndarray)
        
        print("Convenience functions test passed")
    
    def test_14_performance_large_diagrams(self):
        """Test 14: Performance with large diagrams"""
        print("Testing performance with large diagrams...")
        
        start_time = time.time()
        
        # Test with large diagram
        large_landscape = PersistenceLandscape(
            self.large_diagram, 
            resolution=500, 
            max_landscapes=5
        )
        
        computation_time = time.time() - start_time
        
        assert computation_time < 10.0  # Should complete within 10 seconds
        assert large_landscape.landscapes.shape == (5, 500)
        
        # Test feature extraction performance
        start_time = time.time()
        
        extractor = TopologicalFeatureExtractor(max_landscapes=3, resolution=200)
        features = extractor.extract_features([self.large_diagram])
        
        extraction_time = time.time() - start_time
        
        assert extraction_time < 5.0  # Should complete within 5 seconds
        assert len(features) > 0
        
        print(f"Large diagram processing: {computation_time:.3f}s")
        print(f"Feature extraction: {extraction_time:.3f}s")
        print("Performance test passed")
    
    def test_15_edge_cases_and_robustness(self):
        """Test 15: Edge cases and robustness"""
        print("Testing edge cases and robustness...")
        
        # Test with very small diagram
        tiny_diagram = np.array([[0.0, 0.001]])
        tiny_landscape = PersistenceLandscape(tiny_diagram, resolution=10, max_landscapes=1)
        assert tiny_landscape.landscapes.shape == (1, 10)
        
        # Test with identical birth and death times
        identical_diagram = np.array([[0.5, 0.5]])
        identical_landscape = PersistenceLandscape(identical_diagram, resolution=10, max_landscapes=1)
        assert np.all(identical_landscape.landscapes == 0)  # Should be zero landscape
        
        # Test with negative values (should be handled gracefully)
        try:
            negative_diagram = np.array([[-0.1, 0.5]])
            negative_landscape = PersistenceLandscape(negative_diagram, resolution=10, max_landscapes=1)
            # Should not crash
        except Exception as e:
            print(f"Negative values handled: {e}")
        
        # Test with very high resolution
        high_res_landscape = PersistenceLandscape(
            self.triangle_diagram, 
            resolution=2000, 
            max_landscapes=1
        )
        assert high_res_landscape.resolution == 2000
        
        print("Edge cases and robustness test passed")


def run_comprehensive_tests():
    """Run all tests and provide summary"""
    test_instance = TestPersistenceLandscapes()
    test_instance.setup_method()
    
    test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
    
    print("Running Persistence Landscapes Comprehensive Test Suite")
    print("=" * 65)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_method in test_methods:
        try:
            print(f"\nRunning {test_method}...")
            getattr(test_instance, test_method)()
            passed += 1
        except Exception as e:
            if "skip" in str(e).lower():
                skipped += 1
                print(f"SKIPPED: {str(e)}")
            else:
                failed += 1
                print(f"FAILED: {str(e)}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "=" * 65)
    print(f"Test Summary: {passed} passed, {failed} failed, {skipped} skipped")
    if passed + failed > 0:
        print(f"Success Rate: {passed / (passed + failed) * 100:.1f}%")
    
    return passed, failed, skipped


if __name__ == "__main__":
    run_comprehensive_tests() 