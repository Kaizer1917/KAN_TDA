"""
Integration Test Suite for TDA-Enhanced KAN_TDA Components

Tests:
1. TDAFrequencyDecomp functionality and backward compatibility
2. TopologyGuidedMKAN adaptive complexity
3. TDAKAN_TDA end-to-end integration
4. Performance comparison with baseline KAN_TDA
5. Configuration validation and error handling
"""

import torch
import torch.nn as nn
import numpy as np
import time
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.KAN_TDA import (
    Model, TDAKAN_TDA, TDAFrequencyDecomp, TopologyGuidedMKAN, 
    TDAFrequencyMixing, FrequencyDecomp, M_KAN, FrequencyMixing
)


class MockConfig:
    """Mock configuration for testing"""
    def __init__(self, **kwargs):
        # Base KAN_TDA configuration
        self.task_name = 'long_term_forecast'
        self.seq_len = 96
        self.label_len = 48
        self.pred_len = 96
        self.d_model = 16
        self.e_layers = 2
        self.down_sampling_layers = 0
        self.down_sampling_window = 1
        self.begin_order = 1
        self.moving_avg = 25
        self.enc_in = 7
        self.c_out = 7
        self.embed = 'timeF'
        self.freq = 'h'
        self.dropout = 0.1
        self.use_norm = 1
        self.channel_independence = 0
        self.use_future_temporal_feature = 0
        
        # TDA-specific configuration
        self.enable_tda = False
        self.tda_mode = 'full'
        self.takens_dims = [2, 3, 5]
        self.takens_delays = [1, 2, 4]
        self.homology_backend = 'ripser'
        self.max_homology_dim = 2
        self.landscape_resolution = 100
        self.tda_weight = 0.3
        self.enable_tda_guidance = True
        self.max_kan_order = 4
        self.topo_attention_heads = 4
        self.topo_attention_dropout = 0.1
        self.enable_topology_guidance = True
        self.enable_tda_caching = True
        self.enable_cross_frequency_attention = True
        
        # Override with provided kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_tda_frequency_decomp():
    """Test TDA-enhanced frequency decomposition"""
    print("Testing TDAFrequencyDecomp...")
    
    config = MockConfig(down_sampling_layers=2)
    batch_size = 4
    seq_len = 96
    d_model = 16
    
    # Create test data
    test_data = [
        torch.randn(batch_size, seq_len, d_model),
        torch.randn(batch_size, seq_len // 2, d_model),
        torch.randn(batch_size, seq_len // 4, d_model)
    ]
    
    # Test initialization
    decomp = TDAFrequencyDecomp(config)
    assert hasattr(decomp, 'takens_embedding')
    assert hasattr(decomp, 'homology_computer')
    assert hasattr(decomp, 'feature_extractor')
    print("✓ Initialization successful")
    
    # Test forward pass
    with torch.no_grad():
        output = decomp(test_data)
        assert len(output) == len(test_data)
        for i, out in enumerate(output):
            assert out.shape == test_data[i].shape
    print("✓ Forward pass successful")
    
    # Test backward compatibility
    config_no_tda = MockConfig(enable_tda_guidance=False, down_sampling_layers=2)
    original_decomp = FrequencyDecomp(config_no_tda)
    tda_decomp = TDAFrequencyDecomp(config_no_tda)
    
    with torch.no_grad():
        original_out = original_decomp(test_data)
        tda_out = tda_decomp(test_data)
        
        assert len(original_out) == len(tda_out)
        for orig, tda in zip(original_out, tda_out):
            assert orig.shape == tda.shape
    print("✓ Backward compatibility verified")


def test_topology_guided_mkan():
    """Test topology-guided M-KAN implementation"""
    print("Testing TopologyGuidedMKAN...")
    
    config = MockConfig()
    d_model = 16
    seq_len = 96
    order = 2
    batch_size = 4
    
    test_input = torch.randn(batch_size, seq_len, d_model)
    
    # Test initialization
    tg_mkan = TopologyGuidedMKAN(d_model, seq_len, order, config)
    assert hasattr(tg_mkan, 'complexity_analyzer')
    assert hasattr(tg_mkan, 'adaptive_kan_layers')
    assert hasattr(tg_mkan, 'topo_attention')
    print("✓ Initialization successful")
    
    # Test complexity analysis
    with torch.no_grad():
        complexity = tg_mkan._analyze_topological_complexity(test_input)
        assert complexity.shape == (batch_size, 1)
        assert (complexity >= 0).all() and (complexity <= 1).all()
    print("✓ Complexity analysis working")
    
    # Test adaptive order selection
    with torch.no_grad():
        selected_order, order_weights = tg_mkan._adaptive_order_selection(test_input)
        assert tg_mkan.min_order <= selected_order <= tg_mkan.max_order
        if order_weights is not None:
            assert order_weights.shape[0] == batch_size
    print("✓ Adaptive order selection working")
    
    # Test forward pass
    with torch.no_grad():
        output = tg_mkan(test_input)
        assert output.shape == test_input.shape
        
        # Test with frequency context
        frequency_context = torch.randn(batch_size, seq_len, d_model)
        output_with_context = tg_mkan(test_input, frequency_context=frequency_context)
        assert output_with_context.shape == test_input.shape
    print("✓ Forward pass successful")


def test_tda_timekan():
    """Test complete TDA-KAN_TDA integration"""
    print("Testing TDAKAN_TDA...")
    
    config = MockConfig(enable_tda=True)
    batch_size = 4
    seq_len = 96
    enc_in = 7
    
    test_input = torch.randn(batch_size, seq_len, enc_in)
    
    # Test initialization
    model = TDAKAN_TDA(config)
    assert hasattr(model, 'tda_res_blocks')
    assert hasattr(model, 'tda_add_blocks')
    assert model.enable_tda == True
    print("✓ Initialization successful")
    
    # Test backward compatibility
    config_original = MockConfig(enable_tda=False)
    original_model = Model(config_original)
    tda_model = TDAKAN_TDA(config_original)
    
    assert not tda_model.enable_tda
    
    with torch.no_grad():
        original_out = original_model.forecast(test_input)
        tda_out = tda_model.forecast(test_input)
        assert original_out.shape == tda_out.shape
    print("✓ Backward compatibility verified")
    
    # Test forecasting functionality
    with torch.no_grad():
        output = model.forecast(test_input)
        expected_shape = (batch_size, config.pred_len, config.c_out)
        assert output.shape == expected_shape
        assert torch.isfinite(output).all()
    print("✓ Forecasting functionality working")
    
    # Test performance monitoring
    stats = model.get_tda_performance_stats()
    assert stats['tda_enabled'] == True
    
    with torch.no_grad():
        _ = model.forecast(test_input)
    
    updated_stats = model.get_tda_performance_stats()
    assert updated_stats['total_tda_time'] >= 0.0
    print("✓ Performance monitoring working")


def test_performance_comparison():
    """Performance comparison between original and TDA-enhanced KAN_TDA"""
    print("Testing Performance Comparison...")
    
    config_original = MockConfig(enable_tda=False)
    config_tda = MockConfig(enable_tda=True, tda_mode='full')
    batch_size = 2  # Smaller batch for performance tests
    seq_len = 96
    enc_in = 7
    
    test_input = torch.randn(batch_size, seq_len, enc_in)
    
    original_model = Model(config_original)
    tda_model = TDAKAN_TDA(config_tda)
    
    # Test computational overhead
    with torch.no_grad():
        # Warm up
        _ = original_model.forecast(test_input)
        _ = tda_model.forecast(test_input)
        
        # Time original model
        start_time = time.time()
        for _ in range(3):
            _ = original_model.forecast(test_input)
        original_time = time.time() - start_time
        
        # Time TDA model
        start_time = time.time()
        for _ in range(3):
            _ = tda_model.forecast(test_input)
        tda_time = time.time() - start_time
        
        overhead_ratio = tda_time / original_time
        print(f"✓ TDA overhead ratio: {overhead_ratio:.2f}x")
        assert overhead_ratio < 20.0  # Reasonable overhead threshold
    
    # Test memory usage
    original_params = sum(p.numel() for p in original_model.parameters())
    tda_params = sum(p.numel() for p in tda_model.parameters())
    param_ratio = tda_params / original_params
    print(f"✓ TDA parameter ratio: {param_ratio:.2f}x")
    assert param_ratio < 10.0  # Reasonable parameter increase threshold
    
    # Test output quality
    with torch.no_grad():
        original_out = original_model.forecast(test_input)
        tda_out = tda_model.forecast(test_input)
        
        assert torch.isfinite(original_out).all()
        assert torch.isfinite(tda_out).all()
        
        original_std = original_out.std()
        tda_std = tda_out.std()
        std_ratio = tda_std / original_std
        
        assert 0.01 < std_ratio < 100.0  # Outputs in reasonable ranges
    print("✓ Output quality verified")


def test_configuration_validation():
    """Test configuration validation and error handling"""
    print("Testing Configuration Validation...")
    
    # Test with missing required attributes
    class IncompleteConfig:
        # Add minimal required attributes
        task_name = 'long_term_forecast'
        seq_len = 96
        label_len = 48
        pred_len = 96
        d_model = 16
        e_layers = 2
        down_sampling_layers = 0
        down_sampling_window = 1
        begin_order = 1
        moving_avg = 25
        enc_in = 7
        c_out = 7
        embed = 'timeF'
        freq = 'h'
        dropout = 0.1
        use_norm = 1
        channel_independence = 0
        use_future_temporal_feature = 0
    
    incomplete_config = IncompleteConfig()
    
    # Should handle missing attributes gracefully with defaults
    try:
        model = TDAKAN_TDA(incomplete_config)
        print("✓ Handles missing config attributes gracefully")
    except AttributeError:
        print("✗ Failed to handle missing config attributes")
        return False
    
    # Test with extreme TDA parameters
    config = MockConfig(
        enable_tda=True,
        takens_dims=[2, 3, 5, 10],
        tda_weight=1.5,
        max_kan_order=8
    )
    
    model = TDAKAN_TDA(config)
    assert model.enable_tda == True
    print("✓ Handles extreme TDA parameters")
    
    # Test device compatibility
    config = MockConfig(enable_tda=True)
    model = TDAKAN_TDA(config)
    
    test_input = torch.randn(2, 96, 7)
    with torch.no_grad():
        output = model.forecast(test_input)
        assert output.device == test_input.device
    print("✓ Device compatibility verified")
    
    return True


def run_integration_tests():
    """Run all integration tests"""
    print("=" * 60)
    print("TDA-KAN_TDA Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("TDA Frequency Decomposition", test_tda_frequency_decomp),
        ("Topology Guided M-KAN", test_topology_guided_mkan),
        ("TDA KAN_TDA Integration", test_tda_timekan),
        ("Performance Comparison", test_performance_comparison),
        ("Configuration Validation", test_configuration_validation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            if result is False:
                print(f"✗ {test_name} FAILED")
            else:
                print(f"✓ {test_name} PASSED")
                passed += 1
        except Exception as e:
            print(f"✗ {test_name} FAILED: {str(e)}")
    
    print("\n" + "=" * 60)
    print(f"Test Summary: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1) 