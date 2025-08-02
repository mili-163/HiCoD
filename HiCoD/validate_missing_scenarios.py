#!/usr/bin/env python3
"""
Test script for missing modality scenarios
"""

import torch
import numpy as np
from data_loader import HiCoDDataset


def create_dummy_data(batch_size: int = 10):
    """Create dummy data for testing"""
    data = []
    
    for i in range(batch_size):
        item = {
            'text': torch.randn(768),
            'audio': torch.randn(74),
            'vision': torch.randn(35),
            'label': torch.randint(0, 3, (1,)).item()
        }
        data.append(item)
    
    return data


def test_fixed_missing_scenarios():
    """Test fixed missing modality scenarios"""
    print("Testing fixed missing modality scenarios...")
    
    # Create dummy data
    dummy_data = create_dummy_data(10)
    
    # Test scenarios
    scenarios = [
        {'name': 'missing_text', 'modalities': ['text']},
        {'name': 'missing_audio', 'modalities': ['audio']},
        {'name': 'missing_vision', 'modalities': ['vision']},
        {'name': 'missing_text_audio', 'modalities': ['text', 'audio']},
        {'name': 'missing_text_vision', 'modalities': ['text', 'vision']},
        {'name': 'missing_audio_vision', 'modalities': ['audio', 'vision']},
    ]
    
    for scenario in scenarios:
        print(f"Testing scenario: {scenario['name']}")
        
        # Create dataset with fixed missing scenario
        dataset = HiCoDDataset('dummy_path', 'train', missing_rate=0.0, 
                              missing_scenario='fixed', missing_modalities=scenario['modalities'])
        # Override data directly to avoid file loading
        dataset.data = dummy_data
        
        # Test a few samples
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            
            # Check if modalities are properly masked
            modalities = ['text', 'audio', 'vision']
            masked_count = sum(1 for mod in modalities if sample[mod] is None)
            
            # Verify masking
            for mod in scenario['modalities']:
                assert sample[mod] is None, f"Modality {mod} should be masked in scenario {scenario['name']}"
            
            # Verify non-masked modalities are present
            for mod in modalities:
                if mod not in scenario['modalities']:
                    assert sample[mod] is not None, f"Modality {mod} should not be masked in scenario {scenario['name']}"
        
        print(f"Scenario {scenario['name']} passed")
    
    print("All fixed missing scenarios passed!")
    return True


def test_random_missing_scenarios():
    """Test random missing modality scenarios"""
    print("Testing random missing modality scenarios...")
    
    # Create dummy data
    dummy_data = create_dummy_data(20)
    
    # Test different missing rates
    missing_rates = [0.0, 0.1, 0.3, 0.5, 0.7]
    
    for mr in missing_rates:
        print(f"Testing missing rate: {mr*100:.0f}%")
        
        # Create dataset with random missing scenario
        dataset = HiCoDDataset('dummy_path', 'train', missing_rate=mr, 
                              missing_scenario='random', missing_modalities=None)
        # Override data directly to avoid file loading
        dataset.data = dummy_data
        
        # Test multiple samples to check randomness
        masked_counts = []
        for i in range(min(10, len(dataset))):
            sample = dataset[i]
            
            # Count masked modalities
            modalities = ['text', 'audio', 'vision']
            masked_count = sum(1 for mod in modalities if sample[mod] is None)
            masked_counts.append(masked_count)
        
        # For missing rate 0.0, no modalities should be masked
        if mr == 0.0:
            assert all(count == 0 for count in masked_counts), f"With missing rate 0.0, no modalities should be masked"
        else:
            # For other rates, some modalities should be masked
            assert any(count > 0 for count in masked_counts), f"With missing rate {mr}, some modalities should be masked"
        
        print(f"Missing rate {mr*100:.0f}% passed")
    
    print("All random missing scenarios passed!")
    return True


def test_missing_rate_calculation():
    """Test missing rate calculation formula"""
    print("Testing missing rate calculation...")
    
    # Test the formula: r_miss = (1 - Σ(m_i)/(N×M)) × 100%
    # where N = number of samples, M = number of modalities (3), m_i = available modalities for sample i
    
    # Example 1: No missing modalities
    N = 10  # 10 samples
    M = 3   # 3 modalities
    m_i = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]  # All modalities available
    r_miss = (1 - sum(m_i)/(N*M)) * 100
    assert r_miss == 0.0, f"Expected 0%, got {r_miss}%"
    
    # Example 2: 50% missing rate
    m_i = [2, 1, 2, 1, 2, 1, 2, 1, 2, 1]  # Average 1.5 modalities per sample
    r_miss = (1 - sum(m_i)/(N*M)) * 100
    expected = (1 - 15/(10*3)) * 100  # 15 total modalities / 30 total possible = 0.5
    assert abs(r_miss - 50.0) < 0.1, f"Expected ~50%, got {r_miss}%"
    
    # Example 3: 70% missing rate (maximum)
    m_i = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # Only 1 modality per sample
    r_miss = (1 - sum(m_i)/(N*M)) * 100
    expected = (1 - 10/(10*3)) * 100  # 10 total modalities / 30 total possible = 0.33
    assert abs(r_miss - 66.67) < 0.1, f"Expected ~66.67%, got {r_miss}%"
    
    print("Missing rate calculation passed!")
    return True


def main():
    """Run all missing scenario tests"""
    print("Running missing modality scenario tests...")
    print("="*60)
    
    tests = [
        test_fixed_missing_scenarios,
        test_random_missing_scenarios,
        test_missing_rate_calculation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"Test failed: {e}")
    
    print("="*60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All missing scenario tests passed!")
    else:
        print("Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    main() 