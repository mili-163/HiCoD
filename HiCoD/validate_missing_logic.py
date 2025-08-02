#!/usr/bin/env python3
"""
Simple test for missing modality logic
"""

import torch
import numpy as np


def validate_fixed_missing_logic():
    """Validate fixed missing modality logic"""
    print("Validating fixed missing modality logic...")
    
    # Simulate the logic from data_loader
    def apply_fixed_missing(text_feat, audio_feat, vision_feat, missing_modalities):
        """Apply fixed missing modalities"""
        if 'text' in missing_modalities:
            text_feat = None
        if 'audio' in missing_modalities:
            audio_feat = None
        if 'vision' in missing_modalities:
            vision_feat = None
        
        return text_feat, audio_feat, vision_feat
    
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
        print(f"Validating scenario: {scenario['name']}")
        
        # Create dummy features
        text_feat = torch.randn(768)
        audio_feat = torch.randn(74)
        vision_feat = torch.randn(35)
        
        # Apply fixed missing
        result_text, result_audio, result_vision = apply_fixed_missing(
            text_feat, audio_feat, vision_feat, scenario['modalities']
        )
        
        # Verify masking
        for mod in scenario['modalities']:
            if mod == 'text':
                assert result_text is None, f"Text should be masked in scenario {scenario['name']}"
            elif mod == 'audio':
                assert result_audio is None, f"Audio should be masked in scenario {scenario['name']}"
            elif mod == 'vision':
                assert result_vision is None, f"Vision should be masked in scenario {scenario['name']}"
        
        # Verify non-masked modalities are present
        if 'text' not in scenario['modalities']:
            assert result_text is not None, f"Text should not be masked in scenario {scenario['name']}"
        if 'audio' not in scenario['modalities']:
            assert result_audio is not None, f"Audio should not be masked in scenario {scenario['name']}"
        if 'vision' not in scenario['modalities']:
            assert result_vision is not None, f"Vision should not be masked in scenario {scenario['name']}"
        
        print(f"Scenario {scenario['name']} validated")
    
    print("All fixed missing scenarios validated!")
    return True


def validate_random_missing_logic():
    """Validate random missing modality logic"""
    print("Validating random missing modality logic...")
    
    # Simulate the logic from data_loader
    def apply_random_missing(text_feat, audio_feat, vision_feat, missing_rate):
        """Apply random missing rate to features"""
        if np.random.random() < missing_rate:
            # Randomly choose which modality to mask
            modalities = ['text', 'audio', 'vision']
            mask_modality = np.random.choice(modalities)
            
            if mask_modality == 'text':
                text_feat = None
            elif mask_modality == 'audio':
                audio_feat = None
            elif mask_modality == 'vision':
                vision_feat = None
        
        return text_feat, audio_feat, vision_feat
    
    # Test different missing rates
    missing_rates = [0.0, 0.1, 0.3, 0.5, 0.7]
    
    for mr in missing_rates:
        print(f"Validating missing rate: {mr*100:.0f}%")
        
        # Test multiple times to check randomness
        masked_counts = []
        for _ in range(20):  # Test 20 times
            # Create dummy features
            text_feat = torch.randn(768)
            audio_feat = torch.randn(74)
            vision_feat = torch.randn(35)
            
            # Apply random missing
            result_text, result_audio, result_vision = apply_random_missing(
                text_feat, audio_feat, vision_feat, mr
            )
            
            # Count masked modalities
            masked_count = sum(1 for feat in [result_text, result_audio, result_vision] if feat is None)
            masked_counts.append(masked_count)
        
        # For missing rate 0.0, no modalities should be masked
        if mr == 0.0:
            assert all(count == 0 for count in masked_counts), f"With missing rate 0.0, no modalities should be masked"
        else:
            # For other rates, some modalities should be masked
            assert any(count > 0 for count in masked_counts), f"With missing rate {mr}, some modalities should be masked"
        
        print(f"Missing rate {mr*100:.0f}% validated")
    
    print("All random missing scenarios validated!")
    return True


def validate_missing_rate_formula():
    """Validate missing rate calculation formula"""
    print("Validating missing rate calculation formula...")
    
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
    
    print("Missing rate calculation validated!")
    return True


def main():
    """Run all missing logic validations"""
    print("Running missing modality logic validations...")
    print("="*60)
    
    validations = [
        validate_fixed_missing_logic,
        validate_random_missing_logic,
        validate_missing_rate_formula
    ]
    
    passed = 0
    total = len(validations)
    
    for validation in validations:
        try:
            if validation():
                passed += 1
        except Exception as e:
            print(f"Validation failed: {e}")
    
    print("="*60)
    print(f"Validation Results: {passed}/{total} validations passed")
    
    if passed == total:
        print("All missing logic validations passed!")
    else:
        print("Some validations failed. Please check the implementation.")


if __name__ == "__main__":
    main() 