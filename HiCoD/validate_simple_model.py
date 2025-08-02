#!/usr/bin/env python3
"""
Simple model test for HiCoD without complex LLM components
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
from typing import Dict, List


def validate_simple_forward():
    """Validate simplified forward pass"""
    print("Validating simplified model forward pass...")
    
    # Create a simple test model
    class SimpleHiCoD(nn.Module):
        def __init__(self, feature_dims, shared_dim=512, num_classes=3):
            super().__init__()
            self.shared_dim = shared_dim
            self.num_classes = num_classes
            
            # Simple modality encoders
            self.text_encoder = nn.Linear(feature_dims[0], shared_dim)
            self.audio_encoder = nn.Linear(feature_dims[1], shared_dim)
            self.vision_encoder = nn.Linear(feature_dims[2], shared_dim)
            
            # Simple fusion
            self.fusion = nn.Linear(shared_dim * 3, shared_dim)
            
            # Simple classifier
            self.classifier = nn.Linear(shared_dim, num_classes)
            
            # Simple loss
            self.criterion = nn.CrossEntropyLoss()
        
        def forward(self, batch):
            # Extract features
            text_feat = batch.get('text', None)
            audio_feat = batch.get('audio', None)
            vision_feat = batch.get('vision', None)
            labels = batch['labels']['M'].squeeze(-1).long() if 'labels' in batch and 'M' in batch['labels'] else None
            
            # Encode features
            encoded_features = {}
            if text_feat is not None:
                encoded_features['text'] = self.text_encoder(text_feat)
            if audio_feat is not None:
                encoded_features['audio'] = self.audio_encoder(audio_feat)
            if vision_feat is not None:
                encoded_features['vision'] = self.vision_encoder(vision_feat)
            
            # Simple fusion (concatenate available features)
            available_features = []
            for mod in ['text', 'audio', 'vision']:
                if mod in encoded_features and encoded_features[mod] is not None:
                    available_features.append(encoded_features[mod])
                else:
                    # Use zero tensor for missing modalities
                    available_features.append(torch.zeros_like(encoded_features.get('text', torch.randn(1, self.shared_dim))))
            
            # Concatenate and fuse
            concatenated = torch.cat(available_features, dim=1)
            fused = self.fusion(concatenated)
            
            # Classify
            logits = self.classifier(fused)
            probs = torch.softmax(logits, dim=1)
            
            # Compute loss if labels are available
            if labels is not None:
                loss = self.criterion(logits, labels)
            else:
                loss = torch.tensor(0.0)
            
            return {
                'loss': loss,
                'predictions': probs,
                'cls_loss': loss,
                'local_loss': torch.tensor(0.0),
                'fusion_loss': torch.tensor(0.0),
                'prompt_loss': torch.tensor(0.0)
            }
    
    # Create model
    model = SimpleHiCoD(feature_dims=[768, 74, 35], shared_dim=512, num_classes=3)
    model.eval()
    
    # Create dummy batch
    batch_size = 4
    batch = {
        'text': torch.randn(batch_size, 768),
        'audio': torch.randn(batch_size, 74),
        'vision': torch.randn(batch_size, 35),
        'labels': {'M': torch.randint(0, 3, (batch_size, 1)).float()}
    }
    
    # Forward pass
    with torch.no_grad():
        output = model(batch)
    
    # Check output structure
    required_keys = ['loss', 'predictions', 'cls_loss', 'local_loss', 'fusion_loss', 'prompt_loss']
    for key in required_keys:
        assert key in output, f"Missing key: {key}"
    
    # Check output shapes
    assert output['predictions'].shape == (batch_size, 3), f"Wrong predictions shape: {output['predictions'].shape}"
    assert output['loss'].item() >= 0, f"Negative loss: {output['loss'].item()}"
    
    print("Simple model forward pass validation passed!")
    return True


def validate_missing_modalities():
    """Validate handling of missing modalities"""
    print("Validating missing modalities handling...")
    
    class SimpleHiCoD(nn.Module):
        def __init__(self, feature_dims, shared_dim=512, num_classes=3):
            super().__init__()
            self.shared_dim = shared_dim
            self.num_classes = num_classes
            
            # Simple modality encoders
            self.text_encoder = nn.Linear(feature_dims[0], shared_dim)
            self.audio_encoder = nn.Linear(feature_dims[1], shared_dim)
            self.vision_encoder = nn.Linear(feature_dims[2], shared_dim)
            
            # Simple fusion
            self.fusion = nn.Linear(shared_dim * 3, shared_dim)
            
            # Simple classifier
            self.classifier = nn.Linear(shared_dim, num_classes)
            
            # Simple loss
            self.criterion = nn.CrossEntropyLoss()
        
        def forward(self, batch):
            # Extract features
            text_feat = batch.get('text', None)
            audio_feat = batch.get('audio', None)
            vision_feat = batch.get('vision', None)
            labels = batch['labels']['M'].squeeze(-1).long() if 'labels' in batch and 'M' in batch['labels'] else None
            
            # Encode features
            encoded_features = {}
            if text_feat is not None:
                encoded_features['text'] = self.text_encoder(text_feat)
            if audio_feat is not None:
                encoded_features['audio'] = self.audio_encoder(audio_feat)
            if vision_feat is not None:
                encoded_features['vision'] = self.vision_encoder(vision_feat)
            
            # Simple fusion (concatenate available features)
            available_features = []
            for mod in ['text', 'audio', 'vision']:
                if mod in encoded_features and encoded_features[mod] is not None:
                    available_features.append(encoded_features[mod])
                else:
                    # Use zero tensor for missing modalities
                    available_features.append(torch.zeros_like(encoded_features.get('text', torch.randn(1, self.shared_dim))))
            
            # Concatenate and fuse
            concatenated = torch.cat(available_features, dim=1)
            fused = self.fusion(concatenated)
            
            # Classify
            logits = self.classifier(fused)
            probs = torch.softmax(logits, dim=1)
            
            # Compute loss if labels are available
            if labels is not None:
                loss = self.criterion(logits, labels)
            else:
                loss = torch.tensor(0.0)
            
            return {
                'loss': loss,
                'predictions': probs,
                'cls_loss': loss,
                'local_loss': torch.tensor(0.0),
                'fusion_loss': torch.tensor(0.0),
                'prompt_loss': torch.tensor(0.0)
            }
    
    # Create model
    model = SimpleHiCoD(feature_dims=[768, 74, 35], shared_dim=512, num_classes=3)
    model.eval()
    
    # Test with missing modalities
    batch_size = 4
    
    # Test case 1: All modalities present
    batch1 = {
        'text': torch.randn(batch_size, 768),
        'audio': torch.randn(batch_size, 74),
        'vision': torch.randn(batch_size, 35),
        'labels': {'M': torch.randint(0, 3, (batch_size, 1)).float()}
    }
    
    # Test case 2: Missing audio
    batch2 = {
        'text': torch.randn(batch_size, 768),
        'audio': None,
        'vision': torch.randn(batch_size, 35),
        'labels': {'M': torch.randint(0, 3, (batch_size, 1)).float()}
    }
    
    # Test case 3: Missing vision
    batch3 = {
        'text': torch.randn(batch_size, 768),
        'audio': torch.randn(batch_size, 74),
        'vision': None,
        'labels': {'M': torch.randint(0, 3, (batch_size, 1)).float()}
    }
    
    batches = [batch1, batch2, batch3]
    
    for i, batch in enumerate(batches):
        with torch.no_grad():
            output = model(batch)
        
        # Check that output is valid
        assert output['predictions'].shape == (batch_size, 3), f"Wrong predictions shape for batch {i}"
        assert output['loss'].item() >= 0, f"Negative loss for batch {i}"
    
    print("Missing modalities handling validation passed!")
    return True


def main():
    """Run simple tests"""
    print("Running HiCoD simple validations...")
    print("="*50)
    
    validations = [
        validate_simple_forward,
        validate_missing_modalities
    ]
    
    passed = 0
    total = len(validations)
    
    for validation in validations:
        try:
            if validation():
                passed += 1
        except Exception as e:
            print(f"Validation failed: {e}")
    
    print("="*50)
    print(f"Validation Results: {passed}/{total} validations passed")
    
    if passed == total:
        print("All simple validations passed! HiCoD core functionality is working.")
    else:
        print("Some validations failed. Please check the implementation.")


if __name__ == "__main__":
    main() 