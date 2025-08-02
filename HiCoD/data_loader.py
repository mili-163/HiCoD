import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from typing import Dict, List, Tuple, Optional


def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


class HiCoDDataset(Dataset):
    """Dataset for HiCoD"""
    
    def __init__(self, data_path: str, split: str = 'train', missing_rate: float = 0.0, 
                 missing_scenario: str = 'random', missing_modalities: List[str] = None):
        self.data_path = data_path
        self.split = split
        self.missing_rate = missing_rate
        self.missing_scenario = missing_scenario  # 'random' or 'fixed'
        self.missing_modalities = missing_modalities or []  # List of modalities to always mask
        
        # Load data
        self.data = self.load_data()
        
    def load_data(self):
        """Load dataset from pickle file"""
        file_path = os.path.join(self.data_path, f"{self.split}.pkl")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract features
        text_feat = torch.tensor(item['text'], dtype=torch.float32) if 'text' in item else None
        audio_feat = torch.tensor(item['audio'], dtype=torch.float32) if 'audio' in item else None
        vision_feat = torch.tensor(item['vision'], dtype=torch.float32) if 'vision' in item else None
        
        # Apply missing rate for training and validation
        if self.split in ['train', 'val']:
            if self.missing_scenario == 'fixed' and self.missing_modalities:
                text_feat, audio_feat, vision_feat = self.apply_fixed_missing(text_feat, audio_feat, vision_feat)
            elif self.missing_scenario == 'random' and self.missing_rate > 0:
                text_feat, audio_feat, vision_feat = self.apply_random_missing(text_feat, audio_feat, vision_feat)
        
        # Extract label
        label = torch.tensor(item['label'], dtype=torch.float32)
        
        return {
            'text': text_feat,
            'audio': audio_feat,
            'vision': vision_feat,
            'labels': {'M': label}
        }
    
    def apply_fixed_missing(self, text_feat, audio_feat, vision_feat):
        """Apply fixed missing modalities"""
        if 'text' in self.missing_modalities:
            text_feat = None
        if 'audio' in self.missing_modalities:
            audio_feat = None
        if 'vision' in self.missing_modalities:
            vision_feat = None
        
        return text_feat, audio_feat, vision_feat
    
    def apply_random_missing(self, text_feat, audio_feat, vision_feat):
        """Apply random missing rate to features"""
        if np.random.random() < self.missing_rate:
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


class HiCoDDataLoader:
    """Data loader for HiCoD"""
    
    def __init__(self, dataset_name: str, config: Dict, missing_rate: float = 0.0, 
                 missing_scenario: str = 'random', missing_modalities: List[str] = None):
        self.dataset_name = dataset_name
        self.config = config
        self.data_path = config['data_path']
        self.batch_size = config.get('batch_size', 32)
        self.missing_rate = missing_rate
        self.missing_scenario = missing_scenario
        self.missing_modalities = missing_modalities or []
        self.device = get_device()
        
        # Create datasets
        self.train_dataset = HiCoDDataset(self.data_path, 'train', missing_rate, 
                                         missing_scenario, missing_modalities)
        self.val_dataset = HiCoDDataset(self.data_path, 'val', missing_rate, 
                                       missing_scenario, missing_modalities)
        self.test_dataset = HiCoDDataset(self.data_path, 'test', missing_rate, 
                                        missing_scenario, missing_modalities)
        
        # Create data loaders with GPU pinning if available
        pin_memory = self.device == 'cuda'
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4 if self.device == 'cuda' else 0,
            pin_memory=pin_memory
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4 if self.device == 'cuda' else 0,
            pin_memory=pin_memory
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4 if self.device == 'cuda' else 0,
            pin_memory=pin_memory
        )
    
    def get_data_loaders(self) -> Dict[str, DataLoader]:
        """Get train and validation data loaders"""
        return {
            'train': self.train_loader,
            'val': self.val_loader
        }
    
    def get_test_loader(self) -> DataLoader:
        """Get test data loader"""
        return self.test_loader
    
    def get_feature_dims(self) -> List[int]:
        """Get feature dimensions for each modality"""
        return self.config['feature_dims']
