#!/usr/bin/env python3
"""
Main script for running HiCoD (Hierarchical Consistency-Guided Prompt Distillation) experiments
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from data_loader import HiCoDDataLoader
from trains.ATIO import ATIO
from trains.singleTask.model import create_model


def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def HiCoD_run(model_name='hicod', dataset_name='mosi', seeds=[1111, 2222, 3333, 4444, 5555], 
              mr=0.1, missing_scenario='random', missing_modalities=None):
    """
    Run HiCoD experiments
    
    Args:
        model_name: Name of the model to use
        dataset_name: Name of the dataset (mosi/mosei)
        seeds: List of random seeds for reproducibility
        mr: Missing rate for simulating missing modalities
        missing_scenario: 'random' or 'fixed' missing modality scenario
        missing_modalities: List of modalities to mask in fixed scenario
    """
    print(f"Running HiCoD experiments with model: {model_name}, dataset: {dataset_name}")
    print(f"Missing scenario: {missing_scenario}")
    if missing_scenario == 'fixed':
        print(f"Fixed missing modalities: {missing_modalities}")
    else:
        print(f"Random missing rate: {mr}")
    
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load configuration
    config_path = 'config/config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    if model_name not in config:
        print(f"Error: Model '{model_name}' not found in config")
        return
    
    model_config = config[model_name]
    
    # Run experiments for each seed
    results = []
    for seed in seeds:
        print(f"\nRunning experiment with seed: {seed}")
        
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Create data loader with missing rate
        data_loader = HiCoDDataLoader(dataset_name, model_config['datasetParams'][dataset_name], 
                                     missing_rate=mr, missing_scenario=missing_scenario, 
                                     missing_modalities=missing_modalities)
        
        # Create model
        args = argparse.Namespace(**model_config['commonParams'])
        args.device = device
        args.feature_dims = data_loader.get_feature_dims()
        args.num_classes = 3  # For sentiment classification
        
        model = create_model(args)
        model.to(args.device)
        
        # Create trainer
        trainer = ATIO()
        
        # Train and test
        try:
            # Training
            print("Starting training...")
            train_results = trainer.do_train(model, data_loader.get_data_loaders(), return_epoch_results=True)
            
            # Validation
            print("Starting validation...")
            test_results = trainer.do_validate(model, data_loader.get_test_loader())
            
            # Save results
            result = {
                'seed': seed,
                'train_results': train_results,
                'test_results': test_results
            }
            results.append(result)
            
            print(f"Seed {seed} completed successfully")
            
        except Exception as e:
            print(f"Error in seed {seed}: {e}")
            continue
    
    # Print summary
    print(f"\nExperiment Summary:")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Seeds: {seeds}")
    print(f"Completed runs: {len(results)}/{len(seeds)}")
    
    # Save results
    output_dir = f"results/{model_name}_{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_dir}/results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run HiCoD experiments')
    parser.add_argument('--model', type=str, default='hicod', help='Model name')
    parser.add_argument('--dataset', type=str, default='mosi', help='Dataset name')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1111, 2222, 3333, 4444, 5555], help='Random seeds')
    parser.add_argument('--mr', type=float, default=0.1, help='Missing rate for random scenario')
    parser.add_argument('--scenario', type=str, default='random', choices=['random', 'fixed'], help='Missing modality scenario')
    parser.add_argument('--missing_modalities', type=str, nargs='+', default=None, help='Modalities to mask in fixed scenario')
    
    args = parser.parse_args()
    
    HiCoD_run(
        model_name=args.model,
        dataset_name=args.dataset,
        seeds=args.seeds,
        mr=args.mr,
        missing_scenario=args.scenario,
        missing_modalities=args.missing_modalities
    )