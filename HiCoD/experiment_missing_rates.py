#!/usr/bin/env python3
"""
Experiment script for testing HiCoD with different missing rates
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from run import HiCoD_run


def run_missing_rate_experiments(dataset_name='mosi', seeds=[1111, 2222, 3333], missing_rates=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]):
    """
    Run experiments with different missing rates
    
    Args:
        dataset_name: Name of the dataset (mosi/mosei)
        seeds: List of random seeds for reproducibility
        missing_rates: List of missing rates to test
    """
    print(f"Running missing rate experiments for {dataset_name}")
    print(f"Missing rates: {missing_rates}")
    print(f"Seeds: {seeds}")
    
    all_results = {}
    
    for mr in missing_rates:
        print(f"\n{'='*50}")
        print(f"Testing missing rate: {mr}")
        print(f"{'='*50}")
        
        # Run experiment for this missing rate
        try:
            HiCoD_run(
                model_name='hicod',
                dataset_name=dataset_name,
                seeds=seeds,
                mr=mr
            )
            
            # Load results
            results_file = f"results/hicod_{dataset_name}/results.json"
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results = json.load(f)
                all_results[f"mr_{mr}"] = results
                print(f"Results saved for missing rate {mr}")
            else:
                print(f"Warning: Results file not found for missing rate {mr}")
                
        except Exception as e:
            print(f"Error in missing rate {mr}: {e}")
            continue
    
    # Save combined results
    combined_results_file = f"results/hicod_{dataset_name}/missing_rate_experiments.json"
    os.makedirs(os.path.dirname(combined_results_file), exist_ok=True)
    
    with open(combined_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAll missing rate experiments completed!")
    print(f"Combined results saved to: {combined_results_file}")
    
    # Print summary
    print(f"\nExperiment Summary:")
    print(f"Dataset: {dataset_name}")
    print(f"Missing rates tested: {missing_rates}")
    print(f"Seeds per experiment: {len(seeds)}")
    print(f"Total experiments: {len(missing_rates)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run HiCoD missing rate experiments')
    parser.add_argument('--dataset', type=str, default='mosi', help='Dataset name (mosi/mosei)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1111, 2222, 3333], help='Random seeds')
    parser.add_argument('--missing_rates', type=float, nargs='+', default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], help='Missing rates to test')
    
    args = parser.parse_args()
    
    run_missing_rate_experiments(
        dataset_name=args.dataset,
        seeds=args.seeds,
        missing_rates=args.missing_rates
    ) 