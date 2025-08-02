#!/usr/bin/env python3
"""
Comprehensive experiment script for HiCoD with both fixed and random missing modality scenarios
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from run import HiCoD_run


def run_fixed_missing_experiments(dataset_name='mosi', seeds=[1111, 2222, 3333]):
    """
    Run experiments with fixed missing modalities
    
    Fixed missing scenarios:
    - Single modality missing: {L}, {A}, {V}
    - Two modalities missing: {L, A}, {L, V}, {A, V}
    """
    print("Running fixed missing modality experiments...")
    print("="*60)
    
    # Define fixed missing scenarios
    fixed_scenarios = [
        # Single modality missing
        {'name': 'missing_text', 'modalities': ['text']},
        {'name': 'missing_audio', 'modalities': ['audio']},
        {'name': 'missing_vision', 'modalities': ['vision']},
        # Two modalities missing
        {'name': 'missing_text_audio', 'modalities': ['text', 'audio']},
        {'name': 'missing_text_vision', 'modalities': ['text', 'vision']},
        {'name': 'missing_audio_vision', 'modalities': ['audio', 'vision']},
    ]
    
    all_results = {}
    
    for scenario in fixed_scenarios:
        print(f"\nTesting scenario: {scenario['name']}")
        print(f"Missing modalities: {scenario['modalities']}")
        print("-" * 40)
        
        try:
            # Run experiment for this scenario
            HiCoD_run(
                model_name='hicod',
                dataset_name=dataset_name,
                seeds=seeds,
                mr=0.0,  # Not used in fixed scenario
                missing_scenario='fixed',
                missing_modalities=scenario['modalities']
            )
            
            # Load results
            results_file = f"results/hicod_{dataset_name}/results.json"
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results = json.load(f)
                all_results[scenario['name']] = results
                print(f"Results saved for scenario: {scenario['name']}")
            else:
                print(f"Warning: Results file not found for scenario {scenario['name']}")
                
        except Exception as e:
            print(f"Error in scenario {scenario['name']}: {e}")
            continue
    
    # Save combined results
    combined_results_file = f"results/hicod_{dataset_name}/fixed_missing_experiments.json"
    os.makedirs(os.path.dirname(combined_results_file), exist_ok=True)
    
    with open(combined_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nFixed missing experiments completed!")
    print(f"Combined results saved to: {combined_results_file}")
    
    return all_results


def run_random_missing_experiments(dataset_name='mosi', seeds=[1111, 2222, 3333]):
    """
    Run experiments with random missing modalities
    
    Random missing rates: [0%, 10%, 20%, 30%, 40%, 50%, 60%, 70%]
    """
    print("Running random missing modality experiments...")
    print("="*60)
    
    # Define random missing rates (0% to 70% in 10% increments)
    missing_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    all_results = {}
    
    for mr in missing_rates:
        print(f"\nTesting missing rate: {mr*100:.0f}%")
        print("-" * 40)
        
        try:
            # Run experiment for this missing rate
            HiCoD_run(
                model_name='hicod',
                dataset_name=dataset_name,
                seeds=seeds,
                mr=mr,
                missing_scenario='random',
                missing_modalities=None
            )
            
            # Load results
            results_file = f"results/hicod_{dataset_name}/results.json"
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results = json.load(f)
                all_results[f"mr_{mr*100:.0f}"] = results
                print(f"Results saved for missing rate {mr*100:.0f}%")
            else:
                print(f"Warning: Results file not found for missing rate {mr*100:.0f}%")
                
        except Exception as e:
            print(f"Error in missing rate {mr*100:.0f}%: {e}")
            continue
    
    # Save combined results
    combined_results_file = f"results/hicod_{dataset_name}/random_missing_experiments.json"
    os.makedirs(os.path.dirname(combined_results_file), exist_ok=True)
    
    with open(combined_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nRandom missing experiments completed!")
    print(f"Combined results saved to: {combined_results_file}")
    
    return all_results


def run_comprehensive_experiments(dataset_name='mosi', seeds=[1111, 2222, 3333]):
    """
    Run comprehensive experiments including both fixed and random missing scenarios
    """
    print("Running comprehensive HiCoD experiments...")
    print("="*60)
    print(f"Dataset: {dataset_name}")
    print(f"Seeds: {seeds}")
    print("="*60)
    
    # Run fixed missing experiments
    print("\n1. Fixed Missing Modality Experiments")
    fixed_results = run_fixed_missing_experiments(dataset_name, seeds)
    
    # Run random missing experiments
    print("\n2. Random Missing Modality Experiments")
    random_results = run_random_missing_experiments(dataset_name, seeds)
    
    # Combine all results
    all_results = {
        'fixed_missing': fixed_results,
        'random_missing': random_results,
        'experiment_info': {
            'dataset': dataset_name,
            'seeds': seeds,
            'fixed_scenarios': [
                'missing_text', 'missing_audio', 'missing_vision',
                'missing_text_audio', 'missing_text_vision', 'missing_audio_vision'
            ],
            'random_rates': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        }
    }
    
    # Save comprehensive results
    comprehensive_results_file = f"results/hicod_{dataset_name}/comprehensive_experiments.json"
    os.makedirs(os.path.dirname(comprehensive_results_file), exist_ok=True)
    
    with open(comprehensive_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*60)
    print("Comprehensive experiments completed!")
    print(f"All results saved to: {comprehensive_results_file}")
    
    # Print summary
    print("\nExperiment Summary:")
    print(f"Dataset: {dataset_name}")
    print(f"Fixed missing scenarios: {len(fixed_results)}")
    print(f"Random missing rates: {len(random_results)}")
    print(f"Total experiments: {len(fixed_results) + len(random_results)}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run comprehensive HiCoD experiments')
    parser.add_argument('--dataset', type=str, default='mosi', help='Dataset name (mosi/mosei)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1111, 2222, 3333], help='Random seeds')
    parser.add_argument('--scenario', type=str, default='comprehensive', 
                       choices=['comprehensive', 'fixed', 'random'], 
                       help='Experiment scenario to run')
    
    args = parser.parse_args()
    
    if args.scenario == 'comprehensive':
        run_comprehensive_experiments(args.dataset, args.seeds)
    elif args.scenario == 'fixed':
        run_fixed_missing_experiments(args.dataset, args.seeds)
    elif args.scenario == 'random':
        run_random_missing_experiments(args.dataset, args.seeds) 