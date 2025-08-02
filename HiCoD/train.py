#!/usr/bin/env python3
"""
Training script for HiCoD (Hierarchical Consistency-Guided Prompt Distillation)
"""

from run import HiCoD_run

if __name__ == "__main__":
    # Run HiCoD training with default parameters
HiCoD_run(
    model_name='hicod',
        dataset_name='mosi',
        seeds=[1111, 2222, 3333, 4444, 5555],
        mr=0.1
    )
