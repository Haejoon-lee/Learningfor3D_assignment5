#!/usr/bin/env python3
"""
Main entry point for Assignment 5: PointNet for Classification and Segmentation

This script provides a unified interface to run all training and evaluation tasks.
"""

import argparse
import subprocess
import sys
import os

def run_command(cmd):
    """Run a command and handle errors."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Error: Command failed with return code {result.returncode}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Assignment 5: PointNet Main Entry Point')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'eval_cls', 'eval_seg', 'eval_cls_rotation', 
                                'eval_seg_rotation', 'eval_cls_num_points', 'eval_seg_num_points',
                                'train_pointnet2', 'compare_cls', 'compare_seg'],
                        help='Mode to run: train, eval_cls, eval_seg, etc.')
    
    # Parse known args to get mode, pass rest to subcommands
    args, remaining = parser.parse_known_args()
    
    if args.mode == 'train':
        # Training PointNet models
        from train import main as train_main
        train_main()
    
    elif args.mode == 'eval_cls':
        # Evaluate classification model
        from eval_cls import main as eval_cls_main
        eval_cls_main()
    
    elif args.mode == 'eval_seg':
        # Evaluate segmentation model
        from eval_seg import main as eval_seg_main
        eval_seg_main()
    
    elif args.mode == 'eval_cls_rotation':
        # Evaluate classification with rotation
        from eval_cls_rotation import main as eval_cls_rotation_main
        eval_cls_rotation_main()
    
    elif args.mode == 'eval_seg_rotation':
        # Evaluate segmentation with rotation
        from eval_seg_rotation import main as eval_seg_rotation_main
        eval_seg_rotation_main()
    
    elif args.mode == 'eval_cls_num_points':
        # Evaluate classification with different number of points
        from eval_cls_num_points import main as eval_cls_num_points_main
        eval_cls_num_points_main()
    
    elif args.mode == 'eval_seg_num_points':
        # Evaluate segmentation with different number of points
        from eval_seg_num_points import main as eval_seg_num_points_main
        eval_seg_num_points_main()
    
    elif args.mode == 'train_pointnet2':
        # Training PointNet++ models
        from train_pointnet2 import main as train_pointnet2_main
        train_pointnet2_main()
    
    elif args.mode == 'compare_cls':
        # Compare PointNet vs PointNet++ for classification
        from compare_cls import main as compare_cls_main
        compare_cls_main()
    
    elif args.mode == 'compare_seg':
        # Compare PointNet vs PointNet++ for segmentation
        from compare_seg import main as compare_seg_main
        compare_seg_main()
    
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)

if __name__ == '__main__':
    main()



