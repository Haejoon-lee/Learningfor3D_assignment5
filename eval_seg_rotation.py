import numpy as np
import argparse
import csv
import os

import torch
from models import seg_model
from utils import create_dir, rotate_point_cloud, viz_seg

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_seg_class', type=int, default=6, help='The number of segmentation classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--test_data', type=str, default='./data/seg/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/seg/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')
    
    parser.add_argument('--visualize', action='store_true', help='Save visualization GIFs for sample object')
    parser.add_argument('--viz_idx', type=int, default=0, help='Index of object to visualize')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # Initialize Model for Segmentation Task
    model = seg_model(num_seg_classes=args.num_seg_class).to(args.device) 
    
    # Load Model Checkpoint
    model_path = './checkpoints/seg/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print("Successfully loaded checkpoint from {}".format(model_path))

    # Load test data
    ind = np.random.choice(10000, args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label = torch.from_numpy((np.load(args.test_label))[:,ind])
    
    print("\n" + "="*60)
    print("Rotation Robustness Test - Segmentation")
    print("="*60)
    print(f"Test set size: {test_data.size(0)} objects")
    print(f"Points per object: {args.num_points}")
    if args.visualize:
        print(f"Visualizing object index: {args.viz_idx}")
    print()

    # Test at different rotation angles
    rotation_angles = [0, 30, 60, 90, 180]
    results = []
    baseline_accuracy = None

    for angle in rotation_angles:
        # Apply rotation if angle > 0
        if angle == 0:
            rotated_data = test_data.clone()
            seed = None  # No rotation, no seed needed
        else:
            # Use fixed seed for reproducibility (same rotation for all objects at same angle)
            seed = 42 + angle  # Different seed for each angle
            rotated_data = []
            for i in range(test_data.size(0)):
                rotated_pc = rotate_point_cloud(test_data[i], angle, seed=seed + i)
                rotated_data.append(rotated_pc)
            rotated_data = torch.stack(rotated_data, dim=0)

        # Make predictions
        batch_size = 32
        pred_label = []
        with torch.no_grad():
            for i in range(0, rotated_data.size(0), batch_size):
                batch_data = rotated_data[i:i+batch_size].to(args.device)
                # Model outputs (B, N, num_seg_classes), get argmax to get class predictions
                batch_pred = model(batch_data).argmax(dim=2)  # (B, N)
                pred_label.append(batch_pred.cpu())
        pred_label = torch.cat(pred_label, dim=0)  # (num_objects, num_points)

        # Compute accuracy
        test_accuracy = pred_label.eq(test_label.data).cpu().sum().item() / (test_label.reshape((-1,1)).size()[0])
        
        # Calculate accuracy drop vs baseline
        if angle == 0:
            baseline_accuracy = test_accuracy
            accuracy_drop = 0.0
        else:
            accuracy_drop = baseline_accuracy - test_accuracy

        results.append({
            'angle': angle,
            'accuracy': test_accuracy,
            'accuracy_drop': accuracy_drop
        })

        print(f"Rotation: {angle:3d}° | Accuracy: {test_accuracy:.4f} | Drop: {accuracy_drop:.4f}")

        # Save visualization if requested
        if args.visualize:
            viz_dir = os.path.join(args.output_dir, 'rotation_viz')
            create_dir(viz_dir)
            
            # Visualize ground truth (only once at angle 0)
            if angle == 0:
                viz_seg(
                    rotated_data[args.viz_idx], 
                    test_label[args.viz_idx], 
                    os.path.join(viz_dir, f"gt_angle_{angle}.gif"), 
                    args.device
                )
            
            # Visualize prediction
            viz_seg(
                rotated_data[args.viz_idx], 
                pred_label[args.viz_idx], 
                os.path.join(viz_dir, f"pred_angle_{angle}.gif"), 
                args.device
            )

    # Save results to CSV
    csv_path = os.path.join(args.output_dir, 'rotation_results_seg.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['angle', 'accuracy', 'accuracy_drop'])
        writer.writeheader()
        writer.writerows(results)
    
    print("\n" + "="*60)
    print(f"Results saved to: {csv_path}")
    if args.visualize:
        print(f"Visualizations saved to: {os.path.join(args.output_dir, 'rotation_viz')}")
    print("="*60)
    print("\nSummary:")
    print(f"Baseline accuracy (0°): {baseline_accuracy:.4f}")
    print(f"Accuracy at 30°:  {results[1]['accuracy']:.4f} (drop: {results[1]['accuracy_drop']:.4f})")
    print(f"Accuracy at 60°:  {results[2]['accuracy']:.4f} (drop: {results[2]['accuracy_drop']:.4f})")
    print(f"Accuracy at 90°:  {results[3]['accuracy']:.4f} (drop: {results[3]['accuracy_drop']:.4f})")
    print(f"Accuracy at 180°: {results[4]['accuracy']:.4f} (drop: {results[4]['accuracy_drop']:.4f})")

