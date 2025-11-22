import numpy as np
import argparse
import csv
import os

import torch
from models import cls_model
from utils import create_dir

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--load_checkpoint', type=str, default='best_model')
    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    # Initialize Model for Classification Task
    model = cls_model(num_classes=args.num_cls_class).to(args.device) 
    
    # Load Model Checkpoint
    model_path = './checkpoints/cls/{}.pt'.format(args.load_checkpoint)
    with open(model_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model.load_state_dict(state_dict)
    model.eval()
    print("Successfully loaded checkpoint from {}".format(model_path))

    # Load full test data
    full_test_data = np.load(args.test_data)  # (num_objects, 10000, 3)
    full_test_label = np.load(args.test_label)  # (num_objects,)
    
    print("\n" + "="*60)
    print("Number of Points Robustness Test - Classification")
    print("="*60)
    print(f"Test set size: {full_test_data.shape[0]} objects")
    print(f"Full points per object: 10000")
    print()

    # Test at different point counts
    point_counts = [10000, 5000, 2000, 1000, 500]
    results = []
    baseline_accuracy = None

    # Use fixed seed for reproducibility and nested subsets
    np.random.seed(42)
    # Sample 10000 indices once
    indices_10000 = np.random.choice(10000, 10000, replace=False)

    for num_points in point_counts:
        # Get nested subset indices
        if num_points == 10000:
            indices = indices_10000
        else:
            # Take first N indices from the 10000 (nested subset)
            indices = indices_10000[:num_points]

        # Sample points using nested indices
        test_data = torch.from_numpy(full_test_data[:, indices, :])
        test_label = torch.from_numpy(full_test_label)

        # Make predictions
        batch_size = 32
        pred_label = []
        with torch.no_grad():
            for i in range(0, test_data.size(0), batch_size):
                batch_data = test_data[i:i+batch_size].to(args.device)
                batch_pred = model(batch_data).argmax(dim=1)
                pred_label.append(batch_pred.cpu())
        pred_label = torch.cat(pred_label, dim=0)

        # Compute accuracy
        accuracy = pred_label.eq(test_label.data).cpu().sum().item() / test_label.size()[0]
        
        # Calculate accuracy drop vs baseline
        if num_points == 10000:
            baseline_accuracy = accuracy
            accuracy_drop = 0.0
        else:
            accuracy_drop = baseline_accuracy - accuracy

        results.append({
            'num_points': num_points,
            'accuracy': accuracy,
            'accuracy_drop': accuracy_drop
        })

        print(f"Points: {num_points:5d} | Accuracy: {accuracy:.4f} | Drop: {accuracy_drop:.4f}")

    # Save results to CSV
    csv_path = os.path.join(args.output_dir, 'num_points_results_cls.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['num_points', 'accuracy', 'accuracy_drop'])
        writer.writeheader()
        writer.writerows(results)
    
    print("\n" + "="*60)
    print(f"Results saved to: {csv_path}")
    print("="*60)
    print("\nSummary:")
    print(f"Baseline accuracy (10000 points): {baseline_accuracy:.4f}")
    print(f"Accuracy at 5000 points: {results[1]['accuracy']:.4f} (drop: {results[1]['accuracy_drop']:.4f})")
    print(f"Accuracy at 2000 points: {results[2]['accuracy']:.4f} (drop: {results[2]['accuracy_drop']:.4f})")
    print(f"Accuracy at 1000 points: {results[3]['accuracy']:.4f} (drop: {results[3]['accuracy_drop']:.4f})")
    print(f"Accuracy at 500 points:  {results[4]['accuracy']:.4f} (drop: {results[4]['accuracy_drop']:.4f})")

