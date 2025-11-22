import numpy as np
import argparse
import csv
import os

import torch
from models import cls_model, cls_model_pointnet2
from utils import create_dir

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_cls_class', type=int, default=3, help='The number of classes')
    parser.add_argument('--num_points', type=int, default=10000, help='The number of points per object to be included in the input data')
    parser.add_argument('--k', type=int, default=32, help='Number of neighbors for PointNet++')

    parser.add_argument('--pointnet_checkpoint', type=str, default='best_model', help='PointNet checkpoint name')
    parser.add_argument('--pointnet2_checkpoint', type=str, default='best_model_pointnet2', help='PointNet++ checkpoint name')

    parser.add_argument('--test_data', type=str, default='./data/cls/data_test.npy')
    parser.add_argument('--test_label', type=str, default='./data/cls/label_test.npy')
    parser.add_argument('--output_dir', type=str, default='./output')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    create_dir(args.output_dir)

    print("="*70)
    print("Classification Model Comparison: PointNet vs PointNet++")
    print("="*70)

    # Initialize PointNet model
    print("\nLoading PointNet model...")
    model_pointnet = cls_model(num_classes=args.num_cls_class).to(args.device)
    pointnet_path = './checkpoints/cls/{}.pt'.format(args.pointnet_checkpoint)
    
    if not os.path.exists(pointnet_path):
        print(f"Error: PointNet checkpoint not found at {pointnet_path}")
        exit(1)
    
    with open(pointnet_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model_pointnet.load_state_dict(state_dict)
    model_pointnet.eval()
    print(f"Successfully loaded PointNet checkpoint from {pointnet_path}")

    # Initialize PointNet++ model
    print("\nLoading PointNet++ model...")
    model_pointnet2 = cls_model_pointnet2(num_classes=args.num_cls_class, k=args.k).to(args.device)
    pointnet2_path = './checkpoints/cls/{}.pt'.format(args.pointnet2_checkpoint)
    
    if not os.path.exists(pointnet2_path):
        print(f"Error: PointNet++ checkpoint not found at {pointnet2_path}")
        exit(1)
    
    with open(pointnet2_path, 'rb') as f:
        state_dict = torch.load(f, map_location=args.device)
        model_pointnet2.load_state_dict(state_dict)
    model_pointnet2.eval()
    print(f"Successfully loaded PointNet++ checkpoint from {pointnet2_path}")

    # Load test data with fixed seed for fair comparison
    np.random.seed(42)
    ind = np.random.choice(10000, args.num_points, replace=False)
    test_data = torch.from_numpy((np.load(args.test_data))[:,ind,:])
    test_label = torch.from_numpy(np.load(args.test_label))
    
    print(f"\nTest set size: {test_data.size(0)} objects")
    print(f"Points per object: {args.num_points}")
    print()

    # Evaluate PointNet
    print("Evaluating PointNet...")
    batch_size = 32
    pred_pointnet = []
    with torch.no_grad():
        for i in range(0, test_data.size(0), batch_size):
            batch_data = test_data[i:i+batch_size].to(args.device)
            batch_pred = model_pointnet(batch_data).argmax(dim=1)
            pred_pointnet.append(batch_pred.cpu())
    pred_pointnet = torch.cat(pred_pointnet, dim=0)
    accuracy_pointnet = pred_pointnet.eq(test_label.data).cpu().sum().item() / test_label.size()[0]

    # Evaluate PointNet++
    print("Evaluating PointNet++...")
    pred_pointnet2 = []
    with torch.no_grad():
        for i in range(0, test_data.size(0), batch_size):
            batch_data = test_data[i:i+batch_size].to(args.device)
            batch_pred = model_pointnet2(batch_data).argmax(dim=1)
            pred_pointnet2.append(batch_pred.cpu())
    pred_pointnet2 = torch.cat(pred_pointnet2, dim=0)
    accuracy_pointnet2 = pred_pointnet2.eq(test_label.data).cpu().sum().item() / test_label.size()[0]

    # Calculate improvement
    improvement = accuracy_pointnet2 - accuracy_pointnet
    improvement_pct = (improvement / accuracy_pointnet * 100) if accuracy_pointnet > 0 else 0

    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"{'Model':<20} {'Accuracy':<15} {'Improvement':<15}")
    print("-"*70)
    print(f"{'PointNet':<20} {accuracy_pointnet:.4f}        {'-':<15}")
    print(f"{'PointNet++':<20} {accuracy_pointnet2:.4f}        {improvement:+.4f} ({improvement_pct:+.2f}%)")
    print("="*70)

    # Save results to CSV
    csv_path = os.path.join(args.output_dir, 'comparison_cls.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'accuracy', 'improvement', 'improvement_pct'])
        writer.writeheader()
        writer.writerow({
            'model': 'PointNet',
            'accuracy': accuracy_pointnet,
            'improvement': 0.0,
            'improvement_pct': 0.0
        })
        writer.writerow({
            'model': 'PointNet++',
            'accuracy': accuracy_pointnet2,
            'improvement': improvement,
            'improvement_pct': improvement_pct
        })
    
    print(f"\nResults saved to: {csv_path}")

