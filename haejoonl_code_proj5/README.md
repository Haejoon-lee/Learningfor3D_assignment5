# Assignment 5: PointNet for Classification and Segmentation

**Author:** Haejoon Lee (andrewid: haejoonl)

## Overview

This assignment implements PointNet-based architectures for point cloud classification and segmentation tasks. The codebase includes:

- **Q1**: PointNet classification model for 3-class classification (chairs, vases, lamps)
- **Q2**: PointNet segmentation model for per-point semantic segmentation (6 classes)
- **Q3**: Robustness analysis experiments (rotation and number of points)
- **Q4**: Simplified PointNet++ implementation with locality

## File Structure

```
haejoonl_code_proj5/
├── main.py                 # Main entry point for all tasks
├── models.py              # Model architectures (PointNet, PointNet++)
├── train.py               # Training script for PointNet
├── train_pointnet2.py     # Training script for PointNet++
├── eval_cls.py            # Evaluation for classification
├── eval_seg.py            # Evaluation for segmentation
├── eval_cls_rotation.py   # Rotation robustness for classification
├── eval_seg_rotation.py   # Rotation robustness for segmentation
├── eval_cls_num_points.py # Number of points robustness for classification
├── eval_seg_num_points.py # Number of points robustness for segmentation
├── compare_cls.py         # Compare PointNet vs PointNet++ (classification)
├── compare_seg.py         # Compare PointNet vs PointNet++ (segmentation)
├── data_loader.py          # Data loading utilities
├── utils.py                # Utility functions (visualization, checkpoints)
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Environment Setup

### Using Apptainer (Recommended for Cluster)

```bash
# Use the provided Apptainer container
apptainer exec --nv ~/containers/pytorch3d_t4.sif python main.py --mode <mode> [options]
```

### Local Setup

```bash
pip install -r requirements.txt
```

## Data Preparation

Download the dataset from Hugging Face:

```bash
sudo apt install git-lfs
git lfs install
git clone https://huggingface.co/datasets/learning3dvision/assignment5
unzip ./assignment5/a5_data.zip -d ./
```

This creates a `data/` folder with `cls/` and `seg/` subdirectories containing training and test `.npy` files.

## Usage

### Main Entry Point

All tasks can be run through `main.py`:

```bash
python main.py --mode <mode> [additional arguments]
```

### Training

**Train PointNet Classification:**
```bash
python main.py --mode train --task cls --num_epochs 200 --batch_size 32
```

**Train PointNet Segmentation:**
```bash
python main.py --mode train --task seg --num_epochs 200 --batch_size 32
```

**Train PointNet++ (both tasks):**
```bash
python main.py --mode train_pointnet2 --task cls --num_epochs 200 --batch_size 16
python main.py --mode train_pointnet2 --task seg --num_epochs 200 --batch_size 2
```

### Evaluation

**Evaluate Classification:**
```bash
python main.py --mode eval_cls --load_checkpoint best_model --num_cls_class 3
```

**Evaluate Segmentation:**
```bash
python main.py --mode eval_seg --load_checkpoint best_model --i 0 --exp_name seg_eval
```

### Robustness Experiments

**Rotation Robustness:**
```bash
python main.py --mode eval_cls_rotation --load_checkpoint best_model
python main.py --mode eval_seg_rotation --load_checkpoint best_model
```

**Number of Points Robustness:**
```bash
python main.py --mode eval_cls_num_points --load_checkpoint best_model
python main.py --mode eval_seg_num_points --load_checkpoint best_model
```

### Comparison

**Compare PointNet vs PointNet++:**
```bash
python main.py --mode compare_cls
python main.py --mode compare_seg --i 0
```

### Direct Script Execution

Alternatively, you can run scripts directly:

```bash
# Training
python train.py --task cls
python train.py --task seg
python train_pointnet2.py --task cls
python train_pointnet2.py --task seg

# Evaluation
python eval_cls.py --load_checkpoint best_model
python eval_seg.py --load_checkpoint best_model --i 0

# Robustness
python eval_cls_rotation.py --load_checkpoint best_model
python eval_seg_rotation.py --load_checkpoint best_model
python eval_cls_num_points.py --load_checkpoint best_model
python eval_seg_num_points.py --load_checkpoint best_model

# Comparison
python compare_cls.py
python compare_seg.py --i 0
```

## Model Architectures

### PointNet Classification
- Shared MLP: 3→64→128→1024
- Global max pooling
- Classifier: 1024→512→256→3

### PointNet Segmentation
- Encoder: 3→64→128→1024
- Global feature concatenation
- Decoder: 1088→512→256→128→6

### PointNet++ Classification
- SA1: 512 centers, k=32 neighbors, 128-dim features
- SA2: 128 centers, k=32 neighbors, 512-dim features
- Global max pooling + MLP: 512→256→128→3

### PointNet++ Segmentation (Simplified)
- Per-point MLP: 3→64→64
- Local aggregation: k-NN (k=16) with relative coordinates
- Global feature concatenation: 256-dim
- Decoder: 256→256→128→6

**Note:** The segmentation PointNet++ is a simplified version that does not implement the full feature propagation architecture from the original paper.

## Results

### Classification
- **PointNet**: 98.22% test accuracy
- **PointNet++**: 98.64% test accuracy

### Segmentation
- **PointNet**: 90.45% test accuracy
- **PointNet++**: 88.44% test accuracy (simplified architecture)

### Robustness Analysis
- **Rotation**: Significant degradation with rotation (30°: -20% classification, -15% segmentation)
- **Number of Points**: Robust to point reduction (500 points: 96.96% classification, 88.55% segmentation)

## Output Files

- **Checkpoints**: Saved in `checkpoints/{task}/`
- **Logs**: TensorBoard logs in `logs/`
- **Results**: CSV files in `output/`
- **Visualizations**: GIF files in `output/{exp_name}/`, `output/rotation_viz/`, `output/num_points_viz/`, `output/comparison_seg/`

## Key Arguments

### Training
- `--task`: `cls` or `seg`
- `--num_epochs`: Number of training epochs (default: 200)
- `--batch_size`: Batch size (default: 32, use 2 for PointNet++ segmentation)
- `--lr`: Learning rate (default: 0.001)
- `--num_points`: Number of points per object (default: 10000)

### Evaluation
- `--load_checkpoint`: Checkpoint name (e.g., `best_model`)
- `--checkpoint_dir`: Directory containing checkpoints (default: `./checkpoints/{task}/`)
- `--i`: Object index for visualization (segmentation only)
- `--exp_name`: Experiment name for output directory

## Notes

- PointNet++ segmentation requires smaller batch sizes (batch_size=2) due to memory constraints from k-NN computation at full resolution
- All models use CUDA if available, otherwise CPU
- Checkpoints include model state, optimizer state, epoch, and best accuracy for resuming training

## Dependencies

See `requirements.txt` for full list. Key dependencies:
- PyTorch
- NumPy
- PyTorch3D
- TensorBoard
- scipy (for rotation experiments)
- matplotlib (for visualization)



