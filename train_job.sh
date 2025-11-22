#!/bin/bash
#SBATCH -J pointnet2_resume
#SBATCH -o slurm_logs/train_job_%j.out
#SBATCH -e slurm_logs/train_job_%j.err
#SBATCH -t 3-23:00:00
#SBATCH --nodes 1
#SBATCH --cpus-per-task 1
#SBATCH --ntasks 1
#SBATCH --mem=64G
#SBATCH --gres=gpu:H100:1

# Change to the working directory
cd ~/Learningfor3D_assignment5/Learningfor3D_assignment5

# Create log directory if it doesn't exist
mkdir -p slurm_logs

# Set Python to unbuffered mode for real-time output
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "Starting PointNet++ Training Resume"
echo "Date: $(date)"
echo "Node: $SLURM_NODELIST"
echo "=========================================="


# #=============== Train Classification Model
# echo ""
# echo "=========================================="
# echo "Training PointNet++ Classification Model"
# echo "=========================================="
# echo "Start time: $(date)"

# apptainer exec --nv ~/containers/pytorch3d_t4.sif \
#   python -u ~/Learningfor3D_assignment5/Learningfor3D_assignment5/train_pointnet2.py \
#   --task cls \
#   --exp_name pointnet2_cls \
#   --load_checkpoint best_model_pointnet2 \
#   --num_epochs 250

# echo "Classification training end time: $(date)"
# echo ""

# =============Train Segmentation Model
echo "=========================================="
echo "Training PointNet++ Segmentation Model"
echo "=========================================="
echo "Start time: $(date)"

apptainer exec --nv ~/containers/pytorch3d_t4.sif \
  python -u ~/Learningfor3D_assignment5/Learningfor3D_assignment5/train_pointnet2.py \
  --task seg \
  --exp_name pointnet2_seg \
  --load_checkpoint best_model_pointnet2 \
  --batch_size 2 \
  --num_epochs 250

# echo "Segmentation training end time: $(date)"
# echo ""

echo "=========================================="
echo "All training completed!"
echo "End time: $(date)"
echo "=========================================="

