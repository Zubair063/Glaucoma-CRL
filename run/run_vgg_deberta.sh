#!/bin/bash

# Causal Multimodal Learning Training Command (VGG + DeBERTa)
# Usage: bash run/run_vgg_deberta.sh [experiment_name] [gpu_id]

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Add project root to Python path
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Set experiment name (default or from argument)
EXPERIMENT_NAME=${1:-"vgg_deberta"}

# Set GPU ID (default or from argument)
GPU_ID=${2:-1}

# Dataset paths
DATA_DIR="/medailab/medailab/shilab/FairCLIP"
CSV_PATH="/medailab/medailab/shilab/FairCLIP/data_summary.csv"

# Training hyperparameters
BATCH_SIZE=16
MAX_EPOCHS=50
LR=2e-5
WEIGHT_DECAY=1e-4
LAMBDA_V=1.0
LAMBDA_FE=1.0
PATIENCE=15
WARMUP_EPOCHS=2
GRAD_CLIP=1.0
LABEL_SMOOTHING=0.1
SEED=42

# Model hyperparameters
VGG_VARIANT="vgg16"  # Options: vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
DEBERTA_MODEL="microsoft/deberta-base"
HIDDEN_SZ=768  # DeBERTa output dimension
IMG_HIDDEN_SZ=4096  # VGG output dimension

echo "=========================================="
echo "Causal Multimodal Learning Training"
echo "Backbone: VGG + DeBERTa"
echo "=========================================="
echo "Experiment: $EXPERIMENT_NAME"
echo "GPU ID: $GPU_ID"
echo "Data Dir: $DATA_DIR"
echo "Batch Size: $BATCH_SIZE"
echo "Max Epochs: $MAX_EPOCHS"
echo "Learning Rate: $LR"
echo "Weight Decay: $WEIGHT_DECAY"
echo "Warmup Epochs: $WARMUP_EPOCHS"
echo "Gradient Clip: $GRAD_CLIP"
echo "Label Smoothing: $LABEL_SMOOTHING"
echo "VGG Variant: $VGG_VARIANT"
echo "DeBERTa Model: $DEBERTA_MODEL"
echo "Lambda V: $LAMBDA_V"
echo "Lambda FE: $LAMBDA_FE"
echo "Patience: $PATIENCE"
echo "=========================================="
echo ""

# Set GPU visibility
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Run training
python train/train_vgg_deberta.py \
    --data_dir "$DATA_DIR" \
    --csv_path "$CSV_PATH" \
    --batch_size $BATCH_SIZE \
    --max_epochs $MAX_EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --lambda_v $LAMBDA_V \
    --lambda_fe $LAMBDA_FE \
    --patience $PATIENCE \
    --warmup_epochs $WARMUP_EPOCHS \
    --grad_clip $GRAD_CLIP \
    --label_smoothing $LABEL_SMOOTHING \
    --vgg_variant $VGG_VARIANT \
    --deberta_model "$DEBERTA_MODEL" \
    --hidden_sz $HIDDEN_SZ \
    --img_hidden_sz $IMG_HIDDEN_SZ \
    --seed $SEED \
    --gpu_id $GPU_ID \
    --name "$EXPERIMENT_NAME"

echo ""
echo "Training completed! Checkpoints saved in: ./checkpoints/$EXPERIMENT_NAME/"

