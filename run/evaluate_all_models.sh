#!/bin/bash

# Evaluate All Trained Models
# This script finds all model_best.pt checkpoints and evaluates them on the test set
# Usage: bash run/evaluate_all_models.sh [gpu_id] [output_dir]

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Add project root to Python path
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

GPU_ID=${1:-0}
OUTPUT_DIR=${2:-"./evaluation_results"}

# Validate GPU_ID is a number
if ! [[ "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "Error: GPU_ID must be a number, got: $GPU_ID"
    echo "Usage: bash run/evaluate_all_models.sh [gpu_id] [output_dir]"
    exit 1
fi

# Dataset paths
DATA_DIR="/medailab/medailab/shilab/FairCLIP"
CSV_PATH="/medailab/medailab/shilab/FairCLIP/data_summary.csv"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Set GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "=========================================="
echo "EVALUATING ALL TRAINED MODELS"
echo "=========================================="
echo "GPU ID: $GPU_ID"
echo "Output Directory: $OUTPUT_DIR"
echo "Data Dir: $DATA_DIR"
echo "CSV Path: $CSV_PATH"
echo "=========================================="
echo ""

# Find all model_best.pt files
CHECKPOINTS=$(find ./checkpoints -name "model_best.pt" -type f 2>/dev/null | sort)

if [ -z "$CHECKPOINTS" ]; then
    echo "Error: No model_best.pt files found in ./checkpoints directory"
    exit 1
fi

# Count checkpoints
NUM_CHECKPOINTS=$(echo "$CHECKPOINTS" | wc -l)
echo "Found $NUM_CHECKPOINTS checkpoint(s) to evaluate"
echo ""

# Counter for tracking
COUNT=0
SUCCESS_COUNT=0
FAIL_COUNT=0

# Loop through each checkpoint
while IFS= read -r checkpoint_path; do
    COUNT=$((COUNT + 1))
    
    # Extract experiment name from path
    # e.g., ./checkpoints/causal_base_fairclip/model_best.pt -> causal_base_fairclip
    experiment_name=$(dirname "$checkpoint_path" | sed 's|^\./checkpoints/||' | sed 's|^checkpoints/||')
    
    echo "=========================================="
    echo "[$COUNT/$NUM_CHECKPOINTS] Evaluating: $experiment_name"
    echo "Checkpoint: $checkpoint_path"
    echo "=========================================="
    
    # Run evaluation
    python evaluation/evaluate_checkpoint.py \
        --checkpoint_path "$checkpoint_path" \
        --data_dir "$DATA_DIR" \
        --csv_path "$CSV_PATH" \
        --gpu_id "$GPU_ID" \
        --output_dir "$OUTPUT_DIR" \
        --save_txt \
        --save_json \
        --model_type auto
    
    # Check if evaluation was successful
    if [ $? -eq 0 ]; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo "✓ Successfully evaluated: $experiment_name"
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "✗ Failed to evaluate: $experiment_name"
    fi
    
    echo ""
    
done <<< "$CHECKPOINTS"

# Summary
echo "=========================================="
echo "EVALUATION SUMMARY"
echo "=========================================="
echo "Total checkpoints: $NUM_CHECKPOINTS"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAIL_COUNT"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

# List all output files
echo ""
echo "Generated output files:"
find "$OUTPUT_DIR" -name "evaluation_results_*.txt" -o -name "evaluation_results_*.json" 2>/dev/null | sort

# Create summary CSV if we have JSON files
SUMMARY_CSV="$OUTPUT_DIR/summary_all_models.csv"
JSON_FILES=$(find "$OUTPUT_DIR" -name "evaluation_results_*.json" 2>/dev/null | sort)

if [ ! -z "$JSON_FILES" ]; then
    echo ""
    echo "Creating summary CSV: $SUMMARY_CSV"
    
    # Create CSV header
    echo "Model,Overall_AUC,Overall_Sensitivity,Overall_Specificity,Overall_Accuracy,Asian_AUC,Asian_Sensitivity,Asian_Specificity,Asian_N,Black_AUC,Black_Sensitivity,Black_Specificity,Black_N,White_AUC,White_Sensitivity,White_Specificity,White_N" > "$SUMMARY_CSV"
    
    # Extract data from each JSON file
    while IFS= read -r json_file; do
        # Extract model name from filename
        model_name=$(basename "$json_file" | sed 's/evaluation_results_//' | sed 's/_[0-9]\{8\}_[0-9]\{6\}\.json$//')
        
        # Use python to extract metrics from JSON
        python3 << EOF
import json
import sys

try:
    with open("$json_file", 'r') as f:
        data = json.load(f)
    
    metrics = data['metrics']
    overall = metrics['overall']
    groups = metrics['groups']
    
    # Extract group metrics with defaults
    asian = groups.get('Asian', {'auc': 0, 'sensitivity': 0, 'specificity': 0, 'n_samples': 0})
    black = groups.get('Black', {'auc': 0, 'sensitivity': 0, 'specificity': 0, 'n_samples': 0})
    white = groups.get('White', {'auc': 0, 'sensitivity': 0, 'specificity': 0, 'n_samples': 0})
    
    # Write CSV row
    row = [
        "$model_name",
        f"{overall['auc']:.4f}",
        f"{overall['sensitivity']:.4f}",
        f"{overall['specificity']:.4f}",
        f"{overall['accuracy']:.4f}",
        f"{asian['auc']:.4f}",
        f"{asian['sensitivity']:.4f}",
        f"{asian['specificity']:.4f}",
        f"{asian['n_samples']}",
        f"{black['auc']:.4f}",
        f"{black['sensitivity']:.4f}",
        f"{black['specificity']:.4f}",
        f"{black['n_samples']}",
        f"{white['auc']:.4f}",
        f"{white['sensitivity']:.4f}",
        f"{white['specificity']:.4f}",
        f"{white['n_samples']}"
    ]
    print(','.join(row))
except Exception as e:
    print(f"Error processing $json_file: {e}", file=sys.stderr)
    sys.exit(1)
EOF
    done <<< "$JSON_FILES" >> "$SUMMARY_CSV"
    
    echo "Summary CSV created: $SUMMARY_CSV"
fi

echo ""
echo "All evaluations completed!"

