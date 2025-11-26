#!/usr/bin/env python3
"""
Evaluate saved checkpoint on test dataset
Computes Overall AUC, Sensitivity, Specificity, and group-specific AUCs
Supports all model architectures: ViT+BERT, EfficientNet+DistilBERT, ResNet+RoBERTa, VGG+DeBERTa, Base pattern

Usage:
    python evaluation/evaluate_checkpoint.py --checkpoint_path ./checkpoints/causal_base_fairclip/model_best.pt
    
    # With custom GPU
    python evaluation/evaluate_checkpoint.py --checkpoint_path ./checkpoints/causal_base_fairclip/model_best.pt --gpu_id 1
    
    # With custom data paths
    python evaluation/evaluate_checkpoint.py \
        --checkpoint_path ./checkpoints/causal_base_fairclip/model_best.pt \
        --data_dir /path/to/FairCLIP \
        --csv_path /path/to/data_summary.csv
"""

import argparse
import os
import sys
import warnings

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*TensorFlow.*')
warnings.filterwarnings('ignore', message='.*oneDNN.*')
warnings.filterwarnings('ignore', message='.*Protobuf.*')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json
from datetime import datetime
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
from tqdm import tqdm

from data.fairclip_dataset import get_fairclip_loaders
# Import all model types
from models.causal_multimodal_vit_bert import CausalMultimodalModelViTBERT
from models.causal_multimodal_base import CausalMultimodalModelBase
from models.causal_multimodal_efficientnet_distilbert import CausalMultimodalModelEfficientNetDistilBERT
from models.causal_multimodal_resnet_roberta import CausalMultimodalModelResNetRoBERTa
from models.causal_multimodal_vgg_deberta import CausalMultimodalModelVGGDeBERTa
from utils.utils import set_seed, load_checkpoint


def detect_model_type(checkpoint_path):
    """Detect model type from checkpoint keys"""
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    keys = list(state_dict.keys())
    
    # Convert to lowercase for case-insensitive matching
    keys_lower = [k.lower() for k in keys]
    
    # Check for specific architectures
    has_efficientnet = any('efficientnet' in k for k in keys_lower)
    has_distilbert = any('distilbert' in k for k in keys_lower)
    has_resnet = any('resnet' in k for k in keys_lower) and not has_efficientnet
    has_roberta = any('roberta' in k for k in keys_lower)
    has_vgg = any('vgg' in k for k in keys_lower)
    has_deberta = any('deberta' in k for k in keys_lower)
    has_vit = any('vit' in k for k in keys_lower) and not has_efficientnet
    has_bert = any('bert' in k for k in keys_lower) and not has_distilbert and not has_roberta and not has_deberta
    
    # Check for base pattern vs original pattern
    has_attention_scale = any('attention_scale' in k for k in keys)
    has_feature_extractor = any('feature_extractor' in k for k in keys)
    has_cross_attn = any('cross_attn' in k for k in keys)
    has_fusion_mlp = any('fusion_mlp' in k for k in keys)
    has_calibrator = any('calibrator' in k for k in keys)
    
    # Determine model type
    if has_efficientnet and has_distilbert:
        return 'efficientnet_distilbert', {
            'hidden_sz': 768,  # DistilBERT
            'img_hidden_sz': 1280,  # EfficientNet-B0
            'num_image_embeds': 1
        }
    elif has_resnet and has_roberta:
        return 'resnet_roberta', {
            'hidden_sz': 768,  # RoBERTa
            'img_hidden_sz': 2048,  # ResNet50
            'num_image_embeds': 1
        }
    elif has_vgg and has_deberta:
        # Try to detect img_hidden_sz from checkpoint's img_proj.weight shape
        img_hidden_sz = 4096  # Default for VGG
        if 'img_proj.weight' in state_dict:
            # img_proj.weight shape is [hidden_sz, img_hidden_sz * num_image_embeds]
            # For VGG: [768, 4096] means img_hidden_sz = 4096 (with num_image_embeds=1)
            img_proj_shape = state_dict['img_proj.weight'].shape
            if len(img_proj_shape) == 2:
                detected_img_hidden_sz = img_proj_shape[1]  # Second dimension
                img_hidden_sz = detected_img_hidden_sz
        return 'vgg_deberta', {
            'hidden_sz': 768,  # DeBERTa
            'img_hidden_sz': img_hidden_sz,  # VGG outputs 4096 features (or detected from checkpoint)
            'num_image_embeds': 1
        }
    elif has_attention_scale and has_feature_extractor and not has_cross_attn:
        return 'base', {
            'hidden_sz': 768,  # BERT/ViT
            'img_hidden_sz': 768,  # ViT
            'num_image_embeds': 1
        }
    elif has_cross_attn or has_fusion_mlp or has_calibrator:
        return 'original', {
            'hidden_sz': 768,  # BERT
            'img_hidden_sz': 768,  # ViT
            'num_image_embeds': 1
        }
    else:
        # Default to original if can't determine
        return 'original', {
            'hidden_sz': 768,
            'img_hidden_sz': 768,
            'num_image_embeds': 1
        }


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on test set")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="/medailab/medailab/shilab/FairCLIP",
                        help="Directory containing FairCLIP dataset")
    parser.add_argument("--csv_path", type=str, default="/medailab/medailab/shilab/FairCLIP/data_summary.csv",
                        help="Path to data_summary.csv")
    
    # Model arguments
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to checkpoint file")
    parser.add_argument("--model_type", type=str, default="auto",
                        choices=["auto", "original", "base", "efficientnet_distilbert", "resnet_roberta", "vgg_deberta"],
                        help="Model type (auto-detect if not specified)")
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased",
                        help="BERT model name (for original/base models)")
    parser.add_argument("--hidden_sz", type=int, default=768,
                        help="Hidden size (will be auto-detected if model_type=auto)")
    parser.add_argument("--img_hidden_sz", type=int, default=768,
                        help="Image encoder hidden size (will be auto-detected if model_type=auto)")
    parser.add_argument("--num_image_embeds", type=int, default=1,
                        help="Number of image embeddings")
    parser.add_argument("--n_classes", type=int, default=2,
                        help="Number of classes")
    
    # Evaluation arguments
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for evaluation")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="Maximum sequence length for text")
    parser.add_argument("--n_workers", type=int, default=4,
                        help="Number of data loader workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID to use")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save evaluation results (default: same as checkpoint directory)")
    parser.add_argument("--save_json", action="store_true",
                        help="Save results as JSON file")
    parser.add_argument("--save_txt", action="store_true", default=True,
                        help="Save results as text file (default: True)")
    
    return parser.parse_args()


def create_model(model_type, args):
    """Create the appropriate model instance based on type"""
    if model_type == 'efficientnet_distilbert':
        from models.causal_multimodal_efficientnet_distilbert import CausalMultimodalModelEfficientNetDistilBERT
        return CausalMultimodalModelEfficientNetDistilBERT(args)
    elif model_type == 'resnet_roberta':
        from models.causal_multimodal_resnet_roberta import CausalMultimodalModelResNetRoBERTa
        return CausalMultimodalModelResNetRoBERTa(args)
    elif model_type == 'vgg_deberta':
        from models.causal_multimodal_vgg_deberta import CausalMultimodalModelVGGDeBERTa
        return CausalMultimodalModelVGGDeBERTa(args)
    elif model_type == 'base':
        from models.causal_multimodal_base import CausalMultimodalModelBase
        return CausalMultimodalModelBase(args)
    elif model_type == 'vit_bert':
        from models.causal_multimodal_vit_bert import CausalMultimodalModelViTBERT
        return CausalMultimodalModelViTBERT(args)
    else:  # default to vit_bert
        from models.causal_multimodal_vit_bert import CausalMultimodalModelViTBERT
        return CausalMultimodalModelViTBERT(args)


def evaluate_model(model, test_loader, csv_path, args):
    """Evaluate model on test set and compute metrics by group"""
    model.eval()
    
    all_logits = []
    all_labels = []
    all_filenames = []
    
    print("Running inference on test set...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            img = batch['image'].cuda()
            txt = batch['txt'].cuda()
            mask = batch['mask'].cuda()
            segment = batch['segment'].cuda()
            labels = batch['label'].cuda()
            
            # Get filenames from batch
            filenames = batch.get('filename', [])
            if len(filenames) == 0:
                # Fallback: get from dataset indices
                start_idx = batch_idx * args.batch_size
                end_idx = min(start_idx + len(labels), len(test_loader.dataset))
                filenames = [test_loader.dataset.df.iloc[i]['filename'] for i in range(start_idx, end_idx)]
            
            # Forward pass (no labels for inference)
            fusion_logits, txt_logits, img_logits = model(txt, mask, segment, img, None)
            
            all_logits.append(fusion_logits.cpu())
            all_labels.append(labels.cpu())
            all_filenames.extend(filenames)
    
    # Concatenate all predictions
    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # Get probabilities and predictions
    all_probs = torch.softmax(torch.from_numpy(all_logits), dim=1)[:, 1].numpy()
    all_preds = all_logits.argmax(axis=1)
    
    # Load CSV to get demographic information
    df = pd.read_csv(csv_path)
    df_test = df[df['use'] == 'test'].reset_index(drop=True)
    
    # Ensure we have the same number of filenames as predictions
    if len(all_filenames) != len(all_probs):
        # Get filenames from test dataset directly
        test_dataset = test_loader.dataset
        all_filenames = [test_dataset.df.iloc[i]['filename'] for i in range(len(test_dataset))]
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'filename': all_filenames[:len(all_probs)],
        'label': all_labels[:len(all_probs)],
        'pred': all_preds[:len(all_probs)],
        'prob': all_probs[:len(all_probs)]
    })
    
    # Merge with CSV to get demographic info
    results_df = results_df.merge(df_test[['filename', 'race', 'ethnicity']], on='filename', how='left')
    
    # Compute overall metrics
    print("\n" + "="*60)
    print("OVERALL METRICS")
    print("="*60)
    
    # Overall AUC
    overall_auc = roc_auc_score(all_labels, all_probs)
    print(f"Overall AUC: {overall_auc:.4f}")
    
    # Confusion matrix for Sensitivity and Specificity
    cm = confusion_matrix(all_labels, all_preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        accuracy = accuracy_score(all_labels, all_preds)
        
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Sensitivity (Recall): {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:  {tn}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
        print(f"  True Positives:  {tp}")
    else:
        print("Warning: Could not compute Sensitivity/Specificity (binary classification expected)")
        sensitivity = 0.0
        specificity = 0.0
        accuracy = 0.0
    
    # Group-specific metrics
    print("\n" + "="*60)
    print("GROUP-SPECIFIC METRICS")
    print("="*60)
    
    # Define race groups (case-insensitive matching)
    race_groups = {
        'Asian': ['asian', 'asia'],
        'Black': ['black', 'african', 'african american'],
        'White': ['white', 'caucasian']
    }
    
    group_metrics = {}
    
    for group_name, keywords in race_groups.items():
        # Filter by race (case-insensitive)
        mask = results_df['race'].notna()
        group_mask = mask & results_df['race'].str.lower().str.contains('|'.join(keywords), case=False, na=False, regex=True)
        
        if group_mask.sum() == 0:
            print(f"\n{group_name} group: No samples found")
            group_metrics[group_name] = {
                'auc': 0.0,
                'sensitivity': 0.0,
                'specificity': 0.0,
                'n_samples': 0
            }
            continue
        
        group_labels = results_df[group_mask]['label'].values
        group_probs = results_df[group_mask]['prob'].values
        group_preds = results_df[group_mask]['pred'].values
        
        n_samples = len(group_labels)
        
        # Compute AUC
        if len(np.unique(group_labels)) > 1:
            group_auc = roc_auc_score(group_labels, group_probs)
        else:
            group_auc = 0.0
            print(f"Warning: {group_name} group has only one class, AUC set to 0.0")
        
        # Compute Sensitivity and Specificity
        group_cm = confusion_matrix(group_labels, group_preds)
        if group_cm.shape == (2, 2):
            tn, fp, fn, tp = group_cm.ravel()
            group_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            group_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            group_sensitivity = 0.0
            group_specificity = 0.0
        
        print(f"\n{group_name} group (n={n_samples}):")
        print(f"  AUC: {group_auc:.4f}")
        print(f"  Sensitivity: {group_sensitivity:.4f}")
        print(f"  Specificity: {group_specificity:.4f}")
        
        group_metrics[group_name] = {
            'auc': group_auc,
            'sensitivity': group_sensitivity,
            'specificity': group_specificity,
            'n_samples': n_samples
        }
    
    # Summary
    summary_lines = []
    summary_lines.append("\n" + "="*60)
    summary_lines.append("SUMMARY")
    summary_lines.append("="*60)
    summary_lines.append(f"Overall AUC: {overall_auc:.4f}")
    summary_lines.append(f"Overall Sensitivity: {sensitivity:.4f}")
    summary_lines.append(f"Overall Specificity: {specificity:.4f}")
    summary_lines.append(f"\nGroup-specific AUCs:")
    for group_name, metrics in group_metrics.items():
        if metrics['n_samples'] > 0:
            summary_lines.append(f"  {group_name}: {metrics['auc']:.4f} (n={metrics['n_samples']})")
    
    # Print summary
    for line in summary_lines:
        print(line)
    
    results = {
        'overall_auc': float(overall_auc),
        'overall_sensitivity': float(sensitivity),
        'overall_specificity': float(specificity),
        'overall_accuracy': float(accuracy),
        'group_metrics': {
            group_name: {
                'auc': float(metrics['auc']),
                'sensitivity': float(metrics['sensitivity']),
                'specificity': float(metrics['specificity']),
                'n_samples': int(metrics['n_samples'])
            }
            for group_name, metrics in group_metrics.items()
        },
        'summary_text': '\n'.join(summary_lines)
    }
    
    return results


def main():
    args = get_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    
    # Set seed
    set_seed(args.seed)
    
    print("="*60)
    print("CHECKPOINT EVALUATION")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Data Dir: {args.data_dir}")
    print(f"CSV Path: {args.csv_path}")
    print(f"GPU ID: {args.gpu_id}")
    print("="*60)
    
    # Detect or use specified model type
    if args.model_type == "auto":
        print("\nDetecting model type from checkpoint...")
        model_type, model_config = detect_model_type(args.checkpoint_path)
        print(f"Detected model type: {model_type}")
        
        # Update args with detected configuration
        args.hidden_sz = model_config['hidden_sz']
        args.img_hidden_sz = model_config['img_hidden_sz']
        args.num_image_embeds = model_config['num_image_embeds']
    else:
        model_type = args.model_type
        print(f"\nUsing specified model type: {model_type}")
    
    # Load data
    print("\nLoading test data...")
    _, _, test_loader = get_fairclip_loaders(
        args.data_dir,
        args.csv_path,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        max_seq_len=args.max_seq_len
    )
    
    if len(test_loader.dataset) == 0:
        print("Error: Test set is empty!")
        return
    
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print(f"\nCreating {model_type} model...")
    print(f"  hidden_sz: {args.hidden_sz}")
    print(f"  img_hidden_sz: {args.img_hidden_sz}")
    print(f"  num_image_embeds: {args.num_image_embeds}")
    
    model = create_model(model_type, args)
    model.cuda()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint_path}...")
    if not os.path.exists(args.checkpoint_path):
        print(f"Error: Checkpoint not found at {args.checkpoint_path}")
        return
    
    try:
        load_checkpoint(model, args.checkpoint_path)
        print("Checkpoint loaded successfully!")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("\nTrying to load with strict=False...")
        checkpoint = torch.load(args.checkpoint_path, weights_only=False, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        model.load_state_dict(state_dict, strict=False)
        print("Checkpoint loaded with strict=False (some weights may not match)")
    
    # Evaluate
    metrics = evaluate_model(model, test_loader, args.csv_path, args)
    
    # Save results
    if args.output_dir is None:
        # Default to checkpoint directory
        args.output_dir = os.path.dirname(args.checkpoint_path)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filename based on checkpoint name
    checkpoint_name = os.path.basename(os.path.dirname(args.checkpoint_path))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.save_txt:
        txt_path = os.path.join(args.output_dir, f"evaluation_results_{checkpoint_name}_{timestamp}.txt")
        with open(txt_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("EVALUATION RESULTS\n")
            f.write("="*60 + "\n")
            f.write(f"Checkpoint: {args.checkpoint_path}\n")
            f.write(f"Model Type: {model_type}\n")
            f.write(f"Test Samples: {len(test_loader.dataset)}\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            f.write(metrics['summary_text'])
            f.write("\n\n")
            f.write("="*60 + "\n")
            f.write("DETAILED METRICS\n")
            f.write("="*60 + "\n")
            f.write(f"Overall AUC: {metrics['overall_auc']:.4f}\n")
            f.write(f"Overall Sensitivity: {metrics['overall_sensitivity']:.4f}\n")
            f.write(f"Overall Specificity: {metrics['overall_specificity']:.4f}\n")
            f.write(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}\n\n")
            f.write("Group-specific Metrics:\n")
            for group_name, group_metrics in metrics['group_metrics'].items():
                if group_metrics['n_samples'] > 0:
                    f.write(f"\n{group_name}:\n")
                    f.write(f"  AUC: {group_metrics['auc']:.4f}\n")
                    f.write(f"  Sensitivity: {group_metrics['sensitivity']:.4f}\n")
                    f.write(f"  Specificity: {group_metrics['specificity']:.4f}\n")
                    f.write(f"  N Samples: {group_metrics['n_samples']}\n")
        print(f"\nResults saved to: {txt_path}")
    
    if args.save_json:
        json_path = os.path.join(args.output_dir, f"evaluation_results_{checkpoint_name}_{timestamp}.json")
        json_data = {
            'checkpoint_path': args.checkpoint_path,
            'model_type': model_type,
            'test_samples': len(test_loader.dataset),
            'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': {
                'overall': {
                    'auc': metrics['overall_auc'],
                    'sensitivity': metrics['overall_sensitivity'],
                    'specificity': metrics['overall_specificity'],
                    'accuracy': metrics['overall_accuracy']
                },
                'groups': metrics['group_metrics']
            }
        }
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"JSON results saved to: {json_path}")
    
    print("\n" + "="*60)
    print("Evaluation completed!")
    print("="*60)


if __name__ == "__main__":
    main()
