import argparse
import os
import sys
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*TensorFlow.*')
warnings.filterwarnings('ignore', message='.*oneDNN.*')
warnings.filterwarnings('ignore', message='.*Protobuf.*')

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm

from data.fairclip_dataset import get_fairclip_loaders
from models.causal_multimodal_resnet_roberta import CausalMultimodalModelResNetRoBERTa
from utils.utils import set_seed, save_checkpoint, load_checkpoint, log_metrics
from logs.logger import create_logger

# Import model_forward, train_epoch, evaluate from train_efficientnet_distilbert
# They are the same, so we can reuse them
import sys
sys.path.insert(0, os.path.dirname(__file__))
from train_efficientnet_distilbert import model_forward, train_epoch, evaluate


def get_args():
    parser = argparse.ArgumentParser(description="Train Causal Multimodal Model (ResNet + RoBERTa) on FairCLIP")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="/medailab/medailab/shilab/FairCLIP",
                        help="Directory containing FairCLIP dataset")
    parser.add_argument("--csv_path", type=str, default="/medailab/medailab/shilab/FairCLIP/data_summary.csv",
                        help="Path to data_summary.csv")
    
    # Model arguments
    parser.add_argument("--roberta_model", type=str, default="roberta-base",
                        help="RoBERTa model name")
    parser.add_argument("--resnet_variant", type=str, default="resnet50",
                        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
                        help="ResNet variant")
    parser.add_argument("--hidden_sz", type=int, default=768,
                        help="Hidden size (RoBERTa output dimension)")
    parser.add_argument("--img_hidden_sz", type=int, default=2048,
                        help="Image encoder hidden size (ResNet50 outputs 2048)")
    parser.add_argument("--num_image_embeds", type=int, default=1,
                        help="Number of image embeddings")
    parser.add_argument("--n_classes", type=int, default=2,
                        help="Number of classes (binary: glaucoma yes/no)")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--max_epochs", type=int, default=50,
                        help="Maximum number of epochs")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--min_lr", type=float, default=1e-7,
                        help="Minimum learning rate for scheduler")
    parser.add_argument("--warmup_epochs", type=int, default=2,
                        help="Number of warmup epochs")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping value")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing factor")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    
    # Causal loss arguments
    parser.add_argument("--lambda_v", type=float, default=1.0,
                        help="Weight for KL divergence loss")
    parser.add_argument("--lambda_fe", type=float, default=1.0,
                        help="Weight for feature extraction loss")
    
    # Other arguments
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--n_workers", type=int, default=4,
                        help="Number of data loader workers")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="Maximum sequence length for text")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--savedir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--name", type=str, default="causal_resnet_roberta",
                        help="Experiment name")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID to use")
    
    return parser.parse_args()


def main():
    args = get_args()
    
    # Set GPU
    if args.gpu_id >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        torch.cuda.set_device(0)
    
    # Set seed
    set_seed(args.seed)
    
    # Create save directory
    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)
    
    # Create logger
    logger = create_logger(os.path.join(args.savedir, "logfile.log"), args)
    
    # Load data
    logger.info("Loading data...")
    train_loader, val_loader, test_loader = get_fairclip_loaders(
        args.data_dir,
        args.csv_path,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        max_seq_len=args.max_seq_len
    )
    logger.info(f"Train samples: {len(train_loader.dataset)}, "
                f"Val samples: {len(val_loader.dataset)}, "
                f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    logger.info("Creating model with ResNet + RoBERTa...")
    model = CausalMultimodalModelResNetRoBERTa(args)
    model.cuda()
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer with different learning rates for encoders vs new layers
    encoder_params = []
    new_params = []
    for name, param in model.named_parameters():
        if 'txtclf' in name or 'imgclf' in name:
            encoder_params.append(param)
        else:
            new_params.append(param)
    
    optimizer = torch.optim.AdamW(
        [
            {'params': encoder_params, 'lr': args.lr * 0.5, 'weight_decay': args.weight_decay},
            {'params': new_params, 'lr': args.lr, 'weight_decay': args.weight_decay}
        ],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler - Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=args.min_lr
    )
    
    # Warmup scheduler
    warmup_scheduler = None
    if args.warmup_epochs > 0:
        from torch.optim.lr_scheduler import LambdaLR
        warmup_scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: min(epoch / args.warmup_epochs, 1.0)
        )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_metric = -np.inf
    n_no_improve = 0
    
    if args.resume:
        checkpoint = torch.load(args.resume, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_metric = checkpoint.get('best_metric', -np.inf)
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, args.max_epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, args, epoch, logger)
        
        # Validate
        val_metrics = evaluate(model, val_loader, args, "Val")
        log_metrics("Val", val_metrics, args, logger)
        
        # Update learning rate
        if warmup_scheduler is not None and epoch < args.warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Current learning rate: {current_lr:.2e}")
        
        # Check for improvement based on AUC
        current_metric = val_metrics['auc']
        is_best = current_metric > best_metric
        
        if is_best:
            best_metric = current_metric
            n_no_improve = 0
        else:
            n_no_improve += 1
        
        # Save checkpoint
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_metric': best_metric,
                'n_no_improve': n_no_improve,
            },
            is_best,
            args.savedir
        )
        
        # Early stopping
        if n_no_improve >= args.patience:
            logger.info(f"No improvement for {args.patience} epochs. Early stopping.")
            break
    
    # Test
    if len(test_loader.dataset) > 0:
        logger.info("Evaluating on test set...")
        load_checkpoint(model, os.path.join(args.savedir, 'model_best.pt'))
        test_metrics = evaluate(model, test_loader, args, "Test")
        log_metrics("Test", test_metrics, args, logger)
    else:
        logger.info("Test set is empty. Skipping test evaluation.")
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()

