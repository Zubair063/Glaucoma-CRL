import argparse
import os
import sys
import warnings

# Suppress TensorFlow warnings (transformers library pulls in TensorFlow dependencies)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings
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
from models.causal_multimodal_vit_bert import CausalMultimodalModelViTBERT
from utils.utils import set_seed, save_checkpoint, load_checkpoint, log_metrics
from logs.logger import create_logger


def get_args():
    parser = argparse.ArgumentParser(description="Train Causal Multimodal Model (Base Pattern) on FairCLIP")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="/medailab/medailab/shilab/FairCLIP",
                        help="Directory containing FairCLIP dataset")
    parser.add_argument("--csv_path", type=str, default="/medailab/medailab/shilab/FairCLIP/data_summary.csv",
                        help="Path to data_summary.csv")
    
    # Model arguments
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased",
                        help="BERT model name")
    parser.add_argument("--hidden_sz", type=int, default=768,
                        help="Hidden size (BERT output dimension)")
    parser.add_argument("--img_hidden_sz", type=int, default=768,
                        help="Image encoder hidden size (ViT base outputs 768)")
    parser.add_argument("--num_image_embeds", type=int, default=1,
                        help="Number of image embeddings")
    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["avg", "max"],
                        help="Image embedding pool type")
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
    parser.add_argument("--warmup_epochs", type=int, default=2,
                        help="Warmup epochs for learning rate")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping value")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing factor")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="Maximum sequence length for text")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    
    # Causal learning arguments
    parser.add_argument("--lambda_v", type=float, default=1.0,
                        help="Weight for KL divergence loss")
    parser.add_argument("--lambda_fe", type=float, default=1.0,
                        help="Weight for feature extraction loss")
    
    # Other arguments
    parser.add_argument("--n_workers", type=int, default=4,
                        help="Number of data loader workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--savedir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--name", type=str, default="causal_base_fairclip",
                        help="Experiment name")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience")
    parser.add_argument("--min_lr", type=float, default=1e-7,
                        help="Minimum learning rate for cosine annealing")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID to use")
    
    return parser.parse_args()


def model_forward(model, batch, args, compute_causal=True):
    """Forward pass through model"""
    img = batch['image'].cuda()
    txt = batch['txt'].cuda()
    mask = batch['mask'].cuda()
    segment = batch['segment'].cuda()
    labels = batch['label'].cuda()
    
    if compute_causal:
        fusion_logits, txt_logits, img_logits, total_loss, c3_risk, loss_v, loss_fe = model(
            txt, mask, segment, img, labels
        )
        return {
            'fusion_logits': fusion_logits,
            'txt_logits': txt_logits,
            'img_logits': img_logits,
            'total_loss': total_loss,
            'c3_risk': c3_risk,
            'loss_v': loss_v,
            'loss_fe': loss_fe,
            'labels': labels
        }
    else:
        fusion_logits, txt_logits, img_logits = model(txt, mask, segment, img, None)
        return {
            'fusion_logits': fusion_logits,
            'txt_logits': txt_logits,
            'img_logits': img_logits,
            'labels': labels
        }


def train_epoch(model, train_loader, optimizer, args, epoch, logger):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_c3 = 0
    total_v = 0
    total_fe = 0
    
    all_fusion_logits = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch_idx, batch in enumerate(pbar):
        outputs = model_forward(model, batch, args, compute_causal=True)
        
        loss = outputs['total_loss']
        c3_risk = outputs['c3_risk']
        loss_v = outputs['loss_v']
        loss_fe = outputs['loss_fe']
        
        # Collect logits and labels for AUC calculation
        fusion_logits = outputs['fusion_logits']
        labels = outputs['labels']
        all_fusion_logits.append(fusion_logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
        
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        
        loss.backward()
        
        if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * args.gradient_accumulation_steps
        total_c3 += c3_risk.item()
        total_v += loss_v.item()
        total_fe += loss_fe.item()
        
        pbar.set_postfix({
            'loss': f"{loss.item() * args.gradient_accumulation_steps:.4f}",
            'c3': f"{c3_risk.item():.4f}"
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_c3 = total_c3 / len(train_loader)
    avg_v = total_v / len(train_loader)
    avg_fe = total_fe / len(train_loader)
    
    # Calculate AUC for training set
    all_fusion_logits = torch.cat(all_fusion_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    fusion_probs = torch.softmax(all_fusion_logits, dim=1)[:, 1].numpy()
    all_labels_np = all_labels.numpy()
    try:
        train_auc = roc_auc_score(all_labels_np, fusion_probs)
    except ValueError:
        train_auc = 0.0  # In case of single class in batch
    
    logger.info(f"Train - Loss: {avg_loss:.4f}, AUC: {train_auc:.4f}, C3: {avg_c3:.4f}, Loss_V: {avg_v:.4f}, Loss_FE: {avg_fe:.4f}")
    
    return {'loss': avg_loss, 'auc': train_auc}


def evaluate(model, data_loader, args, split_name="Val"):
    """Evaluate model"""
    model.eval()
    
    # Check if data loader is empty
    if len(data_loader.dataset) == 0:
        print(f"Warning: {split_name} set is empty! Skipping evaluation.")
        return {
            'loss': 0.0,
            'auc': 0.0,
            'fusion_acc': 0.0,
            'fusion_f1': 0.0,
            'fusion_precision': 0.0,
            'fusion_recall': 0.0,
            'txt_acc': 0.0,
            'txt_f1': 0.0,
            'txt_auc': 0.0,
            'img_acc': 0.0,
            'img_f1': 0.0,
            'img_auc': 0.0,
        }
    
    all_fusion_logits = []
    all_txt_logits = []
    all_img_logits = []
    all_labels = []
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=split_name):
            # During evaluation, don't compute causal loss (no gradients needed)
            outputs = model_forward(model, batch, args, compute_causal=False)
            
            fusion_logits = outputs['fusion_logits']
            txt_logits = outputs['txt_logits']
            img_logits = outputs['img_logits']
            labels = outputs['labels']
            
            # Compute simple classification loss for evaluation
            criterion = nn.CrossEntropyLoss()
            loss = criterion(fusion_logits, labels)
            
            all_fusion_logits.append(fusion_logits.cpu())
            all_txt_logits.append(txt_logits.cpu())
            all_img_logits.append(img_logits.cpu())
            all_labels.append(labels.cpu())
            total_loss += loss.item()
            num_batches += 1
    
    # Check if we have any data
    if num_batches == 0 or len(all_fusion_logits) == 0:
        print(f"Warning: {split_name} set produced no batches! Skipping evaluation.")
        return {
            'loss': 0.0,
            'auc': 0.0,
            'fusion_acc': 0.0,
            'fusion_f1': 0.0,
            'fusion_precision': 0.0,
            'fusion_recall': 0.0,
            'txt_acc': 0.0,
            'txt_f1': 0.0,
            'txt_auc': 0.0,
            'img_acc': 0.0,
            'img_f1': 0.0,
            'img_auc': 0.0,
        }
    
    # Concatenate all predictions
    all_fusion_logits = torch.cat(all_fusion_logits, dim=0)
    all_txt_logits = torch.cat(all_txt_logits, dim=0)
    all_img_logits = torch.cat(all_img_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # Get predictions
    fusion_preds = all_fusion_logits.argmax(dim=1).numpy()
    txt_preds = all_txt_logits.argmax(dim=1).numpy()
    img_preds = all_img_logits.argmax(dim=1).numpy()
    
    # Get probabilities for AUC calculation
    fusion_probs = torch.softmax(all_fusion_logits, dim=1)[:, 1].numpy()
    txt_probs = torch.softmax(all_txt_logits, dim=1)[:, 1].numpy()
    img_probs = torch.softmax(all_img_logits, dim=1)[:, 1].numpy()
    
    # Compute metrics including AUC
    metrics = {
        'loss': total_loss / len(data_loader),
        'auc': roc_auc_score(all_labels, fusion_probs) if len(np.unique(all_labels)) > 1 else 0.0,
        'fusion_acc': accuracy_score(all_labels, fusion_preds),
        'fusion_f1': f1_score(all_labels, fusion_preds),
        'fusion_precision': precision_score(all_labels, fusion_preds),
        'fusion_recall': recall_score(all_labels, fusion_preds),
        'txt_acc': accuracy_score(all_labels, txt_preds),
        'txt_f1': f1_score(all_labels, txt_preds),
        'txt_auc': roc_auc_score(all_labels, txt_probs) if len(np.unique(all_labels)) > 1 else 0.0,
        'img_acc': accuracy_score(all_labels, img_preds),
        'img_f1': f1_score(all_labels, img_preds),
        'img_auc': roc_auc_score(all_labels, img_probs) if len(np.unique(all_labels)) > 1 else 0.0,
    }
    
    return metrics


def main():
    args = get_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    
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
    logger.info("Creating model...")
    model = CausalMultimodalModelViTBERT(args)
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
        # Use weights_only=False to allow loading checkpoints with numpy scalars (PyTorch 2.6+)
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

