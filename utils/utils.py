import os
import random
import numpy as np
import torch
import shutil


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state, is_best, checkpoint_dir):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pt')
    torch.save(state, checkpoint_path)
    
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'model_best.pt')
        shutil.copyfile(checkpoint_path, best_path)
        print(f"Saved best model to {best_path}")


def load_checkpoint(model, checkpoint_path):
    """Load model checkpoint"""
    # Use weights_only=False to allow loading checkpoints with numpy scalars (PyTorch 2.6+)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    return checkpoint


def log_metrics(split, metrics, args, logger):
    """Log metrics"""
    log_str = f"{split} - "
    for key, value in metrics.items():
        log_str += f"{key}: {value:.4f} "
    logger.info(log_str)

