"""
Test script to verify the setup is correct
"""
import sys
import torch
import numpy as np
from data.fairclip_dataset import get_fairclip_loaders
from models.causal_multimodal_vit_bert import CausalMultimodalModelViTBERT


class Args:
    """Mock args for testing"""
    def __init__(self):
        self.data_dir = "/medailab/medailab/shilab/FairCLIP"
        self.csv_path = "/medailab/medailab/shilab/FairCLIP/data_summary.csv"
        self.bert_model = "bert-base-uncased"
        self.hidden_sz = 768
        self.img_hidden_sz = 2048
        self.num_image_embeds = 1
        self.img_embed_pool_type = "avg"
        self.n_classes = 2
        self.lambda_v = 1.0
        self.lambda_fe = 1.0


def test_data_loader():
    """Test data loading"""
    print("Testing data loader...")
    args = Args()
    
    try:
        train_loader, val_loader, test_loader = get_fairclip_loaders(
            args.data_dir,
            args.csv_path,
            batch_size=2,
            num_workers=0,
            max_seq_len=512
        )
        
        # Get one batch
        batch = next(iter(train_loader))
        print(f"✓ Data loader works!")
        print(f"  - Image shape: {batch['image'].shape}")
        print(f"  - Text shape: {batch['txt'].shape}")
        print(f"  - Mask shape: {batch['mask'].shape}")
        print(f"  - Labels shape: {batch['label'].shape}")
        print(f"  - Train batches: {len(train_loader)}")
        print(f"  - Val batches: {len(val_loader)}")
        print(f"  - Test batches: {len(test_loader)}")
        return True
    except Exception as e:
        print(f"✗ Data loader failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model():
    """Test model forward pass"""
    print("\nTesting model...")
    args = Args()
    
    try:
        model = CausalMultimodalModelViTBERT(args)
        print(f"✓ Model created!")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create dummy inputs
        batch_size = 2
        txt = torch.randint(0, 1000, (batch_size, 512))
        mask = torch.ones(batch_size, 512)
        segment = torch.zeros(batch_size, 512)
        img = torch.randn(batch_size, 3, 224, 224)
        labels = torch.randint(0, 2, (batch_size,))
        
        # Forward pass without labels
        fusion_logits, txt_logits, img_logits = model(txt, mask, segment, img, None)
        print(f"✓ Forward pass (no labels) works!")
        print(f"  - Fusion logits: {fusion_logits.shape}")
        print(f"  - Text logits: {txt_logits.shape}")
        print(f"  - Image logits: {img_logits.shape}")
        
        # Forward pass with labels (causal loss)
        fusion_logits, txt_logits, img_logits, total_loss, c3_risk, loss_v, loss_fe = model(
            txt, mask, segment, img, labels
        )
        print(f"✓ Forward pass (with causal loss) works!")
        print(f"  - Total loss: {total_loss.item():.4f}")
        print(f"  - C3 risk: {c3_risk.item():.4f}")
        print(f"  - Loss V: {loss_v.item():.4f}")
        print(f"  - Loss FE: {loss_fe.item():.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("Testing Causal Multimodal Learning Setup")
    print("=" * 60)
    
    data_ok = test_data_loader()
    model_ok = test_model()
    
    print("\n" + "=" * 60)
    if data_ok and model_ok:
        print("✓ All tests passed! Setup is correct.")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

