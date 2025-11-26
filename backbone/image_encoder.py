import torch
import torch.nn as nn
try:
    from transformers import ViTModel, ViTConfig
    USE_TRANSFORMERS = True
except ImportError:
    USE_TRANSFORMERS = False
    raise ImportError("transformers library is required for ViT. Please install: pip install transformers")


class ImageEncoder(nn.Module):
    def __init__(self, args):
        super(ImageEncoder, self).__init__()
        self.args = args
        # Use ViT ImageNet-21k pretrained (same as MultiFair: google/vit-base-patch16-224-in21k)
        vit_variant = 'google/vit-base-patch16-224-in21k'  # Same as MultiFair
        pretrained = getattr(args, 'pretrained', True)
        
        if pretrained:
            try:
                self.vit = ViTModel.from_pretrained(vit_variant, local_files_only=False)
            except Exception as e:
                print(f"Warning: Failed to load model with local files, forcing download: {e}")
                self.vit = ViTModel.from_pretrained(vit_variant)
        else:
            config = ViTConfig.from_pretrained(vit_variant)
            self.vit = ViTModel(config)
        
        # ViT base patch16-224-in21k outputs hidden_size of 768
        self.hidden_size = self.vit.config.hidden_size
        # Store the expected output dimension for compatibility
        self.output_dim = self.hidden_size

    def forward(self, x):
        # x: [B, 3, 224, 224]
        # ViT expects pixel_values
        outputs = self.vit(pixel_values=x, output_hidden_states=False)
        # Use pooler_output like MultiFair implementation
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            vit_features = outputs.pooler_output  # [B, hidden_size]
        else:
            # Fallback to [CLS] token if pooler_output not available
            last_hidden_state = outputs.last_hidden_state  # [B, num_patches+1, hidden_size]
            vit_features = last_hidden_state[:, 0]  # [B, hidden_size]
        
        # Return in format compatible with existing code: [B, num_image_embeds, hidden_size]
        num_embeds = getattr(self.args, 'num_image_embeds', 1)
        if num_embeds == 1:
            return vit_features.unsqueeze(1)  # [B, 1, hidden_size]
        else:
            # If num_image_embeds > 1, use CLS token repeated
            return vit_features.unsqueeze(1).expand(-1, num_embeds, -1)  # [B, num_image_embeds, hidden_size]


class ImageClf(nn.Module):
    def __init__(self, args):
        super(ImageClf, self).__init__()
        self.args = args
        self.img_encoder = ImageEncoder(args)
        # ViT base outputs 768-dim features
        # Update img_hidden_sz to 768 for ViT
        img_hidden_sz = 768  # ViT base hidden size
        num_image_embeds = getattr(args, 'num_image_embeds', 1)
        self.clf = nn.Linear(img_hidden_sz * num_image_embeds, args.n_classes)

    def forward(self, x):
        x = self.img_encoder(x)  # [B, num_image_embeds, hidden_size]
        x = torch.flatten(x, start_dim=1)  # [B, num_image_embeds * hidden_size]
        return self.clf(x)

