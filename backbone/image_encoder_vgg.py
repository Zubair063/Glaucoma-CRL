import torch
import torch.nn as nn
try:
    from torchvision import models
    USE_TORCHVISION = True
except ImportError:
    USE_TORCHVISION = False
    raise ImportError("torchvision library is required for VGG. Please install: pip install torchvision")


class ImageEncoderVGG(nn.Module):
    """Image encoder using VGG (classic CNN architecture)"""
    def __init__(self, args):
        super(ImageEncoderVGG, self).__init__()
        self.args = args
        
        # Use VGG variant (VGG11, VGG13, VGG16, VGG19)
        vgg_variant = getattr(args, 'vgg_variant', 'vgg16')
        pretrained = getattr(args, 'pretrained', True)
        
        if pretrained:
            if vgg_variant == 'vgg11':
                self.vgg = models.vgg11(pretrained=True)
            elif vgg_variant == 'vgg13':
                self.vgg = models.vgg13(pretrained=True)
            elif vgg_variant == 'vgg16':
                self.vgg = models.vgg16(pretrained=True)
            elif vgg_variant == 'vgg19':
                self.vgg = models.vgg19(pretrained=True)
            elif vgg_variant == 'vgg11_bn':
                self.vgg = models.vgg11_bn(pretrained=True)
            elif vgg_variant == 'vgg13_bn':
                self.vgg = models.vgg13_bn(pretrained=True)
            elif vgg_variant == 'vgg16_bn':
                self.vgg = models.vgg16_bn(pretrained=True)
            elif vgg_variant == 'vgg19_bn':
                self.vgg = models.vgg19_bn(pretrained=True)
            else:
                # Default to VGG16
                self.vgg = models.vgg16(pretrained=True)
        else:
            if vgg_variant == 'vgg11':
                self.vgg = models.vgg11(pretrained=False)
            elif vgg_variant == 'vgg13':
                self.vgg = models.vgg13(pretrained=False)
            elif vgg_variant == 'vgg16':
                self.vgg = models.vgg16(pretrained=False)
            elif vgg_variant == 'vgg19':
                self.vgg = models.vgg19(pretrained=False)
            elif vgg_variant == 'vgg11_bn':
                self.vgg = models.vgg11_bn(pretrained=False)
            elif vgg_variant == 'vgg13_bn':
                self.vgg = models.vgg13_bn(pretrained=False)
            elif vgg_variant == 'vgg16_bn':
                self.vgg = models.vgg16_bn(pretrained=False)
            elif vgg_variant == 'vgg19_bn':
                self.vgg = models.vgg19_bn(pretrained=False)
            else:
                self.vgg = models.vgg16(pretrained=False)
        
        # VGG outputs 4096 features for VGG16/19, 4096 for VGG11/13
        # Get the actual feature dimension from the classifier
        if hasattr(self.vgg, 'classifier'):
            # VGG classifier typically has: Linear(25088, 4096) -> ReLU -> Dropout -> Linear(4096, 4096) -> ReLU -> Dropout -> Linear(4096, 1000)
            # We want the second-to-last layer (4096)
            if len(self.vgg.classifier) >= 6:
                self.hidden_size = self.vgg.classifier[3].in_features  # Second Linear layer input
            else:
                self.hidden_size = 4096  # Default for VGG
        else:
            self.hidden_size = 4096  # Default for VGG
        
        # Store the expected output dimension for compatibility
        self.output_dim = self.hidden_size
        
        # Remove the classifier head (we'll use features only)
        # Keep only up to the second-to-last layer (before final classification)
        if hasattr(self.vgg, 'classifier'):
            # Replace classifier with just the feature extraction part
            # VGG classifier: Linear(25088, 4096) -> ReLU -> Dropout -> Linear(4096, 4096) -> ReLU -> Dropout -> Linear(4096, 1000)
            # We want: Linear(25088, 4096) -> ReLU -> Dropout -> Linear(4096, 4096) -> ReLU
            if len(self.vgg.classifier) >= 6:
                self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier.children())[:-3])  # Remove last 3 layers (Dropout, Linear)
            else:
                # Fallback: just remove the last layer
                self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier.children())[:-1])

    def forward(self, x):
        # x: [B, 3, 224, 224]
        # VGG forward pass
        # VGG processes through features (conv layers), then avgpool, then classifier
        features = self.vgg(x)  # [B, hidden_size] = [B, 4096]
        
        # Return in format compatible with existing code: [B, num_image_embeds, hidden_size]
        num_embeds = getattr(self.args, 'num_image_embeds', 1)
        if num_embeds == 1:
            return features.unsqueeze(1)  # [B, 1, hidden_size]
        else:
            # If num_image_embeds > 1, repeat features
            return features.unsqueeze(1).expand(-1, num_embeds, -1)  # [B, num_image_embeds, hidden_size]


class ImageClfVGG(nn.Module):
    """Image classifier using VGG encoder"""
    def __init__(self, args):
        super(ImageClfVGG, self).__init__()
        self.args = args
        self.img_encoder = ImageEncoderVGG(args)
        
        # Get the actual hidden size from encoder
        img_hidden_sz = self.img_encoder.hidden_size
        num_image_embeds = getattr(args, 'num_image_embeds', 1)
        self.clf = nn.Linear(img_hidden_sz * num_image_embeds, args.n_classes)

    def forward(self, x):
        x = self.img_encoder(x)  # [B, num_image_embeds, hidden_size]
        x = torch.flatten(x, start_dim=1)  # [B, num_image_embeds * hidden_size]
        return self.clf(x)

