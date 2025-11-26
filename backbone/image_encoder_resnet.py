import torch
import torch.nn as nn
try:
    from torchvision import models
    USE_TORCHVISION = True
except ImportError:
    USE_TORCHVISION = False
    raise ImportError("torchvision library is required for ResNet. Please install: pip install torchvision")


class ImageEncoderResNet(nn.Module):
    """Image encoder using ResNet (classic and proven architecture)"""
    def __init__(self, args):
        super(ImageEncoderResNet, self).__init__()
        self.args = args
        
        # Use ResNet variant (ResNet50, ResNet101, ResNet152)
        resnet_variant = getattr(args, 'resnet_variant', 'resnet50')
        pretrained = getattr(args, 'pretrained', True)
        
        if pretrained:
            if resnet_variant == 'resnet50':
                self.resnet = models.resnet50(pretrained=True)
            elif resnet_variant == 'resnet101':
                self.resnet = models.resnet101(pretrained=True)
            elif resnet_variant == 'resnet152':
                self.resnet = models.resnet152(pretrained=True)
            elif resnet_variant == 'resnet34':
                self.resnet = models.resnet34(pretrained=True)
            elif resnet_variant == 'resnet18':
                self.resnet = models.resnet18(pretrained=True)
            else:
                # Default to ResNet50
                self.resnet = models.resnet50(pretrained=True)
        else:
            if resnet_variant == 'resnet50':
                self.resnet = models.resnet50(pretrained=False)
            elif resnet_variant == 'resnet101':
                self.resnet = models.resnet101(pretrained=False)
            elif resnet_variant == 'resnet152':
                self.resnet = models.resnet152(pretrained=False)
            elif resnet_variant == 'resnet34':
                self.resnet = models.resnet34(pretrained=False)
            elif resnet_variant == 'resnet18':
                self.resnet = models.resnet18(pretrained=False)
            else:
                self.resnet = models.resnet50(pretrained=False)
        
        # ResNet outputs 2048 features for ResNet50/101/152, 512 for ResNet18/34
        # Get the actual feature dimension
        if hasattr(self.resnet, 'fc'):
            self.hidden_size = self.resnet.fc.in_features
        else:
            # Fallback: ResNet50/101/152 = 2048, ResNet18/34 = 512
            if '50' in resnet_variant or '101' in resnet_variant or '152' in resnet_variant:
                self.hidden_size = 2048
            else:
                self.hidden_size = 512
        
        # Store the expected output dimension for compatibility
        self.output_dim = self.hidden_size
        
        # Remove the classifier head (we'll use features only)
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        # x: [B, 3, 224, 224]
        # ResNet forward pass
        # ResNet processes through conv layers, then avgpool, then fc (which we removed)
        features = self.resnet(x)  # [B, hidden_size]
        
        # Return in format compatible with existing code: [B, num_image_embeds, hidden_size]
        num_embeds = getattr(self.args, 'num_image_embeds', 1)
        if num_embeds == 1:
            return features.unsqueeze(1)  # [B, 1, hidden_size]
        else:
            # If num_image_embeds > 1, repeat features
            return features.unsqueeze(1).expand(-1, num_embeds, -1)  # [B, num_image_embeds, hidden_size]


class ImageClfResNet(nn.Module):
    """Image classifier using ResNet encoder"""
    def __init__(self, args):
        super(ImageClfResNet, self).__init__()
        self.args = args
        self.img_encoder = ImageEncoderResNet(args)
        
        # Get the actual hidden size from encoder
        img_hidden_sz = self.img_encoder.hidden_size
        num_image_embeds = getattr(args, 'num_image_embeds', 1)
        self.clf = nn.Linear(img_hidden_sz * num_image_embeds, args.n_classes)

    def forward(self, x):
        x = self.img_encoder(x)  # [B, num_image_embeds, hidden_size]
        x = torch.flatten(x, start_dim=1)  # [B, num_image_embeds * hidden_size]
        return self.clf(x)

