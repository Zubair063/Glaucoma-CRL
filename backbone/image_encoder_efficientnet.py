import torch
import torch.nn as nn
try:
    from torchvision import models
    USE_TORCHVISION = True
except ImportError:
    USE_TORCHVISION = False
    raise ImportError("torchvision library is required for EfficientNet. Please install: pip install torchvision")


class ImageEncoderEfficientNet(nn.Module):
    """Image encoder using EfficientNet (same style as MultiFair)"""
    def __init__(self, args):
        super(ImageEncoderEfficientNet, self).__init__()
        self.args = args
        
        # Use EfficientNet-B0 (or B1, B2, etc. based on args)
        efficientnet_variant = getattr(args, 'efficientnet_variant', 'efficientnet_b0')
        pretrained = getattr(args, 'pretrained', True)
        
        if pretrained:
            if efficientnet_variant == 'efficientnet_b0':
                self.efficientnet = models.efficientnet_b0(pretrained=True)
            elif efficientnet_variant == 'efficientnet_b1':
                self.efficientnet = models.efficientnet_b1(pretrained=True)
            elif efficientnet_variant == 'efficientnet_b2':
                self.efficientnet = models.efficientnet_b2(pretrained=True)
            elif efficientnet_variant == 'efficientnet_b3':
                self.efficientnet = models.efficientnet_b3(pretrained=True)
            elif efficientnet_variant == 'efficientnet_b4':
                self.efficientnet = models.efficientnet_b4(pretrained=True)
            else:
                # Default to B0
                self.efficientnet = models.efficientnet_b0(pretrained=True)
        else:
            if efficientnet_variant == 'efficientnet_b0':
                self.efficientnet = models.efficientnet_b0(pretrained=False)
            elif efficientnet_variant == 'efficientnet_b1':
                self.efficientnet = models.efficientnet_b1(pretrained=False)
            elif efficientnet_variant == 'efficientnet_b2':
                self.efficientnet = models.efficientnet_b2(pretrained=False)
            elif efficientnet_variant == 'efficientnet_b3':
                self.efficientnet = models.efficientnet_b3(pretrained=False)
            elif efficientnet_variant == 'efficientnet_b4':
                self.efficientnet = models.efficientnet_b4(pretrained=False)
            else:
                self.efficientnet = models.efficientnet_b0(pretrained=False)
        
        # EfficientNet-B0 outputs 1280 features, B1=1280, B2=1408, B3=1536, B4=1792
        # Get the actual feature dimension
        if hasattr(self.efficientnet, 'classifier'):
            # For EfficientNet, features are in the classifier input
            if hasattr(self.efficientnet.classifier, 'in_features'):
                self.hidden_size = self.efficientnet.classifier.in_features
            else:
                # Fallback: check the last layer
                self.hidden_size = 1280  # Default for B0
        else:
            self.hidden_size = 1280  # Default for B0
        
        # Store the expected output dimension for compatibility
        self.output_dim = self.hidden_size
        
        # Store the classifier input dimension
        # For EfficientNet, we'll extract features before the classifier
        # The features() method gives us the convolutional features

    def forward(self, x):
        # x: [B, 3, 224, 224]
        # EfficientNet forward pass - extract features before classifier
        # EfficientNet has a 'features' attribute that contains the conv layers
        if hasattr(self.efficientnet, 'features'):
            features = self.efficientnet.features(x)  # [B, C, H, W]
        else:
            # Fallback: use avgpool and flatten
            features = self.efficientnet.avgpool(self.efficientnet.features(x))
            features = torch.flatten(features, 1)
            return features.unsqueeze(1) if getattr(self.args, 'num_image_embeds', 1) == 1 else features.unsqueeze(1).expand(-1, getattr(self.args, 'num_image_embeds', 1), -1)
        
        # Global average pooling
        features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))  # [B, C, 1, 1]
        features = features.squeeze(-1).squeeze(-1)  # [B, C]
        
        # Return in format compatible with existing code: [B, num_image_embeds, hidden_size]
        num_embeds = getattr(self.args, 'num_image_embeds', 1)
        if num_embeds == 1:
            return features.unsqueeze(1)  # [B, 1, hidden_size]
        else:
            # If num_image_embeds > 1, repeat features
            return features.unsqueeze(1).expand(-1, num_embeds, -1)  # [B, num_image_embeds, hidden_size]


class ImageClfEfficientNet(nn.Module):
    """Image classifier using EfficientNet encoder"""
    def __init__(self, args):
        super(ImageClfEfficientNet, self).__init__()
        self.args = args
        self.img_encoder = ImageEncoderEfficientNet(args)
        
        # Get the actual hidden size from encoder
        img_hidden_sz = self.img_encoder.hidden_size
        num_image_embeds = getattr(args, 'num_image_embeds', 1)
        self.clf = nn.Linear(img_hidden_sz * num_image_embeds, args.n_classes)

    def forward(self, x):
        x = self.img_encoder(x)  # [B, num_image_embeds, hidden_size]
        x = torch.flatten(x, start_dim=1)  # [B, num_image_embeds * hidden_size]
        return self.clf(x)

