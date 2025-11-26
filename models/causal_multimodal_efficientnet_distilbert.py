import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.image_encoder_efficientnet import ImageClfEfficientNet
from backbone.text_encoder_distilbert import DistilBertClf


class CausalMultimodalModelEfficientNetDistilBERT(nn.Module):
    """
    Causal Representation Learning Model for Multimodal (Image + Text) Learning
    Uses EfficientNet for images and DistilBERT for text
    Implements C3 risk (sufficiency + necessity) for causal inference
    """
    
    def __init__(self, args):
        super(CausalMultimodalModelEfficientNetDistilBERT, self).__init__()
        self.args = args
        
        # Encoders - EfficientNet for images, DistilBERT for text
        self.txtclf = DistilBertClf(args)
        self.imgclf = ImageClfEfficientNet(args)
        
        # Feature dimension - DistilBERT outputs 768, EfficientNet-B0 outputs 1280
        # We'll use DistilBERT dimension (768) as the common dimension
        d = args.hidden_sz  # Should be 768 for DistilBERT
        
        # Attention mechanism for multimodal fusion
        self.W_Q = nn.Linear(d, d)
        self.W_K = nn.Linear(d, d)
        self.W_V = nn.Linear(d, d)
        
        # Improved cross-attention fusion mechanism
        self.cross_attn = nn.MultiheadAttention(d, num_heads=8, dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        
        # Enhanced fusion network with residual connections
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        self.fusion_classifier = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.LayerNorm(d // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d // 2, args.n_classes)
        )
        
        # Calibrator for causal representation
        self.calibrator = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Linear(d, d)
        )
        
        # Causal classifier
        self.causal_classifier = nn.Linear(d, args.n_classes)
        
        # Feature selection/gating
        self.feature_selection = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, d),
            nn.Sigmoid()
        )
        
        # Loss functions - with label smoothing support
        label_smoothing = getattr(args, 'label_smoothing', 0.0)
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()
        
        # Hyperparameters
        self.lambda_v = getattr(args, 'lambda_v', 1.0)
        self.lambda_fe = getattr(args, 'lambda_fe', 1.0)
        
        # Image projection layer (EfficientNet-B0 outputs 1280, need to project to 768)
        # Image encoder output after flattening: num_image_embeds * img_hidden_sz
        img_feat_dim = args.num_image_embeds * args.img_hidden_sz
        # For EfficientNet-B0: img_hidden_sz = 1280, so if num_image_embeds=1, then img_feat_dim=1280
        # Need to project from 1280 to 768 (d)
        if img_feat_dim != d:
            self.img_proj = nn.Linear(img_feat_dim, d)
        else:
            self.img_proj = nn.Identity()
        
        # Store metrics
        self.c3_metrics = {}
        
        # Initialize fusion layers
        for m in [self.fusion_mlp, self.fusion_classifier]:
            for module in m:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        
        # Initialize attention weights
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.zeros_(self.W_Q.bias)
        nn.init.zeros_(self.W_K.bias)
        nn.init.zeros_(self.W_V.bias)
    
    def forward(self, txt, mask, segment, img, labels=None):
        """
        Forward pass
        
        Args:
            txt: Text token IDs [B, seq_len]
            mask: Attention mask [B, seq_len]
            segment: Token type IDs [B, seq_len]
            img: Image tensor [B, 3, H, W]
            labels: Ground truth labels [B] (required for causal loss)
        
        Returns:
            If labels provided: (fusion_logits, txt_logits, img_logits, total_loss, c3_risk, loss_v, loss_fe)
            Otherwise: (fusion_logits, txt_logits, img_logits)
        """
        # Extract features
        hidden_t = self.txtclf.enc(txt, mask, segment)  # [B, hidden_sz] = [B, 768]
        hidden_i = self.imgclf.img_encoder(img)  # [B, num_image_embeds, img_hidden_sz] = [B, 1, 1280]
        hidden_i_flat = torch.flatten(hidden_i, start_dim=1)  # [B, num_image_embeds * img_hidden_sz] = [B, 1280]
        
        # Project image features to same dimension as text (1280 -> 768)
        hidden_i_proj = self.img_proj(hidden_i_flat)  # [B, hidden_sz] = [B, 768]
        
        # Get logits for standard classification
        txt_logits = self.txtclf.clf(hidden_t)
        img_logits = self.imgclf.clf(hidden_i_flat)
        
        # Enhanced fusion with cross-attention
        # Expand dimensions for attention: [B, d] -> [B, 1, d]
        txt_attn = hidden_t.unsqueeze(1)  # [B, 1, d]
        img_attn = hidden_i_proj.unsqueeze(1)  # [B, 1, d]
        
        # Cross-attention: text attends to image
        txt_attended, _ = self.cross_attn(txt_attn, img_attn, img_attn)  # [B, 1, d]
        txt_attended = txt_attended.squeeze(1)  # [B, d]
        txt_fused = self.norm1(hidden_t + txt_attended)  # Residual connection
        
        # Cross-attention: image attends to text
        img_attended, _ = self.cross_attn(img_attn, txt_attn, txt_attn)  # [B, 1, d]
        img_attended = img_attended.squeeze(1)  # [B, d]
        img_fused = self.norm2(hidden_i_proj + img_attended)  # Residual connection
        
        # Concatenate and fuse
        fusion_features = torch.cat([txt_fused, img_fused], dim=-1)  # [B, 2*d]
        fusion_hidden = self.fusion_mlp(fusion_features)  # [B, d]
        fusion_logits = self.fusion_classifier(fusion_hidden)  # [B, n_classes]
        
        # If labels provided, compute causal loss
        if labels is not None:
            B, d = hidden_t.size()
            
            # Attention mechanism
            Q = self.W_Q(hidden_t)  # [B, d]
            K = self.W_K(hidden_i_proj)  # [B, d]
            scores = torch.sum(Q * K, dim=-1) / (d ** 0.5)  # [B]
            weights = torch.sigmoid(scores).unsqueeze(-1)  # [B, 1]
            V = weights * self.W_V(hidden_i_proj) + (1 - weights) * self.W_V(hidden_t)  # [B, d]
            
            # Feature selection
            t_selected = hidden_t * self.feature_selection(hidden_t)
            i_selected = hidden_i_proj * self.feature_selection(hidden_i_proj)
            
            # Causal representation
            Z_cat = torch.cat([t_selected, i_selected], dim=-1)  # [B, 2*d]
            Z_c = self.calibrator(Z_cat)  # [B, d]
            
            # Main classification loss using fusion logits
            loss_ce = self.ce_loss(fusion_logits, labels)
            
            # Counterfactual generation using gradient-based intervention
            # Create a copy that requires grad for gradient computation
            Z_c_cf = Z_c.clone().detach().requires_grad_(True)
            
            # Compute counterfactual loss
            logits_cf = self.causal_classifier(Z_c_cf)
            loss_cf = self.ce_loss(logits_cf, labels)
            
            # Compute gradient for counterfactual
            grad = torch.autograd.grad(
                outputs=loss_cf, 
                inputs=Z_c_cf, 
                retain_graph=True, 
                create_graph=True,
                allow_unused=False
            )[0]
            # Counterfactual representation with gradient step
            with torch.no_grad():
                Z_bar = Z_c_cf - 0.1 * grad  # Use smaller step size for stability
            
            # C3 Risk computation
            # Sufficiency risk: real prediction != true label
            y_pred_real = self.causal_classifier(Z_c).argmax(dim=1)
            risk_suff = (y_pred_real != labels).float().mean()
            
            # Necessity risk: counterfactual prediction == true label
            y_pred_counterfactual = self.causal_classifier(Z_bar).argmax(dim=1)
            risk_nec = (y_pred_counterfactual == labels).float().mean()
            c3_risk = risk_suff + risk_nec
            
            # Additional losses
            # Use Z_c (calibrated representation) and V for KL divergence
            # Normalize features before KL divergence
            p_z = F.log_softmax(Z_c / (torch.norm(Z_c, dim=-1, keepdim=True) + 1e-8), dim=-1)  # [B, d]
            p_v = F.softmax(V / (torch.norm(V, dim=-1, keepdim=True) + 1e-8), dim=-1)  # [B, d]
            loss_v = self.kl_loss(p_z, p_v)
            
            # Feature extraction loss: measure difference between counterfactual and original
            # Use detached Z_bar and Z_c to avoid affecting gradients but still compute differentiable loss
            loss_fe = self.mse_loss(Z_bar.detach(), Z_c.detach())
            
            # Total loss - focus more on main classification, reduce causal regularization
            # Use much smaller weights for causal losses to avoid interference
            total_loss = (
                loss_ce
                + 0.01 * self.lambda_v * loss_v
                + 0.01 * self.lambda_fe * loss_fe
                + 0.01 * c3_risk
            )
            
            # Store metrics
            self.c3_metrics = {
                'loss_ce': loss_ce.item(),
                'loss_v': loss_v.item(),
                'loss_fe': loss_fe.item(),
                'c3_risk': c3_risk.item(),
                'risk_suff': risk_suff.item(),
                'risk_nec': risk_nec.item(),
                'total_loss': total_loss.item()
            }
            
            return fusion_logits, txt_logits, img_logits, total_loss, c3_risk, loss_v, loss_fe
        
        return fusion_logits, txt_logits, img_logits

