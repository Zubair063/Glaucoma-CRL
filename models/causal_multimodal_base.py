import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.image_encoder import ImageClf
from backbone.text_encoder import BertClf


class CausalMultimodalModelBase(nn.Module):
    """
    Causal Representation Learning Model following Multi-Modal-Base pattern
    Uses ViT + BERT encoders with C3 risk (sufficiency + necessity) for causal inference
    """
    
    def __init__(self, args):
        super(CausalMultimodalModelBase, self).__init__()
        self.args = args
        
        # Encoders
        self.txtclf = BertClf(args)
        self.imgclf = ImageClf(args)
        
        # Feature dimension (should match hidden_sz from BERT)
        d = args.hidden_sz
        
        # Image projection layer (image features may need projection to match text dim)
        img_feat_dim = args.num_image_embeds * args.img_hidden_sz
        if img_feat_dim != d:
            self.img_proj = nn.Linear(img_feat_dim, d)
        else:
            self.img_proj = nn.Identity()
        
        # Attention mechanism for multimodal fusion
        self.W_Q = nn.Linear(d, d)
        self.W_K = nn.Linear(d, d)
        self.W_V = nn.Linear(d, d)
        
        # Feature extractor (calibrator) for causal representation
        self.feature_extractor = nn.Sequential(
            nn.Linear(d * 2, d),
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
        
        # Additional parameters from base_vta_causal pattern
        # attention_scale: scalar parameter to scale attention scores
        self.attention_scale = nn.Parameter(torch.ones(1))
        self.W_causal = nn.Parameter(torch.ones(d, d))
        
        # Hyperparameters
        self.lambda_v = getattr(args, 'lambda_v', 1.0)
        self.lambda_fe = getattr(args, 'lambda_fe', 1.0)
        
        # Store metrics
        self.c3_metrics = {}
    
    def forward(self, txt, mask, segment, img, labels=None):
        """
        Forward pass with causal representation learning following base_vta_causal pattern
        
        Args:
            txt: Text token ids [B, seq_len]
            mask: Attention mask [B, seq_len]
            segment: Segment ids [B, seq_len]
            img: Image tensor [B, 3, H, W]
            labels: Ground truth labels [B] (required for causal loss)
        
        Returns:
            If labels provided: (fusion_logits, txt_logits, img_logits, total_loss, c3_risk, loss_v, loss_fe)
            Otherwise: (fusion_logits, txt_logits, img_logits)
        """
        # Extract features
        hidden_t = self.txtclf.enc(txt, mask, segment)  # [B, hidden_sz]
        hidden_i = self.imgclf.img_encoder(img)  # [B, num_image_embeds, img_hidden_sz]
        hidden_i_flat = torch.flatten(hidden_i, start_dim=1)  # [B, num_image_embeds * img_hidden_sz]
        
        # Project image features to same dimension as text
        hidden_i_proj = self.img_proj(hidden_i_flat)  # [B, hidden_sz]
        
        # Get logits for standard classification
        txt_logits = self.txtclf.clf(hidden_t)
        img_logits = self.imgclf.clf(hidden_i_flat)
        
        # Simple fusion for standard prediction
        fusion_logits = (txt_logits + img_logits) / 2
        
        # If labels provided, compute causal loss following base_vta_causal pattern
        if labels is not None:
            # Feature selection
            t_out = hidden_t * self.feature_selection(hidden_t)  # [B, d]
            i_out = hidden_i_proj * self.feature_selection(hidden_i_proj)  # [B, d]
            
            # Create multimodal features for attention [B, 2, d]
            multimodal_features = torch.cat([t_out.unsqueeze(1), i_out.unsqueeze(1)], dim=1)
            
            # Attention mechanism
            Q = self.W_Q(multimodal_features)  # [B, 2, d]
            K = self.W_K(multimodal_features)  # [B, 2, d]
            V = self.W_V(multimodal_features)  # [B, 2, d]
            
            # Compute attention scores
            attention_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.args.hidden_sz ** 0.5)  # [B, 2, 2]
            attention_scores = attention_scores * self.attention_scale  # Scale by scalar parameter [B, 2, 2]
            attention_weights = F.softmax(attention_scores, dim=-1)  # [B, 2, 2]
            V_attended = torch.bmm(attention_weights, V).sum(dim=1)  # [B, d]
            
            # Causal representation using feature extractor
            z_c = self.feature_extractor(torch.cat([t_out, i_out], dim=1))  # [B, d]
            z_c.requires_grad_(True)
            
            # Counterfactual generation using gradient-based intervention
            # Following base_vta_causal pattern: use log_softmax gradient
            # But fix the grad_outputs to match the original pattern (though it seems wrong, let's try it)
            log_probs = self.causal_classifier(z_c).log_softmax(dim=1)  # [B, n_classes]
            
            # Try using sum of log_probs as scalar output (more stable gradient)
            log_probs_sum = log_probs.sum(dim=1).mean()  # Scalar
            delta = -torch.autograd.grad(
                outputs=log_probs_sum,
                inputs=z_c,
                retain_graph=True,
                create_graph=True
            )[0]
            # Use a smaller step size for stability
            step_size = 0.01  # Smaller step for more stable counterfactual
            z_c_bar = z_c + step_size * delta  # Counterfactual representation
            
            # C3 Risk computation
            y_pred_real = self.causal_classifier(z_c).argmax(dim=1)
            y_pred_counterfactual = self.causal_classifier(z_c_bar).argmax(dim=1)
            
            sufficiency_risk = (y_pred_real != labels).float().mean()
            necessity_risk = (y_pred_counterfactual == labels).float().mean()
            c3_risk = sufficiency_risk + necessity_risk
            
            # Loss computation following base_vta_causal pattern
            # loss_v: KL divergence between counterfactual and original (on feature distributions)
            # Normalize features to probability distributions over feature dimensions
            loss_v = F.kl_div(
                z_c_bar.log_softmax(dim=1),
                z_c.softmax(dim=1),
                reduction='batchmean'
            )
            
            # loss_fe: KL divergence between counterfactual and original (same as loss_v in base pattern)
            loss_fe = F.kl_div(
                z_c_bar.log_softmax(dim=1),
                z_c.softmax(dim=1),
                reduction='batchmean'
            )
            
            # Main classification loss using fusion logits (primary learning signal)
            # This is the main task loss that drives learning
            loss_ce = F.cross_entropy(fusion_logits, labels)
            
            # Total loss: prioritize classification, scale down causal regularization
            # Following the working pattern: use small weights for causal losses
            # C3 risk of 1.0 would dominate without scaling
            total_loss = (
                loss_ce
                + 0.01 * c3_risk  # Scale down C3 risk (it's in [0, 2] range)
                + 0.01 * self.lambda_v * loss_v  # Scale down KL divergence
                + 0.01 * self.lambda_fe * loss_fe  # Scale down feature extraction loss
            )
            
            # Store metrics
            self.c3_metrics = {
                'loss_ce': loss_ce.item(),
                'c3_risk': c3_risk.item(),
                'risk_suff': sufficiency_risk.item(),
                'risk_nec': necessity_risk.item(),
                'loss_v': loss_v.item(),
                'loss_fe': loss_fe.item(),
                'total_loss': total_loss.item()
            }
            
            return fusion_logits, txt_logits, img_logits, total_loss, c3_risk, loss_v, loss_fe
        
        return fusion_logits, txt_logits, img_logits

