import torch.nn as nn
try:
    from transformers import AutoModel
    USE_TRANSFORMERS = True
except ImportError:
    USE_TRANSFORMERS = False
    raise ImportError("transformers library is required for RoBERTa. Please install: pip install transformers")


class RoBERTaEncoder(nn.Module):
    """Text encoder using RoBERTa (improved BERT with better pre-training)"""
    def __init__(self, args):
        super(RoBERTaEncoder, self).__init__()
        self.args = args
        
        # Use RoBERTa model
        roberta_model = getattr(args, 'roberta_model', 'roberta-base')
        
        if USE_TRANSFORMERS:
            try:
                # Use AutoModel for RoBERTa
                self.roberta = AutoModel.from_pretrained(roberta_model, local_files_only=False)
            except Exception as e:
                print(f"Warning: Failed to load model with local files, forcing download: {e}")
                self.roberta = AutoModel.from_pretrained(roberta_model)
            self.bert_dim = self.roberta.config.hidden_size  # Store for external use
            self.use_transformers = True
        else:
            raise ImportError("transformers library is required for RoBERTa")

    def forward(self, txt, mask, segment):
        """
        Args:
            txt: input_ids [B, seq_len]
            mask: attention_mask [B, seq_len]
            segment: token_type_ids [B, seq_len] (RoBERTa doesn't use this, but kept for compatibility)
        Returns:
            [B, bert_dim] - RoBERTa [CLS] token features
        """
        if self.use_transformers:
            # RoBERTa doesn't use token_type_ids, so we ignore segment
            outputs = self.roberta(
                input_ids=txt,
                attention_mask=mask,
                output_hidden_states=False,
            )
            # Use [CLS] token representation (last_hidden_state[:, 0])
            out = outputs.last_hidden_state[:, 0]  # [B, bert_dim]
        else:
            raise RuntimeError("transformers library is required for RoBERTa")
        
        return out


class RoBERTaClf(nn.Module):
    """Text classifier using RoBERTa encoder"""
    def __init__(self, args):
        super(RoBERTaClf, self).__init__()
        self.args = args
        self.enc = RoBERTaEncoder(args)
        
        # RoBERTa base outputs 768 dimensions
        # Use the actual dimension from the encoder
        hidden_sz = self.enc.bert_dim
        self.clf = nn.Linear(hidden_sz, args.n_classes)
        
        # Initialize classifier
        nn.init.xavier_uniform_(self.clf.weight)
        nn.init.zeros_(self.clf.bias)

    def forward(self, txt, mask, segment):
        x = self.enc(txt, mask, segment)
        out = self.clf(x)
        return out

