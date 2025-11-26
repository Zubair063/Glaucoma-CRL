import torch.nn as nn
try:
    from transformers import AutoModel
    USE_TRANSFORMERS = True
except ImportError:
    USE_TRANSFORMERS = False
    raise ImportError("transformers library is required for DeBERTa. Please install: pip install transformers")


class DeBERTaEncoder(nn.Module):
    """Text encoder using DeBERTa (Enhanced BERT with disentangled attention)"""
    def __init__(self, args):
        super(DeBERTaEncoder, self).__init__()
        self.args = args
        
        # Use DeBERTa model
        deberta_model = getattr(args, 'deberta_model', 'microsoft/deberta-base')
        
        if USE_TRANSFORMERS:
            try:
                # Use AutoModel for DeBERTa
                self.deberta = AutoModel.from_pretrained(deberta_model, local_files_only=False)
            except Exception as e:
                print(f"Warning: Failed to load model with local files, forcing download: {e}")
                self.deberta = AutoModel.from_pretrained(deberta_model)
            self.bert_dim = self.deberta.config.hidden_size  # Store for external use
            self.use_transformers = True
        else:
            raise ImportError("transformers library is required for DeBERTa")

    def forward(self, txt, mask, segment):
        """
        Args:
            txt: input_ids [B, seq_len]
            mask: attention_mask [B, seq_len]
            segment: token_type_ids [B, seq_len] (DeBERTa can use this)
        Returns:
            [B, bert_dim] - DeBERTa [CLS] token features
        """
        if self.use_transformers:
            # DeBERTa can use token_type_ids
            outputs = self.deberta(
                input_ids=txt,
                attention_mask=mask,
                token_type_ids=segment,
                output_hidden_states=False,
            )
            # Use [CLS] token representation (last_hidden_state[:, 0])
            out = outputs.last_hidden_state[:, 0]  # [B, bert_dim]
        else:
            raise RuntimeError("transformers library is required for DeBERTa")
        
        return out


class DeBERTaClf(nn.Module):
    """Text classifier using DeBERTa encoder"""
    def __init__(self, args):
        super(DeBERTaClf, self).__init__()
        self.args = args
        self.enc = DeBERTaEncoder(args)
        
        # DeBERTa base outputs 768 dimensions
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

