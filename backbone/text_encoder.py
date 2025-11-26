import torch.nn as nn
try:
    from transformers import AutoModel
    USE_TRANSFORMERS = True
except ImportError:
    from pytorch_pretrained_bert.modeling import BertModel
    USE_TRANSFORMERS = False


class BertEncoder(nn.Module):
    """Text encoder using AutoModel (same as MultiFair implementation)"""
    def __init__(self, args):
        super(BertEncoder, self).__init__()
        self.args = args
        bert_model = getattr(args, 'bert_model', 'bert-base-uncased')
        
        if USE_TRANSFORMERS:
            try:
                # Use AutoModel like MultiFair
                self.bert = AutoModel.from_pretrained(bert_model, local_files_only=False)
            except Exception as e:
                print(f"Warning: Failed to load model with local files, forcing download: {e}")
                self.bert = AutoModel.from_pretrained(bert_model)
            self.bert_dim = self.bert.config.hidden_size  # Store for external use
            self.use_transformers = True
        else:
            try:
                self.bert = BertModel.from_pretrained(bert_model, local_files_only=False)
            except Exception as e:
                print(f"Warning: Failed to load model with local files, forcing download: {e}")
                self.bert = BertModel.from_pretrained(bert_model, force_download=True)
            self.bert_dim = 768  # BERT base dimension
            self.use_transformers = False

    def forward(self, txt, mask, segment):
        """
        Args:
            txt: input_ids [B, seq_len]
            mask: attention_mask [B, seq_len]
            segment: token_type_ids [B, seq_len] (may not be used)
        Returns:
            [B, bert_dim] - BERT [CLS] token features
        """
        if self.use_transformers:
            # Use AutoModel like MultiFair - get last_hidden_state[:, 0]
            outputs = self.bert(
                input_ids=txt,
                attention_mask=mask,
                output_hidden_states=False,
            )
            # Use [CLS] token representation (last_hidden_state[:, 0]) like MultiFair
            out = outputs.last_hidden_state[:, 0]  # [B, bert_dim]
        else:
            _, out = self.bert(
                txt,
                token_type_ids=segment,
                attention_mask=mask,
                output_all_encoded_layers=False,
            )
        return out


class BertClf(nn.Module):
    def __init__(self, args):
        super(BertClf, self).__init__()
        self.args = args
        self.enc = BertEncoder(args)
        self.clf = nn.Linear(args.hidden_sz, args.n_classes)
        if not self.enc.use_transformers and hasattr(self.enc.bert, 'init_bert_weights'):
            self.clf.apply(self.enc.bert.init_bert_weights)
        else:
            nn.init.xavier_uniform_(self.clf.weight)
            nn.init.zeros_(self.clf.bias)

    def forward(self, txt, mask, segment):
        x = self.enc(txt, mask, segment)
        out = self.clf(x)
        return out

