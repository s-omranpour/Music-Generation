import math
import torch
from torch import nn

class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x.long()) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=20000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1), :]


class CPEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        emb_layers = []
        for k in config['attributes']:
            emb_layers += [[k, Embeddings(n_token=config['n_tokens'][k], d_model=config['emb_sizes'][k])]]
        self.emb_layers = nn.ModuleDict(emb_layers)
        sum_emb_dims = sum(config['emb_sizes'].values())
        self.proj = nn.Linear(sum_emb_dims, config['d_model']) if config['d_model'] != sum_emb_dims else None
        self.pos_emb = None
        if config['pos_emb']:
            self.pos_emb = PositionalEncoding(d_model=config['d_model'], max_len=config['max_len'])
        self.dropout = nn.Dropout(p=config['dropout'])

    def forward(self, x):
        embs = []
        for i, k in enumerate(self.emb_layers):
            embs += [self.emb_layers[k](x[..., i])]
        embs = torch.cat(embs, dim=-1)
        if self.proj is not None:
            embs = self.proj(embs)
        if self.pos_emb is not None:
            embs += self.pos_emb(embs)
        return self.dropout(embs)
    
    
class RemiEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.emb = nn.Embedding(config['n_vocab'], config['d_model'])
        if config['pos_emb']:
            self.pos_emb = PositionalEncoding(d_model=config['d_model'], max_len=config['max_len'])
        self.dropout = nn.Dropout(p=config['dropout'])
        
    def forward(self, x):
        h = self.emb(x)
        return self.dropout(self.pos_emb(h) + h)

