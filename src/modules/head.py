import torch
from torch import nn
from torch.nn import functional as F

class CPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        heads = []
        for k in config['attributes']:
            heads += [[k, nn.Linear(config['d_model'], config['n_tokens'][k])]]
        self.heads = nn.ModuleDict(heads)
        self.proj_cat = nn.Linear(config['d_model'] + config['emb_sizes']['ttype'], config['d_model'])

    def register_type_embedding(self, embedding):
        self.type_emb = embedding.emb_layers['ttype']

    def forward_type(self, h):
        return self.heads['ttype'](h)

    def forward(self, h, y_type=None):
        if y_type is None:
            type_prob = F.softmax(self.forward_type(h), dim=-1)
            n,s,t = type_prob.shape
            y_type = torch.multinomial(type_prob.view(-1, t), 1, replacement=True).view(n, s)
        tf_skip_type = self.type_emb(y_type)

        # project other
        y_concat_type = torch.cat([h, tf_skip_type], dim=-1)
        y_  = self.proj_cat(y_concat_type)
        logits = []
        for k in self.config['attributes'][1:]:
            logits += [self.heads[k](y_)]
        return logits
    
    
class CPSimpleHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        heads = []
        for k in config['attributes']:
            heads += [[k, nn.Linear(config['d_model'], config['n_tokens'][k])]]
        self.heads = nn.ModuleDict(heads)

    def forward(self, h):
        logits = []
        for k in self.config['attributes']:
            logits += [self.heads[k](h)]
        return logits
    
class RemiHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head = nn.Linear(config['d_model'], config['n_vocab'])
        
    def forward(self, h):
        return self.head(h)