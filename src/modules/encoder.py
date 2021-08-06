import torch
from torch import nn

from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.builders import RecurrentEncoderBuilder
from fast_transformers.masking import TriangularCausalMask, LengthMask

class LinearTransformer(nn.Module):
    def __init__(self, config, is_training):
        super().__init__()
        self.is_training = is_training
        d_head = config['d_model']//config['n_head']
        if is_training:
            self.encoder = TransformerEncoderBuilder.from_kwargs(
                n_layers=config['n_layer'],
                n_heads=config['n_head'],
                query_dimensions=d_head,
                value_dimensions=d_head,
                feed_forward_dimensions=config['d_inner'],
                activation=config['activation'],
                dropout=config['dropout'],
                attention_type='causal-linear',
            ).get()
        else:
            self.encoder = RecurrentEncoderBuilder.from_kwargs(
                n_layers=config['n_layer'],
                n_heads=config['n_head'],
                query_dimensions=d_head,
                value_dimensions=d_head,
                feed_forward_dimensions=config['d_inner'],
                activation=config['activation'],
                dropout=config['dropout'],
                attention_type='causal-linear'
            ).get()
    
    def forward(self, h, lengths=None, state=None):
        if self.is_training:
            attn_mask = TriangularCausalMask(h.size(1), device=h.device)
            length_mask = None if lengths is None else LengthMask(lengths, device=h.device)
            h = self.encoder(h, attn_mask, length_mask)
            return h, None

        h, state = self.encoder(h.squeeze(0), state=state)
        return h, state
    
    
class VanillaTransformer(nn.Module):
    def __init__(self,  config):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=config['d_model'], 
            nhead=config['n_head'], 
            dim_feedforward=config['d_inner'], 
            dropout=config['dropout'], 
            activation=config['activation']
        )
        
        self.encoder = nn.TransformerEncoder(layer, num_layers=config['n_layer'])
    
    def forward(self, src, src_len=None):
        def _generate_square_subsequent_mask(sz):
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            return mask

        if src_len is not None:
            length_mask = torch.ones(src.shape[0], src.shape[1]).to(src.device).bool()
            for i, l in enumerate(src_len):
                length_mask[i, :l] = False
        else:
            length_mask = None
        
        h = src.permute(1,0,2)
        att_mask = _generate_square_subsequent_mask(h.shape[0]).to(src.device)
        h = self.encoder(h, mask=att_mask, src_key_padding_mask=length_mask)
        return h.permute(1,0,2)
    

class VanillaRNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rnn = nn.GRU(
            input_size=config['d_model'], 
            hidden_size=config['d_model'], 
            num_layers=config['n_layer'], 
            batch_first=True, 
            dropout=config['dropout'],
            bidirectional=config['bidirectional']
        )
    
    def forward(self, x, h0=None):
        if h0 is None:
            d = 2 if self.config['bidirectional'] else 1
            h0 = torch.zeros(d*self.config['n_layer'], x.shape[0], self.config['d_model']).to(x.device)
        return self.rnn(x, h0)