import numpy as np
import torch
from tqdm.notebook import tqdm
from torch import nn
import pytorch_lightning as pl
from deepnote import Constants, MusicRepr

from src.modules import RemiEmbedding, RemiHead, sampling, VanillaTransformer


class RemiTransformer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        
        self.embedding = RemiEmbedding(config['embedding'])
        self.transformer = VanillaTransformer(**config['transformer'])
        self.head = RemiHead(config['head'])
        if config['tie_emb']:
            self.head.head.weight = self.embedding.emb.weight.T
        

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=0.)
        return [opt]#, [sch]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, x_len=None, y=None):
        emb = self.embedding(x.long())
        h = self.transformer(
            src=emb, 
            src_len=x_len
        )
        logits = self.head(h)
        loss = None if y is None else self.compute_loss(logits,  y, x_len)
        return logits, loss

    def compute_loss(self, predict, target, lengths):
        loss_mask = torch.zeros_like(target).to(target.device)
        for i, l in enumerate(lengths):
            loss_mask[i, :l] = 1.

        loss = self.loss_func(predict.transpose(1,2), target.long())
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss

    def step(self, batch, mode='train'):
        logits, loss  = self.forward(batch['X'], batch['X_len'], batch['labels'])
        self.log(mode+'_loss', loss.item())
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.step(batch)
    
    def validation_step(self, batch, batch_idx):
        self.step(batch, 'val')       

    @torch.no_grad()
    def generate(self, prompt=None, max_len=1000, cuda=False, top_p=1., temperature=1.):
        self.eval()
        self.to('cuda' if cuda else 'cpu')
        
        if prompt is None:
            const = Constants()
            init = [0] ## bar
        else:
            const = prompt.const
            init = prompt.to_remi(ret='index')

        for _ in tqdm(range(max_len)):
            # forward
            input_ = torch.tensor(init).unsqueeze(0).to('cuda' if cuda else 'cpu').long()
            logits, _ = self.forward(input_)
            
            next_tok = sampling(
                logits[0,-1], 
                t=temperature,
                p=top_p
            )
            init += [next_tok]

        return np.array(init)