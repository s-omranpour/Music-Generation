import numpy as np
import torch
from tqdm.notebook import tqdm
from torch import nn
import pytorch_lightning as pl
from deepnote import Constants, MusicRepr

from src.modules import sampling, RemiHead, RemiEmbedding, LinearTransformer


class RemiLinearTransformer(pl.LightningModule):
    def __init__(self, config, is_training=True):
        super().__init__()
        self.config = config
        self.is_training = is_training
        self.embedding = RemiEmbedding(config['embedding'])
        self.transformer = LinearTransformer(config['transformer'], is_training)
        self.head = RemiHead(config['head'])
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=0.)
        return [opt], [sch]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, x_len=None, y=None, state=None):
        emb = self.embedding(x)
        h, state = self.transformer(emb, x_len, state)
        logits = self.head(h)
        loss = None if y is None else self.compute_loss(logits,  y, x_len)
        return h, logits, state, loss

    def compute_loss(self, predict, target, lengths):
        loss_mask = torch.zeros_like(target).to(target.device)
        for i, l in enumerate(lengths):
            loss_mask[i, :l] = 1.

        loss = self.loss_func(predict.transpose(1,2), target.long())
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss

    def step(self, batch, mode='train'):
        _, _, _, loss  = self.forward(x=batch['X'], x_len=batch['X_len'], y=batch['labels'])
        self.log(mode+'_loss', loss.item())
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch)
    
    def validation_step(self, batch, batch_idx):
        self.step(batch, 'val')

    @torch.no_grad()
    def generate(self, prompt=None, max_len=1000, cuda=False, top_p=1., temperature=1.):
        self.to('cuda' if cuda else 'cpu')
        
        if prompt is None:
            const = Constants()
            init = [0] ## bar
        else:
            const = prompt.const
            init = prompt.to_remi(ret='index')

        final_res = []
        state = None
        h = None
        init_t = torch.tensor(init).long().to('cuda' if cuda else 'cpu')

        for step in range(len(init)):
            input_ = init_t[step:step+1].unsqueeze(0)
            final_res.append(init[step])
            _, logits, state, _ = self.forward(input_, state=state)

        for _ in tqdm(range(max_len)):
            # sample others
            next_tok = sampling(
                logits[-1], 
                t=temperature,
                p=top_p
            )
            final_res.append(next_tok)

            # forward
            input_ = torch.tensor([next_tok]).long().to('cuda' if cuda else 'cpu').unsqueeze(0)
            _, logits, state, _ = self.forward(input_, state=state)

        return np.array(final_res)