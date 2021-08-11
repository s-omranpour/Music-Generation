import numpy as np
import torch
from tqdm.notebook import tqdm
from torch import nn
import pytorch_lightning as pl
from deepnote import Constants, MusicRepr

from src.modules import CPEmbedding, CPSimpleHead, sampling, VanillaTransformer


class CPSimpleTransformer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        
        self.embedding = CPEmbedding(config['embedding'])
        self.transformer = VanillaTransformer(config['transformer'])
        self.head = CPSimpleHead(config['head'])

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=0.)
        return [opt]#, [sch]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, x_len=None):
        emb = self.embedding(x.long())
        h = self.transformer(
            src=emb, 
            src_len=x_len
        )
        return self.head(h)

    def compute_loss(self, predict, target, mask):
        loss = self.loss_func(predict.transpose(1,2), target.long())
        loss = loss * mask
        loss = torch.sum(loss) / torch.sum(mask)
        return loss
    
    def compute_acc(self, predict, target, mask):
        correct = torch.argmax(predict, dim=-1) == target
        correct = correct * mask
        return torch.sum(correct) / torch.sum(mask)
        

    def step(self, batch, mode='train'):
        x, target, lengths = batch['X'], batch['labels'], batch['X_len']
        clm_logits = self.forward(x, lengths)
        ae_logits = self.head(self.embedding(x.long()))
        
        
        ## mask
        loss_mask = torch.zeros_like(target).to(x.device)
        for i, l in enumerate(lengths):
            loss_mask[i, :l, 0] = 1.                                       ## ttype
            loss_mask[i, :l, 1:4] = (target[i, :l, :1] == 0).repeat(1,3)   ## metrical
            loss_mask[i, :l, 4:8] = (target[i, :l, :1] == 1).repeat(1,4)   ## note
            
        # loss
        total_clm_loss = 0
        total_ae_loss = 0
        total_clm_acc = []
        total_ae_acc = []
        for i,k in enumerate(self.config['attributes']):
            clm_loss = self.compute_loss(clm_logits[i],  target[..., i], loss_mask[..., i])
            clm_acc = self.compute_acc(clm_logits[i],  target[..., i], loss_mask[..., i])
            ae_loss = self.compute_loss(ae_logits[i],  x[..., i], loss_mask[..., i])
            ae_acc = self.compute_acc(ae_logits[i],  x[..., i], loss_mask[..., i])
            total_clm_loss += clm_loss
            total_ae_loss += ae_loss
            total_clm_acc += [clm_acc]
            total_ae_acc += [ae_acc]
            self.log(mode +'_' + k + '_clm_loss', clm_loss.item())
            self.log(mode +'_' + k + '_ae_loss', ae_loss.item())
            self.log(mode +'_' + k + '_clm_acc', clm_acc.item())
            self.log(mode +'_' + k + '_ae_acc', ae_acc.item())
        
        total_loss = total_clm_loss + total_ae_loss
        self.log(mode + '_clm_loss', total_clm_loss.item())
        self.log(mode + '_ae_loss', total_ae_loss.item())
        self.log(mode + '_loss', total_loss.item())
        self.log(mode + '_clm_acc', sum(total_clm_acc).item() / len(total_clm_acc))
        self.log(mode + '_ae_acc', sum(total_ae_acc).item() / len(total_ae_acc))
        return total_loss

    def training_step(self, batch, batch_idx):
        return self.step(batch)

    def validation_step(self, batch, batch_idx):
        self.step(batch, mode='val')

    def sample_output(self, logits, cuda=False, gen_conf=None):
        word = []
        for i,k in enumerate(self.config['attributes']):
            word += [
                int(torch.argmax(logits[i][0, -1]))
#                 sampling(
#                     logits[i][0, -1], 
#                     t=1. if gen_conf is None else gen_conf['t_'+k],
#                     p=1. if gen_conf is None else gen_conf['p_'+k]
#                 )
            ]
        return np.array(word)        

    @torch.no_grad()
    def generate(self, prompt=None, max_len=1000, window=1000, cuda=False, gen_conf=None):
        self.eval()
        self.to('cuda' if cuda else 'cpu')
        
        if prompt is None:
            const = Constants()
            init = [[0, 0, 0, 0, 0, 0, 0, 0]] ## bar
        else:
            const = prompt.const
            init = prompt.to_cp().tolist()

        for _ in tqdm(range(max_len)):
            s = max(0, len(init) - window)
            input_ = torch.tensor(init[s:]).unsqueeze(0).to('cuda' if cuda else 'cpu').long()
            logits = self.forward(input_)
            
            # sample others
            next_word = self.sample_output(logits, cuda=cuda, gen_conf=gen_conf)
            init += [next_word]

        return np.array(init)