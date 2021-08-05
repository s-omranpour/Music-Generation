import numpy as np
import torch
from tqdm.notebook import tqdm
from torch import nn
import pytorch_lightning as pl
from deepnote import Constants, MusicRepr

from src.modules import CPEmbedding, CPHead, sampling, VanillaTransformer


class CPTransformer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        
        self.embedding = CPEmbedding(config['embedding'])
        self.transformer = VanillaTransformer(config['transformer'])
        self.head = CPHead(config['head'])
        self.head.register_type_embedding(self.embedding)
        

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=0.)
        return [opt]#, [sch]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

#     def forward_hidden(self, x, lengths=None, training=False):
#         emb = self.embedding(x)
        
# #         if lengths is None:
# #             mask = None
# #         else:
# #             mask = torch.zeros(x.shape[0], x.shape[1]).to(x.device)
# #             for i,l in enumerate(lengths):
# #                 mask[i, :l] = 1
#         h = self.transformer(inputs_embeds=emb).last_hidden_state #attention_mask=mask
#         y_type = self.head.forward_type(h)
#         return h, y_type

    def forward_hidden(self, x, x_len=None):
        emb = self.embedding(x.long())
        h = self.transformer(
            src=emb, 
            src_len=x_len
        )
        return h, self.head.forward_type(h)

    def compute_loss(self, predict, target, loss_mask):
        # reshape (b, s, f) -> (b, f, s)
        loss = self.loss_func(predict.transpose(1,2), target.long())
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss

    def training_step(self, batch, batch_idx):
        x, target, lengths = batch['X'], batch['labels'], batch['X_len']
        h, y_type  = self.forward_hidden(x, lengths)
        logits = [y_type] + self.head(h, target[..., 0])
        
        ## mask
        loss_mask = torch.zeros_like(target).to(x.device)
        for i, l in enumerate(lengths):
            loss_mask[i, :l, 0] = 1.                                       ## ttype
            loss_mask[i, :l, 1:4] = (target[i, :l, :1] == 0).repeat(1,3)   ## metrical
            loss_mask[i, :l, 4:7] = (target[i, :l, :1] == 1).repeat(1,3)   ## note
            
        # loss
        total_loss = 0
        for i,k in enumerate(self.config['attributes']):
            loss = self.compute_loss(logits[i],  target[..., i], loss_mask[..., i])
            total_loss += loss
            self.log('train_'+k+'_loss', loss.item())
        self.log('train_loss', total_loss.item())
        return total_loss

    def forward_output_sampling(self, h, y_type, cuda=False, gen_conf=None):
        '''
        for inference
        '''
        # sample type
        y_type_logit = y_type[0, -1]
        cur_word_type = sampling(
            y_type_logit, 
            t=1. if gen_conf is None else gen_conf['t_ttype'],
            p=1. if gen_conf is None else gen_conf['p_ttype']
        )

        type_word_t = torch.tensor([cur_word_type]*h.shape[1]).unsqueeze(0).to('cuda' if cuda else 'cpu').long()
        logits = self.head(h, type_word_t)
        word = [cur_word_type]
        for i,k in enumerate(self.config['attributes'][1:]):
            word += [
                sampling(
                    logits[i][0, -1], 
                    t=1. if gen_conf is None else gen_conf['t_'+k],
                    p=1. if gen_conf is None else gen_conf['p_'+k]
                )
            ]
        return np.array(word)        

    @torch.no_grad()
    def generate(self, prompt=None, max_len=1000, cuda=False, gen_conf=None):
        self.eval()
        self.to('cuda' if cuda else 'cpu')
        
        if prompt is None:
            const = Constants()
            init = [[0, 0, 0, 0, 0, 0, 0]] ## bar
        else:
            const = prompt.const
            init = prompt.to_cp()
            init = np.concatenate([init[:, :4], init[:, 5:]], axis=1).tolist()

        for _ in tqdm(range(max_len)):
            # forward
            input_ = torch.tensor(init).unsqueeze(0).to('cuda' if cuda else 'cpu').long()
            h, y_type = self.forward_hidden(input_)
            
            # sample others
            next_word = self.forward_output_sampling(h, y_type, cuda=cuda, gen_conf=gen_conf)
            init += [next_word]

        return np.array(init)