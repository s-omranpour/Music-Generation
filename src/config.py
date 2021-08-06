from typing import List, Dict
from deepnote import Constants

ATTRIBUTES = ['ttype' , 'barbeat', 'tempo', 'chord', 'pitch', 'duration', 'velocity']
EMB_SIZES = {
    'ttype' : 8, 
    'barbeat' : 32,
    'tempo' : 32,
    'chord' : 128,
    'pitch' : 128,
    'duration' : 32,
    'velocity' : 32
}

def make_config(
    const : Constants,
    mode : str = 'cp',
    model : str = 'transformer',
    d_model : int = 512, 
    max_len : int = 1000,
    dropout : float = 0.1, 
    lr : float = 1e-4,
    tie_emb : bool = False,
    bidirectional : bool = True,
    attributes : List[str] = ATTRIBUTES,
    emb_sizes : Dict[str, int] = EMB_SIZES, 
    pos_emb : bool = True, 
    n_layer : int = 1, 
    n_head : int = 8, 
    d_inner : int = 512, 
    activation : str = 'gelu'):
    
    assert mode in ['cp', 'remi']
    assert model in ['rnn', 'transformer']
    
    
    config = {
        'lr' : lr,
        'embedding': {
            'd_model' : d_model, 
            'dropout' : dropout,
            'max_len' : max_len,
            'positional_embedding' : pos_emb
        },
        'head' : {
            'd_model' : d_model
        }
    }
    if model == 'transformer':
        config['transformer'] = {
            'd_model' : d_model,
            'n_layer' : n_layer,
            'n_head' : n_head,
            'd_inner' : d_inner,
            'dropout' : dropout,
            'activation' : activation
        }
    else:
        config['rnn'] = {
            'd_model' : d_model,
            'n_layer' : n_layer,
            'dropout' : dropout,
            'bidirectional' : bidirectional
        }
            
    if mode == 'remi':
        config['tie_emb'] = tie_emb
        config['embedding']['n_vocab'] = len(const.all_tokens)
        config['head']['n_vocab'] = len(const.all_tokens)
    else:
        n_tokens = {
            'ttype' : 2, 
            'barbeat' : const.n_bar_steps,
            'tempo' : const.num_tempo_bins + 1,
            'chord' : len(const.chords) + 1,
            'pitch' : 128,
            'duration' : const.n_bar_steps,
            'velocity' : const.num_velocity_bins
        }
        config['attributes'] = attributes
        config['embedding']['attributes'] = config['head']['attributes']= attributes
        config['embedding']['n_tokens'] = config['head']['n_tokens'] = n_tokens
        config['embedding']['emb_sizes'] = config['head']['emb_sizes'] = emb_sizes
    return config
        