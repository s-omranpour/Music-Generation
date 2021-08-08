import os
from typing import List
import numpy as np
import torch
from tqdm.notebook import tqdm
from torch.utils.data import Dataset
from joblib import Parallel, delayed
from deepnote import MusicRepr, Constants

def get_track(file_path, const, window, instruments, mode):
    try:
        seq = MusicRepr.from_file(file_path, const=const)
        tracks = seq.separate_tracks()
        tracks = dict([(inst, tracks[inst]) for inst in tracks if inst in instruments])
        if len(tracks):
            seq = MusicRepr.merge_tracks(tracks)
            if mode == 'cp':
                cp = seq.to_cp()
                return np.pad(cp, ((0, 1), (0, 0)))
            return seq.to_remi(ret='index', add_eos=True)
        return None
    except Exception as e:
        print(e)
        return None

class MidiDataset(Dataset):
    def __init__(
        self, 
        data_dir : str, 
        const : Constants = None, 
        instruments : List[str] = ['piano'],
        mode : str = 'remi',
        pad_value : int = 0,
        max_files : int = 100, 
        window_len : int = 10,
        n_jobs : int = 2):

        super().__init__()

        assert mode in ['cp', 'remi']
        self.const = Constants() if const is None else const
        self.window_len = window_len
        self.mode = mode
        self.pad_value = pad_value

        ## loading midis
        files = sorted(
            list(
                filter(lambda x: x.endswith('.mid'), os.listdir(data_dir))
            )[:max_files], 
            key=lambda x: os.stat(data_dir + x).st_size, 
            reverse=True
        )
        
        # tracks = [get_track(data_dir + file, const, window_len, instruments, mode) for file in tqdm(files)]
        tracks = Parallel(n_jobs=n_jobs)(
            delayed(get_track)(data_dir + file, const, window_len, instruments, mode) for file in tqdm(files)
        )
        

        self.tracks = list(filter(lambda x: x is not None, tracks))
        lens = list(map(len, self.tracks))
        self.lens = [max(0, l - self.window_len) + 1 for l in lens]
        self.cum_lens = [0] + [sum(self.lens[:i+1]) for i in range(len(self.lens))]

    def __len__(self):
        return self.cum_lens[-1]

    def get_idx(self, idx):
        for i, cl in enumerate(self.cum_lens):
            if idx < cl:
                return i-1, idx - self.cum_lens[i-1]

    def __getitem__(self, idx):
        ind, offset = self.get_idx(idx)
        return self.tracks[ind][offset:offset+self.window_len]


    def fn(self, batch):
        def pad(x, l):
            if self.mode == 'cp':
                return np.pad(x, ((0, l), (0, 0)), constant_values=self.pad_value)
            return np.pad(x, (0, l), constant_values=self.pad_value)
        
        x_len = torch.tensor([len(x)-1 for x in batch])
        M = max(x_len)
        return {
            'X': torch.tensor([pad(x[:-1], M-l) for x,l in zip(batch, x_len)]),
            'X_len': x_len,
            'labels': torch.tensor([pad(x[1:], M-l) for x,l in zip(batch, x_len)])
        }