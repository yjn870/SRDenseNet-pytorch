import random
import h5py
import numpy as np
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, h5_file, patch_size, scale):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file
        self.patch_size = patch_size // scale
        self.scale = scale

    def random_crop(self, lr, hr):
        x = random.randint(0, lr.shape[-1] - self.patch_size)
        y = random.randint(0, lr.shape[-2] - self.patch_size)
        lr = lr[:, y:y+self.patch_size, x:x+self.patch_size]
        hr = hr[:, y*self.scale:y*self.scale+self.patch_size*self.scale, x*self.scale:x*self.scale+self.patch_size*self.scale]
        return lr, hr

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            lr = np.expand_dims(f['lr'][str(idx)][::].astype(np.float32), 0) / 255.0
            hr = np.expand_dims(f['hr'][str(idx)][::].astype(np.float32), 0) / 255.0
            lr, hr = self.random_crop(lr, hr)
            return lr, hr

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class EvalDataset(Dataset):
    def __init__(self, h5_file):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            lr = np.expand_dims(f['lr'][str(idx)][::].astype(np.float32), 0) / 255.0
            hr = np.expand_dims(f['hr'][str(idx)][::].astype(np.float32), 0) / 255.0
            return lr, hr

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])
