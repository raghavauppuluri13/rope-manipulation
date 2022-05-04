import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F


class RopeDataset(Dataset):
    def __init__(self, data_path, transforms=None):
        super().__init__()
        self.data_path = data_path
        self.num_batches = []
        self.batch_sizes = []
        _, dirnames, _ = next(os.walk(f"{self.data_path}"))
        self.num_batches = len(dirnames)
        sample_batch = dirnames[0]
        _, _, filenames = next(os.walk(f"{self.data_path}/{sample_batch}/obs"))
        self.batch_size = len(filenames) // 2
        self.transforms = transforms

    def __len__(self):
        return self.num_batches * self.batch_size

    def __getitem__(self, idx):
        sample_i = (idx % (2 * self.batch_size)) // 2
        batch_i = idx // (2 * self.num_batches)
        o = Image.open(f"{self.data_path}/batch_{batch_i}/obs/{sample_i}_1.png")
        o_next = Image.open(f"{self.data_path}/batch_{batch_i}/obs/{sample_i}_2.png")
        return self.transforms(o), self.transforms(o_next)


class RopeActionDataset(RopeDataset):
    def __init__(self, data_path, transforms=None, **kwargs):
        self.p_dim = kwargs["rope_site_dim"]
        self.th_dim = kwargs["angle_dim"]
        self.len_dim = kwargs["length_dim"]

        self.p_range = kwargs["rope_site_range"]
        self.th_range = kwargs["angle_range"]
        self.len_range = kwargs["length_range"]

        super().__init__(data_path, transforms=transforms)

    def __getitem__(self, idx):
        o, o_next = super().__getitem__(idx)

        batch_i = idx // (2 * self.num_batches)
        sample_i = (idx % (2 * self.batch_size)) // 2
        acts = np.load(
            f"{self.data_path}/batch_{batch_i}/action_hats.npy", allow_pickle=True
        )
        p_label = acts[sample_i]["rope_site"]
        th_label = acts[sample_i]["angle"]
        len_label = acts[sample_i]["length"]

        p_label = torch.tensor(
            self.discretize(p_label, self.p_range, self.p_dim)
        ).long()
        p_label = F.one_hot(p_label, self.p_dim)

        th_label = torch.tensor(
            self.discretize(th_label, self.th_range, self.th_dim)
        ).long()
        th_label = F.one_hot(th_label, self.th_dim)

        len_label = torch.tensor(
            self.discretize(len_label, self.len_range, self.len_dim)
        ).long()
        len_label = F.one_hot(len_label, self.len_dim)
        return o, o_next, (p_label, th_label, len_label)

    def discretize(self, val, range, disc_steps):
        step = np.abs(range[1] - range[0]) / disc_steps
        out = val / step
        if out.min() < 0:
            out -= out.min()
        return out
