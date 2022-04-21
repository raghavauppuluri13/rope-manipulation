import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class RopeDataset(Dataset):
    def __init__(self, data_path, transforms=None):
        super().__init__()
        self.data_path = data_path
        _, dirnames,_ = next(os.walk(f"{self.data_path}"))
        self.num_batches = len(dirnames)
        sample_batch = dirnames[0]
        _, _,filenames = next(os.walk(f"{self.data_path}/{sample_batch}/obs"))
        self.batch_size = len(filenames)
        self.transforms = transforms

    def __len__(self):
        return self.num_batches * self.batch_size

    def __getitem__(self, idx):
        sample_i = (idx % (2 * self.batch_size)) // 2
        state_i = idx % 2
        batch_i = idx // (2 * self.num_batches)
        img = torch.Tensor(Image.open(f"{self.data_path}/batch_{batch_i}/obs/{sample_i}_{state_i}.png"))

        return self.transforms(img)


class RopeActionDataset(RopeDataset):
    def __init__(self, data_path, transforms=None):
        super().__init__(data_path, transforms=transforms)

    def __getitem__(self, idx):
        img = super().__getitem__(idx)

        batch_i = idx // (2 * self.num_batches)
        act = torch.Tensor(np.load(f"{self.data_path}/batch_{batch_i}/action_hats.npy"))

        return img,act