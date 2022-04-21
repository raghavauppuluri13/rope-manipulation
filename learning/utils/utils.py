import glob

import numpy as np
import torch
from PIL import Image

dtype_long = torch.LongTensor

def get_normalization_params(dataset_path):
    images = []
    for f in glob.iglob(f"{dataset_path}/obs*"):
        images.append(np.asarray(Image.open(f)))
    images = np.array(images)
    print(images.shape)
    data_mean = images.mean(axis=(0, 1, 2)) / 255
    data_std = images.std(axis=(0, 1, 2)) / 255
    return data_mean, data_std

