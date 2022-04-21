# Load Data
import os

from datasets.data_loader import RopeDataset

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import random

import torch
import torch.optim as optim
import torchvision.transforms as transforms
import wandb
from trainer import Trainer
from utils import get_normalization_params

torch.manual_seed(32)
random.seed(32)


data_path = os.path.abspath("../datasets")
_, dirnames, _ = next(os.walk(data_path))
dirnames.sort()

train_data_path = os.path.join(data_path, dirnames[-1])

data_mean, data_std_dev = get_normalization_params(train_data_path + "/images")
norm_transform = transforms.Normalize(data_mean, 1)
preprocess = transforms.Compose([norm_transform,transforms.CenterCrop(320,320),transforms.Resize((64,64)),transforms.RandomRotation(180)])

infogan_dataset = RopeDataset(train_data_path, preprocess)
hparams = {
    "batch_size_train": 10,
    "batch_size_test": 1000,
    "lr": 0.0002,
    "lr_drop": 0.9,
    "epochs": 35,
    "margin": 0.5,
}
device = torch.device("cuda")

optimizer = optim.Adam(model.parameters(), lr=hparams["lr"], weight_decay=1e-4)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=hparams["batch_size_train"], shuffle=True
)

trainer = Trainer(device, model, train_loader, optimizer, hparams)

wandb.init(
    entity="raghavauppuluri", project="rope-manipulation-training", config=hparams
)

wandb.watch(trainer.model)

for epoch in range(1, hparams["epochs"] + 1):
    trainer.train()
    trainer.test()

wandb.unwatch(model)
