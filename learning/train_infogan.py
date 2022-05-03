# Load Data
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import random
import yaml
import wandb
from trainers import CausalInfoganTrainer
import torch

torch.manual_seed(512)
random.seed(512)

PARAMS = '/home/ruppulur/repos/rope-manipulation/learning/config/causal_infogan.yaml' 

with open(PARAMS, "r") as f:
    params = yaml.safe_load(f)


trainer = CausalInfoganTrainer(**params)
#trainer.train()
trainer.display_sample()