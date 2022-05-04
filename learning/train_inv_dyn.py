# Load Data
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import random
import yaml
from trainers import CausalInfoganTrainer, InverseDynamicsTrainer
import torch

torch.manual_seed(512)
random.seed(512)

PARAMS = "/home/ruppulur/repos/rope-manipulation/learning/config/inverse_dynamics.yaml"

with open(PARAMS, "r") as f:
    params = yaml.safe_load(f)


trainer = InverseDynamicsTrainer(**params)
trainer.display_sample()
trainer.train_test()
