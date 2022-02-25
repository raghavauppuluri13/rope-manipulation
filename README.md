# rope-manipulation

Manipulating rope to different goal states. From high-dimensional images, generate

## Quick Start

1. Create conda environment + install dm_control

```
conda env create --file environment.yaml
```

Install [dm_control](https://github.com/deepmind/dm_control#requirements-and-installation)

1. Run the rope_env

```
python rope_env.py
```

## To-do

- [x] Basic env with rope + panda
- [x] Position Controller for Panda
- [ ] Torque Controller for Panda
- [ ] Learn low-dim plan representation from image start/goal configurations according to [Learning Plannable Representations with Causal InfoGAN](https://arxiv.org/abs/1807.09341)
- [ ] Execute plans using MPC