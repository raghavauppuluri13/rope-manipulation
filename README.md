# rope-manipulation

Manipulating rope to different goal states.

## Quick Start

1. Create conda environment + install dm_control

```
conda env create --file environment.yaml
```

Install [dm_control](https://github.com/deepmind/dm_control#requirements-and-installation)


2. Run the tuning.py to view the PID controller tuning results

```
python tuning.py
```

4. Run `exploration.py` to run a sample rollout interaction with the rope
```
python exploration.py
```

## Run 

## To-do

- [x] Basic env with rope + panda
- [x] Position Controller for Panda
- [ ] Learn low-dim plan representation from image start/goal configurations according to [Learning Plannable Representations with Causal InfoGAN](https://arxiv.org/abs/1807.09341)
- [ ] Execute plans using MPC