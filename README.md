# rope-manipulation

Manipulating rope to different goal states. From high-dimensional images, generate

## Quick Start

1. Create conda environment + install dm_control

```
conda env create --file environment.yaml
```

Install [dm_control](https://github.com/deepmind/dm_control#requirements-and-installation)

2. Run the pd_control.py to get the PD controller rollout visualized in env_render.gif

```
python pd_control.py
```

3. Run the tuning.py to get the MPC controller error graphs (written to `graphs`) for a single setpoint

```
python tuning.py
```
4. Run `traj_follow_mpc.py` to get the MPC controller error following a trajectory 


## Run 

## To-do

- [x] Basic env with rope + panda
- [x] Position Controller for Panda
- [ ] Learn low-dim plan representation from image start/goal configurations according to [Learning Plannable Representations with Causal InfoGAN](https://arxiv.org/abs/1807.09341)
- [ ] Execute plans using MPC