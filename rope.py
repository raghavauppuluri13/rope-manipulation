import numpy as np
from dm_control import manipulation
import matplotlib.pyplot as plt
env = _rope() 
action_spec = env.action_spec()

def sample_random_action():
    return env.random_state.uniform(
        low=action_spec.minimum,
        high=action_spec.maximum,
    ).astype(action_spec.dtype, copy=False)


# Step the environment through a full episode using random actions and record
# the camera observations.
frames = []
timestep = env.reset()
frames.append(timestep.observation["front_close"])
while not timestep.last():
    timestep = env.step(sample_random_action())
    frames.append(timestep.observation["front_close"])
    plt.imshow(frames[-1][0])
    plt.pause(0.001)
all_frames = np.concatenate(frames, axis=0)