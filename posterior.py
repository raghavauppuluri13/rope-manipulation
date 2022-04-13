from matplotlib import pyplot as plt
import numpy as np
from rope_env import rope_env

from dm_control import viewer


env = rope_env()
#viewer.launch(env)

with env.physics.reset_context():
    env.physics.named.model.jnt_axis['rope/CB9'] = [0,0,0,1]
obs = env.physics.render()
print(env.physics.named.model.body_pos)
obs = env.step(np.zeros_like(env.action_spec()))
obs = obs.observation['robot0_robotview'][0]

plt.imshow(obs)
plt.show()