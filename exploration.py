from turtle import setpos
from controllers import PositionController
from rope_env import rope_env
from dm_control import viewer
import matplotlib.pyplot as plt
from utils import Interpolator
import numpy as np

env = rope_env()
action_spec = env.action_spec()

def sample_random_action():
    act = env.random_state.uniform(
                low=action_spec.minimum,
                high=action_spec.maximum,
            ).astype(action_spec.dtype, copy=False)
    return act
#viewer.launch(env)
timestep = env.reset()
print(env.observation_spec())
obs = timestep.observation['rope/position']
goal_pose = obs[:,-1,:]
ctrl_steps = 75
controller = PositionController(env,'config/panda.yaml')
interpolator = Interpolator(0.1)
print(goal_pose)
print(controller.ee_pos)
interpolator.set_goal(controller.ee_pos,goal_pose)

err = []


fig,ax = plt.subplots(nrows=controller.dof,ncols=1)

fig,ax1 = plt.subplots()

for t in range(len(interpolator)): 
    controller.set_cartesian_goal(interpolator.next())
    for i in range(ctrl_steps):
        err.append(controller.setpoint - controller.qpos)
        t = np.arange(len(err))
        action = controller.step()
        timestep = env.step(action)
        plt.plot(t,err)

        for i in range(controller.dof):
            ax[i].cla()
            ax[i].plot(t, np.array(err)[:,i])
            ax[i].set_xlabel("Time")
            ax[i].set_ylabel("Error")
            ax[i].set_title("MPC Error Plot at Joint {}".format(i))
        obs = timestep.observation
        img = obs['frontview']
        ax1.imshow(img[0])
        plt.pause(0.001)