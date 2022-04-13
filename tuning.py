from rope_env import TASK_VIEW, rope_env

from observer import Observer
from planner import ArmPlanner
from controllers import ArmPositionController
import matplotlib.pyplot as plt
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
obs = timestep.observation['rope/position']
goal_pose = obs[:,5,:]

observer = Observer(env.physics,obs_camera=TASK_VIEW,show=False)

arm_controller = ArmPositionController(env,'config/panda.yaml')
arm_planner = ArmPlanner(env,arm_controller,observer,interpolator_step=0.02) 

err = []
arm_planner.set_goal(goal_pose)

while not arm_planner.done:
    err.append(arm_controller.setpoint - arm_controller.qpos)
    arm_action = arm_planner.get_action()
    timestep = env.step(arm_action)
    observer.step(timestep)

fig,ax = plt.subplots(nrows=arm_controller.dof,ncols=1)
t = np.arange(len(err))
for i in range(arm_controller.dof):
    ax[i].cla()
    ax[i].plot(t, np.array(err)[:,i])
    ax[i].set_xlabel("Time")
    ax[i].set_ylabel("Error")
    ax[i].set_title("MPC Error Plot at Joint {}".format(i))
    ax[i].grid(True)

pos_err = np.sum((arm_controller.ee_pos - goal_pose) ** 2)
print("final pos_error: %f" % pos_err)
plt.show()