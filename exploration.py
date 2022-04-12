from time import time
from controllers import PositionController
from rope_env import rope_env
from dm_control import viewer

from utils import Interpolator

env = rope_env()
action_spec = env.action_spec()

def sample_random_action():
    act = env.random_state.uniform(
                low=action_spec.minimum,
                high=action_spec.maximum,
            ).astype(action_spec.dtype, copy=False)
    return act

timestep = env.reset()
obs = timestep.observation['rope/position']
goal_pose = obs[-1]
ctrl_steps = 75
controller = PositionController(env,'config/panda.yaml')
interpolator = Interpolator(0.01)
interpolator.set_goal(goal_pose)
for t in range(len(interpolator)): 
    controller.set_cartesian_goal(interpolator.next())
    for i in range(ctrl_steps):
        action = controller.step()
        timestep = env.step(action)
        obs = timestep.observation

