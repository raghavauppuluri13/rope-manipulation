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


# viewer.launch(env)

timestep = env.reset()
obs = timestep.observation["rope/position"]
goal_pose = obs[:, 5, :] + [0, 0, 0.1]

observer = Observer(env, obs_camera=TASK_VIEW, show=False)

arm_controller = ArmPositionController(env, "config/panda.yaml")
arm_planner = ArmPlanner(env, arm_controller, observer, interpolator_step=0.02)

err = []
arm_planner.set_goal(goal_pose)

while not arm_planner.done:
    err.append(arm_controller.setpoint - arm_controller.qpos)
    arm_action = arm_planner.get_action()
    timestep = env.step(arm_action)
    observer.step(timestep)

t = np.arange(len(err))
for i in range(arm_controller.dof):
    plt.cla()
    plt.plot(t, np.array(err)[:, i])
    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.title("PID Error Plot at Joint {}".format(i))
    plt.grid(True)
    plt.savefig(f"tune_jnt_{i}.png")

pos_err = np.sum((arm_controller.ee_pos - goal_pose) ** 2)
print("final pos_error: %f" % pos_err)
observer.save("../gifs/tuning.gif")
