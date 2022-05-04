"""Generate training datasets for pick/place with rope"""
import datetime
import json
import os
from PIL import Image
from rope_env import rope_env, FRONT_VIEW, EYE_IN_HAND_VIEW, TASK_VIEW
from collections import deque
from observer import Observer
from planner import ArmPlanner, GripperPlanner
from controllers import ArmPositionController, GripperController
import numpy as np

ARM = "arm"
GRIP = "grip"

_WORKSPACE = [(-0.3, 0.3), (-0.3, 0.3)]

_DATA_PATH = os.path.abspath("../datasets")
_NAME = datetime.datetime.today().strftime("%Y-%m-%d-%H:%M:%S")

_BATCHES = 1000
_BATCH_SIZE = 15

# _SEED = 321

_ACTION_HAT_SPECS = {
    "rope_site": (3, 6),
    "angle": (-3 * np.pi / 4, 3 * np.pi / 4),
    "length": (0.05, 0.15),
}


def _polar_to_xyz(r, angle):
    return [r * np.cos(angle), r * np.sin(angle), 0]


def _valid_act_hat(x):
    if x[0] >= _WORKSPACE[0][0] and x[0] <= _WORKSPACE[0][1]:
        if x[1] >= _WORKSPACE[1][0] and x[1] <= _WORKSPACE[1][1]:
            return True
    return False


def get_rand_action_hat(SPECS):
    angle = np.random.uniform(*SPECS["angle"])
    len = np.random.uniform(*SPECS["length"])
    act_pos = _polar_to_xyz(len, angle)
    while not _valid_act_hat(act_pos):
        angle = np.random.uniform(*SPECS["angle"])
        len = np.random.uniform(*SPECS["length"])
    return {
        "rope_site": np.random.randint(*SPECS["rope_site"]),
        "angle": angle,
        "length": len,
    }


if __name__ == "__main__":

    os.mkdir(f"{_DATA_PATH}/{_NAME}")

    with open(f"{_DATA_PATH}/{_NAME}/hparams.json", "a", encoding="utf-8") as file:
        json.dump(_ACTION_HAT_SPECS, file)

    # np.random.seed(_SEED)

    env = rope_env()

    observer = Observer(env, obs_camera=TASK_VIEW, show=False)

    arm_controller = ArmPositionController(env, "config/panda.yaml")
    arm_planner = ArmPlanner(env, arm_controller, observer, interpolator_step=0.1)

    grip_controller = GripperController(env, "config/panda_hand.yaml")
    grip_planner = GripperPlanner(env, grip_controller, observer, interpolator_step=0.2)

    def sample_setpoints(timestep):
        action_hat = get_rand_action_hat(_ACTION_HAT_SPECS)
        angle = action_hat["angle"]
        l = action_hat["length"]

        obs = timestep.observation["rope/position"]
        hover = obs[:, action_hat["rope_site"], :] + [0, 0, 0.2]
        pick = obs[:, action_hat["rope_site"], :]
        # hover_pick = obs[:,action_hat['rope_site'],:] + [0,0,0.2]
        place = pick + _polar_to_xyz(l, angle) + [0, 0, 0.1]
        retract = place + [0, 0, 0.6]
        return action_hat, [
            (ARM, retract),
            (GRIP, grip_controller.open),
            (ARM, place),
            # (ARM,hover_pick),
            (GRIP, grip_controller.close),
            (ARM, pick),
            (ARM, hover),
            (GRIP, grip_controller.open),
        ]

    for i in range(_BATCHES):
        observer.reset()
        timestep = env.reset()
        arm_controller.set_phys(env.physics)
        grip_controller.set_phys(env.physics)
        os.mkdir(f"{_DATA_PATH}/{_NAME}/batch_{i}")
        os.mkdir(f"{_DATA_PATH}/{_NAME}/batch_{i}/obs")
        action_hats = []
        for j in range(_BATCH_SIZE):
            action_hat, setpoints = sample_setpoints(timestep)
            action_hats.append(action_hat)
            pre_act_obs = timestep.observation[TASK_VIEW][0]
            while setpoints or not arm_planner.done or not grip_planner.done:
                if arm_planner.done and grip_planner.done:
                    planner, setp = setpoints.pop()
                    if planner == ARM:
                        arm_planner.set_goal(setp)
                    elif planner == GRIP:
                        grip_planner.set_goal(setp)
                arm_action = arm_planner.get_action()
                grip_action = grip_planner.get_action()
                action = arm_action + grip_action
                timestep = env.step(action)
                observer.step(timestep)
            post_act_obs = timestep.observation[TASK_VIEW][0]
            Image.fromarray(pre_act_obs).save(
                f"{_DATA_PATH}/{_NAME}/batch_{i}/obs/{j}_1.png"
            )
            Image.fromarray(post_act_obs).save(
                f"{_DATA_PATH}/{_NAME}/batch_{i}/obs/{j}_2.png"
            )
            for action_hat in action_hats:
                rope_site = action_hat["rope_site"]
                action_hat["pick_pos"] = env.physics.named.data.xpos[
                    f"rope/CB{rope_site}"
                ]
            np.save(f"{_DATA_PATH}/{_NAME}/batch_{i}/action_hats.npy", action_hats)
        observer.save(f"{_DATA_PATH}/{_NAME}/batch_{i}/batch.gif")
