"""
Run Rope Env
"""
import numpy as np
from dm_control.mujoco.wrapper.mjbindings import mjlib


import matplotlib.pyplot as plt

from controllers import PositionController
from envs import RopeEnv
from observer import Observer
from utils import lerp

FRAMES = 60


def main():

    env = RopeEnv()
    renderer = Observer(env.physics, camera_id=-1, n_frames=FRAMES)

    arm_controller = PositionController(env.physics, "config/panda_arm.yaml")
    hand_controller = PositionController(env.physics, "config/panda_hand.yaml")

    ee_quat = np.array([0, 1, 0, 0])
    start_config = np.array(
        [
            0,
            np.pi / 16.0,
            0.00,
            -np.pi / 2.0 - np.pi / 3.0,
            0.00,
            np.pi - 0.2,
            np.pi / 4,
        ]
    )
    end_pos = np.array([0.3, -0.4, 0])
    end_config = arm_controller.ik(end_pos, ee_quat)
    grasp_config_hand = np.array([0.04, 0.04])
    traj = np.array([lerp(start_config, end_config, t) for t in np.arange(0, 1, 0.01)])

    with env.physics.reset_context():
        env.physics.named.data.qpos[arm_controller.jnt_names] = start_config
        env.physics.named.data.qpos[hand_controller.jnt_names] = grasp_config_hand

    plt.imsave("start.png", renderer.render())

    arm_controller.update()
    print(arm_controller.ee_pos)

    for i in range(len(traj)):
        arm_controller.set_joint_goal(traj[i])
        hand_controller.set_joint_goal(grasp_config_hand)
        arm_controller.run_controller()
        hand_controller.run_controller()
        env.physics.step()
        renderer.step()

    plt.imsave("end.png", renderer.render())
    renderer.save()


if __name__ == "__main__":
    main()
