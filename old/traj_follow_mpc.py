"""
Runs one iteration of the joint position controller and plots the PID error loops
"""
import numpy as np

import matplotlib.pyplot as plt

from controllers import PositionController, MPC
from envs import RopeEnv
from utils import wrap_traj

EE_SITE = "panda/ee_site"


def main():

    env = RopeEnv()

    arm_controller = MPC(env.physics, "config/panda_arm.yaml")
    hand_controller = PositionController(env.physics, "config/panda_hand.yaml")

    ee_quat = np.array([0, 1, 0, 0])
    start_pos = np.array([-8.52069580e-05, -4.44985263e-01, 1.94763677e-01])
    start_config = arm_controller.ik(start_pos, ee_quat)
    end_config = start_config + 0.01
    grasp_config_hand = np.array([0.04, 0.04])

    start_pos = np.array([-8.52069580e-05, -4.44985263e-01, 1.94763677e-01])
    center_pos = np.array([0, -0.4, 0.05])
    traj = wrap_traj(start_pos, center_pos, 0.5, 0.1)
    # print(traj.shape)
    # plt.plot(traj[:,0],traj[:,1])
    # plt.show()

    with env.physics.reset_context():
        env.physics.named.data.qpos[arm_controller.jnt_names] = start_config
        env.physics.named.data.qpos[hand_controller.jnt_names] = grasp_config_hand

    arm_controller.update()

    act_traj = np.zeros((len(traj), 3))

    arm_controller.set_joint_goal(end_config)

    for i in range(len(traj)):
        print(i)
        arm_controller.set_cartesian_goal(traj[i], ee_quat)
        arm_controller.step()
        act_traj[i, :] = arm_controller.ee_pos
        env.physics.step()

    plt.plot(act_traj[:, 0], act_traj[:, 1])
    plt.plot(traj[:, 0], traj[:, 1])
    plt.show()

    for i in range(arm_controller.dof):
        plt.cla()
        plt.plot(t, e[i, :])
        plt.xlabel("time")
        plt.ylabel("error")
        plt.savefig("graphs/jnt_%d.png" % (i))
    plt.show()


if __name__ == "__main__":
    main()
