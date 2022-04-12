"""
Runs one iteration of the joint position controller and plots the PID error loops
"""
import numpy as np

import matplotlib.pyplot as plt

from controllers import PositionController, MPC
from envs import RopeEnv

SIM_STEPS = 30

def main():

    env = RopeEnv()

    arm_controller = MPC(env.physics, "config/panda_arm.yaml")
    arm_controller.steps = 1
    hand_controller = PositionController(env.physics, "config/panda_hand.yaml")

    ee_quat = np.array([0, 1, 0, 0])
    start_pos = np.array([-8.52069580e-05, -4.44985263e-01, 1.94763677e-01])
    start_config = arm_controller.ik(start_pos, ee_quat)
    end_config = start_config + 0.01
    grasp_config_hand = np.array([0.04, 0.04])

    with env.physics.reset_context():
        env.physics.named.data.qpos[arm_controller.jnt_names] = start_config
        env.physics.named.data.qpos[hand_controller.jnt_names] = grasp_config_hand

    arm_controller.update()

    t = np.arange(SIM_STEPS)
    err = np.zeros((SIM_STEPS, arm_controller.dof))

    arm_controller.set_joint_goal(end_config)

    for i in range(SIM_STEPS):
        arm_controller.step()
        err[i, :] = arm_controller.setpoint - arm_controller.qpos
        env.physics.step()

    for i in range(arm_controller.dof):
        plt.cla()
        plt.plot(t, err[:, i])
        plt.xlabel("Time")
        plt.ylabel("Error")
        plt.title("MPC Error Plot at Joint {}".format(i))
        plt.savefig("graphs/jnt_%d.png" % (i))
    plt.show()


if __name__ == "__main__":
    main()
