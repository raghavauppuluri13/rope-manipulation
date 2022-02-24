"""
Controller implementations for a Panda arm in Mujoco
Author: Raghava Uppuluri
"""
import numpy as np
import yaml

from dm_control import mjcf


class PositionController:
    """Joint position controller"""

    def __init__(self, physics: mjcf.Physics, params_file="panda_cfg.yaml"):
        """Initializes position controller

        Args:
            physics (mjcf.Physics): Mujoco Physics instance
            robot_name (string): name of mujoco model of the robot
            qpos_i (list): indicies corresponding to the robot qpos in mjcf.Physics.data
            ctrl_i (list): indicies corresponding to the robot qpos in mjcf.Physics.data
            params_file (string): path to file for PD controller
        """
        with open(params_file, "r") as f:
            self.params = yaml.safe_load(f)

        self.physics = physics
        self.jnt_names = self.params["joint_names"]
        self.act_names = self.params["act_names"]
        self.setpoint = None
        self.curr = self.physics.named.data.qpos[self.jnt_names]
        self.dof = self.curr.shape[0]

        self.min_ctrl = self.physics.named.model.actuator_ctrlrange[self.act_names][
            :, 0
        ]
        self.max_ctrl = self.physics.named.model.actuator_ctrlrange[self.act_names][
            :, 1
        ]

        self.Kp = np.diag(np.array(self.params["pos"]["Kp"]))
        self.Kd = np.diag(np.array(self.params["pos"]["Kd"]))
        self.prev_error = 0

    def set_goal(self, setpoint):
        self.setpoint = setpoint

    def update(self):
        """Calculates and sets the joint angles
        according to PID parameters
        Args:
            ref (_type_): Reference joint array
        """
        self.curr = self.physics.named.data.qpos[self.jnt_names]
        e = self.setpoint - self.curr
        P = self.Kp @ e
        D = self.Kd @ (self.prev_error - e) / self.physics.timestep()
        ctrl = np.clip(P + D, self.min_ctrl, self.max_ctrl)
        self.physics.set_control(ctrl)


class VelocityController:
    """Joint velocity controller"""


class TorqueController:
    """Torque controller"""
