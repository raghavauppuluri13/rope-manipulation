"""
Controller implementations for a Panda arm in Mujoco
Author: Raghava Uppuluri
"""
from configparser import Interpolation
from multiprocessing import Pool
import numpy as np
import yaml
from ctypes import *

from dm_control.mujoco.wrapper.mjbindings import enums

from mujoco import rollout
from dm_control import mjcf, mujoco,composer
from dm_control.utils.inverse_kinematics import qpos_from_site_pose

DOWN_QUATERNION = np.array([0., 0.70710678118, 0.70710678118, 0.])

class Controller:
    def __init__(self, env: composer.Environment, params_file):
        with open(params_file, "r") as f:
            self.params = yaml.safe_load(f)
        self.params_file = params_file
        self.env = env
        self.phys = env.physics
        self.jnt_names = self.params["jnt_names"]
        self.jnt_inds = np.array(
            self.phys.named.data.qpos.axes.row.convert_key_item(self.jnt_names)
        )
        self.act_names = self.params["act_names"]
        self.ctrl_inds = np.array(
            self.phys.named.data.ctrl.axes.row.convert_key_item(self.act_names)
        )
        self.dof = len(self.jnt_names)

        self.phys.forward()
        self.qpos = self.phys.named.data.qpos[self.jnt_names]
        self.qvel = self.phys.named.data.qvel[self.jnt_names]
        mass_matrix = np.zeros((self.phys.model.nq, self.phys.model.nq))
        mujoco.mj_fullM(self.phys.model.ptr, mass_matrix, self.phys.data.qM)
        self.mass_matrix = mass_matrix[self.jnt_inds, :][:, self.jnt_inds]
        self.grav_comp = self.phys.data.qfrc_bias[self.jnt_inds]
        self.setpoint = self.qpos

    def set_joint_goal(self, setpoint):
        assert isinstance(setpoint, (list, np.ndarray, tuple))
        assert len(setpoint) == self.dof, "Invalid setpoint shape"
        self.setpoint = setpoint

    def update(self):
        self.phys.forward()
        self.qpos = self.phys.named.data.qpos[self.jnt_names]
        self.qvel = self.phys.named.data.qvel[self.jnt_names]
        mass_matrix = np.zeros((self.phys.model.nq, self.phys.model.nq))
        mujoco.mj_fullM(self.phys.model.ptr, mass_matrix, self.phys.data.qM)
        self.mass_matrix = mass_matrix[self.jnt_inds, :][:, self.jnt_inds]
        self.grav_comp = self.phys.data.qfrc_bias[self.jnt_inds]
    
    def set_ctrl(self,ctrl):
        full_ctrl = np.zeros_like(self.phys.data.ctrl)
        full_ctrl[self.ctrl_inds] = ctrl
        return full_ctrl 
        

class ArmController(Controller):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ee_site = self.params["ee_site"]
        self.ee_pos = self.phys.named.data.site_xpos[self.ee_site]
        self.ik_attempts = 10

    def ik(self, target_pos=None, target_quat=None):
        assert target_pos is not None or target_quat is not None
        for _ in range(self.ik_attempts):
            res = qpos_from_site_pose(
                self.phys,
                self.ee_site,
                target_pos=target_pos,
                target_quat=target_quat,
                joint_names=self.jnt_names,
                max_steps=200
            )
            if res.success:
                break
        assert res.success
        return res.qpos[self.jnt_inds]

    def set_cartesian_goal(self, target_pos=None, target_quat=DOWN_QUATERNION):
        goal_qpos = self.ik(target_pos, target_quat)
        self.set_joint_goal(goal_qpos)

    def update(self):
        super().update()
        self.ee_pos = self.phys.named.data.site_xpos[self.ee_site]


class PositionController(Controller):
    """Task pose controller"""

    def __init__(self, **kwargs):
        """Initializes position controller

        Args:
            physics (mjcf.Physics): Mujoco Physics instance
            params_file (string): path to file for PD controller
        """
        super().__init__(**kwargs)
        self.Kp = np.array(self.params["pos"]["Kp"])
        #self.Ki = np.array(self.params["pos"]["Ki"])
        self.Kd = np.array(self.params["pos"]["Kd"]) 

        # self.saturated = False
        # self.err_sum = np.zeros(self.dof)


    def step(self):
        self.update()
        err = self.setpoint - self.qpos
        D_err = -self.qvel

        desired_ctrl = np.multiply(self.Kp, err) + np.multiply(self.Kd, D_err)
        ctrl = np.dot(self.mass_matrix, desired_ctrl) + self.grav_comp
        #self.ctrl = np.clip(desired_ctrl, self.env.action_spec.minimum, self.action_spec.maximum)
        # self.saturated = (
        #     False if np.sum(np.abs(ctrl - desired_ctrl)) == 0 else True
        # )
        return self.set_ctrl(ctrl)

class ArmPositionController(ArmController,PositionController):
    def __init__(self, env, params_file="panda_cfg.yaml"):
        super().__init__(env=env, params_file=params_file)

class GripperController(Controller):
    def __init__(self, env, params_file="panda_cfg.yaml"):
        """Initializes position controller

        Args:
            physics (mjcf.Physics): Mujoco Physics instance
            params_file (string): path to file for PD controller
        """
        super().__init__(env=env, params_file=params_file)
        self.grip_force_sensor = self.params['force_sensor']
        self.open = np.array(self.params['open'])
        self.close = np.array(self.params['closed'])
    
    def step(self):
        self.update()
        return self.set_ctrl(self.setpoint) 


class VelocityController(ArmController):
    """Joint velocity controller"""

    def __init__(self, phys: mjcf.Physics, params_file="panda_cfg.yaml"):
        """Initializes position controller

        Args:
            physics (mjcf.Physics): Mujoco Physics instance
            params_file (string): path to file for PD controller
        """
        super().__init__(phys, params_file)
        self.Kp = np.array(self.params["vel"]["Kp"])
        self.Ki = self.Kp * 0.05 
        self.Kd = self.Kp * 0.01 

        self.saturated = False
        self.err_sum = np.zeros(self.dof)

    def step(self):
        self.update()
        err = self.setpoint - self.qpos
        D_err = -self.qvel

        if not self.saturated:
            self.err_sum += err

        desired_ctrl = np.multiply(self.Kp, err) + np.multiply(self.Kd, D_err)
        self.ctrl = np.dot(self.mass_matrix, desired_ctrl) + self.grav_comp

        self.ctrl = np.clip(desired_ctrl, self.min_ctrl, self.max_ctrl)
        grav_comp = self.phys.named.data.qfrc_bias[self.jnt_names]
        ss_term = np.multiply(self.Ki, self.err_sum)
        self.phys.named.data.qfrc_applied[self.jnt_names] = -grav_comp + ss_term
        self.saturated = (
            False if np.sum(np.abs(self.ctrl - desired_ctrl)) == 0 else True
        )
        super().step()

    def run_controller(self):
        """Calculates and sets the joint angles
        according to PD parameters and inertias. Caller needs to update mjcf.Physics
        state using physics.step
        """
        self.step()

class MPC(ArmController):
    def __init__(self, env, params):
        super().__init__(env, params)
        self.mpc_params = self.params["mpc"]
        self.sample_len = self.mpc_params["sample_len"]
        self.rollout_len = self.mpc_params["rollout_len"]
        self.cem_len = self.mpc_params["cem_len"]
        self.top_traj_len = self.mpc_params["top_traj"]
        self.ctrl_means = np.zeros((self.rollout_len, self.dof)) if "means" not in self.mpc_params else np.array(self.mpc_params["means"])
        self.ctrl_stds = np.ones((self.rollout_len, self.dof)) if "stds" not in self.mpc_params else np.array(self.mpc_params["stds"])
        self.nqva = self.phys.model.nq + self.phys.model.nv + self.phys.model.na
        self.pool = Pool(4)

    def set_joint_goal(self, setpoint):
        assert isinstance(setpoint, (list, np.ndarray, tuple))
        assert setpoint.shape == (self.rollout_len,self.dof), "Invalid setpoint shape"
        self.setpoint = setpoint

    def set_cartesian_goal(self, target_poses=None, target_quats=None):
        goal_qposes = np.zeros((self.rollout_len,self.dof))
        for i in range(self.rollout_len):
            quat = target_quats[i] if target_quats is not None else None
            goal_qposes[i] = self.ik(target_poses[i], quat)
        self.set_joint_goal(goal_qposes)

    def get_cost(self, states, actions):
        pos_c = self.mpc_params["w_pos"] * np.sum(
            (states[:, :, 0] - self.setpoint) ** 2, axis=-1
        ) + np.arange(self.rollout_len)
        vel_c = self.mpc_params["w_vel"] * np.sum(
            (states[:, :, 1]) ** 2, axis=-1
        ) + np.arange(self.rollout_len)
        reg = self.mpc_params["w_reg"] * np.sum((actions**2), axis=-1)
        costs = np.sum(pos_c + vel_c + reg, axis=-1)
        return costs

    def save_params(self,means_file,std_file):
        np.savetxt(means_file,self.ctrl_means)
        np.savetxt(std_file,self.ctrl_stds)
        super().save_params()

    def step(self):
        ctrls = np.tile(
            self.phys.data.ctrl.reshape(1, len(self.phys.data.ctrl)),
            (self.sample_len, self.rollout_len, 1),
        )
        qfrc_applied = np.tile(
            self.phys.data.qfrc_applied.reshape(1, len(self.phys.data.qfrc_applied)),
            (self.sample_len, self.rollout_len, 1),
        )
        for i in range(self.sample_len):
            qfrc_applied[i, :][:, self.ctrl_inds] = self.grav_comp

        init_states = np.tile(
            self.phys.get_state().reshape(1, self.nqva), (self.sample_len, 1)
        )

        for i in range(self.sample_len):
            ctrls[i, :][:, self.ctrl_inds] = np.random.normal(
                self.ctrl_means, self.ctrl_stds
            )
        for t in range(self.cem_len):
            rollouts = self.rollout(init_states, ctrls,qfrc_applied)
            costs = self.get_cost(rollouts, np.array(ctrls))
            top_traj_i = np.argsort(costs)[: self.top_traj_len]
            top_traj = ctrls[top_traj_i, :][:, :, self.ctrl_inds]
            self.ctrl_means = np.mean(top_traj, axis=0)
            self.ctrl_stds = np.std(top_traj, axis=0)
            for i in range(self.sample_len):
                if i not in top_traj_i:
                    ctrls[i, :][:, self.ctrl_inds] = np.random.normal(
                        self.ctrl_means, self.ctrl_stds
                    )
        self.ctrl = self.phys.data.ctrl
        self.ctrl[self.ctrl_inds] = top_traj[0][0]

        return self.ctrl
    
    

    def rollout(self, init_states, ctrls,qfrc_applied):
        # rollouts_in = [(self.phys.model.ptr,self.phys.data.ptr,init_states[i],ctrls[i],qfrc_applied[i]) for i in range(self.sample_len)]
        # next_states,_ = zip(*self.pool.starmap(call_rollout,rollouts_in))
        # next_states = np.array(next_states)
        next_states, _ = rollout.rollout(
            self.phys.model.ptr,
            self.phys.data.ptr,
            initial_state=init_states,
            qfrc_applied=qfrc_applied,
            ctrl=ctrls,
        )
        qpos_rollouts = next_states[:, :, self.jnt_inds]
        qvel_rollouts = next_states[:, :, (self.jnt_inds + self.phys.model.nq)]
        rollouts = np.concatenate((qpos_rollouts, qvel_rollouts), axis=-1)
        rollouts = rollouts.reshape(rollouts.shape[0], rollouts.shape[1], 2, self.dof)
        return rollouts
