from dm_control import composer
from dm_control import mjcf
from dm_control.composer import define
from dm_control.composer.observation import observable
from dm_control.entities.manipulators import base
import robosuite as suite
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

_ARM_JOINTS = [""]
_ARM_ACTS = [""]
_GRIP_ACTS = [""]
_GRIP_JOINTS = [""]
_EEF_SITE = [""]


class PandaRobot:
    """A composer entity representing a Panda arm."""

    def __init__(self, mjcf_arm):
        """Initializes the PandaArm.
        Args:
        name: String, the name of this robot. Used as a prefix in the MJCF name
            name attributes.
        """

        env = suite.make(
            "Lift",
            robots=["Panda"],
            has_renderer=False,
            has_offscreen_renderer=False,
            ignore_done=True,
            use_camera_obs=False,
            control_freq=100,
        )
        env.reset()
        env_xml = env.model.get_xml()

        self._mjcf_root = mjcf.from_xml_string(env_xml)
        # Find MJCF elements that will be exposed as attributes.
        self._joints = [self._mjcf_root.find("joint", name) for name in _ARM_JOINTS]
        self._actuators = [
            self._mjcf_root.find("actuators", name) for name in _ARM_ACTS
        ]
        self._eef_site = self._mjcf_root.find("site", _EEF_SITE)
        self._bodies = self.mjcf_model.find_all("body")

    def _build_observables(self):
        return PandaArmObservables(self)

    @property
    def joints(self):
        """List of joint elements belonging to the arm."""
        return self._joints

    @property
    def actuators(self):
        """List of actuator elements belonging to the arm."""
        return self._actuators

    @property
    def wrist_site(self):
        """Wrist site of the arm (attachment point for the hand)."""
        return self._wrist_site

    @property
    def mjcf_model(self):
        """Returns the `mjcf.RootElement` object corresponding to this robot."""
        return self._mjcf_root


class PandaArmObservables(base.JointsObservables):
    @define.observable
    def joints_pos(self):
        def get_joint_angles(physics):
            joint_pos = physics.bind(self._entity.joints).qpos
            return joint_pos

        return observable.Generic(get_joint_angles)


class PandaHand:
    """A composer entity representing a Panda arm."""

    def __init__(self, mjcf_hand):
        """Initializes the PandaArmHand.
        Args:
        name: String, the name of this robot. Used as a prefix in the MJCF name
            name attributes.
        """

        self._mjcf_root = mjcf_hand
        self._joints = [self._mjcf_root.find("joint", name) for name in _GRIP_JOINTS]
        self._actuators = [
            self._mjcf_root.find("actuators", name) for name in _GRIP_ACTS
        ]
        self._eef_site = self._mjcf_root.find("site", _EEF_SITE)

    def _build_observables(self):
        return PandaHandObservables(self)

    @property
    def joints(self):
        """List of joint elements belonging to the hand."""
        return self._joints

    @property
    def actuators(self):
        """List of actuator elements belonging to the hand."""
        return self._actuators

    @property
    def mjcf_model(self):
        """Returns the `mjcf.RootElement` object corresponding to this hand."""
        return self._mjcf_root
