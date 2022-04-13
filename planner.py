import numpy as np

class Interpolator:
    def __init__(self, step):
        self.i = 0
        self.step = step
        self.traj = []
        self._done = True

    def __len__(self):
        return len(self.traj)

    def lerp_init(self, start, goal):
        self.traj = np.array([self.lerp(start, goal, t) for t in np.arange(0, 1, self.step)])
        self.i = 0 
        self._done = False

    def slerp_init(self, start, goal):
        self.traj = np.array(
            [self.slerp(start, goal, t) for t in np.arange(0, 1, self.step)]
        )
        self.i = 0 
        self._done = False

    def next(self):
        if self.i < len(self.traj) - 1:
            self.i += 1
        else:
            self._done = True 
        return self.traj[self.i]

    def next_n(self, n):
        if self.i < len(self.traj) - 1 - n:
            self.i += 1
        else:
            self._done = True 
        return self.traj[self.i : self.i + n]

    @property
    def done(self):
        return self._done 

    @staticmethod
    def lerp(start, end, t):
        return (1 - t) * start + t * end

    @staticmethod
    def slerp(start, end, t):
        """Spherical Linear intERPolation."""
        return (end * start.inverse()) ** t * start

class Planner:
    def __init__(self, env, controller, observer, interpolator_step=0.02):
        self.interp = Interpolator(interpolator_step)
        self.env = env
        self.controller = controller
        self.observer = observer

    @property
    def done(self):
        return self.interp.done

    def set_goal(self,goal):
        raise NotImplementedError

    def get_action(self):
        raise NotImplementedError

class ArmPlanner(Planner):
    def __init__(self, env, controller, observer, interpolator_step=0.02):
        super().__init__(env, controller, observer, interpolator_step)

    def set_goal(self,goal):
        self.interp.lerp_init(self.controller.ee_pos, goal)

    def get_action(self):
        if not self.done:
            self.controller.set_cartesian_goal(self.interp.next())
        return self.controller.step()


class GripperPlanner(Planner):
    def __init__(self, env, controller, observer, interpolator_step=0.02):
        super().__init__(env, controller, observer, interpolator_step)

    def set_goal(self, goal):
        self.interp.lerp_init(self.controller.qpos, goal)

    def get_action(self):
        if not self.done:
            self.controller.set_joint_goal(self.interp.next())
        return self.controller.step()
