import numpy as np

def lerp(start, end, t):
    return (1 - t) * start + t * end


def circle(c, r, t):
    return c + np.array([r * np.cos(t), r * np.sin(t), 0])


def wrap_traj(start, c, r, step):
    circle_traj = np.array(
        [circle(c, r, t) for t in np.arange(1, 0, -step)]
    )  # -pi to 0
    traj = np.array(
        [lerp(start, circle_traj[0], t) for t in np.arange(0, 1, step)]
    )  # curr to -pi
    return np.vstack((traj, circle_traj))


def slerp(end, start, t):
    """Spherical Linear intERPolation."""
    return (end * start.inverse()) ** t * start

class Interpolator:
    def __init__(self,step):
        self.start = None
        self.goal = None
        self.i = 0
        self.step = step
        self.traj = None

    def __len__(self):
        return len(self.traj)
    
    def set_goal(self,start, goal):
        self.start = start
        self.goal = goal
        self.traj = np.array(
            [lerp(start, goal, t) for t in np.arange(0, 1, self.step)]
        )

    def next(self): 
        if i < len(self.traj):
            i += 1
        return self.traj[i]