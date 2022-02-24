'''
Run
'''
import matplotlib.pyplot as plt
import numpy as np

from controllers import PositionController

from envs import Renderer, RopeEnv

env = RopeEnv()

controller = PositionController(env.physics)
controller.set_goal(np.array([-2,0.5,0.5,0.1,0.5,0.5,0.1,0.04,0.04]))

init_joints = np.array([0] * 9)
with env.physics.reset_context():
    env.physics.data.qpos[range(9)] = init_joints

n_frames = 200
renderer = Renderer(env.physics, camera_id=-1, n_frames=n_frames)
tuning = np.zeros((9, n_frames))

env.physics.reset()
for i in range(n_frames):
    tuning[:, i] = controller.curr
    controller.update()
    while env.physics.time() < i / renderer.fps:
        env.physics.step()
    renderer.render()

renderer.save()