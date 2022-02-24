import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from dm_control import mjcf, mujoco


class Renderer:
    def __init__(
        self,
        physics: mujoco.Physics,
        camera_id,
        n_frames,
        height=480,
        width=640,
        fps=60,
        dpi=50,
        render_factor=72.0,
    ):
        self.width = width
        self.height = height
        self.physics = physics
        self.camera_id = camera_id
        self.frames = np.zeros((n_frames, height, width, 3), dtype=np.uint8)
        self.fps = fps
        self.dpi = dpi
        self.render_i = 0
        self.render_factor = render_factor

    def render(self):
        img = self.physics.render(self.height, self.width, camera_id=self.camera_id)
        self.frames[self.render_i] = img
        self.render_i += 1

    def save(self, path="env_render.gif"):
        """Saves rendered frames as gif
        source: https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553

        Args:
            path (str, optional): _description_. Defaults to 'mujoco_animation.gif'.
        """
        plt.figure(
            figsize=(
                self.frames[0].shape[1] / self.render_factor,
                self.frames[0].shape[0] / self.render_factor,
            ),
            dpi=self.dpi,
        )

        patch = plt.imshow(self.frames[0])
        plt.axis("off")

        def animate(i):
            patch.set_data(self.frames[i])

        anim = animation.FuncAnimation(
            plt.gcf(), animate, frames=len(self.frames), interval=50
        )
        anim.save(path, writer="imagemagick", fps=self.fps)


class BaseEnv:
    def __init__(self):
        self.obj = mjcf.from_path(os.path.abspath("models/arenas/checkers.xml"))
        self.physics = mjcf.Physics.from_mjcf_model(self.obj)


class RopeEnv(BaseEnv):
    def __init__(self):
        super().__init__()

        self.obj.worldbody.add(
            "camera", name="overhead", pos="0 -0.5 1", quat="1 0 0 0"
        )

        panda = mjcf.from_path(os.path.abspath("models/panda.xml"))
        panda.worldbody.body["panda_link0"].pos = [0, 0, 0]
        panda.worldbody.body["panda_link0"].quat = [0.7071068, 0, 0, -0.7071068]
        self.obj.attach(panda)

        rope = mjcf.from_path(os.path.abspath("models/rope/rope.xml"))
        rope.worldbody.body["CB10"].pos = [0, -0.5, 0]
        self.obj.attach(rope)

        self.physics = mjcf.Physics.from_mjcf_model(self.obj)
