import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from dm_control import mujoco


class Observer:
    def __init__(
        self,
        physics: mujoco.Physics,
        camera_id=-1,
        height=480,
        width=640,
        fps=10,
        dpi=50,
        render_factor=72.0,
    ):
        self.width = width
        self.height = height
        self.physics = physics
        self.camera_id = camera_id
        self.frames = [] 
        self.fps = fps
        self.dpi = dpi
        self.frame_i = 0
        self.render_factor = render_factor

    def render(self):
        return self.physics.render(self.height, self.width, camera_id=self.camera_id)

    def show(self):
        img = self.render()
        plt.imshow(img)
        plt.pause(0.001)

    def step(self):
        if (
            self.physics.time() > self.frame_i / self.fps
        ):
            self.frames.append(self.render())
            self.frame_i += 1

    def save(self, path="env_render.gif"):
        """Saves rendered frames as gif
        source: https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553

        Args:
            path (str, optional): _description_. Defaults to 'mujoco_animation.gif'.
        """
        frames = np.array(self.frames)
        plt.figure(
            figsize=(
                frames[0].shape[1] / self.render_factor,
                frames[0].shape[0] / self.render_factor,
            ),
            dpi=self.dpi,
        )

        patch = plt.imshow(frames[0])
        plt.axis("off")

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(
            plt.gcf(), animate, frames=len(frames), interval=50
        )
        anim.save(path, writer="imagemagick", fps=self.fps)
