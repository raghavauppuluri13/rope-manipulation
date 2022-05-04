import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from dm_control import mujoco, composer


class Observer:
    def __init__(
        self,
        env: composer.Environment,
        obs_camera=-1,
        show=False,
        height=480,
        width=640,
        fps=10,
        dpi=50,
        render_factor=72.0,
    ):
        self.env = env
        self.width = width
        self.height = height
        self.obs_camera = obs_camera
        self.frames = []
        self.fps = fps
        self.dpi = dpi
        self.frame_i = 0
        self.render_factor = render_factor
        self.show = show
        if self.show:
            fig, self.ax = plt.subplots()

    def reset(self):
        self.frames = []
        self.frame_i = 0

    def step(self, timestep, hold=False):
        img = timestep.observation[self.obs_camera][0]
        if self.env.physics.time() > self.frame_i / self.fps:
            self.frames.append(img)
            self.frame_i += 1

        if self.show:
            self.ax.imshow(img)
            if hold:
                plt.show()
            else:
                plt.pause(0.001)

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
