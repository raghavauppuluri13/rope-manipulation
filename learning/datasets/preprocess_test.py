from matplotlib import pyplot as plt
import matplotlib

matplotlib.use("Agg")
import numpy as np
from PIL import Image, ImageSequence
from preprocess import mask_rope

gif = "/home/ruppulur/repos/rope-manipulation/datasets/2022-04-13-15:59:58/batch_461/batch.gif"
img = Image.open(gif)

frames = np.array(
    [
        np.array(frame.copy().convert("RGB").getdata(), dtype=np.uint8).reshape(
            frame.size[1], frame.size[0], 3
        )
        for frame in ImageSequence.Iterator(img)
    ]
)
frames = frames[:, :, :, ::-1]
print(frames[0][0])


masked_imgs = []
for frame in frames:
    mask = mask_rope(frame)
    plt.imsave("test.png", mask)
    masked_imgs.append(Image.fromarray(np.array(mask)))
print(len(masked_imgs))
masked_imgs[0].save(
    "array.gif", save_all=True, append_images=masked_imgs[1:], duration=50, loop=0
)
