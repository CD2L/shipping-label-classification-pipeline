import cv2

from utils.label_extractor import Extractor
import matplotlib.pyplot as plt

import numpy as np


def plot(*im, title=None, save=None, cmap="gray"):
    plt.figure(dpi=600)
    plt.axis("off")
    
    im = np.concatenate(im, axis=1)
    plt.imshow(im, interpolation="nearest", cmap=cmap)

    if title is not None:
        plt.title(title)
    if save is not None:
        output_path = save + ".jpg"
        plt.savefig(output_path, bbox_inches="tight")

        
ext = Extractor()


# FOR TESTING
idx=33
filename = f"{idx}.jpg"

im = cv2.imread(filename)
out, hist = ext(im)

plot(*hist[:-1], title=None, save=f"{idx}_viz_steps", cmap="gray")

fig, ax = plt.subplots(nrows=1, ncols=2)
fig.tight_layout()
fig.subplots_adjust(top=0.88)

ax[0].imshow(hist[0], cmap="gray")
ax[0].set_title("Before")
ax[0].axis("off")

ax[1].imshow(hist[-1], cmap="gray")
ax[1].set_title("After")
ax[1].axis("off")

fig.savefig(f"{idx}_viz_out.jpg", dpi=600)