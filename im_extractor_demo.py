from utils.scanner import Extractor
import matplotlib.pyplot as plt

import cv2
import numpy as np
import time


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str)

args = parser.parse_args()
filename = args.image


# FOR TESTING

im = cv2.imread(filename)

start = time.process_time()
ext = Extractor()
hist = ext(im)
end = time.process_time()

print(
    f"Processing time: {round((end - start)*1e3)} milliseconds"
)

ext.plot(*hist[:-1], title=None, save=f"xxx_viz_steps", cmap="gray")

fig, ax = plt.subplots(nrows=1, ncols=2, dpi=1200)
fig.tight_layout()
fig.subplots_adjust(top=0.88)

ax[0].imshow(hist[0], cmap="gray")
ax[0].set_title("Before")
ax[0].axis("off")

ax[1].imshow(hist[-1], cmap="gray")
ax[1].set_title("After")
ax[1].axis("off")

fig.savefig(f"xxx_viz_out.jpg")