import torch
import matplotlib.pyplot as plt
import numpy as np

def smooth_data_np_convolve(arr, span):
    return np.convolve(arr, np.ones(span * 2 + 1) / (span * 2 + 1), mode="same")

checkpoint = torch.load("checkpoints/checkpoint_2000.pkl", map_location='cuda:0')
history = checkpoint["loss_history"]
history_val = np.array(history['val'])

psnr = checkpoint['psnr']
psnr_val = np.array(psnr['val'])

num_epochs = len(history["fit"])

fig, axs = plt.subplots(2, 2, sharey=False, figsize=(10,5))
# axs[0].set_title("All epochs")
axs[0][0].plot(list(range(1, len(history["fit"]) + 1)), history["fit"], label="Training loss")
axs[0][0].plot(list(range(1, len(history["fit"]) + 1)), smooth_data_np_convolve(history_val,10), label="Testing loss")
axs[0][0].grid(True)
axs[0][0].legend()

x = 1900
axs[0][1].set_title("Last {} epochs".format(num_epochs-x))
axs[0][1].plot(list(range(1, len(history["fit"]) + 1))[x:], history["fit"][x:], label="Training loss")
axs[0][1].plot(list(range(1, len(history["fit"]) + 1))[x:], history["val"][x:], label="Testing loss")
axs[0][1].grid(True)
axs[0][1].legend()

axs[1][0].plot(list(range(1, len(psnr["fit"]) + 1)), psnr["fit"], label="Training psnr")
axs[1][0].plot(list(range(1, len(psnr["fit"]) + 1)), smooth_data_np_convolve(psnr_val,5), label="Testing psnr")
axs[1][0].grid(True)
axs[1][0].legend()

axs[1][1].set_title("Last {} epochs".format(num_epochs-x))
axs[1][1].plot(list(range(1, len(psnr["fit"]) + 1))[x:], psnr["fit"][x:], label="Training psnr")
axs[1][1].plot(list(range(1, len(psnr["fit"]) + 1))[x:], psnr["val"][x:], label="Testing psnr")
axs[1][1].grid(True)
axs[1][1].legend()



fig.suptitle("Online Triplet Loss (Batch All Strategy)")
fig.supxlabel("Loss")
fig.supylabel("Loss")
plt.tight_layout()
plt.locator_params(axis="x", integer=True, tight=True)
plt.show()
# plt.close()