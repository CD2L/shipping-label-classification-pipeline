import torch
import matplotlib.pyplot as plt
import numpy as np

def smooth_data_np_convolve(arr, span):
    return np.convolve(arr, np.ones(span * 2 + 1) / (span * 2 + 1), mode="same")

checkpoints_path = '../blurred_img_clf/checkpoints/' 

checkpoint = torch.load("../blurred_img_clf/checkpoints/checkpoint_90.pkl", map_location='cuda:0')
history = checkpoint["loss_history"]
history_val = np.array(history['val'])

num_epochs = len(history["fit"])

fig, axs = plt.subplots(1, 2, sharey=False, figsize=(10,5))
# axs[0].set_title("All epochs")
axs[0].plot(list(range(1, len(history["fit"]) + 1)), history["fit"], label="Training loss")
axs[0].plot(list(range(1, len(history["fit"]) + 1)), history["val"], label="Testing loss")
axs[0].grid(True)
axs[0].legend()

x = int(0.7*len(history['fit']))
axs[1].set_title("Last {} epochs".format(num_epochs-x))
axs[1].plot(list(range(1, len(history["fit"]) + 1))[x:], history["fit"][x:], label="Training loss")
axs[1].plot(list(range(1, len(history["fit"]) + 1))[x:], history["val"][x:], label="Testing loss")
axs[1].grid(True)
axs[1].legend()


fig.suptitle("Online Triplet Loss (Batch All Strategy)")
fig.supxlabel("Loss")
fig.supylabel("Loss")
plt.tight_layout()
plt.locator_params(axis="x", integer=True, tight=True)
plt.show()
# plt.close()