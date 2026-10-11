from task_5.Physical_Init import model, y_ex, u_ex, delta, f, loss_fn
import torch
import math
from task_5.RK4 import runge_kutta

#RMSE test starts here. To evaluate, change the import file of the model and the weight file number
model.load_state_dict(torch.load('model_4_weights.pth'))
y_n = y_ex[0]
v_n = (y_ex[1] - y_ex[0]) / delta
predicted = [y_n]
with torch.no_grad():
    for n in range(1, 4096):     
            u_1 = u_ex[n-1]
            u_2 = u_ex[n]
            y_n, v_n = runge_kutta(y_n, v_n, u_1, u_2, delta, f, model)
            predicted.append(y_n)

output = torch.stack(predicted)
y_train = y_ex[:3072]
y_test = y_ex[3072:]

loss_train = math.sqrt(loss_fn(output[:3072], y_train))
loss_test = math.sqrt(loss_fn(output[3072:], y_test))

print("Train RMSE: ", loss_train, "Test RMSE: ", loss_test, "baseline: ", y_ex[3072:].pow(2).mean().sqrt().item())

import numpy as np
import matplotlib.pyplot as plt

pred = output.numpy()
real = y_ex.numpy()
t = np.arange(len(real)) * delta          # time in seconds
t_split = 3072 * delta                    # where the test part starts

err = np.abs(pred - real)

# moving RMS over ~0.1 s, so the trend is visible through the wiggles
win = 61
rms_moving = np.sqrt(np.convolve(err**2, np.ones(win) / win, mode="same"))

fig, ax = plt.subplots(3, 1, figsize=(11, 10))

# 1) error over the whole free run
ax[0].plot(t, err, color="lightgray", lw=0.6, label="|predicted − real|")
ax[0].plot(t, rms_moving, color="C3", lw=1.5, label="moving RMS (~0.1 s)")
ax[0].axvline(t_split, color="k", ls="--", label="test part starts")
ax[0].set_title("Free-run error over the whole excerpt")
ax[0].set_ylabel("error (V)")
ax[0].legend(loc="upper left")

# 2) predicted vs real on the whole test part
m = t >= t_split
ax[1].plot(t[m], real[m], color="C0", lw=0.8, label="measured y")
ax[1].plot(t[m], pred[m], color="C1", lw=0.8, ls="--", label="free-run prediction")
ax[1].set_title(f"Test part: measured vs predicted   (test RMSE = {loss_test:.4f} V)")
ax[1].set_ylabel("y (V)")
ax[1].legend(loc="upper right")

# 3) zoom into ~100 samples of the test part
z0, z1 = 3600, 3700
ax[2].plot(t[z0:z1], real[z0:z1], "C0.-", label="measured y")
ax[2].plot(t[z0:z1], pred[z0:z1], "C1.--", label="free-run prediction")
ax[2].set_title(f"Zoom: samples {z0}–{z1}")
ax[2].set_ylabel("y (V)")
ax[2].legend(loc="upper right")

for a in ax:
    a.set_xlabel("time (s)")
    a.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("free_run_eval.png", dpi=120)
plt.show()