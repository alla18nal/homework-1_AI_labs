import pandas as pd
import torch
torch.manual_seed(1) 
from task_5.hw1.RK4 import runge_kutta
import math
import torch
import torch.nn as nn

df = pd.read_csv("data.csv")
u_all = torch.tensor(df["V1"].values, dtype=torch.float32)  # input voltage
y_all = torch.tensor(df["V2"].values, dtype=torch.float32)  # output voltage


u_ex = u_all[42650:46746] #excerpt data
y_ex = y_all[42650:46746]
u_ex = u_ex - u_ex.mean()
y_ex = y_ex - y_ex.mean()

u_train, y_train = u_ex[:3072], y_ex[:3072]
u_test,  y_test  = u_ex[3072:], y_ex[3072:]

delta = 1 / 610.35

def f(y, v, u, model):
    x = torch.stack([y, v, u])    
    a_n = model(x).squeeze() 
    return a_n



class LinearForce(nn.Module):
    """acceleration = straight-line part(y, v, u), no initialization, random seed fixed, no activation"""
    def __init__(self, y_scale, v_scale, u_scale, a_scale, hidden=32):
        super().__init__()
        # scaling: divide inputs so they're near 1, multiply output back to real size
        self.register_buffer("in_scale", torch.tensor([y_scale, v_scale, u_scale]))
        self.a_scale = a_scale

        # Path A: straight line in y, v, u (spring, friction, input gain)
        self.linear = nn.Linear(3, 1)

        # Path B: curved part, sees only y (the cubic spring)
        # self.curve = nn.Sequential(
        #     nn.Linear(1, hidden), nn.Tanh(),
        #     nn.Linear(hidden, hidden), nn.Tanh(),
        #     nn.Linear(hidden, 1),          # no activation on the output
        # )

        # Starting point (see notes below)
        # with torch.no_grad():
            # self.linear.weight[0, 0] = -1.0    # y: a 70 Hz spring pulling back
            # self.linear.weight[0, 1] = -0.1    # v: some friction
            # self.linear.weight[0, 2] = 0    
            # self.linear.bias.zero_()
            # self.curve[-1].weight.zero_()      # curved part starts at zero
            # self.curve[-1].bias.zero_()

    def forward(self, x):
        z = x / self.in_scale
        # a = self.linear(z) + self.curve(z[..., 0:1])
        a = self.linear(z)
        return a * self.a_scale


# --- scales from the training data ---
omega = 2 * math.pi * 70
y_scale = float(y_train.std())
v_scale = float(((y_train[1:] - y_train[:-1]) / delta).std())
u_scale = float(u_train.std())
a_scale = omega**2 * y_scale          # rough size of the acceleration

model = LinearForce(y_scale, v_scale, u_scale, a_scale)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


import matplotlib.pyplot as plt

epochs = 20            # more epochs, so you can see where it flattens
epoch_losses = []      # average loss of each epoch
chunk_losses = []      # every single chunk loss, in order

L = 32        # chunk length (hyperparameter)

for epoch in range(epochs):
    total = 0.0
    count = 0

    for n in range(0, 3072 - L, L):
        y_n = y_train[n]
        v_n = (y_train[n+1] - y_train[n]) / delta

        predicted = []
        for i in range(L):
            u_1 = u_train[n + i]
            u_2 = u_train[n + i + 1]
            y_n, v_n = runge_kutta(y_n, v_n, u_1, u_2, delta, f, model)
            predicted.append(y_n)

        output = torch.stack(predicted)
        y_batch = y_train[n+1 : n+1+L]

        loss = loss_fn(output, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item()
        count += 1
        chunk_losses.append(loss.item())

    avg = total / count
    epoch_losses.append(avg)
    print(f"Epoch {epoch:3d}   average loss {avg:.3e}   typical error {avg**0.5:.4f} V")

# --- plot ---
baseline = y_train.var().item()   # loss of predicting zero everywhere
chunks_per_epoch = len(chunk_losses) // epochs

plt.figure(figsize=(9, 5))
x_chunks = [k / chunks_per_epoch for k in range(len(chunk_losses))]
plt.plot(x_chunks, chunk_losses, color="lightgray", lw=1, label="each chunk")
plt.plot(range(1, epochs + 1), epoch_losses, "o-", label="epoch average")
plt.axhline(baseline, color="red", ls="--", label="predict zero (baseline)")
plt.yscale("log")
plt.xlabel("epoch")
plt.ylabel("MSE loss (log scale)")
plt.title("Training loss")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("training_loss.png", dpi=120)
plt.show()

torch.save(model.state_dict(), 'model_1_weights.pth')

#RMSE test starts here

# y_n = y_ex[0]
# v_n = (y_ex[1] - y_ex[0]) / delta
# predicted = []
# torch.no_grad()
# for n in range(1, 4094):

        
#         u_1 = u_ex[n]
#         u_2 = u_ex[n + 1]
#         y_n, v_n = runge_kutta(y_n, v_n, u_1, u_2, delta, f, model)
#         predicted.append(y_n)

# output = torch.stack(predicted)
# y_train = y_ex[:3072]
# y_test = y_ex[3072:]

# loss_train = math.sqrt(loss_fn(output[:3072], y_train))
# loss_test = math.sqrt(loss_fn(output[3072:], y_test))

# print("Train RMSE: ", loss_train, "Test RMSE: ", loss_test, "baseline: ", y_ex[3072:].pow(2).mean().sqrt())

