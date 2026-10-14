import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Load data
df = pd.read_csv(r"C:\Users\NAA\Desktop\XRAY PROJ\task_5\data.csv")
u_all = df['V1'].values
y_all = df['V2'].values

# Extract slice [42650 : 46746]
start_idx = 42650
end_idx = 46746
u_slice = u_all[start_idx:end_idx]
y_slice = y_all[start_idx:end_idx]

# Center signals
u_mean = np.mean(u_slice)
y_mean = np.mean(y_slice)
u = u_slice - u_mean
y = y_slice - y_mean

dt = 1.0 / 610.35

# Train/Test split: first 3072 train, last 1024 test
train_len = 3072
test_len = 1024

u_train = torch.tensor(u[:train_len], dtype=torch.float32)
y_train = torch.tensor(y[:train_len], dtype=torch.float32)
u_test = torch.tensor(u[train_len:], dtype=torch.float32)
y_test = torch.tensor(y[train_len:], dtype=torch.float32)

# Define PINN Acceleration Network
class AccelNet(nn.Module):
    def __init__(self):
        super(AccelNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, y_val, v_val, u_val):
        x = torch.cat([y_val, v_val, u_val], dim=-1)
        return self.net(x)

model = AccelNet()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# RK4 Step function
def rk4_step(y_val, v_val, u_val, net, dt):
    # k1
    a1 = net(y_val, v_val, u_val)
    
    # k2
    y2 = y_val + 0.5 * dt * v_val
    v2 = v_val + 0.5 * dt * a1
    a2 = net(y2, v2, u_val)
    
    # k3
    y3 = y_val + 0.5 * dt * v2
    v3 = v_val + 0.5 * dt * a2
    a3 = net(y3, v3, u_val)
    
    # k4
    y4 = y_val + dt * v3
    v4 = v_val + dt * a3
    a4 = net(y4, v4, u_val)
    
    y_next = y_val + (dt / 6.0) * (v_val + 2.0 * v2 + 2.0 * v3 + v4)
    v_next = v_val + (dt / 6.0) * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
    
    return y_next, v_next

# Training with short rollouts (chunked BPTT)
chunk_size = 128
epochs = 40

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    
    # Initialize state at start of training record
    y_curr = y_train[0].view(1, 1)
    v_curr = ((y_train[1] - y_train[0]) / dt).view(1, 1)
    
    loss = 0.0
    count = 0
    
    for i in range(train_len - 1):
        u_curr = u_train[i].view(1, 1)
        y_next, v_next = rk4_step(y_curr, v_curr, u_curr, model, dt)
        
        target_y = y_train[i+1].view(1, 1)
        loss = loss + torch.abs(y_next - target_y)
        
        y_curr = y_next
        v_curr = v_next
        count += 1
        
        if (i + 1) % chunk_size == 0 or i == train_len - 2:
            loss = loss / count
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Detach to truncate graph
            y_curr = y_curr.detach()
            v_curr = v_curr.detach()
            loss = 0.0
            count = 0

# Free run on full dataset (train + test)
model.eval()
with torch.no_grad():
    y_preds = []
    y_curr = y_train[0].view(1, 1)
    v_curr = ((y_train[1] - y_train[0]) / dt).view(1, 1)
    y_preds.append(y_curr)
    
    all_u = torch.cat([u_train, u_test])
    for i in range(len(all_u) - 1):
        u_curr = all_u[i].view(1, 1)
        y_curr, v_curr = rk4_step(y_curr, v_curr, u_curr, model, dt)
        y_preds.append(y_curr)
        
    y_pred_full = torch.cat(y_preds).squeeze()

y_pred_train = y_pred_full[:train_len]
y_pred_test = y_pred_full[train_len:]

train_mae = torch.mean(torch.abs(y_pred_train - y_train)).item()
test_mae = torch.mean(torch.abs(y_pred_test - y_test)).item()

print(f"TRAIN RMSE: {train_mae:.6f}")
print(f"TEST RMSE: {test_mae:.6f}")
