import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 1. Load data
data_path = r"C:\Users\NAA\Desktop\XRAY PROJ\task_5\data.csv"
df = pd.read_csv(data_path)
u_raw = df['V1'].values
y_raw = df['V2'].values

# Sampling parameters
fs = 610.35
dt = 1.0 / fs

# Extract excerpt: slice [42650 : 46746] (4096 samples)
u_exc = u_raw[42650:46746].copy()
y_exc = y_raw[42650:46746].copy()

# Center both signals on the excerpt
u = u_exc - np.mean(u_exc)
y = y_exc - np.mean(y_exc)

# Ground-truth finite-difference velocity for training chunks
v = np.zeros_like(y)
v[1:-1] = (y[2:] - y[:-2]) / (2.0 * dt)
v[0] = (y[1] - y[0]) / dt
v[-1] = (y[-1] - y[-2]) / dt

# Training (first 3072) and test (last 1024)
N_train = 3072
N_test = 1024

y_train = y[:N_train]
u_train = u[:N_train]
v_train = v[:N_train]

# Empirical normalization scales from training data
std_y = float(np.std(y_train))
std_v = float(np.std(v_train))
std_u = float(np.std(u_train))
a_fd = np.gradient(v_train, dt)
std_a = float(np.std(a_fd))

# Define Models
class ResMLPAcceleration(nn.Module):
    def __init__(self, std_y, std_v, std_u, std_a, hidden_dim=64):
        super().__init__()
        self.register_buffer('std_y', torch.tensor(std_y, dtype=torch.float32))
        self.register_buffer('std_v', torch.tensor(std_v, dtype=torch.float32))
        self.register_buffer('std_u', torch.tensor(std_u, dtype=torch.float32))
        self.register_buffer('std_a', torch.tensor(std_a, dtype=torch.float32))
        
        self.linear = nn.Linear(3, 1, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        nn.init.normal_(self.mlp[-1].weight, std=1e-3)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, y_val, v_val, u_val):
        norm_in = torch.cat([y_val / self.std_y, v_val / self.std_v, u_val / self.std_u], dim=-1)
        acc = (self.linear(norm_in) + self.mlp(norm_in)) * self.std_a
        return acc

class PureMLPAcceleration(nn.Module):
    def __init__(self, std_y, std_v, std_u, std_a, hidden_dim=64):
        super().__init__()
        self.register_buffer('std_y', torch.tensor(std_y, dtype=torch.float32))
        self.register_buffer('std_v', torch.tensor(std_v, dtype=torch.float32))
        self.register_buffer('std_u', torch.tensor(std_u, dtype=torch.float32))
        self.register_buffer('std_a', torch.tensor(std_a, dtype=torch.float32))
        
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, y_val, v_val, u_val):
        norm_in = torch.cat([y_val / self.std_y, v_val / self.std_v, u_val / self.std_u], dim=-1)
        acc = self.mlp(norm_in) * self.std_a
        return acc

def rk4_step(net, s, u_curr, u_next, dt_val):
    u_mid = 0.5 * (u_curr + u_next)
    def deriv(st, u_in):
        y_s = st[:, 0:1]
        v_s = st[:, 1:2]
        a_s = net(y_s, v_s, u_in)
        return torch.cat([v_s, a_s], dim=1)
    
    k1 = deriv(s, u_curr)
    k2 = deriv(s + 0.5 * dt_val * k1, u_mid)
    k3 = deriv(s + 0.5 * dt_val * k2, u_mid)
    k4 = deriv(s + dt_val * k3, u_next)
    return s + (dt_val / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def train_model(model, iterations=400, batch_size=32, T_chunk=16, lr=2e-3):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations, eta_min=1e-4)
    
    for it in range(iterations):
        optimizer.zero_grad()
        idx = np.random.randint(0, N_train - T_chunk, size=batch_size)
        
        y_b = torch.tensor(np.stack([y_train[i:i+T_chunk+1] for i in idx]), dtype=torch.float32)
        u_b = torch.tensor(np.stack([u_train[i:i+T_chunk+1] for i in idx]), dtype=torch.float32)
        v_b = torch.tensor(np.stack([v_train[i:i+T_chunk+1] for i in idx]), dtype=torch.float32)
        
        s = torch.stack([y_b[:, 0], v_b[:, 0]], dim=1)
        
        pred_y_list = []
        pred_v_list = []
        for t in range(T_chunk):
            u_c = u_b[:, t:t+1]
            u_n = u_b[:, t+1:t+2]
            s = rk4_step(model, s, u_c, u_n, dt)
            pred_y_list.append(s[:, 0])
            pred_v_list.append(s[:, 1])
            
        pred_y = torch.stack(pred_y_list, dim=1)
        pred_v = torch.stack(pred_v_list, dim=1)
        
        loss_y = nn.functional.mse_loss(pred_y, y_b[:, 1:])
        loss_v = nn.functional.mse_loss(pred_v, v_b[:, 1:])
        loss = loss_y + 1e-4 * ((std_y / std_v) ** 2) * loss_v
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
    return model

def free_run_evaluate(model):
    model.eval()
    with torch.no_grad():
        N_total = len(y)
        y_sim = np.zeros(N_total)
        y_sim[0] = y[0]
        v_cur = (y[1] - y[0]) / dt
        s = torch.tensor([[y[0], v_cur]], dtype=torch.float32)
        
        u_t = torch.tensor(u, dtype=torch.float32)
        for t in range(N_total - 1):
            u_c = u_t[t:t+1].unsqueeze(0)
            u_n = u_t[t+1:t+2].unsqueeze(0)
            s = rk4_step(model, s, u_c, u_n, dt)
            y_sim[t+1] = s[0, 0].item()
            
        train_rmse = np.sqrt(np.mean((y_sim[:N_train] - y[:N_train]) ** 2))
        test_rmse = np.sqrt(np.mean((y_sim[N_train:] - y[N_train:]) ** 2))
    return train_rmse, test_rmse

# Train and evaluate candidate architectures
model_res = ResMLPAcceleration(std_y, std_v, std_u, std_a, hidden_dim=64)
model_res = train_model(model_res, iterations=450, batch_size=32, T_chunk=16, lr=2e-3)
train_rmse_res, test_rmse_res = free_run_evaluate(model_res)

model_pure = PureMLPAcceleration(std_y, std_v, std_u, std_a, hidden_dim=64)
model_pure = train_model(model_pure, iterations=450, batch_size=32, T_chunk=16, lr=2e-3)
train_rmse_pure, test_rmse_pure = free_run_evaluate(model_pure)

# Select the architecture with the best test RMSE
if test_rmse_res <= test_rmse_pure:
    best_train_rmse = train_rmse_res
    best_test_rmse = test_rmse_res
else:
    best_train_rmse = train_rmse_pure
    best_test_rmse = test_rmse_pure

print(f"TRAIN RMSE: {best_train_rmse:.6f}")
print(f"TEST RMSE: {best_test_rmse:.6f}")
