import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Load data
    csv_path = r"C:\Users\NAA\Desktop\XRAY PROJ\task_5\data.csv"
    df = pd.read_csv(csv_path)

    # Slice the excerpt: 4096 samples starting at index 42650
    if len(df) >= 46746:
        excerpt = df.iloc[42650:46746]
    else:
        excerpt = df

    u_raw = excerpt['V1'].values.astype(np.float32)
    y_raw = excerpt['V2'].values.astype(np.float32)

    # Mean-center the signals on this excerpt
    u = u_raw - np.mean(u_raw)
    y = y_raw - np.mean(y_raw)

    fs = 610.35
    dt = 1.0 / fs
    N = len(y)
    N_train = 3072
    N_test = 1024

    # Precompute finite-difference velocity for training initial conditions
    v = np.zeros_like(y)
    v[0] = (y[1] - y[0]) / dt
    v[1:-1] = (y[2:] - y[:-2]) / (2.0 * dt)
    v[-1] = (y[-1] - y[-2]) / dt

    # Normalization scales based on train record
    y_std = float(np.std(y[:N_train])) + 1e-6
    v_std = float(np.std(v[:N_train])) + 1e-6
    u_std = float(np.std(u[:N_train])) + 1e-6
    # Characteristic acceleration scale ~ omega * v_std
    a_std = (v_std**2) / y_std

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the Neural Force / Acceleration Network
    class AccelerationNet(nn.Module):
        def __init__(self, y_std, v_std, u_std, a_std):
            super().__init__()
            self.y_std = y_std
            self.v_std = v_std
            self.u_std = u_std
            self.a_std = a_std
            
            # Nonlinear MLP with smooth activations
            self.mlp = nn.Sequential(
                nn.Linear(3, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
            # Linear skip connection to directly capture linear stiffness, damping and gain
            self.linear = nn.Linear(3, 1, bias=False)
            
            # Initialize linear branch close to expected harmonic oscillator properties
            with torch.no_grad():
                self.linear.weight.zero_()
                self.mlp[-1].weight.data.mul_(0.1)
                self.mlp[-1].bias.data.zero_()

        def forward(self, y_val, v_val, u_val):
            # Inputs shape: (B,)
            y_n = y_val / self.y_std
            v_n = v_val / self.v_std
            u_n = u_val / self.u_std
            inp = torch.stack([y_n, v_n, u_n], dim=-1) # (B, 3)
            out_n = self.mlp(inp) + self.linear(inp)    # (B, 1)
            return out_n.squeeze(-1) * self.a_std      # (B,)

    # RK4 Step function
    def rk4_step(net, y_curr, v_curr, u_curr, u_next, dt):
        u_mid = 0.5 * (u_curr + u_next)
        
        k1_y = v_curr
        k1_v = net(y_curr, v_curr, u_curr)
        
        y_mid1 = y_curr + 0.5 * dt * k1_y
        v_mid1 = v_curr + 0.5 * dt * k1_v
        k2_y = v_mid1
        k2_v = net(y_mid1, v_mid1, u_mid)
        
        y_mid2 = y_curr + 0.5 * dt * k2_y
        v_mid2 = v_curr + 0.5 * dt * k2_v
        k3_y = v_mid2
        k3_v = net(y_mid2, v_mid2, u_mid)
        
        y_end = y_curr + dt * k3_y
        v_end = v_curr + dt * k3_v
        k4_y = v_end
        k4_v = net(y_end, v_end, u_next)
        
        y_nxt = y_curr + (dt / 6.0) * (k1_y + 2.0 * k2_y + 2.0 * k3_y + k4_y)
        v_nxt = v_curr + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
        return y_nxt, v_nxt

    # Prepare Tensors
    u_train_t = torch.tensor(u[:N_train], dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y[:N_train], dtype=torch.float32, device=device)
    v_train_t = torch.tensor(v[:N_train], dtype=torch.float32, device=device)

    # Instantiate model and optimizer
    model = AccelerationNet(y_std, v_std, u_std, a_std).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1200, eta_min=1e-4)

    # Training with progressive chunk curriculum to guarantee long-term stability
    stages = [
        (300, 32, 32),   # (num_iters, chunk_len, batch_size)
        (400, 64, 32),
        (500, 96, 32)
    ]

    model.train()
    for num_iters, chunk_len, batch_size in stages:
        for it in range(num_iters):
            starts = np.random.randint(0, N_train - chunk_len - 1, size=batch_size)
            
            # Extract initial states
            y_k = y_train_t[starts]
            v_k = v_train_t[starts]
            
            loss = 0.0
            for step in range(chunk_len):
                idx_curr = starts + step
                idx_next = idx_curr + 1
                u_curr = u_train_t[idx_curr]
                u_next = u_train_t[idx_next]
                y_target = y_train_t[idx_next]
                
                y_k, v_k = rk4_step(model, y_k, v_k, u_curr, u_next, dt)
                loss = loss + torch.mean((y_k - y_target)**2)
            
            loss = loss / chunk_len
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

    # Continuous Free-Run Evaluation on the full excerpt (4096 steps)
    model.eval()
    u_full_t = torch.tensor(u, dtype=torch.float32, device=device)
    
    y_sim = np.zeros(N, dtype=np.float32)
    y_sim[0] = y[0]
    
    curr_y = torch.tensor([y[0]], dtype=torch.float32, device=device)
    curr_v = torch.tensor([(y[1] - y[0]) / dt], dtype=torch.float32, device=device)
    
    with torch.no_grad():
        for k in range(N - 1):
            u_curr = u_full_t[k:k+1]
            u_next = u_full_t[k+1:k+2]
            curr_y, curr_v = rk4_step(model, curr_y, curr_v, u_curr, u_next, dt)
            y_sim[k + 1] = curr_y.item()

    # Calculate RMSEs
    train_rmse = np.sqrt(np.mean((y_sim[:N_train] - y[:N_train])**2))
    test_rmse = np.sqrt(np.mean((y_sim[N_train:] - y[N_train:])**2))

    print(f"TRAIN RMSE: {train_rmse:.6f}")
    print(f"TEST RMSE: {test_rmse:.6f}")

if __name__ == '__main__':
    main()
