"""
Script: Experiment 1 - Sampling Rate Sensitivity (Secure Archival Version)
Purpose: Compare MSE-CNN vs PISE across different sampling rates.
"""

import os
import sys
import json
import time
import datetime
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, vgg16, VGG16_Weights

# --- 1. Environment & Logger Setup ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
SAVE_DIR = os.path.join("./experiments", f"{TIMESTAMP}_Exp1_Sensitivity_Secure")

# Create directories
os.makedirs(os.path.join(SAVE_DIR, "weights"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "plots"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "logs"), exist_ok=True)
os.makedirs("./weights_cache", exist_ok=True) # Global cache

# I/O Redirection for logging
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(os.path.join(SAVE_DIR, "logs", "experiment_log.txt"), "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()

sys.stdout = Logger() # Redirect print to file

print(f"[System] Archive initialized at: {SAVE_DIR}")
print(f"[System] Device: {torch.cuda.get_device_name(0)}")

# --- 2. Configuration ---
CONFIG = {
    'RATES': [0.02, 0.05, 0.10, 0.20], 
    'EPOCHS': 30,           
    'BATCH_SIZE': 2048,     
    'LR': 0.002,
    'LAMBDA_PERC': 0.1,
    'SEED': 2026,
    'TRAIN_NOISE': 0.01     
}
# Save Config
with open(os.path.join(SAVE_DIR, "config.json"), "w") as f:
    json.dump(CONFIG, f, indent=4)

# --- 3. Models ---
class GhostImagingPhysics(nn.Module):
    def __init__(self, size=28*28, rate=0.05):
        super().__init__()
        self.N, self.M = size, int(size * rate)
        g = torch.Generator(); g.manual_seed(42)
        # NOTE: Negative values in 'A' simulate differential measurement (Pattern - Inverse).
        self.register_buffer('A', torch.randn(self.M, self.N, generator=g) / (self.M ** 0.5))

    def forward_sensing(self, x, noise_std=0.0):
        y = torch.mm(x.float().view(x.size(0), -1), self.A.t())
        if noise_std > 0: 
            y = y + torch.randn_like(y) * noise_std
        return y

    def backward_proxy(self, y):
        proxy = torch.mm(y, self.A)
        B = proxy.size(0); flat = proxy.view(B, -1)
        mi, ma = flat.min(1, True)[0], flat.max(1, True)[0]
        proxy = (flat - mi) / (ma - mi + 1e-6)
        return proxy.view(B, 1, 28, 28)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        def d_conv(in_c, out_c): 
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, 1, 1), nn.BatchNorm2d(out_c), nn.ReLU(True),
                nn.Conv2d(out_c, out_c, 3, 1, 1), nn.BatchNorm2d(out_c), nn.ReLU(True)
            )
        self.inc = d_conv(1, 32)
        self.down1 = d_conv(32, 64); self.pool = nn.MaxPool2d(2)
        self.down2 = d_conv(64, 128); self.down3 = d_conv(128, 256)
        self.up1 = nn.ConvTranspose2d(256, 128, 2, 2); self.conv1 = d_conv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2); self.conv2 = d_conv(128, 64)
        self.up3 = nn.ConvTranspose2d(64, 32, 2, 2); self.conv3 = d_conv(64, 32)
        self.outc = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(self.pool(x1))
        x3 = self.down2(self.pool(x2)); x4 = self.down3(self.pool(x3))
        x = self.up1(x4)
        if x.shape != x3.shape: x = F.interpolate(x, size=x3.shape[2:])
        x = torch.cat([x3, x], dim=1); x = self.conv1(x)
        x = self.up2(x)
        if x.shape != x2.shape: x = F.interpolate(x, size=x2.shape[2:])
        x = torch.cat([x2, x], dim=1); x = self.conv2(x)
        x = self.up3(x)
        if x.shape != x1.shape: x = F.interpolate(x, size=x1.shape[2:])
        x = torch.cat([x1, x], dim=1); x = self.conv3(x)
        return torch.sigmoid(self.outc(x))

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features[:16].to(DEVICE).eval()
        for p in vgg.parameters(): p.requires_grad = False
        self.slice = vgg
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x, y):
        x = x.repeat(1, 3, 1, 1) if x.size(1)==1 else x
        y = y.repeat(1, 3, 1, 1) if y.size(1)==1 else y
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        return F.l1_loss(self.slice(x), self.slice(y))

# --- 4. Evaluator Logic (Secure Version) ---
class ResNetEval(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.model.fc = nn.Linear(512, 10)
    def forward(self, x): return self.model(x)

def prepare_judge_secure(dataset):
    """
    Manages evaluator lifecycle: validates global cache, initiates training if absent, and creates local snapshot.
    """
    GLOBAL_CACHE_PATH = './weights_cache/ResNetEval.pth'
    LOCAL_SNAPSHOT_PATH = os.path.join(SAVE_DIR, "weights", "Judge_Snapshot.pth")
    
    judge = ResNetEval().to(DEVICE)
    
    # 1. Check Global Cache
    if os.path.exists(GLOBAL_CACHE_PATH):
        print(f"[System] Found global Evaluator at {GLOBAL_CACHE_PATH}. Loading...")
        judge.load_state_dict(torch.load(GLOBAL_CACHE_PATH, map_location=DEVICE))
    else:
        print("\n[WARNING] Global Evaluator NOT found! Training a new Evaluator now...")
        judge.train()
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)
        opt = optim.Adam(judge.parameters(), lr=0.001)
        crit = nn.CrossEntropyLoss()
        
        for ep in range(5): 
            correct, total = 0, 0
            for i, l in loader:
                i, l = i.to(DEVICE), l.to(DEVICE)
                opt.zero_grad()
                out = judge(i)
                loss = crit(out, l)
                loss.backward()
                opt.step()
                correct += (out.argmax(1)==l).sum().item()
                total += l.size(0)
            print(f"  > Evaluator Training Ep {ep+1}/5 | Acc: {correct/total*100:.2f}%")
        
        # Global cache serialization
        print(f"[System] Saving new Evaluator to Global Cache: {GLOBAL_CACHE_PATH}")
        torch.save(judge.state_dict(), GLOBAL_CACHE_PATH)

    # 2. Local snapshot generation 
    print(f"[System] Creating local Judge snapshot at: {LOCAL_SNAPSHOT_PATH}")
    torch.save(judge.state_dict(), LOCAL_SNAPSHOT_PATH)

    # Parameter freezing
    judge.eval()
    for p in judge.parameters(): p.requires_grad = False
    return judge

# --- 5. Training Loop for One Rate ---
def train_for_rate(rate, judge):
    print(f"\n{'='*60}")
    print(f" STARTING EXPERIMENT: SAMPLING RATE {rate*100:.1f}%")
    print(f"{'='*60}")
    
    # 1. Physics Setup
    phy = GhostImagingPhysics(28*28, rate=rate).to(DEVICE)
    
    # 2. Model Setup
    net_mse = UNet().to(DEVICE)
    net_ours = UNet().to(DEVICE)
    
    # 3. Optimizers
    opt_mse = optim.Adam(net_mse.parameters(), lr=CONFIG['LR'])
    opt_ours = optim.Adam(net_ours.parameters(), lr=CONFIG['LR'])
    
    crit_mse = nn.MSELoss()
    crit_perc = VGGPerceptualLoss().to(DEVICE)
    
    # 4. Data Loaders
    ts = transforms.ToTensor()
    ds = torchvision.datasets.FashionMNIST('./data', train=True, download=True, transform=ts)
    dl = torch.utils.data.DataLoader(ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=8, pin_memory=True)
    
    ds_test = torchvision.datasets.FashionMNIST('./data', train=False, transform=ts)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=1000, shuffle=False, num_workers=4)

    # 5. Training
    scaler = torch.cuda.amp.GradScaler() 
    acc_m_final, acc_o_final = 0, 0

    for epoch in range(CONFIG['EPOCHS']):
        net_mse.train(); net_ours.train()
        loss_m_sum, loss_o_sum, batches = 0, 0, 0
        
        for i, _ in dl:
            i = i.to(DEVICE, non_blocking=True)
            
            # Forward sensing model
            with torch.no_grad():
                y = phy.forward_sensing(i, noise_std=CONFIG['TRAIN_NOISE'])
                x_init = phy.backward_proxy(y)
            
            # --- Train MSE-CNN ---
            opt_mse.zero_grad()
            with torch.cuda.amp.autocast():
                rec_mse = net_mse(x_init)
                loss_m = crit_mse(rec_mse, i)
            scaler.scale(loss_m).backward()
            scaler.step(opt_mse)
            
            # --- Train Ours (PISE) ---
            opt_ours.zero_grad()
            with torch.cuda.amp.autocast():
                rec_ours = net_ours(x_init)
                loss_o = crit_mse(rec_ours, i) + CONFIG['LAMBDA_PERC'] * crit_perc(rec_ours, i)
            scaler.scale(loss_o).backward()
            scaler.step(opt_ours)
            
            scaler.update()
            loss_m_sum += loss_m.item()
            loss_o_sum += loss_o.item()
            batches += 1
            
        # Epoch-end validation
        net_mse.eval(); net_ours.eval()
        am, ao, tot = 0, 0, 0
        with torch.no_grad():
            for i, l in dl_test:
                i, l = i.to(DEVICE), l.to(DEVICE)
                y = phy.forward_sensing(i, noise_std=0.0) 
                x_init = phy.backward_proxy(y)
                
                am += (judge(net_mse(x_init)).argmax(1) == l).sum().item()
                ao += (judge(net_ours(x_init)).argmax(1) == l).sum().item()
                tot += l.size(0)
        
        acc_m_final = am/tot*100
        acc_o_final = ao/tot*100
        print(f"   [Ep {epoch+1:02d}/{CONFIG['EPOCHS']}] Loss M:{loss_m_sum/batches:.4f} O:{loss_o_sum/batches:.4f} | Acc MSE:{acc_m_final:.2f}% PISE:{acc_o_final:.2f}%")
    
    # Weight serialization per sampling rate
    path_mse = os.path.join(SAVE_DIR, "weights", f"Model_MSE_Rate_{int(rate*100)}pct.pth")
    path_ours = os.path.join(SAVE_DIR, "weights", f"Model_PISE_Rate_{int(rate*100)}pct.pth")
    
    torch.save(net_mse.state_dict(), path_mse)
    torch.save(net_ours.state_dict(), path_ours)
    print(f"[Archival] Weights saved:\n   -> {path_mse}\n   -> {path_ours}")
    
    return acc_m_final, acc_o_final

# --- 6. Main Execution ---
def main():
    torch.manual_seed(CONFIG['SEED'])
    np.random.seed(CONFIG['SEED'])
    
    # Dataset initialization for evaluator pre-training
    ts = transforms.ToTensor()
    ds_train = torchvision.datasets.FashionMNIST('./data', train=True, download=True, transform=ts)
    
    #  Prepare Judge Securely
    judge = prepare_judge_secure(ds_train)
    
    results = {'Rate': [], 'MSE': [], 'PISE': []}
    
    # Run loop
    for r in CONFIG['RATES']:
        m, o = train_for_rate(r, judge)
        results['Rate'].append(r)
        results['MSE'].append(m)
        results['PISE'].append(o)
    
    # Save CSV Data
    df = pd.DataFrame(results)
    csv_path = os.path.join(SAVE_DIR, "logs", "sensitivity_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[Archival] Data saved to {csv_path}")
    
    # Plotting
    print("\n[Plotting] Generating curves...")
    plt.figure(figsize=(10, 6))
    plt.style.use('default') 
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.plot(df['Rate']*100, df['MSE'], marker='o', linestyle='--', linewidth=2, color='gray', label='MSE-CNN (Baseline)')
    plt.plot(df['Rate']*100, df['PISE'], marker='s', linestyle='-', linewidth=3, color='#D95319', label='PISE (Ours)')
    
    plt.title("Sampling Rate Sensitivity Analysis", fontsize=14, fontweight='bold')
    plt.xlabel("Sampling Rate (%)", fontsize=12)
    plt.ylabel("Classification Accuracy (%)", fontsize=12)
    plt.xticks(np.array(CONFIG['RATES'])*100)
    plt.legend(fontsize=12)
    
    plot_path = os.path.join(SAVE_DIR, "plots", "Fig_Sensitivity_Curve.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[Success] All done! Full archive at: {SAVE_DIR}")

if __name__ == "__main__":
    main()