"""
Script: Experiment 2 - Ablation Study (Sim-to-Real Robustness Mode)
"""

import os
import json
import shutil
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import vgg16, VGG16_Weights, resnet18

# --- 1. Environment & Config ---
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_DIR = os.path.join("./experiments", f"{TIMESTAMP}_Exp2_Ablation_Sim2Real")

os.makedirs(os.path.join(SAVE_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "weights"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "plots"), exist_ok=True)
os.makedirs("./weights_cache", exist_ok=True)

print(f"[System] Archive Directory: {SAVE_DIR}")

CONFIG = {
    'RATE': 0.05,
    'EPOCHS': 50,  # Extended to 50
    'BATCH_SIZE': 512, 
    'LR': 0.001,
    'PHOTON': 4000,
    'LAMBDA_PERC': 0.2,
    'SEED': 2026,
    'WORKERS': 8,
    'NOISE_TRAIN': 'gaussian', # Strategy: Domain Adaptation via Noise Distribution Mismatch 
    'NOISE_TEST': 'poisson'
}

with open(os.path.join(SAVE_DIR, "config.json"), "w") as f:
    json.dump(CONFIG, f, indent=4)

# --- 2. Physics with Dual Noise Mode ---
class GhostImagingPhysics(nn.Module):
    def __init__(self, size=28*28, rate=0.05):
        super().__init__()
        self.N, self.M = size, int(size * rate)
        g = torch.Generator()
        g.manual_seed(42) 
        # NOTE: Negative values simulate differential measurement (Pattern - Inverse),
        # allowing zero-mean sensing matrices despite non-negative light intensity.
        self.register_buffer('A', torch.randn(self.M, self.N, generator=g) / (self.M**0.5))

    def forward_sensing(self, x, noise_type='poisson'):
        x_flat = x.view(x.size(0), -1)
        y = torch.mm(x_flat, self.A.t())
        
        y_min, y_max = y.min(), y.max()
        y_norm = (y - y_min) / (y_max - y_min + 1e-8) 
        y_scaled = y_norm * CONFIG['PHOTON']
        
        #  Domain Gap Simulation: Gaussian Training vs. Poisson Inference 
        if noise_type == 'poisson':
            y_noisy = torch.poisson(y_scaled.clamp(min=0))
        else:
            # Gaussian approximation of Poisson distribution
            noise = torch.randn_like(y_scaled) * torch.sqrt(y_scaled.clamp(min=1e-3))
            y_noisy = y_scaled + noise

        y_out = (y_noisy / CONFIG['PHOTON']) * (y_max - y_min) + y_min
        return y_out

    def backward_proxy(self, y):
        return torch.mm(y, self.A)

# --- 3. Network Architectures ---
class UNetCore(nn.Module):
    def __init__(self):
        super().__init__()
        def block(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(True))
        self.enc1 = block(1, 32); self.pool = nn.MaxPool2d(2)
        self.enc2 = block(32, 64)
        self.enc3 = block(64, 128)
        self.bottleneck = block(128, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2); self.dec3 = block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2); self.dec2 = block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2); self.dec1 = block(64, 32)
        self.head = nn.Sequential(nn.Conv2d(32, 1, 1), nn.Sigmoid())

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        
        u3 = self.up3(b)
        if u3.size() != e3.size(): u3 = F.interpolate(u3, size=e3.shape[2:])
        d3 = self.dec3(torch.cat([u3, e3], 1)) 
        
        u2 = self.up2(d3)
        if u2.size() != e2.size(): u2 = F.interpolate(u2, size=e2.shape[2:])
        d2 = self.dec2(torch.cat([u2, e2], 1))
        
        u1 = self.up1(d2)
        if u1.size() != e1.size(): u1 = F.interpolate(u1, size=e1.shape[2:])
        d1 = self.dec1(torch.cat([u1, e1], 1))
        return self.head(d1)

class RandomInitNet(nn.Module):
    def __init__(self, M, N, core_net):
        super().__init__()
        self.fc = nn.Linear(M, N)
        self.core = core_net
    def forward(self, y):
        x = self.fc(y).view(-1, 1, 28, 28)
        return self.core(x)

class PhysicsInitNet(nn.Module):
    def __init__(self, phy, core_net):
        super().__init__()
        self.phy = phy
        self.core = core_net
        self.adapter = nn.Sequential(nn.BatchNorm2d(1), nn.Conv2d(1, 1, 3, 1, 1))
    def forward(self, y):
        x = self.phy.backward_proxy(y).view(-1, 1, 28, 28) #  Correct method call
        x = self.adapter(x)
        return self.core(x)

# --- 4. Losses ---
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features[:9].to(DEVICE).eval()
        for p in vgg.parameters(): p.requires_grad = False
        self.vgg = vgg
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1,3,1,1))

    def forward(self, x, y):
        x = (x.repeat(1,3,1,1) - self.mean) / self.std
        y = (y.repeat(1,3,1,1) - self.mean) / self.std
        return F.mse_loss(self.vgg(x), self.vgg(y)) # Chained features

# --- 5. Judge (Model serialization Secure) ---
class ResNetJudge(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.model.fc = nn.Linear(512, 10)
    def forward(self, x): return self.model(x)

def get_judge(ds_train):
    global_path = './weights_cache/ResNetEval.pth'
    local_path = os.path.join(SAVE_DIR, "weights", "Judge_Snapshot.pth") #  Local weight persistence
    judge = ResNetJudge().to(DEVICE)
    
    if os.path.exists(global_path):
        judge.load_state_dict(torch.load(global_path, map_location=DEVICE))
    else:
        print("[Evaluator] Cache miss. Training evaluator...") 
        judge.train()
        opt = optim.Adam(judge.parameters(), 1e-3)
        dl = torch.utils.data.DataLoader(ds_train, 256, True, num_workers=4)
        crit = nn.CrossEntropyLoss()
        for _ in range(5):
            for img, label in dl: # Explicit variable unpacking
                opt.zero_grad()
                loss = crit(judge(img.to(DEVICE)), label.to(DEVICE))
                loss.backward(); opt.step()
        torch.save(judge.state_dict(), global_path)
    
    torch.save(judge.state_dict(), local_path)
    judge.eval()
    return judge

# --- 6. Main Loop ---
def main():
    ts = transforms.ToTensor()
    ds_train = torchvision.datasets.FashionMNIST('./data', True, download=True, transform=ts)
    ds_test = torchvision.datasets.FashionMNIST('./data', False, transform=ts)
    dl_train = torch.utils.data.DataLoader(ds_train, CONFIG['BATCH_SIZE'], shuffle=True, num_workers=CONFIG['WORKERS'])
    dl_test = torch.utils.data.DataLoader(ds_test, 1000, False, num_workers=4)
    
    judge = get_judge(ds_train)
    phy = GhostImagingPhysics(rate=CONFIG['RATE']).to(DEVICE)
    vgg = VGGPerceptualLoss().to(DEVICE)
    mse = nn.MSELoss()
    
    models = {
        "A_Rand_MSE": RandomInitNet(phy.M, phy.N, UNetCore().to(DEVICE)).to(DEVICE),
        "B_Phys_MSE": PhysicsInitNet(phy, UNetCore().to(DEVICE)).to(DEVICE),
        "C_Rand_Perc": RandomInitNet(phy.M, phy.N, UNetCore().to(DEVICE)).to(DEVICE),
        "D_Ours_PISE": PhysicsInitNet(phy, UNetCore().to(DEVICE)).to(DEVICE)
    }
    
    optimizers = {k: optim.Adam(v.parameters(), lr=CONFIG['LR']) for k,v in models.items()}
    #  Learning rate scheduling
    schedulers = {k: optim.lr_scheduler.CosineAnnealingLR(optimizers[k], T_max=CONFIG['EPOCHS']) for k in models}
    
    history = {k: {'acc': [], 'loss': [], 'grad': []} for k in models}
    best_acc = {k: 0.0 for k in models}

    print(f">>> Strategy: Train({CONFIG['NOISE_TRAIN']}) -> Test({CONFIG['NOISE_TEST']})")
    
    for ep in range(CONFIG['EPOCHS']):
        for m in models.values(): m.train()
        ep_stats = {k: {'loss': 0.0, 'grad': 0.0} for k in models}
        steps = 0
        
        for img, _ in dl_train:
            img = img.to(DEVICE, non_blocking=True)
            #  Robustness training (Gaussian perturbation)
            y_meas = phy.forward_sensing(img, noise_type=CONFIG['NOISE_TRAIN'])
            
            for k, net in models.items():
                opt = optimizers[k]
                opt.zero_grad()
                rec = net(y_meas)
                
                l_mse = mse(rec, img)
                l_vgg = vgg(rec, img)
                loss = l_mse if "MSE" in k else l_mse + CONFIG['LAMBDA_PERC'] * l_vgg
                loss.backward()
                
                
                gnorm = sum(p.grad.data.norm(2).item()**2 for p in net.parameters() if p.grad is not None)**0.5
                ep_stats[k]['grad'] += gnorm
                ep_stats[k]['loss'] += loss.item()
                opt.step()
            steps += 1
        
        for k in schedulers: schedulers[k].step()

        # --- Evaluation ---
        for m in models.values(): m.eval()
        accs = {k: 0 for k in models}; total = 0
        with torch.no_grad():
            for img, label in dl_test:
                img, label = img.to(DEVICE), label.to(DEVICE)
                #  Robustness Test: Poisson Noise
                y_meas = phy.forward_sensing(img, noise_type=CONFIG['NOISE_TEST'])
                for k, net in models.items():
                    accs[k] += (judge(net(y_meas)).argmax(1) == label).sum().item()
                total += label.size(0)

        log_str = f"Ep {ep+1}"
        for k in models:
            acc = accs[k]/total*100
            history[k]['acc'].append(acc)
            history[k]['loss'].append(ep_stats[k]['loss']/steps)
            history[k]['grad'].append(ep_stats[k]['grad']/steps)
            log_str += f" | {k[0]}:{acc:.1f}%"
            if acc > best_acc[k]: 
                best_acc[k] = acc
                torch.save(models[k].state_dict(), os.path.join(SAVE_DIR, "weights", f"{k}_best.pth")) 
        print(log_str)

    # --- Model serialization ---
    for k in models:
        torch.save(models[k].state_dict(), os.path.join(SAVE_DIR, "weights", f"{k}_final.pth"))
    
    df = pd.DataFrame()
    for k in models:
        df[f"{k}_Acc"] = history[k]['acc']
        df[f"{k}_Grad"] = history[k]['grad']
    df.to_csv(os.path.join(SAVE_DIR, "ablation_metrics.csv"), index=False) 
    
    # Plot gradient norm for stability analysis
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1); 
    for k in models: plt.plot(history[k]['acc'], label=k)
    plt.title("Sim-to-Real Accuracy"); plt.legend(); plt.grid(True)
    plt.subplot(1,2,2); 
    for k in models: plt.plot(history[k]['grad'], label=k)
    plt.title("Gradient Norm (Stability)"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR, "plots", "Ablation_Analysis.png"))
    print(f"\n[Done] Saved to {SAVE_DIR}")

if __name__ == "__main__":
    main()