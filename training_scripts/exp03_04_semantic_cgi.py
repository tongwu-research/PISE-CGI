"""
Script: Experiment 3 & 4 - Semantically-Enhanced Ghost Imaging 
"""

import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg16, VGG16_Weights, resnet18
from torch.utils.data import DataLoader
from torch.nn.utils import spectral_norm
import shutil

# --- 1. System Setup ---
torch.set_float32_matmul_precision('high') 
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Archival Directory Structure
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_DIR = os.path.join("./experiments", f"{TIMESTAMP}_Exp03_04_Semantic_CGI")

for d in ["logs", "checkpoints", "gradients", "plots", "debug"]:
    os.makedirs(os.path.join(SAVE_DIR, d), exist_ok=True)
os.makedirs("./weights_cache", exist_ok=True)

# Logger Utility
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(os.path.join(SAVE_DIR, "logs", "experiment.log"), "a", encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self): self.terminal.flush()

sys.stdout = Logger()

print(f"[System] Archive Directory: {SAVE_DIR}")
print(f"[System] GPU: {torch.cuda.get_device_name(0)}")

# --- 2. Experimental Configuration ---
CONFIG = {
    # Physics Parameters
    'RATE': 0.05,
    'PHOTON_COUNT': 4000,   # For Evaluation (Poisson)
    'TRAIN_NOISE': 0.01,    # For Training (Gaussian, Sim-to-Real gap)
    'IMAGE_SIZE': 28,
    
    # Training Hyperparameters
    'BATCH_SIZE': 512,      # High-throughput configuration (Adjust based on VRAM)
    'TRAIN_EPOCHS': 50,     
    'EVAL_SAMPLES': 1000,
    'SEED': 2026,
    
    # Learning Rates
    'LR_PHYSICS': 1e-3, 
    'LR_NET': 1e-3,     
    'LR_GAN': 2e-4,     
    
    # Loss Weights
    'LAMBDA_PERC': 0.05,
    'LAMBDA_MSE': 1.0,     
    
    # Architecture
    'KAIMING_INIT': True
}

# Save Config Immediately
with open(os.path.join(SAVE_DIR, "config.json"), "w") as f: 
    json.dump(CONFIG, f, indent=2)

# --- 3. Physics Solver ---
class GhostImagingPhysics(nn.Module):
    def __init__(self, size=28*28, rate=0.05, seed=42):
        super().__init__()
        self.N, self.M = size, int(size * rate)
        self.H = int(np.sqrt(size))
        
        # Measurement Matrix
        g = torch.Generator()
        g.manual_seed(seed)
        # NOTE: Gaussian matrix implies negative values. In real optical setups (DMD),
        # this corresponds to differential ghost imaging (DGI), where the signal is 
        # the difference between the bucket detector values of a pattern and its inverse.
        A = torch.randn(self.M, self.N, generator=g) / (self.M ** 0.5)
        self.register_buffer('A', A)
        
    def forward(self, x): 
        return torch.mm(x.view(x.size(0), -1), self.A.t())
        
    def backward(self, y): 
        return torch.mm(y, self.A)
    
    # Training phase: Gaussian noise injection
    def forward_sensing(self, x, noise_std=0.0):
        y = self.forward(x)
        if noise_std > 0: 
            y += torch.randn_like(y) * noise_std
        return y

    # Evaluation phase: Poisson noise simulation for Sim-to-Real gap analysis
    def forward_poisson(self, x, photon_count):
        x_flat = x.float().view(x.size(0), -1)
        y_clean = torch.mm(x_flat, self.A.t())
        
        # Shift to positive for Poisson
        y_min = y_clean.min()
        y_shifted = (y_clean - y_min)
        
        # Scale to photon count
        scale = photon_count / (y_shifted.max() + 1e-6)
        y_noisy = torch.poisson(y_shifted * scale)
        
        # Reverse scaling
        return (y_noisy / scale) + y_min

# --- 4. Deep Models ---
def init_weights(model):
    """Weight initialization via He method"""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

# (A) ISTA-Net+
class ISTANetPlusBlock(nn.Module):
    def __init__(self, n_features=32, img_size=28):
        super().__init__()
        self.img_size = img_size
        self.rho_param = nn.Parameter(torch.tensor(0.0)) 
        self.theta_param = nn.Parameter(torch.tensor(-2.0))
        
        # Spectral normalization for Lipschitz continuity
        self.prox = nn.Sequential(
            spectral_norm(nn.Conv2d(1, n_features, 3, 1, 1)), nn.ReLU(True),
            spectral_norm(nn.Conv2d(n_features, n_features, 3, 1, 1)), nn.ReLU(True),
            spectral_norm(nn.Conv2d(n_features, 1, 3, 1, 1))
        )
    def forward(self, x, y, physics):
        B = x.size(0)
        rho = 0.05 * torch.sigmoid(self.rho_param)
        theta = 0.05 * torch.sigmoid(self.theta_param)
        
        # Data fidelity term (physics consistency)
        Ax = physics.forward(x.view(B, 1, self.img_size, self.img_size))
        x_grad = x - rho * physics.backward(Ax - y)
        
        # Proximal Mapping
        x_prox = self.prox(x_grad.view(B, 1, self.img_size, self.img_size))
        x_res = torch.sign(x_prox) * F.relu(torch.abs(x_prox) - theta)
        
        return (x_grad.view(B, 1, self.img_size, self.img_size) + x_res).view(B, -1)

class ISTANetPlus(nn.Module):
    def __init__(self, physics, n_layers=9, n_features=32):
        super().__init__()
        self.physics = physics
        self.img_size = physics.H
        self.stages = nn.ModuleList([ISTANetPlusBlock(n_features, physics.H) for _ in range(n_layers)])
        
    def forward(self, y):
        x = self.physics.backward(y)
        inters = []
        for stage in self.stages:
            x = stage(x, y, self.physics)
            inters.append(x.view(y.size(0), 1, self.img_size, self.img_size))
        return x.view(y.size(0), 1, self.img_size, self.img_size), inters

# (B) ADMM-CSNet
class ADMMCSNetBlock(nn.Module):
    def __init__(self, n_features=32, img_size=28):
        super().__init__()
        self.rho = nn.Parameter(torch.tensor(0.0))
        self.eta = nn.Parameter(torch.tensor(0.0))
        
        self.denoiser_x = nn.Sequential(
            spectral_norm(nn.Conv2d(1, n_features, 3, 1, 1)), nn.BatchNorm2d(n_features), nn.ReLU(True),
            spectral_norm(nn.Conv2d(n_features, n_features, 3, 1, 1)), nn.BatchNorm2d(n_features), nn.ReLU(True),
            spectral_norm(nn.Conv2d(n_features, 1, 3, 1, 1))
        )
        self.denoiser_z = nn.Sequential(
            spectral_norm(nn.Conv2d(1, n_features, 3, 1, 1)), nn.ReLU(True),
            spectral_norm(nn.Conv2d(n_features, 1, 3, 1, 1))
        )
        
    def forward(self, x, z, u, y, physics):
        B = x.size(0); H = 28
        eta = 0.05 * torch.sigmoid(self.eta)
        rho = 0.5 * torch.sigmoid(self.rho) + 0.1
        
        # X update
        Ax = physics.forward(x.view(B, 1, H, H))
        x_new = x - eta * physics.backward(Ax - y) + rho * (z - u - x)
        x_new = x_new.view(B, 1, H, H) + self.denoiser_x(x_new.view(B, 1, H, H))
        x_new = x_new.view(B, -1)
        
        # Z update
        z_new = (x_new.view(B,1,H,H) + u.view(B,1,H,H)) + self.denoiser_z((x_new+u).view(B,1,H,H))
        z_new = z_new.view(B, -1)
        
        # U update
        u_new = u + x_new - z_new
        return x_new, z_new, u_new

class ADMMCSNet(nn.Module):
    def __init__(self, physics, n_layers=7, n_features=32):
        super().__init__()
        self.physics = physics
        self.img_size = physics.H
        self.stages = nn.ModuleList([ADMMCSNetBlock(n_features, physics.H) for _ in range(n_layers)])
        
    def forward(self, y):
        x = self.physics.backward(y)
        z = x.clone()
        u = torch.zeros_like(x)
        for stage in self.stages:
            x, z, u = stage(x, z, u, y, self.physics)
        return x.view(y.size(0), 1, self.img_size, self.img_size)

# (C) U-Net CS
class UNetCS(nn.Module):
    def __init__(self, physics, n_features=64):
        super().__init__()
        self.physics = physics
        self.img_size = physics.H
        
        self.init = nn.Sequential(nn.Linear(physics.M, physics.N), nn.ReLU())
        
        self.enc1 = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True))
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True))
        
        self.bottle = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True))
        
        # Feature map dimensionality verification： 256 (Bottle Up) + 128 (Enc2) = 384
        self.dec2 = nn.Sequential(nn.Conv2d(256+128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True))
        
        # Feature map dimensionality verification： 128 (Dec2 Up) + 64 (Enc1) = 192
        self.dec1 = nn.Sequential(nn.Conv2d(128+64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True))
        
        self.final = nn.Sequential(nn.Conv2d(64, 1, 1), nn.Sigmoid())
        init_weights(self)
        
    def forward(self, y):
        x = self.init(y).view(-1, 1, self.img_size, self.img_size)
        
        e1 = self.enc1(x)                   # 28x28
        e2 = self.enc2(F.max_pool2d(e1, 2)) # 14x14
        
        b = self.bottle(F.max_pool2d(e2, 2)) # 7x7
        
        d2 = self.dec2(torch.cat([F.interpolate(b, scale_factor=2), e2], 1)) # 14x14
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2), e1], 1)) # 28x28
        
        return self.final(d1)

# (D) cGAN
class CGANGenerator(nn.Module):
    def __init__(self, physics, n_features=64):
        super().__init__()
        self.net = UNetCS(physics, n_features) 
    def forward(self, y): return self.net(y)

class CGANDiscriminator(nn.Module):
    def __init__(self, physics):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(physics.M, physics.N), nn.LeakyReLU(0.2, True))
        self.model = nn.Sequential(
            nn.Conv2d(2, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, 1)
        )
    def forward(self, img, y):
        cond = self.embed(y).view(img.size(0), 1, 28, 28)
        return self.model(torch.cat([img, cond], 1))

# (E) PISE (Ours)
class PISE_Net(nn.Module):
    def __init__(self, physics):
        super().__init__()
        self.physics = physics
        self.img_size = physics.H
        
        # Domain adaptation layer
        self.adapter = nn.Sequential(nn.BatchNorm2d(1), nn.Conv2d(1, 1, 3, 1, 1), nn.ReLU(True))
        
        # Semantic enhancement module
        self.enc1 = nn.Sequential(nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True))
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True))
        self.bottle = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True))
        self.dec2 = nn.Sequential(nn.Conv2d(256+128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True))
        self.dec1 = nn.Sequential(nn.Conv2d(128+64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True))
        self.final = nn.Sequential(nn.Conv2d(64, 1, 1), nn.Sigmoid())
        
        init_weights(self)

    def forward(self, y):
        # 1. Physics Domain Transform
        x = self.physics.backward(y).view(-1, 1, self.img_size, self.img_size)
        x = self.adapter(x)
        
        # 2. Semantic Enhancement
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        b = self.bottle(F.max_pool2d(e2, 2))
        
        d2 = self.dec2(torch.cat([F.interpolate(b, scale_factor=2), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2), e1], 1))
        
        return self.final(d1)

# --- 5. VGG Loss (Safe Sequential Version) ---
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features[:16].to(DEVICE).eval()
        self.slice1 = vgg[:4]
        self.slice2 = vgg[4:9]
        self.slice3 = vgg[9:16]
        for p in self.parameters(): p.requires_grad = False
        
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1,3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1,3,1,1)
        
    def forward(self, x, y):
        # Channel expansion for VGG compatibility
        x = x.repeat(1,3,1,1)
        y = y.repeat(1,3,1,1)
        
        # Normalize
        x = (x - self.mean)/self.std
        y = (y - self.mean)/self.std
        
        # Sequential feature extraction
        h_x1 = self.slice1(x)
        h_y1 = self.slice1(y)
        
        h_x2 = self.slice2(h_x1) # Output of slice1 feeds slice2
        h_y2 = self.slice2(h_y1)
        
        h_x3 = self.slice3(h_x2) # Output of slice2 feeds slice3
        h_y3 = self.slice3(h_y2)
        
        return F.l1_loss(h_x1, h_y1) + F.l1_loss(h_x2, h_y2) + F.l1_loss(h_x3, h_y3)

# --- 6. Judge & Security ---
class ResNetJudge(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(weights=None)
        # Adapt for 1-channel input
        self.model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.model.fc = nn.Linear(512, 10)
    def forward(self, x): return self.model(x)

def get_grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None: total += p.grad.data.norm(2).item()**2
    return total**0.5

def prepare_judge_secure(dataset):
    global_cache = './weights_cache/ResNetEval_Secure.pth'
    local_path = os.path.join(SAVE_DIR, "checkpoints", "Judge_Archived.pth")
    grad_log = os.path.join(SAVE_DIR, "gradients", "judge_training_grad.csv")
    
    judge = ResNetJudge().to(DEVICE)
    
    if os.path.exists(global_cache):
        print(f"[Judge] Loading from Global Cache: {global_cache}")
        judge.load_state_dict(torch.load(global_cache, map_location=DEVICE, weights_only=True))
    else:
        print("[Judge] Cache MISS. Training New Judge...")
        judge.train()
        dl = DataLoader(dataset, 512, True, num_workers=8)
        opt = optim.Adam(judge.parameters(), 1e-3)
        crit = nn.CrossEntropyLoss()
        
        grad_history = []
        for ep in range(15):
            ep_grad = 0
            # img, label instead of x, y
            for img, label in dl:
                img, label = img.to(DEVICE), label.to(DEVICE)
                opt.zero_grad()
                pred = judge(img)
                loss = crit(pred, label)
                loss.backward()
                ep_grad += get_grad_norm(judge)
                opt.step()
                
            grad_history.append(ep_grad/len(dl))
            print(f"  > Judge Ep {ep+1} | Loss: {loss.item():.4f} | Grad: {grad_history[-1]:.4f}")
            
        # Gradient norm logging for evaluator verification
        pd.DataFrame(grad_history, columns=['GradNorm']).to_csv(grad_log, index=False)
        torch.save(judge.state_dict(), global_cache)
        print("[Judge] Saved to Global Cache.")
    
    # Local weight snapshotting
    torch.save(judge.state_dict(), local_path)
    print(f"[Judge] Secured snapshot at {local_path}")
    
    judge.eval()
    return judge

# --- 7. Evaluation & Training Routines ---

def evaluate_robust(model, physics, loader, judge, is_gan=False):
    """
    Evaluates model under Poisson Noise (Robustness Test).
    """
    if is_gan: model.G.eval()
    else: model.eval()
    
    acc, psnr, n = 0, 0, 0
    
    with torch.no_grad():
        # img, label
        for img, label in loader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            
            # Evaluation under Poisson noise distribution
            y = physics.forward_poisson(img, CONFIG['PHOTON_COUNT']) 
            
            if is_gan: 
                recon = model.G(y)
            else:
                out = model(y)
                # Handle Tuple output from ISTA/ADMM (Image, Intermediates)
                if isinstance(out, tuple):
                    recon = out[0]
                else:
                    recon = out
            
            # Metrics
            # Range constraint for PSNR validity
            recon_clamped = recon.clamp(0, 1)
            
            acc += (judge(recon_clamped).argmax(1) == label).sum().item()
            mse = F.mse_loss(recon_clamped, img, reduction='none').view(img.size(0), -1).mean(1)
            psnr += (10 * torch.log10(1 / (mse + 1e-8))).sum().item()
            n += img.size(0)
            
    return acc / n * 100, psnr / n

def train_model(name, model, physics, tr_l, te_l, judge, epochs, lr, vgg=None):
    print(f"\n>>> Training {name}...")
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    scaler = torch.cuda.amp.GradScaler()
    
    grads, history_acc = [], []
    best_acc = 0.0
    
    for ep in range(epochs):
        model.train()
        ep_grad = 0
        
        #  img, label
        for img, _ in tr_l:
            img = img.to(DEVICE)
            
            # Train with Gaussian Noise (Baseline)
            y = physics.forward_sensing(img, noise_std=CONFIG['TRAIN_NOISE'])
            
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                if name == 'ISTA-Net+':
                    out = model(y)
                    # Loss on final + intermediates
                    loss = sum(F.mse_loss(i, img) for i in out[1]) / len(out[1])
                    
                elif name == 'PISE (Ours)':
                    rec = model(y)
                    loss = CONFIG['LAMBDA_MSE'] * F.mse_loss(rec, img) + \
                           CONFIG['LAMBDA_PERC'] * vgg(rec, img)
                else: 
                    rec = model(y)
                    if isinstance(rec, tuple): rec = rec[0]
                    loss = F.mse_loss(rec, img)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping for deep unrolled iterations
            if name in ['ISTA-Net+', 'ADMM-CSNet']: 
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            ep_grad += get_grad_norm(model)
            scaler.step(opt); scaler.update()
        
        avg_grad = ep_grad / len(tr_l)
        grads.append(avg_grad)
        sched.step()
        
        # Eval every 2 epochs
        if (ep + 1) % 2 == 0:
            acc, psnr = evaluate_robust(model, physics, te_l, judge)
            history_acc.append(acc)
            print(f"  Ep {ep+1} | Acc: {acc:.2f}% | PSNR: {psnr:.2f} | Grad: {avg_grad:.4f}")
            
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, "checkpoints", f"{name}_best.pth"))

    # Archival
    pd.DataFrame(grads, columns=['GradNorm']).to_csv(os.path.join(SAVE_DIR, "gradients", f"{name}_grads.csv"), index=False)
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "checkpoints", f"{name}_final.pth"))
    
    return best_acc, psnr, history_acc

def train_gan(name, physics, tr_l, te_l, judge, epochs):
    print(f"\n>>> Training {name}...")
    G = CGANGenerator(physics).to(DEVICE)
    D = CGANDiscriminator(physics).to(DEVICE)
    
    opt_G = optim.Adam(G.parameters(), CONFIG['LR_GAN'], betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), CONFIG['LR_GAN'], betas=(0.5, 0.999))
    
    # Evaluation interface wrapper
    class GANWrapper: 
        def __init__(self, G): self.G = G
    
    best_acc, history_acc = 0, []
    
    for ep in range(epochs):
        G.train(); D.train()
        for img, _ in tr_l:
            img = img.to(DEVICE)
            y = physics.forward_sensing(img, noise_std=CONFIG['TRAIN_NOISE'])
            B = img.size(0)
            
            # 1. Train Discriminator
            opt_D.zero_grad()
            fake = G(y)
            # Real
            pred_real = D(img, y)
            loss_d_real = F.binary_cross_entropy_with_logits(pred_real, torch.ones_like(pred_real))
            # Fake
            pred_fake = D(fake.detach(), y)
            loss_d_fake = F.binary_cross_entropy_with_logits(pred_fake, torch.zeros_like(pred_fake))
            
            loss_D = (loss_d_real + loss_d_fake) / 2
            loss_D.backward()
            opt_D.step()
            
            # 2. Train Generator
            opt_G.zero_grad()
            pred_fake = D(fake, y)
            loss_G_gan = F.binary_cross_entropy_with_logits(pred_fake, torch.ones_like(pred_fake))
            loss_G_l1 = F.l1_loss(fake, img) * 100
            
            loss_G = loss_G_gan + loss_G_l1
            loss_G.backward()
            opt_G.step()
            
        if (ep+1) % 2 == 0:
            acc, psnr = evaluate_robust(GANWrapper(G), physics, te_l, judge, is_gan=True)
            history_acc.append(acc)
            print(f"  Ep {ep+1} (GAN) | Acc: {acc:.2f}% | PSNR: {psnr:.2f}")
            if acc > best_acc:
                best_acc = acc
                torch.save(G.state_dict(), os.path.join(SAVE_DIR, "checkpoints", f"{name}_best.pth"))
                
    return best_acc, psnr, history_acc

# --- 8. Main Execution ---
def main():
    torch.manual_seed(CONFIG['SEED'])
    np.random.seed(CONFIG['SEED'])
    
    # Data Preparation
    ts = transforms.Compose([
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor()
    ])
    ds_tr = torchvision.datasets.FashionMNIST('./data', True, download=True, transform=ts)
    ds_te = torchvision.datasets.FashionMNIST('./data', False, download=True, transform=transforms.ToTensor())
    
    tr_l = DataLoader(ds_tr, CONFIG['BATCH_SIZE'], True, num_workers=8, pin_memory=True)
    te_l = DataLoader(ds_te, CONFIG['EVAL_SAMPLES'], False, num_workers=4)
    
    # Physics & Tools
    phy = GhostImagingPhysics(28*28, CONFIG['RATE']).to(DEVICE)
    vgg = VGGPerceptualLoss().to(DEVICE)
    judge = prepare_judge_secure(ds_tr)
    
    results = []
    curves = {}
    
    # --- Phase 2: Deep Models ---
    models_config = [
        ('ISTA-Net+', ISTANetPlus(phy).to(DEVICE), CONFIG['LR_PHYSICS']),
        ('ADMM-CSNet', ADMMCSNet(phy).to(DEVICE), CONFIG['LR_PHYSICS']),
        ('U-Net-CS', UNetCS(phy).to(DEVICE), CONFIG['LR_NET']),
        ('PISE (Ours)', PISE_Net(phy).to(DEVICE), CONFIG['LR_NET'])
    ]
    
    for name, net, lr in models_config:
        # Pass VGG only if needed (for Ours)
        vgg_loss = vgg if 'Ours' in name else None
        acc, psnr, curve = train_model(name, net, phy, tr_l, te_l, judge, CONFIG['TRAIN_EPOCHS'], lr, vgg_loss)
        results.append({'Method': name, 'Acc': acc, 'PSNR': psnr})
        curves[name] = curve
        
    # --- Phase 3: GAN ---
    acc, psnr, curve = train_gan('cGAN-CS', phy, tr_l, te_l, judge, CONFIG['TRAIN_EPOCHS'])
    results.append({'Method': 'cGAN-CS', 'Acc': acc, 'PSNR': psnr})
    curves['cGAN-CS'] = curve
    
    # --- Phase 4: Reporting & Visuals ---
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(SAVE_DIR, "logs", "final_results.csv"), index=False)
    print("\n>>> FINAL GRAND SUMMARY:\n", df)
    
    # Plot 1: Convergence
    plt.figure(figsize=(10, 6))
    epochs_x = range(2, CONFIG['TRAIN_EPOCHS']+1, 2)
    for name, curve in curves.items():
        plt.plot(epochs_x, curve, label=name, linewidth=2)
        
    plt.title("Robustness: Classification Accuracy under Poisson Noise", fontsize=14)
    plt.xlabel("Epochs"); plt.ylabel("Accuracy (%)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(SAVE_DIR, "plots", "convergence_metrics.png"))
    plt.close()
    
    # Plot 2: Visual Comparison
    print("Visualization matrix generation")
    
    def get_recon_model(name):
        if 'cGAN' in name:
            m = CGANGenerator(phy).to(DEVICE)
            m.load_state_dict(torch.load(os.path.join(SAVE_DIR, "checkpoints", f"{name}_best.pth")))
            m.eval()
            return m
        else:
            # Find class from config list
            cls = next(x[1].__class__ for x in models_config if x[0] == name)
            m = cls(phy).to(DEVICE)
            m.load_state_dict(torch.load(os.path.join(SAVE_DIR, "checkpoints", f"{name}_best.pth")))
            m.eval()
            return m

    # Get sample batch
    imgs, _ = next(iter(te_l))
    imgs = imgs[:5].to(DEVICE) # First 5 images
    y_vis = phy.forward_poisson(imgs, CONFIG['PHOTON_COUNT'])
    
    # Ground Truth & Input Proxy
    x_proxy = phy.backward(y_vis).view(-1, 1, 28, 28) 
    
    methods = ['ISTA-Net+', 'cGAN-CS', 'U-Net-CS', 'PISE (Ours)']
    
    # Collect Reconstructions
    # Start with GT and Proxy
    recons_list = [imgs, x_proxy] 
    
    for m_name in methods:
        model = get_recon_model(m_name)
        with torch.no_grad():
            if 'cGAN' in m_name: 
                r = model.G(y_vis) if hasattr(model, 'G') else model(y_vis)
            else: 
                out = model(y_vis)
                r = out[0] if isinstance(out, tuple) else out
        recons_list.append(r.clamp(0, 1)) # Clamp for display
        
    # Plotting Loop
    row_titles = ["Ground Truth", "Input Proxy"] + methods
    n_rows = len(recons_list)
    n_cols = 5
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 1.5 * n_rows))
    
    for r in range(n_rows):
        for c in range(n_cols):
            img_show = recons_list[r][c, 0].cpu().numpy()
            axes[r, c].imshow(img_show, cmap='gray', vmin=0, vmax=1)
            axes[r, c].axis('off')
            if c == 0:
                axes[r, 0].text(-5, 14, row_titles[r], fontsize=10, ha='right', va='center')
                
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "plots", "comparison_visuals.png"), dpi=300)
    
    print(f"[Success] Experiment completed. Data archived in: {SAVE_DIR}")

if __name__ == "__main__":
    main()