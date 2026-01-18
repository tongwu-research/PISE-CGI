"""
Script: Semantically-Enhanced Deep Computational Ghost Imaging (CGI)
"""

import os
import sys
import gc
import json
import time
import random
import datetime
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from contextlib import nullcontext

# --- 0. Environment & Hardware Setup ---
# Suppress verbose compilation logs
os.environ["TORCHDYNAMO_VERBOSE"] = "0"
os.environ["TORCHINDUCTOR_VERBOSE"] = "0"
os.environ["TRITON_PRINT_AUTOTUNING"] = "0"
# Global Cache for evaluator Weights
os.environ['TORCH_HOME'] = './weights_cache'
os.makedirs('./weights_cache', exist_ok=True)

# Enable deterministic algorithms if needed
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.models import resnet18, vgg16, VGG16_Weights


warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Global Configuration ---
CONFIG = {
    'SAMPLING_RATE': 0.05,      # 5% Sampling Rate
    'BATCH_SIZE': 2048,         # Large batch size to ensure accurate gradient estimation
    'EPOCHS_BASE': 50,          # Epochs for Baseline
    'EPOCHS_OURS': 50,          # Epochs for Ours
    'LAMBDA_PERC': 0.1,         # Perceptual Loss Weight
    'LR': 0.002,                # Learning Rate
    'NUM_RUNS': 5,              # Number of independent runs
    'TRAIN_NOISE_STD': 0.01,    # Noise during training
    'EVAL_NOISE_STD': 0.0,      # Clean evaluation
    'BASE_SEED': 20260109,      # Seed for stochastic reproducibility
    'USE_COMPILE': True,        # JIT compilation flag (disable for debugging)
    'DETERMINISTIC': False      # Enable for deterministic benchmarking (CuDNN deterministic mode)
}

# --- 2. Hardware Optimization ---
def setup_hardware():
    if torch.cuda.is_available():
        if CONFIG['DETERMINISTIC']:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False 
            torch.backends.cudnn.deterministic = True
            print(f"[System] Hardware Setup: Deterministic Mode (Slower, Reproducible)")
        else:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True 
            torch.backends.cudnn.deterministic = False
            print(f"[System] Hardware Setup: Performance Mode (Fast, TF32=On)")
    else:
        print("[System] Hardware Setup: CPU Mode")

def get_amp_config():
    """Returns dtype and whether to use GradScaler."""
    use_scaler = False
    dtype = torch.float32
    
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            use_scaler = False # BF16 usually doesn't need scaling
            print("[System] AMP: Using BFloat16 (No Scaler)")
        else:
            dtype = torch.float16
            use_scaler = True  # FP16 needs scaling to prevent underflow
            print("[System] AMP: Using Float16 (With Scaler)")
    else:
        print("[System] AMP: Disabled (CPU)")
        
    return dtype, use_scaler

setup_hardware()
AMP_DTYPE, USE_SCALER = get_amp_config()

# --- 3. Utilities ---
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def maybe_compile(model):
    """Conditionally compiles the model on CUDA."""
    if CONFIG.get('USE_COMPILE', False) and hasattr(torch, "compile") and DEVICE.type == 'cuda':
        return torch.compile(model, mode="reduce-overhead")
    return model

def setup_experiment_folder():
    base_dir = "./experiments"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{timestamp}_SR{CONFIG['SAMPLING_RATE']}_MainExperiment"
    save_path = os.path.join(base_dir, exp_name)
    os.makedirs(os.path.join(save_path, "weights"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "plots"), exist_ok=True)
    
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=4)
        
    print(f"[System] Experiment Archive Created: {save_path}")
    return save_path

SAVE_DIR = setup_experiment_folder()

def save_clean_checkpoint(model, path):
    state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("_orig_mod.", "")
        new_state_dict[name] = v
    torch.save(new_state_dict, path)

# --- 4. Models ---
class GhostImagingPhysics(nn.Module):
    def __init__(self, size=28*28, rate=0.10, dev=None):
        super().__init__()
        if dev is None:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.N, self.M = size, int(size * rate)
        g = torch.Generator()
        g.manual_seed(42) 
        # NOTE: Simulating differential measurement protocol (DGI) effectively allows 
        # for zero-mean sensing matrices despite non-negative light intensity.
        self.A = torch.randn(self.M, self.N, generator=g).to(dev)
        self.A /= (self.M ** 0.5)

    def forward_sensing(self, x, noise_std=0.0):
        # Enforce FP32 precision for numerical fidelity in physics modeling
        x = x.float() 
        flat_x = x.view(x.size(0), -1)
        y = torch.mm(flat_x, self.A.t())
        if noise_std > 0.0:
            y = y + torch.randn_like(y) * noise_std
        return y

    def backward_proxy(self, y):
        # Physics inversion
        s = int(np.sqrt(self.N))
        proxy = torch.mm(y, self.A)
        B = proxy.size(0); flat = proxy.view(B, -1)
        mi, ma = flat.min(1, True)[0], flat.max(1, True)[0]
        proxy = (flat - mi) / (ma - mi + 1e-6)
        return proxy.view(B, 1, s, s)

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(UNet, self).__init__()
        def d_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, 1, 1), nn.BatchNorm2d(out_c), nn.ReLU(True),
                nn.Conv2d(out_c, out_c, 3, 1, 1), nn.BatchNorm2d(out_c), nn.ReLU(True)
            )
        self.inc = d_conv(n_channels, 32)
        self.down1 = d_conv(32, 64); self.pool = nn.MaxPool2d(2)
        self.down2 = d_conv(64, 128); self.down3 = d_conv(128, 256)
        self.up1 = nn.ConvTranspose2d(256, 128, 2, 2); self.conv1 = d_conv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2); self.conv2 = d_conv(128, 64)
        self.up3 = nn.ConvTranspose2d(64, 32, 2, 2); self.conv3 = d_conv(64, 32)
        self.outc = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(self.pool(x1))
        x3 = self.down2(self.pool(x2)); x4 = self.down3(self.pool(x3))
        x = self.up1(x4)
        if x.shape != x3.shape: x = F.interpolate(x, size=x3.shape[2:], mode='bilinear')
        x = torch.cat([x3, x], dim=1); x = self.conv1(x)
        x = self.up2(x)
        if x.shape != x2.shape: x = F.interpolate(x, size=x2.shape[2:], mode='bilinear')
        x = torch.cat([x2, x], dim=1); x = self.conv2(x)
        x = self.up3(x)
        if x.shape != x1.shape: x = F.interpolate(x, size=x1.shape[2:], mode='bilinear')
        x = torch.cat([x1, x], dim=1); x = self.conv3(x)
        return torch.sigmoid(self.outc(x))

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        try: vgg = vgg16(weights=VGG16_Weights.DEFAULT).features
        except: vgg = vgg16(pretrained=True).features
        self.slice = nn.Sequential()
        for x in range(16): self.slice.add_module(str(x), vgg[x])
        self.slice.eval()
        for p in self.slice.parameters(): p.requires_grad = False
        self.slice.to(DEVICE)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1,3,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1,3,1,1))

    def forward(self, x, y):
        if x.shape[1] == 1: x = x.repeat(1, 3, 1, 1); y = y.repeat(1, 3, 1, 1)
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        y = (y - self.mean.to(y.device)) / self.std.to(y.device)
        return F.l1_loss(self.slice(x), self.slice(y))

# --- 5. Evaluators ---
class ResNetEval(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.model.fc = nn.Linear(512, 10)
    def forward(self, x): return self.model(x)

class VGGEval(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,1,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,1,1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,1,1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128*3*3, 10))
    def forward(self, x): return self.classifier(self.features(x))

# --- 6. Dataset & Helper Functions ---
def build_datasets():
    ts = transforms.ToTensor()
    try:
        trainset = torchvision.datasets.FashionMNIST('./data', train=True, download=True, transform=ts)
        testset = torchvision.datasets.FashionMNIST('./data', train=False, download=True, transform=ts)
    except:
        trainset = torchvision.datasets.FashionMNIST('./data', train=True, download=False, transform=ts)
        testset = torchvision.datasets.FashionMNIST('./data', train=False, download=False, transform=ts)
    return trainset, testset

def make_loader(dataset, batch_size, seed, shuffle=True):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, 
        num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4,
        generator=g if shuffle else None, worker_init_fn=seed_worker 
    )

def prepare_evaluators(dataset, weights_dir):
    """Automated initialization of downstream evaluators"""
    CACHE_DIR = './weights_cache'
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    res_path = os.path.join(CACHE_DIR, "ResNetEval.pth")
    vgg_path = os.path.join(CACHE_DIR, "VGGEval.pth")
    
    res = ResNetEval().to(DEVICE)
    vgg = VGGEval().to(DEVICE)

    if os.path.exists(res_path) and os.path.exists(vgg_path):
        print(f"[System] Loading EXISTING Evaluators from {CACHE_DIR}...")
        res.load_state_dict(torch.load(res_path, map_location=DEVICE))
        vgg.load_state_dict(torch.load(vgg_path, map_location=DEVICE))
    else:
        print(f"\n[System] Evaluators NOT found in {CACHE_DIR}. Training evaluators...")
        judge_loader = make_loader(dataset, 256, seed=999, shuffle=True)
        opt_r = optim.Adam(res.parameters(), lr=0.005)
        opt_v = optim.Adam(vgg.parameters(), lr=0.005)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(15):
            res.train(); vgg.train()
            correct_r, correct_v, total = 0, 0, 0
            for i, l in judge_loader:
                i, l = i.to(DEVICE), l.to(DEVICE)
                
                # Computational optimization: Single forward pass
                out_r = res(i)
                opt_r.zero_grad(); loss_r = criterion(out_r, l); loss_r.backward(); opt_r.step()
                
                out_v = vgg(i)
                opt_v.zero_grad(); loss_v = criterion(out_v, l); loss_v.backward(); opt_v.step()
                
                correct_r += (out_r.argmax(1) == l).sum().item()
                correct_v += (out_v.argmax(1) == l).sum().item()
                total += l.size(0)
            print(f"  > Evaluator Train Ep {epoch+1}/15 | ResNet: {correct_r/total:.2%} | VGG: {correct_v/total:.2%}", end='\r')
        
        print(f"\n[System] Evaluator training completed. Saving to {CACHE_DIR}...")
        torch.save(res.state_dict(), res_path)
        torch.save(vgg.state_dict(), vgg_path)

    res.eval(); vgg.eval()
    for p in res.parameters(): p.requires_grad = False
    for p in vgg.parameters(): p.requires_grad = False
    return res, vgg

# --- 7. Metrics ---
@torch.no_grad()
def calc_metrics(net, res, vgg, phy, loader):
    net.eval()
    acc_r, acc_v, t_psnr, count = 0, 0, 0, 0
    for i, l in loader:
        i, l = i.to(DEVICE, non_blocking=True), l.to(DEVICE, non_blocking=True)
        # Clean Eval
        y = phy.forward_sensing(i, noise_std=CONFIG['EVAL_NOISE_STD'])
        xp = phy.backward_proxy(y)
        xr = net(xp)
        acc_r += (res(xr).argmax(1) == l).sum().item()
        acc_v += (vgg(xr).argmax(1) == l).sum().item()
        mse = torch.mean((xr - i)**2, dim=(1,2,3)).clamp_min(1e-10) # Numerical stability constant to prevent logarithmic singularity
        t_psnr += (20 * torch.log10(1.0 / torch.sqrt(mse))).sum().item()
        count += i.size(0)
    return acc_r/count*100, acc_v/count*100, t_psnr/count

def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5

# --- 8. Main Loop ---
def main():
    set_seed(CONFIG['BASE_SEED'])
    trainset, testset = build_datasets()
    te_l = make_loader(testset, 256, seed=42, shuffle=False)
    
    phy = GhostImagingPhysics(28*28, CONFIG['SAMPLING_RATE'], DEVICE)
    res_eval, vgg_eval = prepare_evaluators(trainset, './weights_cache')
    
    # Initialize VGG Loss once, outside the loop
    perc_crit = VGGPerceptualLoss().to(DEVICE)
    
    history = {
        'base_acc': [], 'base_loss': [], 'base_grad': [],
        'ours_acc': [], 'ours_loss': [], 'ours_grad': []
    }
    stats = {'base': [], 'ours': []} 
    run_metrics_list = [] 
    metrics_csv_path = os.path.join(SAVE_DIR, "run_metrics_live.csv")

    # Conditional Scaler Initialization (Hardware-Adaptive)
    if DEVICE.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler(enabled=USE_SCALER)
        amp_ctx = torch.autocast(device_type='cuda', dtype=AMP_DTYPE)
    else:
        scaler = None
        amp_ctx = nullcontext()

    for run in range(CONFIG['NUM_RUNS']):
        print(f"\n[Run {run+1}/{CONFIG['NUM_RUNS']}]")
        run_seed = CONFIG['BASE_SEED'] + run * 100
        
        # Seed management: Independent weight initialization vs. shared data shuffling
        set_seed(run_seed)
        init_weights = UNet().to(DEVICE).state_dict()
        data_seed = run_seed + 9999
        
        # --- PHASE 1: Baseline ---
        print("  > Training Baseline...")
        set_seed(run_seed) # Ensure noise sequence starts at same state
        
        tr_l = make_loader(trainset, CONFIG['BATCH_SIZE'], seed=data_seed)
        
        net = UNet().to(DEVICE); net.load_state_dict(init_weights); net = maybe_compile(net)
        opt = optim.Adam(net.parameters(), lr=CONFIG['LR'])
        sc = optim.lr_scheduler.OneCycleLR(opt, CONFIG['LR'], steps_per_epoch=len(tr_l), epochs=CONFIG['EPOCHS_BASE'])
        mse_crit = nn.MSELoss()
        
        run_acc, run_loss, run_grad = [], [], []
        
        for e in range(CONFIG['EPOCHS_BASE']):
            net.train()
            epoch_grad, epoch_loss, batches = 0, 0, 0
            for i, _ in tr_l:
                i = i.to(DEVICE, non_blocking=True)
                
                with torch.no_grad():
                    xp = phy.backward_proxy(phy.forward_sensing(i, noise_std=CONFIG['TRAIN_NOISE_STD']))
                
                with amp_ctx:
                    loss = mse_crit(net(xp), i)
                
                # Mixed-precision backward propagation with scaling
                opt.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()
                
                #  Gradient norm computation (interval-based)
                if batches % 10 == 0:
                    epoch_grad += get_grad_norm(net)
                epoch_loss += loss.item()
                batches += 1
                sc.step()
            
            rb, _, pb = calc_metrics(net, res_eval, vgg_eval, phy, te_l)
            run_acc.append(rb); run_loss.append(epoch_loss/batches); run_grad.append(epoch_grad/(batches/10))
            print(f"    Epoch {e+1}: Acc={rb:.2f}%, PSNR={pb:.2f} dB", end='\r')
            
        history['base_acc'].append(run_acc); history['base_loss'].append(run_loss); history['base_grad'].append(run_grad)
        
        rb_final, vb_final, pb_final = calc_metrics(net, res_eval, vgg_eval, phy, te_l)
        stats['base'].append([rb_final, vb_final, pb_final])
        
        # Save Baseline Checkpoint
        save_clean_checkpoint(net, os.path.join(SAVE_DIR, "weights", f"Baseline_Run{run+1}.pth"))
        
        # --- PHASE 2: Ours ---
        print("\n  > Training Ours...")
        set_seed(run_seed) # Reset Seed -> STRICTLY FAIR noise sequence
        
        tr_l = make_loader(trainset, CONFIG['BATCH_SIZE'], seed=data_seed)

        net = UNet().to(DEVICE); net.load_state_dict(init_weights); net = maybe_compile(net)
        opt = optim.Adam(net.parameters(), lr=CONFIG['LR'])
        sc = optim.lr_scheduler.OneCycleLR(opt, CONFIG['LR'], steps_per_epoch=len(tr_l), epochs=CONFIG['EPOCHS_OURS'])
        
        run_acc, run_loss, run_grad = [], [], []
        
        for e in range(CONFIG['EPOCHS_OURS']):
            net.train()
            epoch_grad, epoch_loss, batches = 0, 0, 0
            for i, _ in tr_l:
                i = i.to(DEVICE, non_blocking=True)
                
                with torch.no_grad():
                    xp = phy.backward_proxy(phy.forward_sensing(i, noise_std=CONFIG['TRAIN_NOISE_STD']))
                
                with amp_ctx:
                    xr = net(xp)
                    # Use pre-initialized perc_crit
                    loss = mse_crit(xr, i) + CONFIG['LAMBDA_PERC'] * perc_crit(xr, i)
                
                opt.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()
                
                if batches % 10 == 0:
                    epoch_grad += get_grad_norm(net)
                epoch_loss += loss.item()
                batches += 1
                sc.step()
            
            ro, _, po = calc_metrics(net, res_eval, vgg_eval, phy, te_l)
            run_acc.append(ro); run_loss.append(epoch_loss/batches); run_grad.append(epoch_grad/(batches/10))
            print(f"    Epoch {e+1}: Acc={ro:.2f}%, PSNR={po:.2f} dB", end='\r')
            
        history['ours_acc'].append(run_acc); history['ours_loss'].append(run_loss); history['ours_grad'].append(run_grad)
        
        ro_final, vo_final, po_final = calc_metrics(net, res_eval, vgg_eval, phy, te_l)
        stats['ours'].append([ro_final, vo_final, po_final])
        
        save_clean_checkpoint(net, os.path.join(SAVE_DIR, "weights", f"Ours_Run{run+1}.pth"))
        
        # Real-time metric logging and persistence
        run_data = [
            {"Run": run+1, "Method": "Baseline", "ResNet": rb_final, "VGG": vb_final, "PSNR": pb_final},
            {"Run": run+1, "Method": "Ours", "ResNet": ro_final, "VGG": vo_final, "PSNR": po_final}
        ]
        run_metrics_list.extend(run_data)
        
        df_live = pd.DataFrame(run_data)
        df_live.to_csv(metrics_csv_path, mode='a', header=not os.path.exists(metrics_csv_path), index=False)
        
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    # --- 9. Analysis, CSV Export & Visualization ---
    
    print("\n[Data] Generating Summary Statistics...")
    df = pd.DataFrame(run_metrics_list)
    # Save final complete CSV (overwrites live one with cleaner format if needed, or separate)
    df.to_csv(os.path.join(SAVE_DIR, "run_metrics_final.csv"), index=False)
    
    summary = df.groupby("Method")[["ResNet", "VGG", "PSNR"]].agg(['mean', 'std'])
    summary.to_csv(os.path.join(SAVE_DIR, "summary_metrics.csv"))
    print(summary)

    # 9.2 Convergence Plots
    def plot_convergence(hist_base, hist_ours, title, ylabel, filename):
        base = np.array(hist_base); ours = np.array(hist_ours)
        epochs = range(1, base.shape[1] + 1)
        b_mean, b_std = base.mean(0), base.std(0)
        o_mean, o_std = ours.mean(0), ours.std(0)
        
        plt.figure(figsize=(7, 5))
        plt.plot(epochs, b_mean, label='Baseline', color='tab:blue', linewidth=2)
        plt.fill_between(epochs, b_mean - b_std, b_mean + b_std, color='tab:blue', alpha=0.15)
        plt.plot(epochs, o_mean, label='Ours', color='tab:orange', linewidth=2)
        plt.fill_between(epochs, o_mean - o_std, o_mean + o_std, color='tab:orange', alpha=0.15)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Epochs', fontsize=12); plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, "plots", filename), dpi=300)
        plt.close()

    print("Visualization generation")
    plot_convergence(history['base_acc'], history['ours_acc'], 
                     "Validation Accuracy Convergence", "ResNet Accuracy (%)", "convergence_accuracy.png")
    plot_convergence(history['base_grad'], history['ours_grad'], 
                     "Optimization Stability", "Gradient L2 Norm", "convergence_gradient.png")
    np.save(os.path.join(SAVE_DIR, "history.npy"), history)
    
    # 9.3 Visual Comparison
    avg_psnr = [(stats['base'][r][2] + stats['ours'][r][2])/2 for r in range(CONFIG['NUM_RUNS'])]
    median_idx = int(np.argsort(avg_psnr)[len(avg_psnr)//2])
    print(f"[Info] Median performance visualization (Run {median_idx+1})")

    net_b = UNet().to(DEVICE); net_b.load_state_dict(torch.load(os.path.join(SAVE_DIR, "weights", f"Baseline_Run{median_idx+1}.pth"), map_location=DEVICE)); net_b.eval()
    net_o = UNet().to(DEVICE); net_o.load_state_dict(torch.load(os.path.join(SAVE_DIR, "weights", f"Ours_Run{median_idx+1}.pth"), map_location=DEVICE)); net_o.eval()
    
    imgs, _ = next(iter(te_l)); imgs = imgs[:5].to(DEVICE)
    with torch.no_grad():
        xp = phy.backward_proxy(phy.forward_sensing(imgs))
        rec_b, rec_o = net_b(xp).cpu(), net_o(xp).cpu()
        imgs, xp = imgs.cpu(), xp.cpu()

    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    for k in range(5):
        axes[0,k].imshow(imgs[k,0], cmap='gray'); axes[1,k].imshow(xp[k,0], cmap='gray')
        axes[2,k].imshow(rec_b[k,0], cmap='gray'); axes[3,k].imshow(rec_o[k,0], cmap='gray')
        if k==0:
            axes[0,0].set_ylabel("Ground Truth", fontsize=12); axes[1,0].set_ylabel("Input Proxy", fontsize=12)
            axes[2,0].set_ylabel("Baseline", fontsize=12); axes[3,0].set_ylabel("Ours", fontsize=12)
    [ax.set_xticks([]) for ax in axes.flat]; [ax.set_yticks([]) for ax in axes.flat]
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "plots", "final_comparison.png"))

    # 9.4 ROBUST DATA EXPORT
    print("\n[Robustness] Exporting raw data for offline plotting...")
    net_b.eval(); net_o.eval()
    imgs_tensor, _ = next(iter(te_l)); imgs_tensor = imgs_tensor[:8].to(DEVICE)
    with torch.no_grad():
        xp_tensor = phy.backward_proxy(phy.forward_sensing(imgs_tensor))
        rec_b_tensor = net_b(xp_tensor); rec_o_tensor = net_o(xp_tensor)
    
    export_data = {
        'config': json.dumps(CONFIG),
        'base_acc': np.array(history['base_acc']),
        'ours_acc': np.array(history['ours_acc']),
        'base_grad': np.array(history['base_grad']),
        'ours_grad': np.array(history['ours_grad']),
        'vis_gt': imgs_tensor.cpu().numpy(),
        'vis_proxy': xp_tensor.cpu().numpy(),
        'vis_base': rec_b_tensor.cpu().numpy(),
        'vis_ours': rec_o_tensor.cpu().numpy()
    }
    save_path = os.path.join(SAVE_DIR, "paper_data_package.npz")
    np.savez_compressed(save_path, **export_data)
    print(f"[Success] Data exported to: {save_path}")
    print(f"Experiment Complete. Results saved to {SAVE_DIR}")

if __name__ == "__main__":
    main()