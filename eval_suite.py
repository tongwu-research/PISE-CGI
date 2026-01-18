"""
PISE: Semantically-Enhanced Deep Computational Ghost Imaging
Evaluation Suite for Reproducibility 
"""

import os
import argparse
import glob
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.nn.utils import spectral_norm
from PIL import Image
import numpy as np

# ==========================================
# 0. Device Helper
# ==========================================
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def device_synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()

# ==========================================
# 1. Physics Definitions
# ==========================================

class GhostImagingPhysics_V1(nn.Module):
    def __init__(self, size=28*28, rate=0.05, device='cpu'):
        super().__init__()
        self.N, self.M = size, int(size * rate)
        g = torch.Generator()
        g.manual_seed(42)
        # NOTE: Gaussian matrix implies negative values. In real optical setups,
        # this corresponds to differential ghost imaging (DGI) (Pattern - Inverse).
        self.register_buffer('A', torch.randn(self.M, self.N, generator=g) / (self.M ** 0.5))
        self.to(device)

    def forward_sensing(self, x, noise_std=0.0):
        x_flat = x.view(x.size(0), -1).float()
        y = torch.mm(x_flat, self.A.t())
        if noise_std > 0:
            y = y + torch.randn_like(y) * noise_std
        return y

    def backward_proxy(self, y):
        proxy = torch.mm(y, self.A)
        B = proxy.size(0)
        flat = proxy.view(B, -1)
        mi, ma = flat.min(1, True)[0], flat.max(1, True)[0]
        proxy = (flat - mi) / (ma - mi + 1e-6)
        return proxy.view(B, 1, 28, 28)

class GhostImagingPhysics_V2(GhostImagingPhysics_V1):
    def backward_proxy(self, y):
        return torch.mm(y, self.A)

# ==========================================
# 2. PISE Model Definitions
# ==========================================

class UNet_Exp1(nn.Module):
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

class UNet_Exp2(nn.Module):
    def __init__(self):
        super().__init__()
        def block(i, o): 
            return nn.Sequential(
                nn.Conv2d(i, o, 3, 1, 1), 
                nn.BatchNorm2d(o), 
                nn.ReLU(True)
            )
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
        b  = self.bottleneck(self.pool(e3))
        u3 = self.up3(b);  u3 = F.interpolate(u3, size=e3.shape[2:])
        d3 = self.dec3(torch.cat([u3, e3], 1))
        u2 = self.up2(d3); u2 = F.interpolate(u2, size=e2.shape[2:])
        d2 = self.dec2(torch.cat([u2, e2], 1))
        u1 = self.up1(d2); u1 = F.interpolate(u1, size=e1.shape[2:])
        d1 = self.dec1(torch.cat([u1, e1], 1))
        return self.head(d1)

# --- Architecture Wrappers ---
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
        x = self.phy.backward_proxy(y).view(-1, 1, 28, 28)
        x = self.adapter(x)
        return self.core(x)

# ==========================================
# 3. Baseline Models (For Speed Benchmark)
# ==========================================

class ISTANetPlusBlock(nn.Module):
    def __init__(self, n_features=32, img_size=28):
        super().__init__()
        self.img_size = img_size
        self.rho_param = nn.Parameter(torch.tensor(0.0)) 
        self.theta_param = nn.Parameter(torch.tensor(-2.0))
        self.prox = nn.Sequential(
            spectral_norm(nn.Conv2d(1, n_features, 3, 1, 1)), nn.ReLU(True),
            spectral_norm(nn.Conv2d(n_features, n_features, 3, 1, 1)), nn.ReLU(True),
            spectral_norm(nn.Conv2d(n_features, 1, 3, 1, 1))
        )
    def forward(self, x, y, physics):
        rho = 0.05 * torch.sigmoid(self.rho_param)
        theta = 0.05 * torch.sigmoid(self.theta_param)
        Ax = physics.forward_sensing(x)
        grad = physics.backward_proxy(Ax - y)
        x_grad = x - rho * grad
        x_prox = self.prox(x_grad)
        x_res = torch.sign(x_prox) * F.relu(torch.abs(x_prox) - theta)
        return x_grad + x_res

class ISTANetPlus(nn.Module):
    def __init__(self, physics, n_layers=9, n_features=32):
        super().__init__()
        self.physics = physics
        self.stages = nn.ModuleList([ISTANetPlusBlock(n_features, 28) for _ in range(n_layers)])
    def forward(self, y):
        x = self.physics.backward_proxy(y)
        for stage in self.stages:
            x = stage(x, y, self.physics)
        return x

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
        eta = 0.05 * torch.sigmoid(self.eta)
        rho = 0.5 * torch.sigmoid(self.rho) + 0.1
        Ax = physics.forward_sensing(x)
        grad = physics.backward_proxy(Ax - y)
        x_new = x - eta * grad + rho * (z - u - x)
        x_new = x_new + self.denoiser_x(x_new)
        z_new = (x_new + u) + self.denoiser_z(x_new + u)
        u_new = u + x_new - z_new
        return x_new, z_new, u_new

class ADMMCSNet(nn.Module):
    def __init__(self, physics, n_layers=7, n_features=32):
        super().__init__()
        self.physics = physics
        self.stages = nn.ModuleList([ADMMCSNetBlock(n_features, 28) for _ in range(n_layers)])
    def forward(self, y):
        x = self.physics.backward_proxy(y)
        z = x.clone()
        u = torch.zeros_like(x)
        for stage in self.stages:
            x, z, u = stage(x, z, u, y, self.physics)
        return x

# ==========================================
# 4. Main Reproducer Class
# ==========================================

class PaperReproducer:
    def __init__(self, demo=False):
        self.demo = demo
        self.device = get_device()

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.samples_dir = os.path.join(self.base_dir, "samples")
        self.weights_dir = os.path.join(self.base_dir, "weights")
        self.download_md = os.path.join(self.weights_dir, "download.md")

        print(f"[System] Script Location: {self.base_dir}")
        print(f"[System] Device: {self.device.type.upper()}")
        print(f"[System] Demo Mode: {self.demo}")

        # Sample validation
        self.has_samples = True
        found_files = []
        if os.path.exists(self.samples_dir):
            all_files = os.listdir(self.samples_dir)
            found_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not found_files:
            print(f"[Warning] No samples found in {self.samples_dir}.")
            print("Please create a 'samples' folder and put some 28x28 images (png/jpg) inside.")
            self.has_samples = False
        else:
            print(f"[System] Found {len(found_files)} sample images.")

    def _print_ckpt_help(self):
        print("[Hint] Checkpoints are expected under:")
        print(f"       {self.weights_dir}/")
        if os.path.exists(self.download_md):
            print(f"[Hint] Download guide: {self.download_md}")
        else:
            print("[Hint] download.md not found. You may provide a link in weights/download.md.")

    def _require_checkpoint(self, path, ckpt_name="checkpoint"):
        """Return True if checkpoint exists or demo mode enabled."""
        if os.path.exists(path):
            return True

        print(f"[Error] Missing {ckpt_name}: {path}")
        self._print_ckpt_help()

        if not self.demo:
            print("Abort this task to avoid producing non-paper results.")
            print("Tip: use --demo to run a pipeline sanity-check without paper checkpoints.")
            return False
        else:
            print("[Demo Mode] Continue with random weights (NOT paper results).")
            return True

    def load_samples(self):
        if not self.has_samples:
            raise FileNotFoundError("No samples found.")

        all_files = glob.glob(os.path.join(self.samples_dir, "*"))
        valid_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        imgs = []
        for f in valid_files[:5]:
            try:
                img = Image.open(f).convert('L').resize((28, 28))
                t_img = transforms.ToTensor()(img)
                imgs.append(t_img)
            except Exception as e:
                print(f"[Warning] Failed to load {f}: {e}")

        if not imgs:
            raise ValueError("Found files but failed to load them. Are they valid images?")

        return torch.stack(imgs).to(self.device)

    def calc_psnr(self, img1, img2):
        mse = F.mse_loss(img1, img2)
        if mse == 0:
            return 100
        return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()

    def measure_fps(self, model, dummy_input, physics, rounds=500):
        with torch.no_grad():
            for _ in range(20):
                y = physics.forward_sensing(dummy_input)
                _ = model(y)

        device_synchronize(self.device)
        start = time.time()
        with torch.no_grad():
            for _ in range(rounds):
                y = physics.forward_sensing(dummy_input)
                _ = model(y)
        device_synchronize(self.device)
        end = time.time()
        return rounds / (end - start)

    def reproduce_figure_2(self):
        print("\n>>> Reproduction Routine: Figure 2 (Reconstruction Quality)")
        if not self.has_samples:
            print("[Error] Cannot reproduce Figure 2 without sample images.")
            return

        phy = GhostImagingPhysics_V1(rate=0.05, device=self.device)
        model = UNet_Exp1().to(self.device)

        weight_path = os.path.join(self.weights_dir, "exp01_sampling", "Model_PISE_Rate_5pct.pth")
        if not self._require_checkpoint(weight_path, "PISE main checkpoint (5%)"):
            return

        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path, map_location=self.device))
        model.eval()

        imgs = self.load_samples()
        with torch.no_grad():
            y = phy.forward_sensing(imgs, noise_std=0.0)
            x_proxy = phy.backward_proxy(y)
            rec = model(x_proxy)

        avg_psnr = sum([self.calc_psnr(rec[i], imgs[i]) for i in range(len(imgs))]) / len(imgs)

        if self.demo and (not os.path.exists(weight_path)):
            print(f"[Demo Result] Figure 2 pipeline ran. (NOT paper results) Avg PSNR: {avg_psnr:.2f} dB")
        else:
            print(f"[Result] Figure 2 reproduced on provided samples. Avg PSNR: {avg_psnr:.2f} dB")

    def reproduce_table_1_ablation(self):
        print("\n>>> Reproducing Table 1 (Ablation Study)")
        if not self.has_samples:
            print("[Error] Cannot reproduce Table 1 without sample images.")
            return

        phy = GhostImagingPhysics_V2(rate=0.05, device=self.device)
        imgs = self.load_samples()

        models_cfg = {
            "A_Rand_MSE":  (RandomInitNet(phy.M, phy.N, UNet_Exp2()), "A_Rand_MSE_best.pth"),
            "B_Phys_MSE":  (PhysicsInitNet(phy, UNet_Exp2()),         "B_Phys_MSE_best.pth"),
            "C_Rand_Perc": (RandomInitNet(phy.M, phy.N, UNet_Exp2()), "C_Rand_Perc_best.pth"),
            "D_Ours_PISE": (PhysicsInitNet(phy, UNet_Exp2()),         "D_Ours_PISE_best.pth"),
        }

        # Checkpoint verification for ablation study
        ablation_dir = os.path.join(self.weights_dir, "exp02_ablation")
        if (not os.path.exists(ablation_dir) or len(os.listdir(ablation_dir)) == 0) and (not self.demo):
            print("[Error] Ablation checkpoints are not included in this appendix.")
            print("This task requires optional checkpoints. Please see weights/download.md")
            self._print_ckpt_help()
            return

        print(f"{'Method':<20} | {'PSNR (dB)':<10} | Status")
        print("-" * 55)

        for name, (net, w_name) in models_cfg.items():
            path = os.path.join(self.weights_dir, "exp02_ablation", w_name)
            net = net.to(self.device)

            if not self._require_checkpoint(path, f"Ablation checkpoint ({w_name})"):
                print(f"{name:<20} | {'--':<10} | Missing (Abort)")
                continue

            if os.path.exists(path):
                net.load_state_dict(torch.load(path, map_location=self.device))

            net.eval()
            with torch.no_grad():
                y = phy.forward_sensing(imgs, noise_std=0.0)
                rec = net(y)
            psnr = sum([self.calc_psnr(rec[i], imgs[i]) for i in range(len(imgs))]) / len(imgs)

            if self.demo and (not os.path.exists(path)):
                print(f"{name:<20} | {psnr:<10.2f} | Demo (NOT paper)")
            else:
                print(f"{name:<20} | {psnr:<10.2f} | OK")

    def reproduce_figure_3_robustness(self):
        print("\n>>> Reproducing Figure 3 (Robustness Curve)")
        if not self.has_samples:
            print("[Error] Cannot reproduce Figure 3 without sample images.")
            return

        phy = GhostImagingPhysics_V1(rate=0.05, device=self.device)
        model = UNet_Exp1().to(self.device)

        path = os.path.join(self.weights_dir, "exp01_sampling", "Model_PISE_Rate_5pct.pth")
        if not self._require_checkpoint(path, "PISE main checkpoint (5%)"):
            return

        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval()

        imgs = self.load_samples()
        noise_levels = [0.01, 0.05, 0.1, 0.2]

        print(f"{'Noise (Std)':<15} | {'PSNR (dB)':<10}")
        print("-" * 30)

        for sigma in noise_levels:
            with torch.no_grad():
                y = phy.forward_sensing(imgs, noise_std=sigma)
                x_proxy = phy.backward_proxy(y)
                rec = model(x_proxy)
            psnr = sum([self.calc_psnr(rec[i], imgs[i]) for i in range(len(imgs))]) / len(imgs)
            print(f"{sigma:<15} | {psnr:.2f}")

        if self.demo and (not os.path.exists(path)):
            print("[Demo Result] Robustness pipeline ran. (NOT paper results)")
        else:
            print("[Result] Robustness curve reproduced on provided samples.")

    def benchmark_efficiency(self):
        print("\n>>> Reproducing Table 3 (Real-time Efficiency Benchmark)")
        print(f"    [Config] Hardware: {self.device.type.upper()}")
        print("    [Config] Batch Size: 1 (Edge Inference Simulation)")

        phy = GhostImagingPhysics_V1(rate=0.05, device=self.device)
        dummy = torch.randn(1, 1, 28, 28).to(self.device)

        # Instantiate Models (weights are irrelevant for FPS in this micro-benchmark)
        m_ista = ISTANetPlus(phy, n_layers=9).to(self.device).eval()
        m_admm = ADMMCSNet(phy, n_layers=7).to(self.device).eval()

        class PISE_Full(nn.Module):
            def __init__(self, p, m):
                super().__init__()
                self.p = p
                self.m = m
            def forward(self, y):
                return self.m(self.p.backward_proxy(y))

        m_pise_core = UNet_Exp1().to(self.device).eval()
        m_pise = PISE_Full(phy, m_pise_core)

        # Parameter complexity assessment
        p_ista = sum(p.numel() for p in m_ista.parameters()) / 1e6
        p_admm = sum(p.numel() for p in m_admm.parameters()) / 1e6
        p_pise = sum(p.numel() for p in m_pise_core.parameters()) / 1e6

        # Inference latency measurement
        fps_ista = self.measure_fps(m_ista, dummy, phy)
        fps_admm = self.measure_fps(m_admm, dummy, phy)
        fps_pise = self.measure_fps(m_pise, dummy, phy)

        speedup = fps_pise / max(fps_ista, 1e-6)

        print("-" * 75)
        print(f"{'Method':<15} | {'Params':<8} | {'FPS (Real)':<12} | {'Speedup':<8}")
        print("-" * 75)
        print(f"{'ISTA-Net+':<15} | {p_ista:.2f}M   | {fps_ista:.0f}{'':<8} | 1.0x")
        print(f"{'ADMM-CSNet':<15} | {p_admm:.2f}M   | {fps_admm:.0f}{'':<8} | {fps_admm/fps_ista:.1f}x")
        print(f"{'PISE (Ours)':<15} | {p_pise:.2f}M   | {fps_pise:.0f}{'':<8} | {speedup:.1f}x")
        print("-" * 75)

        print("[Analysis & Disclaimer]")
        if self.device.type == 'cpu':
            print("1) Paper reports GPU speedup (~6x). CPU may show different absolute FPS.")
        elif self.device.type == 'mps':
            print("1) MPS may have high kernel launch overhead on tiny inputs (28x28).")
        print(f"2) Relative speedup here: {speedup:.1f}x (architecture-level efficiency).")
        print("-" * 75)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PISE Evaluation Suite (v6.2)")
    parser.add_argument(
        '--task',
        type=str,
        default='all',
        choices=['all', 'fig2', 'tab1', 'fig3', 'tab3'],
        help='Choose experiment to reproduce'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run without checkpoints (demo only, NOT paper results)'
    )
    args = parser.parse_args()

    r = PaperReproducer(demo=args.demo)

    if args.task in ['all', 'fig2']:
        r.reproduce_figure_2()
    if args.task in ['all', 'tab1']:
        r.reproduce_table_1_ablation()
    if args.task in ['all', 'fig3']:
        r.reproduce_figure_3_robustness()
    if args.task in ['all', 'tab3']:
        r.benchmark_efficiency()
