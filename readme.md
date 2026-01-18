# PISE: Physics-Anchored Semantically-Enhanced Deep CGI

This package provides the source code and evaluation suite for the IEICE letter:
"PISE: Physics-Anchored Semantically-Enhanced Deep Computational Ghost Imaging for Robust Low-Bandwidth Machine Perception".

## Setup
python3 -m pip install -r requirements.txt

## Checkpoints (Important)
- The main checkpoint for PISE at 5% sampling is expected at:
  weights/exp01_sampling/Model_PISE_Rate_5pct.pth

- Optional ablation checkpoints (Table 1) are included at:
  weights/exp02_ablation/

## Reproduce Results
### (1) Figure 2: Visual quality & PSNR (requires main checkpoint)
python eval_suite.py --task fig2

### (2) Figure 3: Robustness curve (requires main checkpoint)
python eval_suite.py --task fig3

### (3) Table 3: Efficiency benchmark (no checkpoint required)
python eval_suite.py --task tab3

> **Note on FPS:** > The FPS reported by this script measures the **end-to-end Python latency** (including interpreter overhead and data transfer) with Batch Size=1. 
> The paper reports the **peak GPU kernel throughput** (ideal hardware capacity). 
> While absolute FPS differs due to Python overhead, the **relative speedup ratio** (e.g., ~3x-6x vs baselines) remains consistent with the paper's conclusion.

### (4) Table 1: Ablation Study (requires ablation checkpoints)
python eval_suite.py --task tab1

## Demo Mode (No checkpoints)
This mode is ONLY for pipeline sanity-check and does NOT reproduce paper results:
python eval_suite.py --task fig2 --demo
