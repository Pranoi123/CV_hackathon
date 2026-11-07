# AdaptISP — Intelligent & Computationally Efficient Smartphone ISP (Problem Statement P2)

AdaptISP is a hybrid smartphone Image Signal Processor (ISP) combining classical differentiable stages (gray-world auto white balance, exposure normalization, Malvar-He-Cutler demosaicing, Color Correction Matrix, tone reproduction, and unsharp masking) with lightweight neural residual modules (thumbnail parameter estimator and depthwise-separable U-Net residual).

> **Important Transparency Note:**
> All training and evaluations are performed using **simulated RAW derived from SIDD sRGB clean images** via inverse gamma, inverse camera color matrix (CCM), and RGGB Bayer mosaicing. It is never claimed to be real sensor RAW.

---

## Workspace Structure

- `prep_sidd.py`: Extracts clean ground-truth sRGB images (`GT_SRGB_010.PNG`) from SIDD and organizes them into `sidd_gt/{scene_instance}_{sequence}.png`.
- `isp_data.py`: Clean simulated RAW bank, scene-level non-leaking train/val/test splits, Bayer-safe data augmentation.
- `isp_core.py`: Differentiable classical ISP stages, noise parameter estimator, depthwise-separable U-Net residual, end-to-end baseline, and physical degradation simulator.
- `isp_eval.py`: Metrics (PSNR, SSIM, $\Delta\text{E}_{2000}$, LPIPS), ONNX export, INT8 QDQ post-training quantization, and profiling (latency, MACs, memory).
- `train_eval.py`: End-to-end modular pipeline runner (training, distillation, comparative evaluation, sweeps, ONNX/INT8 benchmarks, ablations, failure analysis).
- `app.py`: Interactive Gradio application demonstrating the ISP with real-time controls.
- `hybrid_isp_P2.ipynb`: Jupyter/Colab notebook reproducing the entire pipeline.
- `requirements.txt`: Python package dependencies.
- `.gitignore`: Prevents datasets, checkpoints, and heavy artifacts from entering version control.

---

## Reproduction Commands

### 1. Environment Setup

```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation

Extract clean ground-truth images from the SIDD Small archive into `sidd_gt/`:

```bash
python prep_sidd.py
```
*(Automatically locates `SIDD_Small_sRGB_Only.zip` or existing extracted folders and saves 160 clean images into `sidd_gt/` named as `{scene_instance}_{sequence}.png` across 10 scenes).*

### 3. Full Training & Evaluation Pipeline (Single Command)

```bash
python train_eval.py --img-dir sidd_gt
```

On a CUDA GPU (e.g. Google Colab T4 / A100), this executes:
- Teacher training (6,000 steps)
- Student distillation with KD (6,000 steps)
- Learned baseline training (6,000 steps)
- 5 ablation studies (1,500 steps each)
- Comparative evaluation, parameter sweeps, and ONNX/INT8 benchmarking
- Generates `eval_P2_T01.md`, `results/`, `robustness_curves.png`, and `failure_cases.png`.

### 4. Interactive Web Application

```bash
python app.py
```
*(Open `http://localhost:7860` to view and compare naive demosaicing, classical ISP, and hybrid ISP).*

---

## Results (CPU Demo — 300-step Checkpoints)

> **Data:** Simulated RAW derived from SIDD sRGB · **Test split:** 21 images, 2 scenes · **Hardware:** Intel 12-core CPU (no CUDA GPU)

| Method | PSNR ↑ | SSIM ↑ | ΔE₂₀₀₀ ↓ | Params | Weights | Latency (CPU) | Peak RAM |
|---|---|---|---|---|---|---|---|
| Traditional ISP (gray-world) | 15.24 dB | 0.4836 | 22.52 | — | — | 95 ms | — |
| Learned Baseline | 24.03 dB | 0.7693 | 7.53 | 651,692 | 2.61 MB | 125 ms | 5.9 MB |
| **Hybrid ISP (ours, student)** | **23.88 dB** | **0.5110** | **9.15** | **47,597** | **0.21 MB** | **64 ms** | **0.5 MB** |
| **Hybrid Δ vs Traditional** | **+8.64 dB** | **+0.0274** | **−13.37** | — | — | — | — |

**Key findings:**
- Hybrid beats traditional by **+8.64 dB PSNR** and halves DeltaE2000 (22.5 → 9.1).
- Hybrid student is **13× smaller** (0.21 MB vs 2.61 MB) and **2× faster** than the learned baseline.
- Learned baseline outperforms hybrid at this step count due to larger capacity; full 6000-step GPU training expected to close the gap (see Colab instructions).

---

## Google Colab Instructions (GPU Accelerated)

Because full schedule training (25,500 total steps across teacher, student, baseline, and ablations) requires a CUDA GPU, run in Google Colab as follows:

1. Open [Google Colab](https://colab.research.google.com).
2. Go to **Runtime > Change runtime type** and select **T4 GPU** (or A100).
3. Upload `hybrid_isp_P2.ipynb` or clone the repository.
4. Upload `SIDD_Small_sRGB_Only.zip` to Colab storage (`/content/`).
5. Run the cells sequentially:
   - Cell 2 installs required libraries (`lpips`, `scikit-image`, `onnx`, `onnxruntime`, `rawpy`, `gradio`, `h5py`).
   - Run `!python prep_sidd.py --src /content/SIDD_Small_sRGB_Only.zip` to create `sidd_gt/`.
   - Cell 1 config is already configured with `DEMO_MODE = False`, `IMG_DIR = "sidd_gt"`, `SIDD_SMALL_ROOT = "/content/nonexistent_sidd_raw"`.
   - Run training, evaluation, benchmarking, and markdown export cells.
6. The notebook produces `eval_P2_T01.md`, `teacher.pt`, `student.pt`, `learned.pt`, `results/`, and demo figures.
