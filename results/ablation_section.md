## 3.3 Ablation Study

### Setup

Ablation models were trained from scratch on **simulated RAW derived from SIDD sRGB** clean
images processed through inverse gamma, inverse colour-correction matrix (CCM), and RGGB Bayer
mosaicing. The dataset contains 160 images grouped into 10 scenes
(train: 120 images / 7 scenes,
val: 19 images / 1 scenes,
test: 21 images / 2 scenes,
scene-level disjoint split, seed = 0).

Each variant was trained for **500 steps** (AdamW, lr = 2 × 10⁻³, cosine annealing,
batch = 8, curriculum noise schedule matching the main pipeline). Seeds used: [0].
Checkpoints were selected by validation PSNR only; the test set was never used during training
or model selection. Test evaluation used 48 fixed crops (128 x 128) with
degradation seed tied to the training seed, so every variant and every seed sees identical
noisy inputs. Metrics: PSNR (dB, ↑), SSIM (↑), ΔE 2000 (↓), LPIPS/AlexNet (↓).
CPU latency is measured at 256 × 256 (20 runs after 5 warmup, single batch).
Hardware: cpu (CPU-only).
Total wall time: 30.8 min.

> **Limitation**: all data is *simulated* RAW derived from sRGB images. Results may not
> generalise to real sensor RAW where noise statistics, demosaic artefacts, and colour
> characterisation errors differ substantially.

### Variants

| Variant | Description |
|---|---|
| `full` | Full model: HybridISP base=12, sep=True, estimator + noise map + residual + gate |
| `no_estimator` | Fixed gray-world gains and heuristic noise params instead of learned estimator |
| `no_noise_map` | U-Net receives white-balanced RAW only; no heteroscedastic noise-map input |
| `no_residual` | U-Net output directly replaces Malvar demosaic (no additive residual) |
| `learned_only` | End-to-end LearnedISP (base=20, sep=True) — no classical stages |

### Results Table

Single seed only; differences smaller than ~0.3 dB PSNR should be treated as inconclusive given run-to-run variance.

| Variant | PSNR (dB) | SSIM | ΔE 2000 | LPIPS | ΔPSNR vs full | ΔΔE vs full | Params | Latency@256 (ms) |
|---|---|---|---|---|---|---|---|---|
| full | 23.697 | 0.6587 | 9.534 | 0.4916 | +0.000 | +0.000 | 47,597 | 164.1 |
| learned_only | 23.675 | 0.7455 | 7.652 | 0.5398 | -0.022 | -1.883 | 79,752 | 158.5 |
| no_noise_map | 23.313 | 0.6537 | 9.723 | 0.5016 | -0.384 | +0.188 | 47,509 | 233.9 |
| no_residual | 16.887 | 0.3442 | 17.247 | 0.7149 | -6.810 | +7.713 | 47,597 | 39.0 |
| no_estimator | 15.508 | 0.3809 | 18.951 | 0.6325 | -8.189 | +9.417 | 30,104 | 79.3 |

### Per-finding Notes

- **learned_only**: ΔPSNR = -0.022 dB — within the single-seed noise floor (< 0.3 dB); no conclusive effect can be claimed.
- **no_estimator**: ΔPSNR = -8.189 dB, ΔΔE = +9.417 — removing this component is *associated with* lower quality; the removed module *may* contribute to performance.
- **no_noise_map**: ΔPSNR = -0.384 dB, ΔΔE = +0.188 — removing this component is *associated with* lower quality; the removed module *may* contribute to performance.
- **no_residual**: ΔPSNR = -6.810 dB, ΔΔE = +7.713 — removing this component is *associated with* lower quality; the removed module *may* contribute to performance.

### Honest Caveats

- All differences should be interpreted cautiously. With a single seed, differences
  below ~0.3 dB PSNR are within run-to-run noise.
- The ablation uses 500 training steps vs the full 500-step schedule —
  shorter training may penalise components that require more iterations to converge.
- Simulated RAW is a proxy for real sensor RAW; ranking of variants may shift on a real camera.
- If a removed module did *not* hurt, that may indicate redundancy or insufficient training
  to reveal the benefit, not that the module is useless.

### Condition Breakdown

#### Noise level (log₁₀ α)

| condition_value   |   full |   learned_only |   no_estimator |   no_noise_map |   no_residual |
|:------------------|-------:|---------------:|---------------:|---------------:|--------------:|
| high              | 20.396 |         24.051 |         15.63  |         21.362 |        17.89  |
| low               | 26.765 |         24.451 |         13.137 |         25.456 |        18.506 |
| mid               | 24.233 |         23.156 |         17.66  |         23.485 |        15.326 |

#### Illuminant temperature (K)

| condition_value   |   full |   learned_only |   no_estimator |   no_noise_map |   no_residual |
|:------------------|-------:|---------------:|---------------:|---------------:|--------------:|
| cool              | 22.598 |         21.371 |         17.409 |         21.234 |        14.276 |
| neutral           | 23.836 |         24.837 |         14.789 |         24.096 |        18.442 |
| warm              | 24.827 |         24.804 |         14.046 |         24.917 |        17.64  |

#### Exposure (EV)

| condition_value   |   full |   learned_only |   no_estimator |   no_noise_map |   no_residual |
|:------------------|-------:|---------------:|---------------:|---------------:|--------------:|
| bright            | 22.786 |         22.76  |         15.111 |         22.618 |        16.981 |
| dark              | 23.435 |         23.685 |         14.188 |         23.137 |        17.912 |
| normal            | 24.494 |         25.068 |         19.792 |         24.882 |        13.891 |


> Figure: `results/ablation_plot.png` — PSNR and ΔE bar charts per variant with error bars.
