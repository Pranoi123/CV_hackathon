1RUA24CSE0303,1RUA24CSE0312,1RUA24CSE0277,1RUA24CSE0314,1RUA24CSE0317

# Comparative Evaluation: Problem P2 (Smartphone ISP), Team T07

Repository: https://github.com/Pranoi123/CV_hackathon

## 1. Evaluation setup

| Item | Value |
|---|---|
| Data | Simulated RAW derived from SIDD sRGB clean images (inverse gamma, inverse CCM, RGGB mosaicing). This is not real sensor RAW. |
| Split | Scene-level train/val/test split (no scene overlap) |
| Test set | 21 images from 2 held-out scenes |
| Checkpoints | 300-step demo checkpoints trained on CPU (the full 6,000-step GPU schedule has not been run) |
| Hardware | Intel 12-core CPU, no CUDA GPU |
| Compared systems | Classical baseline (gray-world WB, Malvar demosaic, CCM, tone mapping, unsharp mask); Deep learning only (end-to-end U-Net, 651,692 params); Proposed hybrid (classical stages + learned estimator and U-Net residual, student, 47,597 params) |

## 2. Quantitative results (full resolution, 21 test images)

Metrics are listed in the order of the problem statement: PSNR, SSIM, Delta E, LPIPS, model parameters and size, inference latency and memory.

| Method | PSNR (dB) ↑ | SSIM ↑ | Delta E 2000 ↓ | LPIPS ↓ | Parameters | Model size | Latency (ms/frame) | Peak memory |
|---|---|---|---|---|---|---|---|---|
| Classical baseline | 15.24 | 0.4836 | 22.52 | 0.5639 | 0 (no learned parameters) | n/a | 200 | not measured |
| Deep learning only | 14.91 | 0.1588 | 19.17 | 0.7529 | 651,692 | 2.61 MB | 560 | not measured |
| Proposed hybrid (student) | 14.47 | 0.2404 | 25.87 | 0.9392 | 47,597 | 0.21 MB | 299 | not measured |

Efficiency of the hybrid relative to the deep-learning-only model: 13.7x fewer parameters (47,597 vs 651,692), 12.4x smaller weights (0.21 MB vs 2.61 MB) and 1.87x lower latency (299 ms vs 560 ms).

## 3. Earlier evaluation at 256x256 input (CPU)

A separate run on the same 21 test images with 256x256 inputs produced the following. LPIPS was not computed in this run.

| Method | PSNR (dB) ↑ | SSIM ↑ | Delta E 2000 ↓ | Parameters | Model size | Latency (ms) | Peak RAM |
|---|---|---|---|---|---|---|---|
| Classical baseline | 15.24 | 0.4836 | 22.52 | 0 | n/a | 95 | not measured |
| Deep learning only | 24.03 | 0.7693 | 7.53 | 651,692 | 2.61 MB | 125 | 5.9 MB |
| Proposed hybrid (student) | 23.88 | 0.5110 | 9.15 | 47,597 | 0.21 MB | 64 | 0.5 MB |

## 4. Observations

- In the full-resolution run, the classical baseline has the highest PSNR (15.24 dB), the highest SSIM (0.4836) and the lowest LPIPS (0.5639), and it runs faster than both learned models.
- The deep-learning-only model has the best colour accuracy (Delta E 19.17 vs 22.52 for classical) but the lowest SSIM (0.1588) and the highest latency (560 ms).
- The hybrid has the lowest PSNR, the highest Delta E and the highest LPIPS in the full-resolution run, while its SSIM (0.2404) is above the deep-learning-only model's. Its measured advantage is efficiency: 47,597 parameters, 0.21 MB weights and the lower latency of the two learned models.
- In the 256x256 run the learned models reach about 24 dB PSNR and outperform the classical baseline by roughly 8.6 to 8.8 dB. The two evaluation settings give very different quality numbers for the learned models, so these results should be treated as preliminary until the full-resolution evaluation procedure is verified and the models are retrained.

## 5. Limitations

- All data is simulated RAW from SIDD sRGB and all noise and illuminant changes are synthetic. Results may differ on real sensor RAW.
- Checkpoints are 300-step demo models trained on CPU. No claim is made here about performance after full training.
- The test set is small (21 images, 2 scenes), so small differences are not conclusive.
- Latency was measured on a CPU, not on a mobile device.
- Peak memory is available only for the 256x256 run of the learned models.
- The ablation study and failure-case analysis are reported separately.
