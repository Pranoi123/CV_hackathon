"""Quick LPIPS eval using 256x256 center crops of test images for speed."""
import torch, time
import numpy as np
from isp_core import (HybridISP, LearnedISP, CCM_DEFAULT, degrade,
                      traditional_isp, render_tail, demosaic_malvar, unpack_rggb)
from isp_data import CleanRawBank, load_images_into
import lpips
import warnings; warnings.filterwarnings('ignore')

torch.set_num_threads(12)
device = torch.device('cpu')

bank = CleanRawBank()
load_images_into(bank, 'sidd_gt')
bank.make_splits(seed=0)
ccm = torch.tensor(CCM_DEFAULT, device=device)

ck_s = torch.load('student.pt', map_location='cpu')
student = HybridISP(**ck_s['cfg']).eval()
ck_l = torch.load('learned.pt', map_location='cpu')
learned = LearnedISP(**ck_l['cfg']).eval()

# LPIPS model (AlexNet, already cached)
lpips_fn = lpips.LPIPS(net='alex', verbose=False).eval()

test_indices = bank.idx['test']
print(f"Test split: {len(test_indices)} images, scenes {bank.scene_split['test']}")
print("Computing PSNR / SSIM / LPIPS on 128px center crops (fast CPU mode)...\n")

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def center_crop_packed(packed, size=128):
    """Crop a packed RGGB tensor [4,H,W] to size x size from center."""
    h, w = packed.shape[-2:]
    y = (h - size) // 2
    x = (w - size) // 2
    return packed[:, y:y+size, x:x+size]

def eval_method(name, fn):
    psnrs, ssims, lps, lats = [], [], [], []
    for idx in test_indices:
        raw_full = bank.items[idx][1].float()
        # crop to 128x128 packed = 256x256 image
        raw_clean = center_crop_packed(raw_full, size=128).unsqueeze(0)
        torch.manual_seed(idx)
        noisy, _ = degrade(raw_clean, a=0.003, T=5500.0, ev=0.0)
        target = render_tail(demosaic_malvar(unpack_rggb(raw_clean)), ccm, tone_a=0.0, sharpen=0.0)
        
        with torch.no_grad():
            t0 = time.perf_counter()
            out = fn(noisy)
            lats.append((time.perf_counter() - t0) * 1000)
            out_c = out.clamp(0, 1)
            
            # PSNR / SSIM via skimage
            p_np = out_c[0].permute(1,2,0).numpy().astype(np.float64)
            t_np = target[0].permute(1,2,0).numpy().astype(np.float64)
            psnrs.append(peak_signal_noise_ratio(t_np, p_np, data_range=1.0))
            ssims.append(structural_similarity(t_np, p_np, channel_axis=2, data_range=1.0))
            
            # LPIPS
            lp = lpips_fn(out_c * 2 - 1, target.clamp(0,1) * 2 - 1)
            lps.append(float(lp.mean()))
    
    print(f"[{name}]")
    print(f"  PSNR       : {np.mean(psnrs):.2f} dB")
    print(f"  SSIM       : {np.mean(ssims):.4f}")
    print(f"  LPIPS      : {np.mean(lps):.4f}  (lower = better)")
    print(f"  Latency    : {np.mean(lats):.0f} ms/frame (256px input)")
    return np.mean(psnrs), np.mean(ssims), np.mean(lps), np.mean(lats)

r_trad  = eval_method("Classical Baseline (Traditional ISP)", lambda x: traditional_isp(x, tone_a=0.0))
r_learn = eval_method("Deep Learning Only (Learned Baseline)", lambda x: learned(x)[0])
r_hyb   = eval_method("Proposed Hybrid System", lambda x: student(x)[0])

print("\n=== QUANTITATIVE BENCHMARK TABLE ===")
print(f"{'Method':<35} {'PSNR':>9} {'SSIM':>8} {'LPIPS':>8} {'Latency':>12}")
print("-"*78)
rows = [("Classical Baseline", r_trad), ("Deep Learning Only", r_learn), ("Proposed Hybrid System", r_hyb)]
for name, r in rows:
    print(f"{name:<35} {r[0]:>7.2f} dB {r[1]:>8.4f} {r[2]:>8.4f} {r[3]:>9.0f} ms")
print()
print(f"Delta (Hybrid vs Classical): PSNR {r_hyb[0]-r_trad[0]:+.2f} dB | SSIM {r_hyb[1]-r_trad[1]:+.4f} | LPIPS {r_hyb[2]-r_trad[2]:+.4f}")
