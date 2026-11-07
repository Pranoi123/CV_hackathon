"""Full evaluation: PSNR, SSIM, LPIPS, latency for all three methods."""
import torch, time
import numpy as np
from isp_core import (HybridISP, LearnedISP, CCM_DEFAULT, degrade,
                      traditional_isp, render_tail, demosaic_malvar, unpack_rggb)
from isp_data import CleanRawBank, load_images_into
from isp_eval import image_metrics, count_params, model_size_mb, peak_mem_mb

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

test_indices = bank.idx['test']
test_scenes = bank.scene_split['test']
print(f"Test split: {len(test_indices)} images, scenes {test_scenes}\n")

def eval_method(name, fn):
    psnrs, ssims, des, lps, lats = [], [], [], [], []
    for idx in test_indices:
        raw_clean = bank.items[idx][1].float().unsqueeze(0)
        torch.manual_seed(idx)
        noisy, _ = degrade(raw_clean, a=0.003, T=5500.0, ev=0.0)
        target = render_tail(demosaic_malvar(unpack_rggb(raw_clean)), ccm, tone_a=0.0, sharpen=0.0)
        with torch.no_grad():
            t0 = time.perf_counter()
            out = fn(noisy)
            lats.append((time.perf_counter() - t0) * 1000)
            m = image_metrics(out, target, use_lpips=True, device='cpu')[0]
            psnrs.append(m['psnr'])
            ssims.append(m['ssim'])
            des.append(m['de'])
            lps.append(m.get('lpips', float('nan')))
    print(f"[{name}]")
    print(f"  PSNR       : {np.mean(psnrs):.2f} dB")
    print(f"  SSIM       : {np.mean(ssims):.4f}")
    print(f"  DeltaE2000 : {np.mean(des):.2f}")
    print(f"  LPIPS      : {np.nanmean(lps):.4f}")
    print(f"  Latency    : {np.mean(lats):.0f} ms/frame")
    return np.mean(psnrs), np.mean(ssims), np.mean(des), np.nanmean(lps), np.mean(lats)

r_trad = eval_method("Classical Baseline (Traditional ISP)", lambda x: traditional_isp(x, tone_a=0.0))
r_learn = eval_method("Deep Learning Only (Learned Baseline)", lambda x: learned(x)[0])
r_hyb = eval_method("Proposed Hybrid System", lambda x: student(x)[0])

print("\n=== SUMMARY TABLE ===")
print(f"{'Method':<35} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'Latency':>12}")
print("-"*75)
for name, r in [("Classical Baseline", r_trad), ("Deep Learning Only", r_learn), ("Proposed Hybrid", r_hyb)]:
    print(f"{name:<35} {r[0]:>7.2f}dB {r[1]:>8.4f} {r[3]:>8.4f} {r[4]:>9.0f} ms")

# Efficiency
dummy = torch.rand(1, 4, 256, 256)
print("\n=== EFFICIENCY ===")
for nm, model in [("Hybrid Student", student), ("Learned Baseline", learned)]:
    p = count_params(model)
    sz = model_size_mb(model)
    mem = peak_mem_mb(model, dummy)
    print(f"{nm}: {p:,} params | {sz:.2f} MB | {mem:.1f} MB peak RAM")
