import torch, time, os, psutil
import numpy as np
from isp_core import (HybridISP, LearnedISP, CCM_DEFAULT, degrade, traditional_isp, 
                      render_tail, demosaic_malvar, unpack_rggb)
from isp_data import CleanRawBank, load_images_into
from isp_eval import image_metrics, count_params, model_size_mb, peak_mem_mb

torch.set_num_threads(12)
device = torch.device('cpu')
bank = CleanRawBank()
load_images_into(bank, 'sidd_gt')
bank.make_splits(seed=0)
ccm = torch.tensor(CCM_DEFAULT, device=device)

# Load student
ck_s = torch.load('student.pt', map_location='cpu')
student = HybridISP(**ck_s['cfg']).eval().to(device)
student.load_state_dict(ck_s['state'])

# Load learned
ck_l = torch.load('learned.pt', map_location='cpu')
learned = LearnedISP(**ck_l['cfg']).eval().to(device)
learned.load_state_dict(ck_l['state'])

test_indices = bank.idx['test']
test_scenes = bank.scene_split['test']
print(f"Evaluating on test split: {len(test_indices)} images across scenes {test_scenes}")

def eval_method(method_name, fn):
    psnrs, ssims, delta_es = [], [], []
    for idx in test_indices:
        # items[idx] is (scene, packed fp16 tensor [4,h,w])
        raw_clean = bank.items[idx][1].float().unsqueeze(0).to(device)
        torch.manual_seed(idx)
        noisy, info = degrade(raw_clean, a=0.003, T=5500.0, ev=0.0)
        target = render_tail(demosaic_malvar(unpack_rggb(raw_clean)), ccm, tone_a=0.0, sharpen=0.0)
        
        with torch.no_grad():
            out = fn(noisy)
            m = image_metrics(out, target, use_lpips=False)[0]
            psnrs.append(m['psnr'])
            ssims.append(m['ssim'])
            delta_es.append(m['de'])
            
    return np.mean(psnrs), np.mean(ssims), np.mean(delta_es)

p_trad, s_trad, de_trad = eval_method('Traditional', lambda x: traditional_isp(x, tone_a=0.0))
p_learn, s_learn, de_learn = eval_method('Learned', lambda x: learned(x)[0])
p_hyb, s_hyb, de_hyb = eval_method('Hybrid', lambda x: student(x)[0])

print(f"\n=== TEST SET EVALUATION ({len(test_indices)} images across scenes {test_scenes}) ===")
print(f"Dataset: Simulated RAW derived from SIDD sRGB")
print(f"Traditional: PSNR={p_trad:.2f} dB, SSIM={s_trad:.4f}, DeltaE2000={de_trad:.2f}")
print(f"Learned:     PSNR={p_learn:.2f} dB, SSIM={s_learn:.4f}, DeltaE2000={de_learn:.2f}")
print(f"Hybrid:      PSNR={p_hyb:.2f} dB, SSIM={s_hyb:.4f}, DeltaE2000={de_hyb:.2f}")
print(f"Delta (Hybrid vs Traditional): PSNR: {p_hyb - p_trad:+.2f} dB, SSIM: {s_hyb - s_trad:+.4f}, DeltaE2000: {de_hyb - de_trad:+.2f}")

# Model efficiency metrics
dummy = torch.rand(1, 4, 256, 256)
p_count = count_params(student)
sz_mb = model_size_mb(student)
mem = peak_mem_mb(student, dummy, device=device)
print(f"\nHybrid Student Parameters: {p_count:,}")
print(f"Hybrid Student Size (weights): {sz_mb:.2f} MB")
print(f"Hybrid Student Peak CPU Memory: {mem:.2f} MB")
