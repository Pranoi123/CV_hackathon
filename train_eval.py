#!/usr/bin/env python3
"""AdaptISP: Complete training, distillation, evaluation, optimization, and ablation pipeline.

Simulated RAW derived from SIDD sRGB clean images via inverse gamma, inverse CCM, and RGGB mosaicing.
Labels everything honestly as simulated RAW.
"""

import os
import copy
import math
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from isp_core import (
    HybridISP,
    LearnedISP,
    CCM_DEFAULT,
    make_batch,
    traditional_isp,
    isp_loss,
    demosaic_malvar,
    unpack_rggb,
)
from isp_data import CleanRawBank, load_images_into
from isp_eval import (
    build_methods,
    run_eval,
    run_eval_real,
    count_params,
    model_size_mb,
    macs_g,
    latency_ms,
    peak_mem_mb,
    export_onnx,
    quantize_int8,
    ort_session,
    ort_override,
)

# Configuration defaults
USNS = ["1RVU23XXX001", "1RVU23XXX002", "1RVU23XXX003"]
TEAM_ID, PROB_ID = "T01", "P2"
CROP, TEST_CROP, BS = 96, 128, 8


@torch.no_grad()
def val_psnr(model, vcrops, ccm, device, seed=7):
    model.eval()
    if device.type == "cuda":
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(seed)
            noisy, target, _, _ = make_batch(vcrops, ccm)
    else:
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            noisy, target, _, _ = make_batch(vcrops, ccm)
    pred = model(noisy)[0].clamp(0, 1)
    model.train()
    return float(-10 * torch.log10(F.mse_loss(pred, target)))


def train(model, bank, steps, ccm, device, lr=2e-3, teacher=None, kd_w=0.5, tag=""):
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    vcrops = bank.fixed(24, CROP, "val", seed=5).to(device)
    best, best_state, hist = -1.0, None, []

    for it in range(1, steps + 1):
        cl = bank.sample(BS, CROP, "train").to(device)
        rng = (1e-4, 3e-3) if it < 0.3 * steps else (1e-4, 1.5e-2)  # curriculum
        noisy, target, labels, _ = make_batch(cl, ccm, a_range=rng)
        pred, aux = model(noisy)
        loss = isp_loss(pred, target, aux, labels)
        if teacher is not None:
            with torch.no_grad():
                tp = teacher(noisy)[0]
            loss = loss + kd_w * F.l1_loss(pred, tp)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if it % max(50, steps // 10) == 0 or it == steps:
            v = val_psnr(model, vcrops, ccm, device)
            hist.append((it, float(loss), v))
            if v > best or best_state is None:
                best, best_state = v, copy.deepcopy(model.state_dict())
            print(f"[{tag}] it {it:5d}/{steps} | loss: {float(loss):.4f} | val PSNR: {v:.2f} dB")

    model.load_state_dict(best_state)
    model.eval()
    return model, hist


def save_checkpoint(model, path):
    torch.save(dict(cfg=model.cfg, state={k: v.cpu() for k, v in model.state_dict().items()}), path)


def main():
    parser = argparse.ArgumentParser(description="AdaptISP Full Pipeline")
    parser.add_argument("--img-dir", type=str, default="sidd_gt", help="Clean images folder for simulated RAW bank")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--teacher-steps", type=int, default=6000, help="Teacher training steps")
    parser.add_argument("--student-steps", type=int, default=6000, help="Student training steps")
    parser.add_argument("--learned-steps", type=int, default=6000, help="Learned baseline training steps")
    parser.add_argument("--ablation-steps", type=int, default=1500, help="Steps per ablation model")
    parser.add_argument("--n-test", type=int, default=96, help="Test crops per condition")
    args = parser.parse_args()

    dev_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(dev_str)
    print(f"Using device: {device}")

    # 1. Clean RAW Bank
    print(f"\n--- Loading dataset from '{args.img_dir}' (simulated RAW derived from SIDD sRGB) ---")
    bank = CleanRawBank()
    n = load_images_into(bank, args.img_dir)
    print(f"Loaded {n} clean simulated RAW images into bank.")
    bank.make_splits(seed=args.seed)
    summary = bank.summary()
    print("Split summary (images, scenes):", summary)

    ccm = torch.tensor(CCM_DEFAULT, device=device)

    # 2. Train Models
    print("\n--- Training Teacher (HybridISP base=32) ---")
    teacher, h_t = train(
        HybridISP(base=32, sep=False),
        bank,
        args.teacher_steps,
        ccm,
        device,
        tag="teacher",
    )
    save_checkpoint(teacher, "teacher.pt")

    print("\n--- Training Distilled Student (HybridISP base=12 sep=True) ---")
    student, h_s = train(
        HybridISP(base=12, sep=True),
        bank,
        args.student_steps,
        ccm,
        device,
        teacher=teacher,
        tag="student+KD",
    )
    save_checkpoint(student, "student.pt")

    print("\n--- Training End-to-End Learned Baseline (LearnedISP base=32) ---")
    learned, h_l = train(
        LearnedISP(base=32, sep=False),
        bank,
        args.learned_steps,
        ccm,
        device,
        tag="learned-only",
    )
    save_checkpoint(learned, "learned.pt")

    # 3. Comparative Evaluation
    print("\n--- Running Comparative Evaluation on held-out test scenes ---")
    models = {
        "learned": learned.to(device).eval(),
        "hybrid_teacher": teacher.to(device).eval(),
        "hybrid_student": student.to(device).eval(),
    }
    methods = build_methods(models)
    test_crops = bank.fixed(args.n_test, TEST_CROP, "test", seed=123)

    use_lpips = True
    try:
        import lpips
    except ImportError:
        use_lpips = False
        print("Note: lpips package not found; evaluating PSNR, SSIM, DeltaE2000.")

    main_df = run_eval(methods, test_crops, [{}], device, use_lpips=use_lpips)
    cols = ["psnr", "ssim", "de"] + (["lpips"] if use_lpips else [])
    tbl = main_df.groupby("method")[cols].mean().round(4)
    print("\nMain Comparative Metrics:")
    print(tbl)

    # Sweeps
    sweeps = {
        "noise (shot gain a)": ("a", [dict(a=v, T=5500.0, ev=0.0) for v in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]]),
        "illuminant T (K)": ("T", [dict(a=1e-3, T=v, ev=0.0) for v in [2500.0, 3500.0, 5500.0, 7000.0, 8000.0]]),
        "exposure (EV)": ("ev", [dict(a=1e-3, T=5500.0, ev=v) for v in [-1.5, -0.75, 0.0, 0.5]]),
    }
    sw = {}
    for name, (col, conds) in sweeps.items():
        sw[name] = run_eval(methods, test_crops[: max(8, args.n_test // 3)], conds, device, use_lpips=False)

    fig, axs = plt.subplots(2, 3, figsize=(15, 7))
    for j, (name, (col, _)) in enumerate(sweeps.items()):
        g = sw[name].groupby(["method", "cond"]).agg(x=(col, "mean"), psnr=("psnr", "mean"), de=("de", "mean")).reset_index()
        for m_, d in g.groupby("method"):
            axs[0, j].plot(d.x, d.psnr, "o-", label=m_)
            axs[1, j].plot(d.x, d.de, "o-", label=m_)
        axs[0, j].set_title(name)
        axs[0, j].set_ylabel("PSNR (dB)")
        axs[1, j].set_ylabel("DeltaE 2000")
        axs[1, j].set_xlabel(col)
        if col == "a":
            axs[0, j].set_xscale("log")
            axs[1, j].set_xscale("log")
    axs[0, 0].legend()
    plt.tight_layout()
    plt.savefig("robustness_curves.png", dpi=130)
    plt.close()

    # 4. Benchmarking and INT8 Quantization
    print("\n--- Exporting ONNX & INT8 Quantization ---")
    os.makedirs("onnx", exist_ok=True)
    st_cpu = copy.deepcopy(student).cpu().eval()
    export_onnx(st_cpu.net, "onnx/student_fp32.onnx", 8)

    calib = []
    with torch.no_grad():
        for _ in range(32):
            cl = bank.sample(1, TEST_CROP, "train")
            nz, _, _, _ = make_batch(cl, torch.tensor(CCM_DEFAULT))
            st_cpu.net_override = lambda x: (calib.append(x.numpy()), st_cpu.net(x))[1]
            st_cpu(nz)
    st_cpu.net_override = None

    try:
        quantize_int8("onnx/student_fp32.onnx", "onnx/student_int8.onnx", calib)
        sess32 = ort_session("onnx/student_fp32.onnx")
        sess8 = ort_session("onnx/student_int8.onnx")
        has_onnx = True
    except Exception as e:
        print(f"Warning: ONNX quantization encountered an issue: {e}")
        has_onnx = False

    H = 256
    x512 = torch.rand(1, 4, H, H)
    x1080 = torch.rand(1, 4, 540, 960)
    rows = []

    def bench(name, mod, x, override=None, b_device="cpu", dtype=None):
        m = copy.deepcopy(mod).to(b_device).eval()
        if override is not None:
            m.net_override = override
        xx = x.to(b_device)
        if dtype == torch.float16 and b_device == "cuda":
            def run_fp16(z):
                with torch.amp.autocast("cuda"):
                    return m(z)
            lat, sd = latency_ms(run_fp16, xx, n=10, warmup=2)
        else:
            if dtype is not None:
                m, xx = m.to(dtype), xx.to(dtype)
            lat, sd = latency_ms(lambda z: m(z), xx, n=10, warmup=2)
        return dict(
            model=name,
            device=b_device,
            params=count_params(m),
            size_mb=round(model_size_mb(m), 3),
            macs_G=round(macs_g(lambda z: m(z), xx), 2),
            latency_ms=round(lat, 1),
            latency_std=round(sd, 1),
            peak_mem_mb=round(peak_mem_mb(lambda z: m(z), xx), 2),
        )

    for nm, md_ in [("hybrid_teacher", teacher), ("hybrid_student", student), ("learned", learned)]:
        rows.append(bench(nm + " FP32", md_, x512))

    if has_onnx:
        rows.append({
            **bench("hybrid_student INT8-core (ORT)", student, x512, override=ort_override(sess8)),
            "size_mb": round(os.path.getsize("onnx/student_int8.onnx") / 1e6, 3),
        })
        rows.append({
            **bench("hybrid_student FP32-core (ORT)", student, x512, override=ort_override(sess32)),
            "size_mb": round(os.path.getsize("onnx/student_fp32.onnx") / 1e6, 3),
        })

    if dev_str == "cuda":
        rows.append(bench("hybrid_student FP16 (GPU)", student, x512, b_device="cuda", dtype=torch.float16))
        rows.append(bench("hybrid_student FP32 (GPU)", student, x512, b_device="cuda"))

    rows.append({**bench("hybrid_student FP32 @1080p", student, x1080)})
    bench_df = pd.DataFrame(rows)
    macs_fp32 = float(bench_df.loc[bench_df.model == "hybrid_student FP32", "macs_G"].iloc[0])
    bench_df.loc[bench_df.model.str.contains("ORT"), "macs_G"] = macs_fp32
    print("\nBenchmark Summary:")
    print(bench_df.to_string(index=False))

    # INT8 Quality
    q_df = None
    if has_onnx:
        q_methods = {"student_fp32": lambda x: st_cpu(x)[0]}
        st8 = copy.deepcopy(st_cpu)
        st8.net_override = ort_override(sess8)
        q_methods["student_int8"] = lambda x: st8(x)[0]
        q = run_eval(q_methods, test_crops[:16].cpu(), [{}], "cpu", use_lpips=False)
        q_df = q.groupby("method")[["psnr", "ssim", "de"]].mean().round(4)
        print("\nINT8 vs FP32 Quality:")
        print(q_df)

    # 5. Ablations
    print("\n--- Running Ablations ---")
    variants = {
        "full (student cfg)": lambda: HybridISP(base=12, sep=True),
        "no noise map": lambda: HybridISP(base=12, sep=True, use_noise_map=False),
        "no estimator (gray-world)": lambda: HybridISP(base=12, sep=True, use_estimator=False),
        "no demosaic residual": lambda: HybridISP(base=12, sep=True, residual=False),
        "learned-only (no classical)": lambda: LearnedISP(base=20, sep=True),
    }
    abl = {}
    for name, mk in variants.items():
        torch.manual_seed(args.seed)
        m, _ = train(mk(), bank, args.ablation_steps, ccm, device, tag=name)
        abl[name] = m.to(device).eval()

    abl_methods = {k: (lambda x, m=m: m(x)[0]) for k, m in abl.items()}
    abl_df = run_eval(abl_methods, test_crops, [{}], device, use_lpips=False).groupby("method")[["psnr", "ssim", "de"]].mean().round(4)
    print("\nAblation Results:")
    print(abl_df)

    # 6. Failure Analysis
    print("\n--- Analyzing Failure Cases ---")
    d = main_df[main_df.method == "hybrid_student"].copy()
    worst = d.sort_values("de", ascending=False).head(4)
    d["noise_bin"] = pd.cut(np.log10(d.a), [-4.1, -3.3, -2.6, -1.9])
    d["T_bin"] = pd.cut(d["T"], [2400, 3800, 6000, 8100])
    d["ev_bin"] = pd.cut(d.ev, [-1.6, -0.75, 0, 0.8])
    fail_tables = {c: d.groupby(c, observed=True)[["psnr", "de"]].mean().round(3) for c in ["noise_bin", "T_bin", "ev_bin"]}

    show = lambda t: t[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(len(worst), 3, figsize=(9, 3 * len(worst)))
    for r, (_, row) in enumerate(worst.iterrows()):
        cl = test_crops[int(row.idx)][None].to(device)
        nz, tg, _, _ = make_batch(cl, ccm, a=row.a, T=row["T"], ev=row.ev)
        with torch.no_grad():
            out = student(nz)[0]
        for c_, (im, t) in enumerate([(demosaic_malvar(unpack_rggb(nz)).clamp(0, 1) ** (1 / 2.2), "input"), (out, f"hybrid  DeltaE={row.de:.1f}"), (tg, "target")]):
            ax[r, c_].imshow(show(im))
            ax[r, c_].set_title(t + (f"\n a={row.a:.4f} T={row['T']:.0f}K ev={row.ev:+.2f}" if c_ == 0 else ""), fontsize=8)
            ax[r, c_].axis("off")
    plt.tight_layout()
    plt.savefig("failure_cases.png", dpi=130)
    plt.close()

    # 7. Write Results and Report
    print("\n--- Writing Artifacts and Report ---")
    os.makedirs("results", exist_ok=True)
    main_df.to_csv("results/main_per_image.csv", index=False)
    for k, v in sw.items():
        v.to_csv(f"results/sweep_{k.split()[0]}.csv", index=False)
    bench_df.to_csv("results/benchmark.csv", index=False)
    abl_df.to_csv("results/ablation.csv")

    fname = f"eval_{PROB_ID}_{TEAM_ID}.md"
    b = bench_df.set_index("model")
    size_key = {
        "traditional": None,
        "learned": "learned FP32",
        "hybrid_teacher": "hybrid_teacher FP32",
        "hybrid_student": "hybrid_student FP32",
    }
    lp_header = " | LPIPS" if use_lpips else ""
    lp_div = "|---" if use_lpips else ""
    lines = [
        ",".join(USNS),
        "",
        f"# Evaluation - {PROB_ID} - {TEAM_ID}",
        "",
        f"Held-out test scenes, {args.n_test} crops x random degradations (noise, illuminant 2500-8000 K, exposure -1.5..+0.75 EV).",
        "Dataset: simulated RAW derived from SIDD sRGB clean images + synthetic degradations.",
        "",
        "## Comparative results (traditional vs learned vs hybrid)",
        "",
        f"| Method | PSNR (dB) | SSIM | Delta E 2000{lp_header} | Params | Model size (MB) | Latency 512x512 CPU (ms) | Peak memory (MB) |",
        f"|---|---|---|---{lp_div}|---|---|---|---|",
    ]
    for m_ in ["traditional", "learned", "hybrid_teacher", "hybrid_student"]:
        r = tbl.loc[m_]
        kk = size_key[m_]
        p, s_, l_, mem = (
            ("0 (classical)", "0", "-", "-")
            if kk is None
            else (f"{int(b.loc[kk, 'params']):,}", b.loc[kk, "size_mb"], b.loc[kk, "latency_ms"], b.loc[kk, "peak_mem_mb"])
        )
        lp_val = f" | {r.lpips:.4f}" if use_lpips else ""
        lines.append(f"| {m_} | {r.psnr:.2f} | {r.ssim:.4f} | {r.de:.3f}{lp_val} | {p} | {s_} | {l_} | {mem} |")

    lines += [
        "",
        "## Optimisation (hybrid student)",
        "",
        bench_df.to_markdown(index=False),
        "",
    ]
    if q_df is not None:
        lines += [
            "INT8 vs FP32 quality (CPU):",
            "",
            q_df.to_markdown(),
            "",
        ]
    lines += [
        "## Ablation study",
        "",
        abl_df.to_markdown(),
        "",
        "## Failure analysis (hybrid student, error vs condition)",
        "",
    ]
    for c, t in fail_tables.items():
        lines += [f"### Bin: {c}", "", t.to_markdown(), ""]
    lines += ["Figures: robustness_curves.png, failure_cases.png"]

    with open(fname, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Report written to {fname}")
    print("Execution complete!")


if __name__ == "__main__":
    main()
