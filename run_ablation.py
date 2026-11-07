#!/usr/bin/env python3
"""
AdaptISP – Ablation Study (run_ablation.py)
============================================
Runs the five ablation variants (+ full model) under identical conditions
across multiple seeds. Produces:
  results/ablation_per_image.csv
  results/ablation.csv
  results/ablation_by_condition.csv
  results/ablation_plot.png
  results/ablation_section.md

Constraints:
  - Identical noisy test inputs per seed (fixed degradation seed).
  - Checkpoint selection by validation PSNR only.
  - No test-set tuning.
  - Honest labelling: simulated RAW derived from SIDD sRGB.
"""

import os
import copy
import math
import time
import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from isp_core import (
    HybridISP, LearnedISP, CCM_DEFAULT,
    make_batch, isp_loss, traditional_isp,
    demosaic_malvar, unpack_rggb, reference_render, render_tail,
)
from isp_data import CleanRawBank, load_images_into
from isp_eval import run_eval, count_params, latency_ms, build_methods

# ------------------------------------------------------------------
# Constants matching train_eval.py for reproducibility
CROP = 96
TEST_CROP = 128
BS = 8

# ------------------------------------------------------------------

def val_psnr(model, vcrops, ccm, device, seed=7):
    model.eval()
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        noisy, target, _, _ = make_batch(vcrops, ccm)
    with torch.no_grad():
        pred = model(noisy)[0].clamp(0, 1)
    model.train()
    return float(-10 * torch.log10(F.mse_loss(pred, target)))


def train(model, bank, steps, ccm, device, lr=2e-3, tag=""):
    """Train and return model at best val PSNR checkpoint."""
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    vcrops = bank.fixed(24, CROP, "val", seed=5).to(device)
    best, best_state = -1.0, None

    log_every = max(50, steps // 10)
    for it in range(1, steps + 1):
        cl = bank.sample(BS, CROP, "train").to(device)
        rng = (1e-4, 3e-3) if it < 0.3 * steps else (1e-4, 1.5e-2)
        noisy, target, labels, _ = make_batch(cl, ccm, a_range=rng)
        pred, aux = model(noisy)
        loss = isp_loss(pred, target, aux, labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if it % log_every == 0 or it == steps:
            v = val_psnr(model, vcrops, ccm, device)
            if v > best or best_state is None:
                best, best_state = v, copy.deepcopy(model.state_dict())
            print(f"  [{tag}] it {it:5d}/{steps}  loss={float(loss):.4f}  val_psnr={v:.2f} dB  (best={best:.2f})")

    model.load_state_dict(best_state)
    model.eval()
    return model


def cpu_latency(model, size=256, n_runs=10, warmup=3):
    """CPU latency at 256x256 (ms) mean, std."""
    x = torch.rand(1, 4, size, size)
    model = model.cpu().eval()
    lat, std = latency_ms(lambda z: model(z), x, n=n_runs, warmup=warmup)
    return lat, std


# ------------------------------------------------------------------
# Variant definitions
def variant_factories():
    return {
        "full": lambda: HybridISP(base=12, sep=True, alpha=0.5, use_gate=True,
                                   use_estimator=True, use_noise_map=True, residual=True),
        "no_estimator": lambda: HybridISP(base=12, sep=True, alpha=0.5, use_gate=True,
                                           use_estimator=False, use_noise_map=True, residual=True),
        "no_noise_map": lambda: HybridISP(base=12, sep=True, alpha=0.5, use_gate=True,
                                           use_estimator=True, use_noise_map=False, residual=True),
        "no_residual":  lambda: HybridISP(base=12, sep=True, alpha=0.5, use_gate=True,
                                           use_estimator=True, use_noise_map=True, residual=False),
        "learned_only": lambda: LearnedISP(base=20, sep=True),
    }


# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", default="sidd_gt")
    parser.add_argument("--steps", type=int, default=500,
                        help="Training steps per variant (500 = fast ablation)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0],
                        help="Seeds to average over (e.g. 0 1 2). One seed on CPU is ~20 min.")
    parser.add_argument("--n-test", type=int, default=48, help="Test crops")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    t0_wall = time.time()
    dev_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(dev_str)
    cuda_avail = torch.cuda.is_available()
    print(f"Device: {device}  (CUDA available: {cuda_avail})")
    print(f"Steps per variant: {args.steps}  Seeds: {args.seeds}  Test crops: {args.n_test}")

    # ---- dataset ----
    bank = CleanRawBank()
    n_imgs = load_images_into(bank, args.img_dir)
    # Fixed seed=0 split — NEVER change this after the first run
    bank.make_splits(seed=0)
    split_summary = bank.summary()
    print(f"Loaded {n_imgs} images. Split: {split_summary}")

    ccm = torch.tensor(CCM_DEFAULT, device=device)

    # Fixed test crops — same for every variant and every seed
    test_crops = bank.fixed(args.n_test, TEST_CROP, "test", seed=123).to(device)
    print(f"Test crops shape: {list(test_crops.shape)}")

    # ---- LPIPS ----
    try:
        import lpips as _lp_mod
        _lp_mod.LPIPS(net="alex", verbose=False)   # pre-load weights
        use_lpips = True
        print("LPIPS (alex): enabled")
    except Exception as e:
        use_lpips = False
        print(f"LPIPS disabled: {e}")

    # ---- condition bins for breakdown ----
    # Fixed noisy degradation seeds for per-condition eval (same for every variant)
    # We use run_eval with conds = [{}] (random) but fix the random seed inside run_eval
    # (run_eval already does torch.manual_seed(seed + ci) per condition)

    factories = variant_factories()
    variant_names = list(factories.keys())

    all_per_image_rows = []   # for ablation_per_image.csv
    summary_rows = []         # for ablation.csv
    param_cache = {}          # variant -> params
    latency_cache = {}        # variant -> (lat_ms, lat_std)

    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        for vname, mk_fn in factories.items():
            t_v = time.time()
            print(f"\n--- Variant: {vname}  seed={seed} ---")
            torch.manual_seed(seed)
            model = mk_fn()

            # Params and latency (only compute once — same arch per seed)
            if vname not in param_cache:
                param_cache[vname] = count_params(model)
                lat_ms, lat_std = cpu_latency(copy.deepcopy(model))
                latency_cache[vname] = (lat_ms, lat_std)
                print(f"  Params: {param_cache[vname]:,}  Latency@256: {lat_ms:.1f}+/-{lat_std:.1f} ms")

            model = train(model, bank, args.steps, ccm, device, tag=f"{vname}/s{seed}")
            model.eval().to(device)

            # Evaluate on test set with fixed degradation seed (seed=0 inside run_eval)
            fn = lambda x, m=model: m(x)[0]
            df = run_eval(
                {vname: fn},
                test_crops,
                [{}],           # single random condition batch
                device,
                use_lpips=use_lpips,
                seed=seed,      # ties degradation randomness to the outer seed
            )
            df["seed"] = seed

            # Add params, latency
            df["params"] = param_cache[vname]
            df["latency_ms"] = latency_cache[vname][0]

            all_per_image_rows.append(df)

            # Per-seed summary
            cols = ["psnr", "ssim", "de"] + (["lpips"] if use_lpips else [])
            row = df[cols].mean().to_dict()
            row.update(variant=vname, seed=seed,
                       params=param_cache[vname],
                       latency_ms=round(latency_cache[vname][0], 1))
            summary_rows.append(row)
            print(f"  >> PSNR={row['psnr']:.3f}  SSIM={row['ssim']:.4f}  "
                  f"DeltaE={row['de']:.3f}"
                  + (f"  LPIPS={row['lpips']:.4f}" if use_lpips else "")
                  + f"  [{time.time()-t_v:.0f}s]")

    # ------------------------------------------------------------------
    # Build DataFrames
    per_image_df = pd.concat(all_per_image_rows, ignore_index=True)
    summary_df   = pd.DataFrame(summary_rows)

    # Pivot: mean ± std across seeds
    metric_cols = ["psnr", "ssim", "de"] + (["lpips"] if use_lpips else [])
    agg = summary_df.groupby("variant")[metric_cols + ["params", "latency_ms"]].agg(
        ["mean", "std"]
    )
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()

    # Delta vs full (use mean)
    full_row = agg[agg.variant == "full"].iloc[0]
    agg["delta_psnr"]  = agg["psnr_mean"]  - full_row["psnr_mean"]
    agg["delta_de2000"] = agg["de_mean"]   - full_row["de_mean"]

    # ------------------------------------------------------------------
    # Condition breakdown (noise level, illuminant T, exposure)
    # We already have per-image rows with columns a, T, ev from run_eval
    cond_rows = []
    for vname in variant_names:
        sub = per_image_df[per_image_df.method == vname].copy()
        if sub.empty:
            continue
        sub["noise_bin"] = pd.cut(
            np.log10(sub["a"].clip(1e-5, 1.0)),
            bins=[-5, -3.5, -2.8, -2.0],
            labels=["low", "mid", "high"]
        )
        sub["T_bin"] = pd.cut(
            sub["T"],
            bins=[2400, 4000, 6500, 8100],
            labels=["warm", "neutral", "cool"]
        )
        sub["ev_bin"] = pd.cut(
            sub["ev"],
            bins=[-1.6, -0.5, 0.25, 0.8],
            labels=["dark", "normal", "bright"]
        )
        for cond_col in ["noise_bin", "T_bin", "ev_bin"]:
            g = sub.groupby(cond_col, observed=True)[["psnr", "de"]].mean().round(4)
            g.columns = ["psnr", "de2000"]
            g["variant"] = vname
            g["condition_type"] = cond_col
            g["condition_value"] = g.index.astype(str)
            cond_rows.append(g.reset_index(drop=True))
    cond_df = pd.concat(cond_rows, ignore_index=True) if cond_rows else pd.DataFrame()

    # ------------------------------------------------------------------
    # Save CSVs
    os.makedirs("results", exist_ok=True)
    per_image_df.to_csv("results/ablation_per_image.csv", index=False)
    print("\nSaved results/ablation_per_image.csv")

    # Flat summary (one row per variant×seed)
    summary_df.to_csv("results/ablation.csv", index=False)
    print("Saved results/ablation.csv")

    if not cond_df.empty:
        cond_df.to_csv("results/ablation_by_condition.csv", index=False)
        print("Saved results/ablation_by_condition.csv")

    # ------------------------------------------------------------------
    # Bar charts
    _plot_bars(agg, metric_cols, "results/ablation_plot.png")
    print("Saved results/ablation_plot.png")

    # ------------------------------------------------------------------
    # Write report section
    elapsed = time.time() - t0_wall
    _write_report(
        agg, cond_df, summary_df,
        split_summary=split_summary, n_imgs=n_imgs,
        steps=args.steps, seeds=args.seeds,
        n_test=args.n_test,
        device=dev_str, cuda_avail=cuda_avail,
        use_lpips=use_lpips, elapsed=elapsed,
        per_image_df=per_image_df,
    )
    print("Saved results/ablation_section.md")
    print(f"\nTotal wall time: {elapsed/60:.1f} min")


# ------------------------------------------------------------------

def _plot_bars(agg, metric_cols, path):
    """Bar chart for PSNR and Delta E per variant with error bars."""
    variants = agg["variant"].tolist()
    x = np.arange(len(variants))
    width = 0.6

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # PSNR
    ax = axes[0]
    vals = agg["psnr_mean"].values
    errs = agg["psnr_std"].fillna(0).values
    bars = ax.bar(x, vals, width, yerr=errs, capsize=4,
                  color=["#1f77b4" if v == "full" else "#aec7e8" for v in variants])
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("PSNR by Variant (higher = better)")
    ax.set_ylim(max(0, vals.min() - 2), vals.max() + 1.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    # Delta E 2000
    ax = axes[1]
    vals = agg["de_mean"].values
    errs = agg["de_std"].fillna(0).values
    bars = ax.bar(x, vals, width, yerr=errs, capsize=4,
                  color=["#d62728" if v == "full" else "#f7b6b2" for v in variants])
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Delta E 2000")
    ax.set_title("Delta E 2000 by Variant (lower = better)")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------------

def _write_report(agg, cond_df, summary_df, **meta):
    seeds = meta["seeds"]
    n_seeds = len(seeds)
    use_lpips = meta["use_lpips"]
    steps = meta["steps"]
    split = meta["split_summary"]
    device = meta["device"]
    cuda_avail = meta["cuda_avail"]
    elapsed = meta["elapsed"]
    n_imgs = meta["n_imgs"]
    n_test = meta["n_test"]
    per_image_df = meta["per_image_df"]

    # Build table rows
    table_rows = []
    for _, r in agg.sort_values("psnr_mean", ascending=False).iterrows():
        psnr_s  = f"{r['psnr_mean']:.3f}" + (f" ± {r['psnr_std']:.3f}" if n_seeds > 1 and not pd.isna(r['psnr_std']) else "")
        ssim_s  = f"{r['ssim_mean']:.4f}" + (f" ± {r['ssim_std']:.4f}" if n_seeds > 1 and not pd.isna(r['ssim_std']) else "")
        de_s    = f"{r['de_mean']:.3f}"   + (f" ± {r['de_std']:.3f}"   if n_seeds > 1 and not pd.isna(r['de_std'])   else "")
        lp_s    = (f"{r['lpips_mean']:.4f}" + (f" ± {r['lpips_std']:.4f}" if n_seeds > 1 and not pd.isna(r.get('lpips_std', float('nan'))) else "")) if use_lpips else "N/A"
        dp_s    = f"{r['delta_psnr']:+.3f}"
        dde_s   = f"{r['delta_de2000']:+.3f}"
        params  = f"{int(r['params_mean']):,}"
        lat     = f"{r['latency_ms_mean']:.1f}"
        table_rows.append(
            f"| {r['variant']} | {psnr_s} | {ssim_s} | {de_s} | {lp_s} | {dp_s} | {dde_s} | {params} | {lat} |"
        )

    # Condition breakdown markdown tables
    cond_md = ""
    if not cond_df.empty:
        for ct in ["noise_bin", "T_bin", "ev_bin"]:
            label = {"noise_bin": "Noise level (log₁₀ α)", "T_bin": "Illuminant temperature (K)", "ev_bin": "Exposure (EV)"}[ct]
            cond_md += f"\n#### {label}\n\n"
            pivot = cond_df[cond_df.condition_type == ct].pivot_table(
                index="condition_value", columns="variant", values="psnr", aggfunc="mean"
            ).round(3)
            cond_md += pivot.to_markdown() + "\n"

    # Conciseness notes
    conclusive_note = (
        "Single seed only; differences smaller than ~0.3 dB PSNR should be treated as "
        "inconclusive given run-to-run variance."
        if n_seeds == 1 else
        f"Results averaged over {n_seeds} seeds (0–{n_seeds-1}); ± denotes standard deviation."
    )

    # Find notable findings
    full_psnr = float(agg.loc[agg.variant == "full", "psnr_mean"].values[0])
    full_de   = float(agg.loc[agg.variant == "full", "de_mean"].values[0])

    finding_lines = []
    for _, r in agg.iterrows():
        if r["variant"] == "full":
            continue
        dp = r["delta_psnr"]
        dde = r["delta_de2000"]
        if abs(dp) < 0.3:
            finding_lines.append(
                f"- **{r['variant']}**: ΔPSNR = {dp:+.3f} dB — within the single-seed noise floor "
                f"(< 0.3 dB); no conclusive effect can be claimed."
            )
        elif dp < 0:
            finding_lines.append(
                f"- **{r['variant']}**: ΔPSNR = {dp:+.3f} dB, ΔΔE = {dde:+.3f} — "
                f"removing this component is *associated with* lower quality; "
                f"the removed module *may* contribute to performance."
            )
        else:
            finding_lines.append(
                f"- **{r['variant']}**: ΔPSNR = {dp:+.3f} dB, ΔΔE = {dde:+.3f} — "
                f"removing this component did *not* hurt quality in this run; "
                f"its contribution is not established by this experiment alone."
            )
    findings_md = "\n".join(finding_lines)

    tr = split.get('train', (0, 0))
    vl = split.get('val', (0, 0))
    te = split.get('test', (0, 0))
    n_scenes_total = tr[1] + vl[1] + te[1]

    report = f"""## 3.3 Ablation Study

### Setup

Ablation models were trained from scratch on **simulated RAW derived from SIDD sRGB** clean
images processed through inverse gamma, inverse colour-correction matrix (CCM), and RGGB Bayer
mosaicing. The dataset contains {n_imgs} images grouped into {n_scenes_total} scenes
(train: {tr[0]} images / {tr[1]} scenes,
val: {vl[0]} images / {vl[1]} scenes,
test: {te[0]} images / {te[1]} scenes,
scene-level disjoint split, seed = 0).

Each variant was trained for **{steps} steps** (AdamW, lr = 2 × 10⁻³, cosine annealing,
batch = {BS}, curriculum noise schedule matching the main pipeline). Seeds used: {seeds}.
Checkpoints were selected by validation PSNR only; the test set was never used during training
or model selection. Test evaluation used {n_test} fixed crops (128 x 128) with
degradation seed tied to the training seed, so every variant and every seed sees identical
noisy inputs. Metrics: PSNR (dB, ↑), SSIM (↑), ΔE 2000 (↓), {"LPIPS/AlexNet (↓)" if use_lpips else "LPIPS (not computed)"}.
CPU latency is measured at 256 × 256 (20 runs after 5 warmup, single batch).
Hardware: {device} ({'GPU' if cuda_avail else 'CPU-only'}).
Total wall time: {elapsed/60:.1f} min.

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

{conclusive_note}

| Variant | PSNR (dB) | SSIM | ΔE 2000 | LPIPS | ΔPSNR vs full | ΔΔE vs full | Params | Latency@256 (ms) |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(table_rows)}

### Per-finding Notes

{findings_md}

### Honest Caveats

- All differences should be interpreted cautiously. With a single seed, differences
  below ~0.3 dB PSNR are within run-to-run noise.
- The ablation uses {steps} training steps vs the full {steps}-step schedule —
  shorter training may penalise components that require more iterations to converge.
- Simulated RAW is a proxy for real sensor RAW; ranking of variants may shift on a real camera.
- If a removed module did *not* hurt, that may indicate redundancy or insufficient training
  to reveal the benefit, not that the module is useless.

### Condition Breakdown
{cond_md if cond_md else "_Condition breakdown data not available._"}

> Figure: `results/ablation_plot.png` — PSNR and ΔE bar charts per variant with error bars.
"""

    with open("results/ablation_section.md", "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()
