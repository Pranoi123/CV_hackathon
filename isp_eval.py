"""Metrics, benchmarking, ONNX/INT8 helpers."""
import os, time, threading
import numpy as np
import pandas as pd
import torch
import psutil
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.color import rgb2lab, deltaE_ciede2000
from isp_core import CCM_DEFAULT, make_batch, reference_render, traditional_isp

_LP = {}


def _lpips(device):
    if device not in _LP:
        import lpips
        _LP[device] = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    return _LP[device]


def image_metrics(pred, gt, use_lpips=True, device="cpu"):
    """pred, gt: [B,3,H,W] in [0,1] -> list of dict(psnr, ssim, de, lpips)."""
    p = pred.clamp(0, 1).detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.float64)
    g = gt.clamp(0, 1).detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.float64)
    rows = [dict(psnr=peak_signal_noise_ratio(g[i], p[i], data_range=1.0),
                 ssim=structural_similarity(g[i], p[i], channel_axis=2, data_range=1.0),
                 de=float(deltaE_ciede2000(rgb2lab(g[i]), rgb2lab(p[i])).mean())) for i in range(len(p))]
    if use_lpips:
        try:
            with torch.no_grad():
                v = _lpips(device)(pred.clamp(0, 1).to(device) * 2 - 1, gt.clamp(0, 1).to(device) * 2 - 1)
            for r, x in zip(rows, v.flatten().cpu().numpy()):
                r["lpips"] = float(x)
        except Exception:
            for r in rows:
                r["lpips"] = float("nan")
    return rows


def build_methods(models):
    """models: dict name -> nn.Module (HybridISP / LearnedISP). Adds the traditional baseline."""
    m = {"traditional": lambda x: traditional_isp(x)}
    for name, mod in models.items():
        m[name] = (lambda x, mod=mod: mod(x)[0])
    return m


@torch.no_grad()
def run_eval(methods, clean_crops, conds, device, use_lpips=True, seed=0, bs=16):
    """conds: list of dict(a=,T=,ev=) ({} => random). Every method sees identical noisy inputs."""
    ccm = torch.tensor(CCM_DEFAULT, device=device)
    rows = []
    for ci, cond in enumerate(conds):
        torch.manual_seed(seed + ci)
        for s in range(0, len(clean_crops), bs):
            clean = clean_crops[s:s + bs].to(device)
            noisy, target, _, info = make_batch(clean, ccm, **cond)
            for name, fn in methods.items():
                for i, r in enumerate(image_metrics(fn(noisy), target, use_lpips, device)):
                    r.update(method=name, cond=ci, idx=s + i, a=float(info["a"][i]),
                             T=float(info["T"][i]), ev=float(info["ev"][i]))
                    rows.append(r)
    return pd.DataFrame(rows)


@torch.no_grad()
def run_eval_real(methods, noisy, clean, device, use_lpips=True, bs=16):
    ccm = torch.tensor(CCM_DEFAULT, device=device)
    rows = []
    for s in range(0, len(noisy), bs):
        n, c = noisy[s:s + bs].to(device), clean[s:s + bs].to(device)
        target = reference_render(c, ccm)
        for name, fn in methods.items():
            for i, r in enumerate(image_metrics(fn(n), target, use_lpips, device)):
                r.update(method=name, idx=s + i)
                rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ benchmarking
def count_params(m):
    return sum(p.numel() for p in m.parameters())


def model_size_mb(m, path="/tmp/_m.pt" if os.name != "nt" else "_m.pt"):
    torch.save(m.state_dict(), path)
    sz = os.path.getsize(path) / 1e6
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass
    return sz


def macs_g(module, x):
    from torch.utils.flop_counter import FlopCounterMode
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        module(x)
    return fc.get_total_flops() / 2 / 1e9


def latency_ms(fn, x, n=20, warmup=5):
    """mean, std in ms after warm-up."""
    cuda = x.is_cuda if torch.is_tensor(x) else False
    with torch.no_grad():
        for _ in range(warmup):
            fn(x)
        ts = []
        for _ in range(n):
            if cuda:
                torch.cuda.synchronize()
            t = time.perf_counter()
            fn(x)
            if cuda:
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t) * 1000)
    return float(np.mean(ts)), float(np.std(ts))


def peak_mem_mb(fn, x):
    """GPU: peak allocated; CPU: RSS sampling in a background thread (psutil), peak minus baseline."""
    with torch.no_grad():
        if torch.is_tensor(x) and x.is_cuda:
            torch.cuda.reset_peak_memory_stats()
            fn(x)
            torch.cuda.synchronize()
            return float(torch.cuda.max_memory_allocated() / 1e6)

        pr = psutil.Process()
        baseline_rss = pr.memory_info().rss
        peak_rss = baseline_rss
        stop_sampling = False

        def sampler():
            nonlocal peak_rss
            while not stop_sampling:
                try:
                    curr = pr.memory_info().rss
                    if curr > peak_rss:
                        peak_rss = curr
                except Exception:
                    pass
                time.sleep(0.001)

        thread = threading.Thread(target=sampler, daemon=True)
        thread.start()
        try:
            fn(x)
        finally:
            stop_sampling = True
            thread.join(timeout=0.1)

        diff = (peak_rss - baseline_rss) / 1e6
        if diff <= 0:
            diff = (pr.memory_info().rss - baseline_rss) / 1e6
        return float(max(0.5, round(diff, 2)))


def export_onnx(core, path, cin, size=128):
    dummy = torch.randn(1, cin, size, size)
    kw = dict(input_names=["x"], output_names=["y"], opset_version=17,
              dynamic_axes={"x": {0: "b", 2: "h", 3: "w"}, "y": {0: "b", 2: "h", 3: "w"}})
    try:
        torch.onnx.export(core.cpu().eval(), dummy, path, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(core.cpu().eval(), dummy, path, **kw)
    return path


def quantize_int8(fp32_path, int8_path, calib):
    """Static INT8 post-training quantisation with ONNX Runtime."""
    from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod

    class R(CalibrationDataReader):
        def __init__(self):
            self.it = iter([{"x": c} for c in calib])

        def get_next(self):
            return next(self.it, None)

    try:
        quantize_static(fp32_path, int8_path, R(), quant_format=QuantFormat.QDQ,
                        per_channel=True, weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
                        calibrate_method=CalibrationMethod.MinMax,
                        extra_options={"ActivationSymmetric": True})
    except Exception:
        quantize_static(fp32_path, int8_path, R(), quant_format=QuantFormat.QDQ,
                        per_channel=True, weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8)
    return int8_path


def ort_session(path):
    import onnxruntime as ort
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def ort_override(sess):
    name = sess.get_inputs()[0].name

    def f(x):
        return torch.from_numpy(sess.run(None, {name: x.detach().cpu().numpy()})[0]).to(x.device)
    return f
