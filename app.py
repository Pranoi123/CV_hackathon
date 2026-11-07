"""Gradio demo for the hybrid ISP. Run: python app.py (needs student.pt next to this file)."""
import os, time
import numpy as np
import torch
import torch.nn.functional as F
import gradio as gr
from PIL import Image
from isp_core import (HybridISP, LearnedISP, CCM_DEFAULT, degrade, traditional_isp, demosaic_malvar,
                      unpack_rggb, pack_rggb, to_rggb, render_tail, srgb_oetf, _SHIFT)
from isp_data import srgb_to_pseudo_raw

torch.set_num_threads(max(1, os.cpu_count() or 1))
MAX_BYTES = 80e6
MAX_PACKED = 512          # longest side of the packed raw (=1024 px image); larger inputs are 2x2 binned
RAW_EXT = (".dng", ".nef", ".arw", ".cr2", ".cr3", ".orf", ".rw2", ".raf")
CCM = torch.tensor(CCM_DEFAULT)


def _load(path, cls):
    if not os.path.exists(path):
        return None
    ck = torch.load(path, map_location="cpu")
    m = cls(**ck["cfg"])
    m.load_state_dict(ck["state"])
    return m.eval()


HYBRID = _load("student.pt", HybridISP)
LEARNED = _load("learned.pt", LearnedISP)


def read_input(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in RAW_EXT:
        import rawpy
        with rawpy.imread(path) as r:
            raw = r.raw_image_visible.astype(np.float32)
            black, white = float(np.mean(r.black_level_per_channel)), float(r.white_level)
            desc = r.color_desc.decode()
            pat = "".join(desc[i] for i in r.raw_pattern.flatten())
        if pat not in _SHIFT:
            raise gr.Error(f"Unsupported CFA pattern '{pat}' (need a 2x2 RGGB-family Bayer sensor).")
        raw = np.clip((raw - black) / max(white - black, 1.0), 0, 1)
        return pack_rggb(to_rggb(torch.from_numpy(raw), pat)), "real RAW (DNG/vendor)", False
    img = Image.open(path).convert("RGB")
    img.thumbnail((1400, 1400))
    return srgb_to_pseudo_raw(np.array(img)), "simulated RAW derived from SIDD sRGB", True


def to_img(t):
    return (t[0].detach().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def naive_preview(x):
    lin = demosaic_malvar(unpack_rggb(x))
    return srgb_oetf(lin * (0.18 / lin.mean().clamp_min(1e-4)))


def process(file, degrade_on, noise_log, temp, ev, warmth, tone_a):
    global HYBRID
    if HYBRID is None:
        HYBRID = _load("student.pt", HybridISP)
    if HYBRID is None:
        raise gr.Error("Model checkpoint student.pt not found on the server.")
    if file is None:
        raise gr.Error("Please upload a RAW file (.dng/.nef/.arw/...) or an image (.jpg/.png), or click an example.")
    path = file if isinstance(file, str) else file.name
    if os.path.getsize(path) > MAX_BYTES:
        raise gr.Error("File is larger than 80 MB.")
    try:
        p, kind, synthetic = read_input(path)
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Could not read this file as RAW or image ({type(e).__name__}). Try a DNG, JPG or PNG.")
    notes = [f"Input: **{kind}**"]
    binned = 0
    while max(p.shape[-2:]) > MAX_PACKED:
        p = F.avg_pool2d(p[None], 2)[0]
        binned += 1
    if binned:
        notes.append(f"downscaled {2 ** binned}x by 2x2 Bayer binning for speed")
    x = p[None].float()
    info = None
    if degrade_on:
        torch.manual_seed(0)
        x, info = degrade(x, a=10 ** noise_log, T=temp, ev=ev)
    with torch.no_grad():
        t0 = time.perf_counter()
        trad = traditional_isp(x, tone_a=tone_a)
        t1 = time.perf_counter()
        hy, aux = HYBRID(x, warmth=warmth, tone_a=tone_a)
        t2 = time.perf_counter()
        stage = render_tail(demosaic_malvar(unpack_rggb(aux["wb"])), CCM, tone_a=tone_a, sharpen=0.0)
    g = torch.exp(aux["log_gains"])[0].tolist()
    md = " | ".join(notes) + "\n\n"
    md += f"Estimated gains (R,G,B) = {g[0]:.2f}, {g[1]:.2f}, {g[2]:.2f} · estimated noise gain a = {float(torch.exp(aux['log_a'][0])):.4f}"
    if info is not None:
        md += f" (true {float(info['a'][0]):.4f})"
    md += f"\n\nLatency on this server (CPU): traditional {1000 * (t1 - t0):.0f} ms · hybrid {1000 * (t2 - t1):.0f} ms"
    if synthetic:
        md += "\n\n*Note: The input was simulated RAW derived from SIDD sRGB (via inverse gamma, inverse CCM, and Bayer mosaicing); it is not real sensor RAW.*"
    return to_img(naive_preview(x)), to_img(stage), to_img(trad), to_img(hy), md


def build():
    samples = sorted(os.path.join("samples", f) for f in os.listdir("samples")) if os.path.isdir("samples") else []
    with gr.Blocks(title="AdaptISP: hybrid learned smartphone ISP") as demo:
        gr.Markdown("# AdaptISP: noise-adaptive hybrid smartphone ISP\nUpload a RAW/DNG or any JPG/PNG (processed honestly as simulated RAW derived from SIDD sRGB), "
                    "degrade it with the sliders, and compare the classical ISP with the hybrid (classical + learned) ISP.")
        with gr.Row():
            with gr.Column(scale=1):
                f = gr.File(label="RAW (.dng/.nef/.arw/.cr2) or image", type="filepath",
                            file_types=list(RAW_EXT) + [".jpg", ".jpeg", ".png"])
                deg = gr.Checkbox(True, label="Simulate sensor degradation")
                noise = gr.Slider(-4.0, -1.8, -2.8, step=0.1, label="Noise level (log10 shot-noise gain)")
                temp = gr.Slider(2500, 8000, 5500, step=100, label="Illuminant colour temperature (K)")
                ev = gr.Slider(-1.5, 0.75, 0.0, step=0.25, label="Exposure shift (EV)")
                warmth = gr.Slider(-0.5, 0.5, 0.0, step=0.05, label="Manual warmth edit (after estimation)")
                tone = gr.Slider(0.0, 0.5, 0.0, step=0.05, label="Tone contrast")
                btn = gr.Button("Run ISP", variant="primary")
            with gr.Column(scale=2):
                with gr.Row():
                    o1 = gr.Image(label="1. Degraded RAW (naive demosaic)")
                    o2 = gr.Image(label="2. Learned WB/exposure + demosaic only")
                with gr.Row():
                    o3 = gr.Image(label="3. Traditional ISP")
                    o4 = gr.Image(label="4. Hybrid ISP (ours)")
                md = gr.Markdown()
        if samples:
            gr.Examples([[s] for s in samples], inputs=[f])
        btn.click(process, [f, deg, noise, temp, ev, warmth, tone], [o1, o2, o3, o4, md])
    return demo.queue(max_size=8)


demo = build()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), share=True)
