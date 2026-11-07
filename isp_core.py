"""Hybrid learned ISP: differentiable classical stages + small learned modules.

Packed RAW convention: tensor [B,4,h,w] with channel order (R, G1, G2, B), values in [0,1],
black level already subtracted, Bayer pattern already normalised to RGGB.

All simulated RAW data is derived from SIDD sRGB images via inverse gamma, inverse CCM,
and RGGB Bayer mosaicing. It is never claimed to be real sensor RAW.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------ constants
# Generic camera-RGB -> linear sRGB matrix (rows sum to 1). Replace with the camera's own matrix
# (DNG ColorMatrix) for a specific sensor.
CCM_DEFAULT = [[1.60, -0.45, -0.15],
               [-0.25, 1.50, -0.25],
               [-0.05, -0.50, 1.55]]
# Bayer patterns of the SIDD phones (verify against the SIDD README for your download).
PHONE_BAYER = {"GP": "BGGR", "IP": "RGGB", "S6": "GRBG", "N6": "BGGR", "G4": "BGGR"}
_SHIFT = {"RGGB": (0, 0), "GRBG": (0, 1), "GBRG": (1, 0), "BGGR": (1, 1)}
KEY_VALUE = 0.18  # target mean exposure of the reference rendering


# ------------------------------------------------------------------ Bayer helpers
def to_rggb(raw, pattern="RGGB"):
    """Crop a [...,H,W] mosaic (numpy or torch) so that its pattern becomes RGGB."""
    dy, dx = _SHIFT[pattern]
    raw = raw[..., dy:, dx:]
    h, w = raw.shape[-2:]
    return raw[..., : h - h % 2, : w - w % 2]


def pack_rggb(m):
    """[...,H,W] RGGB mosaic -> [...,4,H/2,W/2] (R,G1,G2,B)."""
    return torch.stack([m[..., 0::2, 0::2], m[..., 0::2, 1::2],
                        m[..., 1::2, 0::2], m[..., 1::2, 1::2]], dim=-3)


def unpack_rggb(p):
    """[...,4,h,w] -> [...,2h,2w] mosaic."""
    h, w = p.shape[-2:]
    m = p.new_zeros(*p.shape[:-3], 2 * h, 2 * w)
    m[..., 0::2, 0::2] = p[..., 0, :, :]
    m[..., 0::2, 1::2] = p[..., 1, :, :]
    m[..., 1::2, 0::2] = p[..., 2, :, :]
    m[..., 1::2, 1::2] = p[..., 3, :, :]
    return m


def mosaic_rggb(rgb):
    """[...,3,H,W] -> RGGB mosaic [...,H,W] (used to build pseudo-RAW from sRGB images)."""
    m = torch.zeros_like(rgb[..., 0, :, :])
    m[..., 0::2, 0::2] = rgb[..., 0, 0::2, 0::2]
    m[..., 0::2, 1::2] = rgb[..., 1, 0::2, 1::2]
    m[..., 1::2, 0::2] = rgb[..., 1, 1::2, 0::2]
    m[..., 1::2, 1::2] = rgb[..., 2, 1::2, 1::2]
    return m


# ------------------------------------------------------------------ classical: demosaic (Malvar-He-Cutler)
def _k(a):
    return torch.tensor(a, dtype=torch.float32) / 8.0


_K_G = _k([[0, 0, -1, 0, 0], [0, 0, 2, 0, 0], [-1, 2, 4, 2, -1], [0, 0, 2, 0, 0], [0, 0, -1, 0, 0]])
_K_H = _k([[0, 0, .5, 0, 0], [0, -1, 0, -1, 0], [-1, 4, 5, 4, -1], [0, -1, 0, -1, 0], [0, 0, .5, 0, 0]])
_K_V = _K_H.t().contiguous()
_K_X = _k([[0, 0, -1.5, 0, 0], [0, 2, 0, 2, 0], [-1.5, 0, 6, 0, -1.5], [0, 2, 0, 2, 0], [0, 0, -1.5, 0, 0]])


def demosaic_malvar(mosaic):
    """[B,H,W] RGGB mosaic (H,W even) -> [B,3,H,W] linear RGB."""
    dev = mosaic.device
    x = F.pad(mosaic.unsqueeze(1), (2, 2, 2, 2), mode="reflect")
    ks = torch.stack([_K_G, _K_H, _K_V, _K_X]).unsqueeze(1).to(dev, mosaic.dtype)
    f = F.conv2d(x, ks)
    g_i, h_i, v_i, x_i = f[:, 0], f[:, 1], f[:, 2], f[:, 3]
    H, W = mosaic.shape[-2:]
    yy = torch.arange(H, device=dev).view(H, 1) % 2
    xx = torch.arange(W, device=dev).view(1, W) % 2
    is_r, is_g1 = (yy == 0) & (xx == 0), (yy == 0) & (xx == 1)
    is_g2, is_b = (yy == 1) & (xx == 0), (yy == 1) & (xx == 1)
    R = torch.where(is_r, mosaic, torch.where(is_g1, h_i, torch.where(is_g2, v_i, x_i)))
    G = torch.where(is_r | is_b, g_i, mosaic)
    B = torch.where(is_b, mosaic, torch.where(is_g1, v_i, torch.where(is_g2, h_i, x_i)))
    return torch.stack([R, G, B], 1)


# ------------------------------------------------------------------ classical: WB, exposure, CCM, tone, filters
def apply_wb_packed(p, gains):
    """p [B,4,h,w], gains [B,3] (R,G,B)."""
    g = torch.stack([gains[:, 0], gains[:, 1], gains[:, 1], gains[:, 2]], 1)
    return p * g[:, :, None, None]


def gray_world_gains(p):
    m = p.mean((2, 3)).clamp_min(1e-4)
    g = 0.5 * (m[:, 1] + m[:, 2])
    return torch.stack([g / m[:, 0], torch.ones_like(g), g / m[:, 3]], 1)


def oracle_gains(p):
    """Gray-world WB + auto-exposure so that mean level = KEY_VALUE."""
    gw = gray_world_gains(p)
    wb = apply_wb_packed(p, gw)
    k = KEY_VALUE / wb.mean((1, 2, 3)).clamp_min(5e-3)
    return gw * k[:, None]


def srgb_oetf(x):
    x = x.clamp(0, 1)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.clamp_min(1e-8) ** (1 / 2.4) - 0.055)


def tone_curve(x, a):
    """Mild S-curve blended with identity; a=0 -> identity."""
    if a <= 0:
        return x
    return (1 - a) * x + a * (x * x * (3 - 2 * x))


def gaussian_blur(x, sigma):
    r = max(1, int(math.ceil(3 * sigma)))
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-(t ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def unsharp(x, amount=0.4, sigma=1.0):
    if amount <= 0:
        return x
    return x + amount * (x - gaussian_blur(x, sigma))


def _box(x, r):
    return F.avg_pool2d(F.pad(x, (r, r, r, r), mode="reflect"), 2 * r + 1, stride=1)


def guided_filter(guide, p, r=2, eps=1e-3):
    """Edge-preserving guided filter. guide [B,1,H,W], p [B,C,H,W], eps scalar or [B,1,1,1]."""
    mi, mp = _box(guide, r), _box(p, r)
    cov = _box(guide * p, r) - mi * mp
    var = _box(guide * guide, r) - mi * mi
    a = cov / (var + eps)
    b = mp - a * mi
    return _box(a, r) * guide + _box(b, r)


def render_tail(lin, ccm, tone_a=0.0, sharpen=0.0):
    """linear camera RGB -> CCM -> sRGB gamma -> tone curve -> unsharp mask.
    Defaults to tone_a=0.0, sharpen=0.0 to match the clean sRGB target exactly."""
    x = torch.einsum("ij,bjhw->bihw", ccm.to(lin.dtype), lin).clamp(0, 1)
    x = tone_curve(srgb_oetf(x), tone_a)
    if sharpen > 0:
        x = unsharp(x, sharpen, 1.0)
    return x.clamp(0, 1)


def reference_render(clean, ccm):
    """Ground-truth RGB: clean raw with identity white balance rendered to linear sRGB
    with no tone curve or unsharp (recovering the original sRGB image)."""
    return render_tail(demosaic_malvar(unpack_rggb(clean)), ccm, tone_a=0.0, sharpen=0.0)


# ------------------------------------------------------------------ noise statistics (classical, from the raw itself)
def noise_features(p):
    """Robust, heteroscedastic noise statistics from the difference of the two green planes (G1-G2 has no
    colour signal, only noise + a little texture). Returns feats [B,8] and global sigma [B]."""
    g1, g2 = p[:, 1].flatten(1), p[:, 2].flatten(1)
    d = (g1 - g2) / math.sqrt(2.0)
    lum = 0.5 * (g1 + g2)
    med = d.median(1, keepdim=True).values
    mad = 1.4826 * (d - med).abs().median(1, keepdim=True).values + 1e-6
    d = torch.max(torch.min(d, 4 * mad), -4 * mad)
    edges = [0.0, 0.05, 0.15, 0.4, 1.01]
    n = lum.shape[1]
    feats = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = ((lum >= lo) & (lum < hi)).float()
        cnt = m.sum(1)
        v = ((d ** 2) * m).sum(1) / (cnt + 1.0)
        feats += [torch.log(v + 1e-8) * 0.2, cnt / n]
    return torch.stack(feats, 1), mad.squeeze(1)


# ------------------------------------------------------------------ learned modules
class Estimator(nn.Module):
    """Tiny parameter estimator: thumbnail CNN + noise statistics -> residual log-gains (R,G,B) and
    noise model (log a, log b) with sigma^2(x) = a*x + b."""

    def __init__(self, ch=16):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, ch, 3, 2, 1), nn.ReLU(True),
            nn.Conv2d(ch, 2 * ch, 3, 2, 1), nn.ReLU(True),
            nn.Conv2d(2 * ch, 2 * ch, 3, 2, 1), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.mlp = nn.Sequential(nn.Linear(2 * ch + 8 + 1, 64), nn.ReLU(True), nn.Linear(64, 5))
        nn.init.zeros_(self.mlp[-1].weight)
        with torch.no_grad():
            self.mlp[-1].bias.copy_(torch.tensor([0.0, 0.0, 0.0, math.log(2e-3), math.log(2e-5)]))

    def forward(self, p, feats):
        t = F.adaptive_avg_pool2d(p, (32, 32))
        lm = torch.log(p.mean((1, 2, 3)).clamp_min(1e-4)).unsqueeze(1) * 0.2
        return self.mlp(torch.cat([self.cnn(t * 4.0), feats, lm], 1))


def _conv(cin, cout, sep):
    if sep and cin > 1:  # depthwise-separable
        return nn.Sequential(nn.Conv2d(cin, cin, 3, padding=1, groups=cin), nn.Conv2d(cin, cout, 1))
    return nn.Conv2d(cin, cout, 3, padding=1)


def _block(cin, cout, sep):
    return nn.Sequential(_conv(cin, cout, sep), nn.ReLU(True), _conv(cout, cout, sep), nn.ReLU(True))


class UNet(nn.Module):
    def __init__(self, cin, cout, base=24, sep=False):
        super().__init__()
        b = base
        self.e1 = _block(cin, b, sep)
        self.d1 = nn.Conv2d(b, 2 * b, 3, 2, 1)
        self.e2 = _block(2 * b, 2 * b, sep)
        self.d2 = nn.Conv2d(2 * b, 4 * b, 3, 2, 1)
        self.mid = _block(4 * b, 4 * b, sep)
        self.u2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, 2)
        self.dec2 = _block(4 * b, 2 * b, sep)
        self.u1 = nn.ConvTranspose2d(2 * b, b, 2, 2)
        self.dec1 = _block(2 * b, b, sep)
        self.head = nn.Conv2d(b, cout, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.relu(self.d1(e1)))
        m = self.mid(F.relu(self.d2(e2)))
        d2 = self.dec2(torch.cat([self.u2(m), e2], 1))
        d1 = self.dec1(torch.cat([self.u1(d2), e1], 1))
        return self.head(d1)


def _pad4(x):
    h, w = x.shape[-2:]
    ph, pw = (-h) % 4, (-w) % 4
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, h, w


class HybridISP(nn.Module):
    """RAW(degraded, packed) -> sRGB.
    [classical] noise stats -> [learned] estimator (gains + noise model) -> [classical] WB
    -> [learned] U-Net residual with adaptive gate & bounded tanh on top of [classical] Malvar demosaic
    -> [classical] CCM, gamma, tone curve, unsharp mask."""

    def __init__(self, base=24, sep=False, use_noise_map=True, use_estimator=True,
                 residual=True, alpha=0.5, use_gate=True, gate_k=150.0):
        super().__init__()
        self.cfg = dict(base=base, sep=sep, use_noise_map=use_noise_map,
                        use_estimator=use_estimator, residual=residual,
                        alpha=alpha, use_gate=use_gate, gate_k=gate_k)
        self.use_noise_map, self.use_estimator, self.residual = use_noise_map, use_estimator, residual
        self.alpha = alpha
        self.use_gate = use_gate
        self.gate_k = gate_k
        self.est = Estimator() if use_estimator else None
        self.net = UNet(8 if use_noise_map else 4, 12, base, sep)
        if residual:
            nn.init.zeros_(self.net.head.weight)
            nn.init.zeros_(self.net.head.bias)
        self.register_buffer("ccm", torch.tensor(CCM_DEFAULT))
        self.net_override = None  # callable(x)->y, used to plug in an ONNX/INT8 runtime

    def _run_net(self, x):
        x, h, w = _pad4(x)
        y = self.net_override(x) if self.net_override is not None else self.net(x)
        return F.pixel_shuffle(y, 2)[..., : 2 * h, : 2 * w]

    def forward(self, raw, warmth=0.0, tone_a=0.0, sharpen=0.0):
        feats, sigma = noise_features(raw)
        prior = oracle_gains(raw)
        if self.use_estimator:
            out = self.est(raw, feats)
            gains = prior * torch.exp(out[:, :3])
            log_a, log_b = out[:, 3], out[:, 4]
        else:
            gains = prior
            log_a = torch.full_like(sigma, math.log(1e-8))
            log_b = torch.log(sigma ** 2 + 1e-10)
        if warmth != 0.0:
            w = torch.tensor([math.exp(warmth), 1.0, math.exp(-warmth)], device=raw.device)
            gains = gains * w
        a, b = torch.exp(log_a).view(-1, 1, 1, 1), torch.exp(log_b).view(-1, 1, 1, 1)
        g4 = torch.stack([gains[:, 0], gains[:, 1], gains[:, 1], gains[:, 2]], 1)[:, :, None, None]
        wb = raw * g4
        if self.use_noise_map:
            nm = g4 * torch.sqrt(a * raw.clamp_min(0) + b) * 10.0
            inp = torch.cat([wb, nm], 1)
        else:
            inp = wb
        res_raw = self._run_net(inp)
        if self.residual:
            # Bounded, scaled residual to prevent oversaturation / overexposure
            res = self.alpha * torch.tanh(res_raw)
            if self.use_gate:
                # Differentiable adaptive gate based on estimated noise level sigma
                gate = torch.tanh(self.gate_k * sigma).view(-1, 1, 1, 1)
                res = gate * res
            base = demosaic_malvar(unpack_rggb(wb))
            lin = F.relu(base + res)
        else:
            lin = F.relu(res_raw)
        rgb = render_tail(lin, self.ccm, tone_a=tone_a, sharpen=sharpen)
        aux = dict(log_gains=torch.log(gains.clamp_min(1e-6)), log_a=log_a, log_b=log_b,
                   wb=wb, lin=lin, trainable=self.use_estimator)
        return rgb, aux


class LearnedISP(nn.Module):
    """End-to-end learned baseline: packed RAW -> RGB with a single U-Net (no classical stages)."""

    def __init__(self, base=32, sep=False):
        super().__init__()
        self.cfg = dict(base=base, sep=sep)
        self.net = UNet(4, 12, base, sep)

    def forward(self, raw, **kw):
        x, h, w = _pad4(raw)
        y = F.pixel_shuffle(self.net(x), 2)[..., : 2 * h, : 2 * w]
        return y, {}


@torch.no_grad()
def traditional_isp(raw, denoise=True, tone_a=0.0, sharpen=0.0):
    """Fully classical baseline: gray-world WB + auto-exposure, Malvar demosaic, guided-filter denoise
    (strength from the measured noise), CCM, gamma/tone, unsharp mask."""
    gains = oracle_gains(raw)
    wb = apply_wb_packed(raw, gains)
    lin = demosaic_malvar(unpack_rggb(wb))
    if denoise:
        _, sigma = noise_features(raw)
        sig = (sigma * gains[:, 1]).view(-1, 1, 1, 1)
        lin = guided_filter(lin.mean(1, keepdim=True), lin, r=2, eps=(2.0 * sig) ** 2 + 1e-8)
    ccm = torch.tensor(CCM_DEFAULT, device=raw.device)
    return render_tail(lin, ccm, tone_a=tone_a, sharpen=sharpen)


# ------------------------------------------------------------------ degradation simulator
def _sample(v, B, dev, lo, hi, log=False):
    if v is None:
        u = torch.rand(B, device=dev)
        return torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))) if log else lo + u * (hi - lo)
    return torch.as_tensor(v, dtype=torch.float32, device=dev).expand(B).clone()


def degrade(clean, a=None, T=None, ev=None, a_range=(1e-4, 1.5e-2)):
    """clean packed raw [B,4,h,w] -> degraded raw.
    a: shot-noise gain (ISO-like), T: illuminant colour temperature (K), ev: exposure shift (stops).
    Noise: heteroscedastic Gaussian approximation of Poisson-Gaussian, var = a*x + b."""
    B, dev = clean.shape[0], clean.device
    a = _sample(a, B, dev, a_range[0], a_range[1], log=True)
    T = _sample(T, B, dev, 2500.0, 8000.0)
    ev = _sample(ev, B, dev, -1.5, 0.75)
    x = (5500.0 - T) / 5500.0                     # warm light (low T) -> boost R, cut B
    cR, cB = torch.exp(0.8 * x), torch.exp(-0.9 * x)
    one = torch.ones_like(cR)
    cast = torch.stack([cR, one, one, cB], 1)[:, :, None, None]
    scale = (2.0 ** ev).view(-1, 1, 1, 1)
    x0 = (clean * cast * scale).clamp(0, 1)       # sensor saturation
    b = (torch.sqrt(a) * 0.1 * (0.5 + torch.rand(B, device=dev))) ** 2
    var = a.view(-1, 1, 1, 1) * x0 + b.view(-1, 1, 1, 1)
    noisy = (x0 + torch.randn_like(x0) * torch.sqrt(var)).clamp(0, 1)
    info = dict(a=a, b=b, T=T, ev=ev, cast_rgb=torch.stack([cR, one, cB], 1), scale=scale.view(-1))
    return noisy, info


def make_batch(clean, ccm, **kw):
    """-> noisy raw, target RGB, supervision labels (for the estimator), info."""
    noisy, info = degrade(clean, **kw)
    target = reference_render(clean, ccm)
    # Target gains that undo the simulated degradation
    need = 1.0 / (info["cast_rgb"] * info["scale"].view(-1, 1))
    labels = dict(log_gains=torch.log(need), log_a=torch.log(info["a"]), log_b=torch.log(info["b"]))
    return noisy, target, labels, info


# ------------------------------------------------------------------ losses
def ssim_torch(x, y, win=11, sigma=1.5):
    """Separable 1D Gaussian SSIM for fast, exact structural similarity calculation."""
    c = x.shape[1]
    t = torch.arange(win, device=x.device, dtype=x.dtype) - win // 2
    g = torch.exp(-t ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    kv = g.view(1, 1, win, 1).expand(c, 1, win, 1).contiguous()
    kh = g.view(1, 1, 1, win).expand(c, 1, 1, win).contiguous()

    def f(z):
        return F.conv2d(F.conv2d(z, kv, padding=(win // 2, 0), groups=c), kh, padding=(0, win // 2), groups=c)

    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx * mx, f(y * y) - my * my, f(x * y) - mx * my
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx * mx + my * my + C1) * (sxx + syy + C2))
    return s.mean()


def rgb2lab_torch(x):
    x = x.clamp(0, 1)
    lin = torch.where(x <= 0.04045, x / 12.92, ((x.clamp_min(0.04045) + 0.055) / 1.055) ** 2.4)
    M = x.new_tensor([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]])
    xyz = torch.einsum("ij,bjhw->bihw", M, lin) / x.new_tensor([0.95047, 1.0, 1.08883]).view(1, 3, 1, 1)
    eps, kappa = 216 / 24389, 24389 / 27
    f = torch.where(xyz > eps, xyz.clamp_min(eps) ** (1 / 3), (kappa * xyz + 16) / 116)
    return torch.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


def isp_loss(pred, target, aux=None, labels=None, w_ssim=0.4, w_lab=0.02, w_aux=0.1):
    """L1 + (1-SSIM) + Lab-L1 (colour/Delta-E proxy) + supervision of the estimator."""
    loss = F.l1_loss(pred, target) + w_ssim * (1 - ssim_torch(pred, target)) \
        + w_lab * F.l1_loss(rgb2lab_torch(pred), rgb2lab_torch(target))
    if aux and labels and aux.get("trainable", False):
        loss = loss + w_aux * (F.l1_loss(aux["log_gains"], labels["log_gains"])
                               + 0.5 * F.l1_loss(aux["log_a"], labels["log_a"])
                               + 0.5 * F.l1_loss(aux["log_b"], labels["log_b"]))
    return loss
