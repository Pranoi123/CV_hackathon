"""Data: clean RAW bank (simulated RAW derived from SIDD sRGB clean images) with scene-level splits.

Note: Everything in this pipeline operates on simulated RAW derived from SIDD sRGB (via inverse gamma,
inverse CCM, and RGGB Bayer mosaicing) or synthetic camera degradations. It is never claimed to be real RAW.
"""
import os
import numpy as np
import torch
from isp_core import CCM_DEFAULT, PHONE_BAYER, to_rggb, pack_rggb, mosaic_rggb


def srgb_to_pseudo_raw(img_u8, ccm=CCM_DEFAULT):
    """sRGB uint8 HxWx3 -> packed 'clean' RAW [4,h,w] by inverting gamma and the CCM, then mosaicing.
    Used for simulated RAW bank (labelled 'simulated RAW derived from SIDD sRGB')."""
    x = torch.from_numpy(np.ascontiguousarray(img_u8[..., :3])).float().permute(2, 0, 1) / 255.0
    x = x[:, : x.shape[1] // 2 * 2, : x.shape[2] // 2 * 2]
    lin = torch.where(x <= 0.04045, x / 12.92, ((x.clamp_min(0.04045) + 0.055) / 1.055) ** 2.4)
    cam = torch.linalg.solve(torch.tensor(ccm), lin.reshape(3, -1)).reshape_as(lin).clamp(0, 1)
    return pack_rggb(mosaic_rggb(cam))


def bayer_flip(p, hflip, vflip):
    """Flip a packed raw [4,h,w] without breaking the Bayer layout (returns 1 px smaller along flipped axes)."""
    if hflip:
        f = p.flip(-1)
        p = torch.stack([f[0][..., :-1], f[1][..., 1:], f[2][..., :-1], f[3][..., 1:]])
    if vflip:
        f = p.flip(-2)
        p = torch.stack([f[0][..., :-1, :], f[1][..., :-1, :], f[2][..., 1:, :], f[3][..., 1:, :]])
    return p


class CleanRawBank:
    def __init__(self):
        self.items = []  # (scene_id, packed fp16 tensor [4,h,w])
        self.idx = {}
        self.scene_split = {}

    def add(self, scene, packed):
        self.items.append((str(scene), packed.half()))

    def make_splits(self, test_frac=0.2, val_frac=0.1, seed=0):
        scenes = sorted({s for s, _ in self.items})
        rng = np.random.RandomState(seed)
        rng.shuffle(scenes)
        n = len(scenes)
        if n < 3:
            print("WARNING: <3 scenes, train/val/test overlap!")
            sp = {"train": scenes, "val": scenes, "test": scenes}
        else:
            nt, nv = max(1, round(n * test_frac)), max(1, round(n * val_frac))
            sp = {"test": scenes[:nt], "val": scenes[nt:nt + nv], "train": scenes[nt + nv:]}
            # Verify strict scene-level isolation: no scene appears in more than one split
            train_s = set(sp["train"])
            val_s = set(sp["val"])
            test_s = set(sp["test"])
            assert train_s.isdisjoint(val_s), f"Train and Val share scenes: {train_s.intersection(val_s)}"
            assert train_s.isdisjoint(test_s), f"Train and Test share scenes: {train_s.intersection(test_s)}"
            assert val_s.isdisjoint(test_s), f"Val and Test share scenes: {val_s.intersection(test_s)}"
        self.scene_split = sp
        self.idx = {k: [i for i, (s, _) in enumerate(self.items) if s in set(v)] for k, v in sp.items()}
        print("Split (images, scenes):", self.summary())

    def summary(self):
        return {k: (len(v), len(self.scene_split[k])) for k, v in self.idx.items()}  # (images, scenes)

    def _crop(self, rng, i, crop, aug):
        p = self.items[i][1]
        h, w = p.shape[-2:]
        c = crop + 1  # one spare pixel for Bayer-safe flips
        if h < c or w < c:
            raise ValueError(f"image {self.items[i][0]} too small ({h}x{w}) for crop {crop}")
        y, x = rng.randint(0, h - c + 1), rng.randint(0, w - c + 1)
        p = p[:, y:y + c, x:x + c].float()
        if aug:
            p = bayer_flip(p, rng.rand() < 0.5, rng.rand() < 0.5)
        return p[:, :crop, :crop]

    def sample(self, n, crop, split="train", aug=True):
        rng = np.random
        ids = self.idx[split]
        return torch.stack([self._crop(rng, ids[rng.randint(len(ids))], crop, aug) for _ in range(n)])

    def fixed(self, n, crop, split="test", seed=123):
        rng = np.random.RandomState(seed)
        ids = self.idx[split]
        return torch.stack([self._crop(rng, ids[k % len(ids)], crop, False) for k in range(n)])


# ------------------------------------------------------------------ loaders
def _read_mat(path):
    try:
        from scipy.io import loadmat
        d = loadmat(path)
        key = "x" if "x" in d else [k for k in d if not k.startswith("__")][0]
        return np.asarray(d[key], dtype=np.float32)
    except NotImplementedError:  # MATLAB v7.3 (HDF5)
        import h5py
        with h5py.File(path, "r") as f:
            key = "x" if "x" in f else list(f.keys())[0]
            return np.array(f[key], dtype=np.float32).T


def load_sidd_gt_into(bank, root, max_scenes=80, crop=1536):
    """Adds the clean GT RAW of SIDD Small scenes (one instance per scene). Folder names look like
    0001_001_S6_00100_00060_3200_L -> scene 0001, phone S6."""
    seen, n = set(), 0
    for dp, _, fs in sorted(os.walk(root)):
        for f in sorted(fs):
            if "GT_RAW" in f.upper() and f.lower().endswith(".mat"):
                parts = os.path.basename(dp).split("_")
                scene = parts[0]
                if scene in seen:
                    continue
                phone = parts[2] if len(parts) > 2 else "IP"
                pattern = PHONE_BAYER.get(phone, "RGGB")
                x = _read_mat(os.path.join(dp, f))
                if x.max() > 1.5:
                    x = x / x.max()
                x = to_rggb(np.clip(x, 0, 1), pattern)
                h, w = x.shape
                c = min(crop, h, w) // 2 * 2
                y0, x0 = (h - c) // 2 // 2 * 2, (w - c) // 2 // 2 * 2
                bank.add(scene, pack_rggb(torch.from_numpy(np.ascontiguousarray(x[y0:y0 + c, x0:x0 + c]))))
                seen.add(scene)
                n += 1
                if n >= max_scenes:
                    return n
    return n


def load_images_into(bank, folder, max_size=1024):
    """Load images from a flat folder or directory tree, converting sRGB to simulated RAW.
    Groups images by scene instance (e.g. 001_0001.png -> scene '001') so that splits do not leak scenes.
    Skips unreadable files with a warning.
    """
    from PIL import Image
    n = 0
    for dp, _, fs in sorted(os.walk(folder)):
        for f in sorted(fs):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(dp, f)
                try:
                    with Image.open(full_path) as im:
                        im = im.convert("RGB")
                        im.thumbnail((max_size, max_size))
                        pseudo = srgb_to_pseudo_raw(np.array(im))
                    # Group by scene instance (e.g. 001_0001.png -> scene '001')
                    scene_id = os.path.splitext(f)[0].split("_")[0]
                    bank.add(scene_id, pseudo)
                    n += 1
                except Exception as e:
                    print(f"WARNING: skipping unreadable file {full_path}: {e}")
    return n


def load_skimage_demo_into(bank):
    from skimage import data
    n = 0
    for name in ["astronaut", "coffee", "chelsea", "rocket", "immunohistochemistry", "retina", "hubble_deep_field"]:
        try:
            img = getattr(data, name)()
            if img.ndim == 3:
                bank.add(name, srgb_to_pseudo_raw(img))
                n += 1
        except Exception as e:  # dataset may need a download
            print("skip", name, type(e).__name__)
    return n


def load_sidd_val(noisy_path, gt_path, n=64, seed=0):
    """Real noisy/clean RAW blocks from the SIDD *validation* .mat files -> packed [n,4,128,128] each.
    NOTE: Bayer pattern per block is unknown here (assumed RGGB), so colours are approximate; use for
    denoising/detail metrics on REAL noise."""
    noisy, gt = _read_mat(noisy_path), _read_mat(gt_path)       # (40,32,256,256)
    noisy, gt = noisy.reshape(-1, 256, 256), gt.reshape(-1, 256, 256)
    ids = np.random.RandomState(seed).choice(len(noisy), size=min(n, len(noisy)), replace=False)
    f = lambda a: pack_rggb(torch.from_numpy(np.clip(a[ids], 0, 1)))
    return f(noisy), f(gt)
