#!/usr/bin/env python3
"""Dataset prep script for SIDD Small sRGB clean ground-truth bank.

Extracts ONLY GT_SRGB_010.PNG from each scene folder (from zip archive or unzipped directory).
Saves to destination folder (default 'sidd_gt/') named as:
    {scene_instance}_{sequence}.png  (e.g., 001_0001.png)

This naming guarantees scene-level grouping via filename.split("_")[0].
Noisy files are intentionally ignored because the pipeline applies its own synthetic degradations.
All data produced here is simulated RAW derived from SIDD sRGB.
"""

import os
import sys
import shutil
import zipfile
import argparse
from pathlib import Path


def find_sidd_source(hint_path=None):
    """Finds SIDD source either as zip file or directory."""
    candidates = []
    if hint_path:
        candidates.append(Path(hint_path))
    candidates.extend([
        Path("SIDD_Small_sRGB_Only.zip"),
        Path("../Downloads/SIDD_Small_sRGB_Only.zip"),
        Path("C:/Users/Pranav/Downloads/SIDD_Small_sRGB_Only.zip"),
        Path("SIDD_Small_sRGB_Only/Data"),
        Path("SIDD_Small_sRGB_Only"),
    ])

    for c in candidates:
        if c.exists():
            return c
    return None


def prep_from_zip(zip_path, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    images_found = 0
    scenes = set()

    with zipfile.ZipFile(zip_path, "r") as z:
        for entry in z.infolist():
            if entry.is_dir():
                continue
            fname = Path(entry.filename)
            if fname.name.upper() == "GT_SRGB_010.PNG":
                # Folder name looks like: 0001_001_S6_00100_00060_3200_L
                folder_name = fname.parent.name
                parts = folder_name.split("_")
                if len(parts) >= 2:
                    sequence = parts[0]
                    scene_instance = parts[1]
                    out_name = f"{scene_instance}_{sequence}.png"
                    out_path = out_dir / out_name
                    with z.open(entry) as src, open(out_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    images_found += 1
                    scenes.add(scene_instance)

    return images_found, sorted(list(scenes))


def prep_from_dir(dir_path, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    images_found = 0
    scenes = set()

    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.upper() == "GT_SRGB_010.PNG":
                folder_name = Path(root).name
                parts = folder_name.split("_")
                if len(parts) >= 2:
                    sequence = parts[0]
                    scene_instance = parts[1]
                    out_name = f"{scene_instance}_{sequence}.png"
                    out_path = out_dir / out_name
                    shutil.copy2(Path(root) / f, out_path)
                    images_found += 1
                    scenes.add(scene_instance)

    return images_found, sorted(list(scenes))


def main():
    parser = argparse.ArgumentParser(description="Prepare SIDD GT clean images for pseudo-RAW bank.")
    parser.add_argument("--src", type=str, default=None, help="Path to SIDD zip or folder.")
    parser.add_argument("--out", type=str, default="sidd_gt", help="Output directory (default: sidd_gt).")
    args = parser.parse_args()

    src = find_sidd_source(args.src)
    if not src:
        print("ERROR: Could not locate SIDD_Small_sRGB_Only.zip or extracted directory.")
        sys.exit(1)

    out_dir = Path(args.out)
    print(f"Source: {src}")
    print(f"Target directory: {out_dir.resolve()}")

    if src.is_file() and src.suffix.lower() == ".zip":
        print("Extracting GT images from zip archive...")
        images_found, scenes = prep_from_zip(src, out_dir)
    else:
        print("Copying GT images from directory...")
        images_found, scenes = prep_from_dir(src, out_dir)

    print(f"Images found: {images_found}")
    print(f"Unique scene instances: {len(scenes)}")
    print(f"Scene IDs: {scenes}")


if __name__ == "__main__":
    main()
