# utils.py
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from pipeline_utils import (
    build_pipeline_slug,
    make_run_dirs,
    write_pipeline_meta,
    append_manifest_row,
    dest_from_src,
)

# -------------------------
# Image IO & discovery
# -------------------------

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def is_image_file(p: str | Path) -> bool:
    return Path(p).suffix.lower() in SUPPORTED_EXTS

def list_images(dirs: Sequence[str | Path], recursive: bool = True) -> List[Path]:
    """
    Collect image files from directories and single-file paths.
    Skips missing dirs silently to stay notebook-friendly.
    """
    out: List[Path] = []
    for d in dirs:
        dp = Path(d)
        if not dp.exists():
            continue
        if dp.is_file():
            if is_image_file(dp):
                out.append(dp.resolve())
            continue
        # Directory
        if recursive:
            for p in dp.rglob("*"):
                if p.is_file() and is_image_file(p):
                    out.append(p.resolve())
        else:
            for p in dp.glob("*"):
                if p.is_file() and is_image_file(p):
                    out.append(p.resolve())
    # dedupe preserve order
    seen = set()
    uniq = []
    for p in out:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq

def load_image_rgb(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Failed to read image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def save_image_rgb(img: np.ndarray, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(out_path), bgr)
    if not ok:
        raise IOError(f"Failed to write: {out_path}")

# -------------------------
# Simple preview grid
# -------------------------

def make_preview_grid(
    imgs: Sequence[np.ndarray],
    cols: int = 5,
    pad: int = 4,
    bg: int = 0,
) -> np.ndarray:
    """
    Build a basic (H,W,3) RGB uint8 grid from same-sized images.
    """
    if len(imgs) == 0:
        raise ValueError("No images to grid.")
    fixed = []
    h0, w0 = imgs[0].shape[:2]
    for im in imgs:
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        if im.shape[:2] != (h0, w0):
            im = cv2.resize(im, (w0, h0), interpolation=cv2.INTER_AREA)
        if im.dtype != np.uint8:
            im = np.clip(im, 0, 255).astype(np.uint8)
        fixed.append(im)

    rows = math.ceil(len(fixed) / cols)
    grid_h = rows * h0 + (rows + 1) * pad
    grid_w = cols * w0 + (cols + 1) * pad
    canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    canvas[:] = bg

    for i, im in enumerate(fixed):
        r = i // cols
        c = i % cols
        top = pad + r * (h0 + pad)
        left = pad + c * (w0 + pad)
        canvas[top:top + h0, left:left + w0] = im
    return canvas

# -------------------------
# Pipeline execution
# -------------------------

def run_pipeline(
    img: np.ndarray,
    pipeline: List[Tuple[str, dict]],
) -> np.ndarray:
    """
    Apply an ordered list of (transform_name, params) to an image.
    Each transform must return RGB uint8.
    """
    from transforms import REGISTRY  # late import to keep utils generic
    out = img
    for name, params in pipeline:
        fn = REGISTRY[name]
        out = fn(out, **(params or {}))
        if not isinstance(out, np.ndarray) or out.ndim != 3 or out.shape[2] != 3:
            raise ValueError(f"Transform {name} returned invalid image of shape {getattr(out, 'shape', None)}")
        if out.dtype != np.uint8:
            out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def save_step_previews(
    sample_paths: Sequence[Path],
    pipeline: List[Tuple[str, dict]],
    previews_dir: str | Path,
    cols: int = 6,
    max_samples: int = 12,
) -> None:
    """
    For a small sample, save per-step before/after grids and a final grid.
    """
    previews_dir = Path(previews_dir)
    previews_dir.mkdir(parents=True, exist_ok=True)
    from transforms import REGISTRY

    # Limit sample
    sample_paths = list(sample_paths)[:max_samples]
    if not sample_paths:
        return

    # Preload originals
    originals = [load_image_rgb(p) for p in sample_paths]

    # Per-step previews
    current = originals
    for step_idx, (name, params) in enumerate(pipeline, start=1):
        fn = REGISTRY[name]
        after = [fn(img, **(params or {})) for img in current]
        # interleave before/after pairs (first N)
        pairs = []
        for b, a in zip(current, after):
            pairs.extend([b, a])
        grid = make_preview_grid(pairs, cols=cols)
        out_path = previews_dir / f"step{step_idx:02d}_{name}.png"
        save_image_rgb(grid, out_path)
        current = after

    # Final grid (just finals)
    final_grid = make_preview_grid(current, cols=cols)
    save_image_rgb(final_grid, previews_dir / "final.png")

def apply_pipeline_for_root(
    input_root: str | Path,
    src_paths: Sequence[str | Path],
    pipeline: List[Tuple[str, dict]],
    run_name: str,
    overwrite: bool = False,
    show_progress: bool = True,
    save_previews: bool = True,
    preview_sample: int = 12,
) -> Dict[str, int]:
    """
    Create a run folder next to input_root and process all src_paths (which should live under input_root).
    Returns counters dict.
    """
    # Prepare run folder structure
    slug = build_pipeline_slug(pipeline)
    paths = make_run_dirs(input_root=input_root, run_name=run_name, pipeline_slug=slug, overwrite=overwrite)
    processed_dir = paths["processed_dir"]
    previews_dir = paths["previews_dir"]
    manifest_path = paths["manifest_path"]
    meta_path = paths["meta_path"]

    # Meta snapshot
    write_pipeline_meta(meta_path, run_name=run_name, pipeline=pipeline, extras={"input_root": str(paths["root"])})

    # Optional previews on a subset
    if save_previews and src_paths:
        sample = list(src_paths)[:preview_sample]
        save_step_previews(sample, pipeline, previews_dir)

    # Process all images
    processed = skipped = errors = 0
    iterator = tqdm(src_paths, desc=f"Pipeline {run_name}_{slug}", unit="img") if show_progress else src_paths

    # Prepare roots list for relative mapping (only this root)
    roots_for_rel = [Path(input_root).resolve()]

    for sp in iterator:
        sp = Path(sp)
        try:
            # Compute destination path under processed/
            dest = dest_from_src(sp, processed_dir, roots_for_rel)
            if dest.exists() and not overwrite:
                append_manifest_row(manifest_path, sp, dest, status="skipped", error="exists")
                skipped += 1
                continue

            img = load_image_rgb(sp)
            out = run_pipeline(img, pipeline)
            save_image_rgb(out, dest)
            append_manifest_row(manifest_path, sp, dest, status="ok", error="")
            processed += 1

        except Exception as e:
            append_manifest_row(manifest_path, sp, "", status="error", error=str(e))
            print(f"[apply_pipeline_for_root] Error on {sp}: {e}", file=sys.stderr)
            errors += 1

    return {"processed": processed, "skipped": skipped, "errors": errors}

def split_paths_by_root(
    all_paths: Sequence[str | Path],
    roots: Sequence[str | Path],
) -> Dict[Path, List[Path]]:
    """
    Group a list of file paths by which input root they belong to.
    If a path doesn't belong to any root, it is ignored (silent).
    """
    roots_resolved = [Path(r).resolve() for r in roots]
    buckets: Dict[Path, List[Path]] = {r: [] for r in roots_resolved}

    for p in all_paths:
        pp = Path(p).resolve()
        for r in roots_resolved:
            try:
                _ = pp.relative_to(r)
                buckets[r].append(pp)
                break
            except ValueError:
                continue
    # remove empties
    return {r: ps for r, ps in buckets.items() if ps}
