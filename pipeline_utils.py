# pipeline_utils.py
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

def build_pipeline_slug(pipeline: List[Tuple[str, dict]]) -> str:
    """
    Order-sensitive slug built from transform names only.
    Example: [("crop_dark_borders", {}), ("clahe", {...})] -> 'crop_dark_borders+clahe'
    """
    return "+".join([name for name, _ in pipeline])

def make_run_dirs(
    input_root: str | Path,
    run_name: str,
    pipeline_slug: str,
    overwrite: bool = False,
) -> Dict[str, Path]:
    """
    For an input root directory, create the run folder:
        <input_root>/<run_name>_<pipeline_slug>/
            processed/
            _previews/
            manifest.csv
            pipeline.json
    Returns a dict of important paths.
    """
    root = Path(input_root).resolve()
    if not root.exists() or not root.is_dir():
        raise NotADirectoryError(f"Input root does not exist or is not a directory: {root}")

    run_dir = root / f"{run_name}_{pipeline_slug}"
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Run folder already exists: {run_dir}. Set overwrite=True to reuse."
            )
    else:
        run_dir.mkdir(parents=True, exist_ok=True)

    processed_dir = run_dir / "processed"
    previews_dir = run_dir / "_previews"
    processed_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "manifest.csv"
    meta_path = run_dir / "pipeline.json"

    # Initialize manifest with header if creating fresh or overwriting
    if (not manifest_path.exists()) or overwrite:
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["src_path", "out_path", "status", "error"])

    return {
        "root": root,
        "run_dir": run_dir,
        "processed_dir": processed_dir,
        "previews_dir": previews_dir,
        "manifest_path": manifest_path,
        "meta_path": meta_path,
    }

def write_pipeline_meta(
    meta_path: str | Path,
    run_name: str,
    pipeline: List[Tuple[str, dict]],
    extras: Optional[Dict] = None,
) -> None:
    """
    Write a JSON snapshot of the pipeline and run info.
    We keep it minimal on purpose (no auto timestamp).
    """
    meta = {
        "run_name": run_name,
        "pipeline": [
            {"name": name, "params": (params or {})} for name, params in pipeline
        ],
    }
    if extras:
        meta["extras"] = extras
    meta_path = Path(meta_path)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def append_manifest_row(
    manifest_path: str | Path,
    src_path: str | Path,
    out_path: str | Path,
    status: str,
    error: str = "",
) -> None:
    """Append one row to manifest.csv"""
    with Path(manifest_path).open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([str(src_path), str(out_path), status, error])

def nearest_root_for(path: Path, roots: Sequence[Path]) -> Optional[Path]:
    """
    Return the deepest root that's an ancestor of 'path', else None.
    """
    path = path.resolve()
    best = None
    best_depth = -1
    for r in roots:
        rp = Path(r).resolve()
        try:
            _ = path.relative_to(rp)
            depth = len(rp.parts)
            if depth > best_depth:
                best = rp
                best_depth = depth
        except ValueError:
            continue
    return best

def dest_from_src(
    src_path: Path,
    processed_dir: Path,
    roots_for_rel: Optional[Sequence[Path]],
) -> Path:
    """
    Map a source file path to a destination path under processed_dir,
    preserving the relative structure under the nearest known root.
    Falls back to flat (filename only) if no root matches.
    """
    if roots_for_rel:
        nr = nearest_root_for(src_path, roots_for_rel)
        if nr is not None:
            rel = src_path.resolve().relative_to(nr)
            return processed_dir / rel
    return processed_dir / src_path.name
