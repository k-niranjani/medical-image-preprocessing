import numpy as np
import cv2

def get_name() -> str:
    return "clahe"

def _ensure_rgb_uint8(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("Input image is None.")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def transform(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8),
    space: str = "LAB",
    apply_on: str = "L",  # for LAB: "L"; for HSV: "V"
    **_,
) -> np.ndarray:
    """
    Apply CLAHE to luminance-like channel in LAB (default) or HSV.
    Args:
      clip_limit: higher → more contrast enhancement (risk of noise amplification).
      tile_grid_size: e.g., (8,8).
      space: "LAB" or "HSV".
      apply_on: "L" (LAB) or "V" (HSV).
    """
    img = _ensure_rgb_uint8(img)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid_size))

    if space.upper() == "LAB":
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        if apply_on.upper() != "L":
            # Fallback to L if an invalid channel is requested
            pass
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        out = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
        return out.astype(np.uint8)

    elif space.upper() == "HSV":
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        v_eq = clahe.apply(v)
        hsv_eq = cv2.merge([h, s, v_eq])
        out = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2RGB)
        return out.astype(np.uint8)

    else:
        raise ValueError("space must be 'LAB' or 'HSV'.")

# ... existing imports and code ...

PARAM_SPECS = {
    "clip_limit": {
        "default": 2.0,
        "type": "float",
        "desc": "Contrast limiting threshold; higher values yield stronger enhancement but may amplify noise.",
        "min": 0.0,
    },
    "tile_grid_size": {
        "default": (8, 8),
        "type": "tuple[int,int]",
        "desc": "CLAHE tile grid size (rows, cols) controlling local histogram regions.",
        "min": (1, 1),
    },
    "space": {
        "default": "LAB",
        "type": "str",
        "desc": "Colorspace used for enhancement: 'LAB' (L channel) or 'HSV' (V channel).",
        "choices": ["LAB", "HSV"],
    },
    "apply_on": {
        "default": "L",
        "type": "str",
        "desc": "Channel to enhance: 'L' for LAB, 'V' for HSV.",
        "choices": ["L", "V"],
    },
}

def get_param_specs():
    return dict(PARAM_SPECS)

def get_params():
    return {k: v["default"] for k, v in PARAM_SPECS.items()}
