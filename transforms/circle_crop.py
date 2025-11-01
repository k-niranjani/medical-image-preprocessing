import numpy as np
import cv2
from .crop_dark_borders import transform as crop_dark_borders

def get_name() -> str:
    return "circle_crop"

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
    center: tuple | None = None,
    radius: int | None = None,
    pad: int = 0,
    **_,
) -> np.ndarray:
    """
    Keep a circular region around the center; zero out outside; then trim dark borders.
    Args:
      center: (x, y) in pixel coords. Default = image center.
      radius: circle radius in pixels. Default = min(H, W)//2 - pad.
      pad: reduce radius by this many pixels (>=0).
    """
    img = _ensure_rgb_uint8(img)

    h, w = img.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    if radius is None:
        radius = max(1, min(h, w) // 2 - max(0, pad))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, int(radius), 255, thickness=-1)
    masked = cv2.bitwise_and(img, img, mask=mask)

    # Final tidy crop to remove zeroed borders
    cropped = crop_dark_borders(masked, tol=1)
    return cropped

# ... existing imports and code ...

PARAM_SPECS = {
    "center": {
        "default": None,
        "type": "tuple[int,int] | None",
        "desc": "Center (x,y) of the circle in pixels. None = image center.",
    },
    "radius": {
        "default": None,
        "type": "int | None",
        "desc": "Circle radius in pixels. None = min(H,W)//2 minus pad.",
        "min": 1,
    },
    "pad": {
        "default": 0,
        "type": "int ≥ 0",
        "desc": "Shrinks the default radius by this many pixels.",
        "min": 0,
    },
}

def get_param_specs():
    return dict(PARAM_SPECS)

def get_params():
    return {k: v["default"] for k, v in PARAM_SPECS.items()}
