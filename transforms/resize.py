import numpy as np
import cv2

def get_name() -> str:
    return "resize"

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
    width: int = 512,
    height: int = 512,
    keep_aspect: bool = False,
    interp: int = cv2.INTER_AREA,
    **_,
) -> np.ndarray:
    """
    Resize image. If keep_aspect=True, letterbox to target (height, width).
    """
    img = _ensure_rgb_uint8(img)

    if not keep_aspect:
        return cv2.resize(img, (width, height), interpolation=interp)

    # Aspect-preserving letterbox
    h, w = img.shape[:2]
    scale = min(width / w, height / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=interp)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    top = (height - nh) // 2
    left = (width - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas

# ... existing imports and code ...

PARAM_SPECS = {
    "width": {
        "default": 512,
        "type": "int",
        "desc": "Target width in pixels (ignored if keep_aspect=True and used only for letterbox canvas).",
        "min": 1,
    },
    "height": {
        "default": 512,
        "type": "int",
        "desc": "Target height in pixels (ignored if keep_aspect=True and used only for letterbox canvas).",
        "min": 1,
    },
    "keep_aspect": {
        "default": False,
        "type": "bool",
        "desc": "If True, resizes with aspect ratio kept and letterboxes into (height,width).",
    },
    "interp": {
        "default": 3,  # cv2.INTER_AREA
        "type": "int (OpenCV interpolation flag)",
        "desc": "Interpolation method (e.g., cv2.INTER_AREA=3, cv2.INTER_LINEAR=1).",
    },
}

def get_param_specs():
    return dict(PARAM_SPECS)

def get_params():
    return {k: v["default"] for k, v in PARAM_SPECS.items()}
