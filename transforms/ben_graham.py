import numpy as np
import cv2
from .crop_dark_borders import transform as crop_dark_borders

def get_name() -> str:
    return "ben_graham"

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
    size: int = 512,
    interp: int = cv2.INTER_AREA,
    **_,
) -> np.ndarray:
    """
    Applies Ben Graham's preprocessing method to improve lighting condition
    """
    img = _ensure_rgb_uint8(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (size, size), interpolation=interp)
    img = cv2.addWeighted (img,4, cv2.GaussianBlur( img , (0,0) , size/10) ,-4 ,128)
    return img

# ... existing imports and code ...

PARAM_SPECS = {
    "size": {
        "default": 512,
        "type": "int",
        "desc": "Target height and width in pixels."
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
