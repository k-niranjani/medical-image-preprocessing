import numpy as np
import cv2

def get_name() -> str:
    return "unsharp_mask"

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
    sigma: float = 10.0,
    amount_a: float = 4.0,
    amount_b: float = -4.0,
    bias: float = 128.0,
    **_,
) -> np.ndarray:
    """
    Ben-Graham style local contrast boost via unsharp masking:
      out = img*amount_a + gaussian(img, sigma)*amount_b + bias
    """
    img = _ensure_rgb_uint8(img)
    blurred = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=float(sigma))
    out = cv2.addWeighted(img, amount_a, blurred, amount_b, bias)
    # ensure uint8
    return np.clip(out, 0, 255).astype(np.uint8)

# ... existing imports and code ...

PARAM_SPECS = {
    "sigma": {
        "default": 10.0,
        "type": "float",
        "desc": "Gaussian blur sigma for the low-pass image (larger = stronger local contrast region).",
        "min": 0.1,
    },
    "amount_a": {
        "default": 4.0,
        "type": "float",
        "desc": "Weight for original image in addWeighted(). Controls sharpening strength.",
    },
    "amount_b": {
        "default": -4.0,
        "type": "float",
        "desc": "Weight for blurred image in addWeighted(). Negative to subtract blur.",
    },
    "bias": {
        "default": 128.0,
        "type": "float",
        "desc": "Added bias after weighted sum; can shift mid-tones.",
    },
}

def get_param_specs():
    return dict(PARAM_SPECS)

def get_params():
    return {k: v["default"] for k, v in PARAM_SPECS.items()}
