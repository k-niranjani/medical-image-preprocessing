import numpy as np
import cv2

def get_name() -> str:
    return "crop_dark_borders"

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

def _crop_image_from_gray(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """
    Crops away dark borders. Works for RGB or grayscale.
    tol: pixels with intensity <= tol are considered background.
    """
    if img.ndim == 2:
        mask = img > tol
        if not mask.any():
            return img
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray > tol
        if not mask.any():
            return img
        ys = np.where(mask.any(1))[0]
        xs = np.where(mask.any(0))[0]
        return img[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1, :]
    else:
        raise ValueError("Unsupported image shape for cropping.")

def transform(img: np.ndarray, tol: int = 7, **_) -> np.ndarray:
    """
    Remove dark borders from an RGB fundus image.
    Args:
      img: RGB uint8 image.
      tol: threshold (0–255) for background removal.
    """
    img = _ensure_rgb_uint8(img)
    return _crop_image_from_gray(img, tol=tol)

# ... existing imports and code ...

PARAM_SPECS = {
    "tol": {
        "default": 7,
        "type": "int (0–255)",
        "desc": "Pixel intensity threshold; values <= tol are treated as background when cropping.",
        "min": 0,
        "max": 255,
    }
}

def get_param_specs():
    # Return a shallow copy to avoid accidental mutation by callers
    return dict(PARAM_SPECS)

def get_params():
    # Backward-compat: just the defaults
    return {k: v["default"] for k, v in PARAM_SPECS.items()}
