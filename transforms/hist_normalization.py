import numpy as np
import cv2

def get_name() -> str:
    return "histogram_normalization"

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
    bg_thresh: int = 0,
    **_,
) -> np.ndarray:
    """
    Applies Histogram Normalization while excluding background pixels.
    Any pixels with intensity < bg_thresh are considered background.
    """
    img = _ensure_rgb_uint8(img)
    img_enhanced = np.zeros_like(img)
        
    for channel in range(3):
        single_channel = img[:, :, channel]
        single_channel = single_channel[single_channel > bg_thresh]
        hist, _ = np.histogram(single_channel.flatten(), 256, [0,256])
        cdf = hist.cumsum()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        img_enhanced[:, :, channel] = cdf[single_channel]
    
    return img_enhanced

# ... existing imports and code ...

PARAM_SPECS = {
    "bg_thresh": {
        "default": 0,
        "type": "int",
        "desc": "The threshold for removing background pixels",
        "min": 0,
    }
}

def get_param_specs():
    return dict(PARAM_SPECS)

def get_params():
    return {k: v["default"] for k, v in PARAM_SPECS.items()}
