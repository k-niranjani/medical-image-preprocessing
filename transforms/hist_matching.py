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
    template: np.ndarray,
    size: int = 512,
    interp: int = cv2.INTER_AREA,
    **_,
) -> np.ndarray:
    """
    Matches the histogram of the input image to the template image. It's a common technique
    to enhance an image with respect to a perfect template.
    """
    img = _ensure_rgb_uint8(img)
    template = _ensure_rgb_uint8(template)

    img = cv2.resize(img, (size, size), interpolation=interp)
    template = cv2.resize(template, (size, size), interpolation=interp)

    result = []
    for channel in range(3):
        source_flat = img[:,:,channel].ravel()
        template_flat = template[:,:,channel].ravel()

        s_values, bin_idx, s_counts = np.unique(source_flat, 
                                               return_inverse=True, 
                                               return_counts=True)
        
        t_values, t_counts = np.unique(template_flat, return_counts=True)
        
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]  # Normalize to [0,1]
        
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]  # Normalize to [0,1]
        
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
        
        interp_t_values_int = interp_t_values.astype(np.uint8)
        
        fractional_part = interp_t_values - interp_t_values_int
        interp_t_values_int[fractional_part > 0.5] += 1
        
        matched_channel = cv2.resize(interp_t_values_int[bin_idx], (size, size), interpolation=interp)
        result.append(matched_channel)
    
    result = cv2.merge((result[0], result[1], result[2]))
    return result


# ... existing imports and code ...

PARAM_SPECS = {
    "size": {
        "default": 512,
        "type": "int",
        "desc": "Target height and width in pixels",
        "min": 1,
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

