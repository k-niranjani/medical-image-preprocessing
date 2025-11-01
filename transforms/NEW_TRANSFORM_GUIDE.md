# Writing a New Preprocessing Transform

This guide explains how to add your own preprocessing transform so it integrates seamlessly with the pipeline system, previews, and metadata listings.

---

## ✅ Required API

Each new transform file must define at least two functions:

```
def get_name() -> str:
    """Return a short, unique name for this transform (e.g., 'gamma_correction')."""

def transform(img, **kwargs):
    """
    Apply the transform to an RGB uint8 NumPy array and return the transformed image.
    - `img` is always an RGB uint8 np.ndarray of shape (H, W, 3).
    - Return value must also be RGB uint8.
    - Avoid any file I/O inside this function; keep it purely computational.
    """
```

> After you add the file, import it in `transforms/__init__.py` like the others.  
> The registry automatically exposes your transform through `REGISTRY`.

---

## ⭐ Parameter Metadata (Highly Recommended)

In addition to `get_name()` and `transform()`, every transform can define **parameter metadata** for documentation and UI display.

### Why include metadata?
- It allows users to **discover tunable hyperparameters** directly in the notebook.  
- It enables **default filling** and **validation** later if we build a UI or CLI layer.  
- It makes your transform self-documenting.

---

### 🧱 Structure

Add this dictionary at the **end of your file**:

```
PARAM_SPECS = {
    "param_name": {
        "default": <value>,           # required
        "type": "int | float | str | bool | tuple[int,int] | None",  # short type hint
        "desc": "Short human-readable explanation of what this parameter does.",
        # Optional metadata:
        "min": <value>,               # numeric lower bound (if applicable)
        "max": <value>,               # numeric upper bound
        "choices": ["opt1", "opt2"],  # valid string options (if applicable)
    },
    # Add more parameters here...
}

def get_param_specs():
    """Return PARAM_SPECS (dict)."""
    return dict(PARAM_SPECS)

def get_params():
    """Return defaults only, for backward compatibility."""
    return {k: v["default"] for k, v in PARAM_SPECS.items()}
```

Each key corresponds to a keyword argument that `transform()` accepts.

---

### 🧩 Example: Gamma Correction Transform

Here’s a complete template you can copy and adapt:

```
import numpy as np
import cv2

def _ensure_rgb_uint8(img):
    if img is None:
        raise ValueError("Input image is None.")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def get_name() -> str:
    return "gamma_correction"

def transform(img, gamma: float = 1.0, **_):
    """
    Apply gamma correction: output = (input / 255) ** (1/gamma) * 255
    """
    img = _ensure_rgb_uint8(img)
    gamma = max(1e-6, float(gamma))
    inv = 1.0 / gamma
    table = (np.arange(256) / 255.0) ** inv * 255
    table = np.clip(table, 0, 255).astype(np.uint8)
    return cv2.LUT(img, table)

# ---- Parameter metadata ----
PARAM_SPECS = {
    "gamma": {
        "default": 1.0,
        "type": "float > 0",
        "desc": "Exponent for intensity correction (gamma < 1 brightens, > 1 darkens).",
        "min": 0.1,
        "max": 5.0,
    }
}

def get_param_specs():
    return dict(PARAM_SPECS)

def get_params():
    return {k: v["default"] for k, v in PARAM_SPECS.items()}
```

Once added and imported, this will appear in the notebook like:

```
• gamma_correction
    - gamma: default=1.0 (float > 0) [min=0.1, max=5.0]
      Exponent for intensity correction (gamma < 1 brightens, > 1 darkens).
```

---

## 🧪 Quick Local Test

Run these lines in a notebook or REPL:

```
from transforms import REGISTRY, SPECS

print("Available transforms:", sorted(REGISTRY.keys()))
print("Specs for gamma_correction:", SPECS.get("gamma_correction", {}))
```

You should see your transform listed with its metadata.

---

## 📏 Design Rules

- Always input/output **RGB uint8** (`dtype=np.uint8`, shape `(H, W, 3)`).
- Keep transforms **pure** (no side effects, no file I/O).
- Provide **sensible defaults**.
- Clip final output to `[0, 255]` and cast to `uint8`.
- Avoid importing heavy or niche libraries unless necessary.
- Validate parameter ranges inside `transform()` if they can cause crashes.

---

## 🧰 Common Patterns to Reuse

| Pattern | Example transforms | Tips |
|----------|-------------------|------|
| Brightness / Gamma | `gamma_correction`, `unsharp_mask` | Apply via lookup tables or addWeighted |
| Cropping | `crop_dark_borders`, `circle_crop` | Use intensity masks or contours |
| Local enhancement | `clahe` | Work in LAB or HSV, modify one channel |
| Resize / Aspect | `resize` | Offer `keep_aspect` and `interp` options |

---

## ❓ FAQ

**Do I need to edit `transforms/__init__.py`?**  
Yes, for now each new file should be imported there (both `transform` and `get_param_specs`).

**What happens if I skip `get_param_specs()`?**  
Your transform still works — it just won’t show up in the parameter documentation list.

**Can I depend on OpenCV and NumPy?**  
Yes. These are standard in the environment. Avoid extra dependencies unless essential.

**What if my transform fails on some images?**  
Raise a clear `ValueError` or handle gracefully — failures will be logged in the manifest.

---

## 🔧 Quick Scaffold Template

To start fast, copy this block into a new file like `transforms/my_new_method.py`:

```
import numpy as np
import cv2

def get_name():
    return "my_new_method"

def transform(img, param1=1.0, param2=False, **_):
    # --- your processing here ---
    return img

PARAM_SPECS = {
    "param1": {"default": 1.0, "type": "float", "desc": "Example numeric parameter."},
    "param2": {"default": False, "type": "bool", "desc": "Example toggle flag."},
}

def get_param_specs(): return dict(PARAM_SPECS)
def get_params(): return {k: v["default"] for k, v in PARAM_SPECS.items()}
```

Save, import, and run the notebook — your new transform will appear automatically.
