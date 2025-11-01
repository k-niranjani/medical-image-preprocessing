"""
Transforms registry and parameter specs.
"""

from .crop_dark_borders import transform as crop_dark_borders, get_name as _n1, get_param_specs as _s1
from .circle_crop import       transform as circle_crop,       get_name as _n2, get_param_specs as _s2
from .resize import            transform as resize,            get_name as _n3, get_param_specs as _s3
from .unsharp_mask import      transform as unsharp_mask,      get_name as _n4, get_param_specs as _s4
from .clahe import             transform as clahe,             get_name as _n5, get_param_specs as _s5

REGISTRY = {
    _n1(): crop_dark_borders,
    _n2(): circle_crop,
    _n3(): resize,
    _n4(): unsharp_mask,
    _n5(): clahe,
}

SPECS = {
    _n1(): _s1(),
    _n2(): _s2(),
    _n3(): _s3(),
    _n4(): _s4(),
    _n5(): _s5(),
}

# Convenience: defaults-only view derived from SPECS
DEFAULTS = {name: {k: v.get("default") for k, v in spec.items()} for name, spec in SPECS.items()}

__all__ = ["REGISTRY", "SPECS", "DEFAULTS"]

