from __future__ import annotations

import numpy as np

from .config import DEFAULT_MASKED_SCL_CLASSES


def scl_cloud_mask(
    scl: np.ndarray,
    *,
    masked_classes: frozenset[int] = DEFAULT_MASKED_SCL_CLASSES,
) -> np.ndarray:
    """Return True for pixels masked by the selected Sentinel-2 SCL classes."""
    if not np.issubdtype(scl.dtype, np.number):
        raise TypeError("SCL data must be numeric.")
    return np.isin(scl, list(masked_classes))


def apply_cloud_mask(data: np.ndarray, mask: np.ndarray) -> np.ma.MaskedArray:
    if data.shape != mask.shape:
        raise ValueError("Cloud mask and data must have identical shapes.")
    return np.ma.array(data, mask=mask | ~np.isfinite(data))
