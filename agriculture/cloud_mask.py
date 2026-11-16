from __future__ import annotations

import numpy as np

from .config import (
    CLOUD_SCL_CLASSES,
    DEFAULT_MASKED_SCL_CLASSES,
    SHADOW_SCL_CLASSES,
)


def scl_cloud_mask(
    scl: np.ndarray,
    *,
    masked_classes: frozenset[int] = DEFAULT_MASKED_SCL_CLASSES,
) -> np.ndarray:
    """Return True for pixels masked by the selected Sentinel-2 SCL classes (default: combined obstruction)."""
    if not np.issubdtype(scl.dtype, np.number):
        raise TypeError("SCL data must be numeric.")
    return np.isin(scl, list(masked_classes))


def scl_pure_cloud_mask(scl: np.ndarray) -> np.ndarray:
    """Return True strictly for medium/high probability cloud and thin cirrus (classes 8, 9, 10)."""
    if not np.issubdtype(scl.dtype, np.number):
        raise TypeError("SCL data must be numeric.")
    return np.isin(scl, list(CLOUD_SCL_CLASSES))


def scl_shadow_mask(scl: np.ndarray) -> np.ndarray:
    """Return True strictly for cloud shadow pixels (class 3)."""
    if not np.issubdtype(scl.dtype, np.number):
        raise TypeError("SCL data must be numeric.")
    return np.isin(scl, list(SHADOW_SCL_CLASSES))


def apply_cloud_mask(data: np.ndarray, mask: np.ndarray) -> np.ma.MaskedArray:
    if data.shape != mask.shape:
        raise ValueError("Cloud mask and data must have identical shapes.")
    return np.ma.array(data, mask=mask | ~np.isfinite(data))
