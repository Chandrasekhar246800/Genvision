from __future__ import annotations

import numpy as np


def _normalized_difference(
    numerator: np.ndarray,
    denominator_band: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> np.ma.MaskedArray:
    a = np.ma.asarray(numerator, dtype=np.float32)
    b = np.ma.asarray(denominator_band, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError("Index bands must have identical shapes.")
    combined_mask = np.ma.getmaskarray(a) | np.ma.getmaskarray(b)
    values_a = np.asarray(np.ma.getdata(a))
    values_b = np.asarray(np.ma.getdata(b))
    denominator = values_a + values_b
    invalid = ~np.isfinite(values_a) | ~np.isfinite(values_b) | (denominator == 0)
    if mask is not None:
        if mask.shape != a.shape:
            raise ValueError("Additional mask must match band shapes.")
        invalid |= mask
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (values_a - values_b) / denominator
    return np.ma.array(result.astype(np.float32), mask=combined_mask | invalid)


def ndvi(nir: np.ndarray, red: np.ndarray, *, mask: np.ndarray | None = None) -> np.ma.MaskedArray:
    """Normalized Difference Vegetation Index: (B08 - B04) / (B08 + B04)."""
    return _normalized_difference(nir, red, mask=mask)


def ndwi(green: np.ndarray, nir: np.ndarray, *, mask: np.ndarray | None = None) -> np.ma.MaskedArray:
    """Green-NIR NDWI (McFeeters): (B03 - B08) / (B03 + B08)."""
    return _normalized_difference(green, nir, mask=mask)


def ndmi(nir: np.ndarray, swir: np.ndarray, *, mask: np.ndarray | None = None) -> np.ma.MaskedArray:
    """NDMI moisture index: (B08 - B11) / (B08 + B11)."""
    return _normalized_difference(nir, swir, mask=mask)


def validate_index_range(index: np.ma.MaskedArray, *, tolerance: float = 1e-5) -> None:
    values = index.compressed()
    if values.size and (not np.isfinite(values).all() or values.min() < -1 - tolerance or values.max() > 1 + tolerance):
        raise ValueError("Normalized difference index contains values outside the expected [-1, 1] range.")
