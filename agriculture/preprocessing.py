from __future__ import annotations

from dataclasses import replace

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from .data_loader import BandRaster, SceneValidationError, Sentinel2Scene


def scale_reflectance(
    data: np.ndarray,
    *,
    scale_factor: float,
    offset: float = 0.0,
    nodata: float | int | None = None,
    clamp_min_zero: bool = True,
) -> tuple[np.ma.MaskedArray, dict]:
    """Apply source-provided reflectance metadata with optional physical non-negative clamping.
    
    Reflectance = (DN + offset) / scale_factor
    For Sentinel-2 L2A PB 04.00+: offset = -1000.0, scale_factor = 10000.0.
    """
    if scale_factor == 0:
        raise ValueError("Reflectance scale factor must be non-zero.")
    mask = ~np.isfinite(data)
    if nodata is not None:
        mask |= data == nodata
    scaled = (data.astype(np.float32) + offset) / scale_factor
    if clamp_min_zero:
        scaled = np.maximum(scaled, 0.0)
    scaled = np.ma.array(scaled, mask=mask)
    return scaled, {
        "scale_factor": scale_factor,
        "offset": offset,
        "nodata_masked": True,
        "clamped_min_zero": clamp_min_zero,
    }


def band_valid_mask(band: BandRaster) -> np.ndarray:
    mask = np.isfinite(band.data)
    if band.nodata is not None:
        mask &= band.data != band.nodata
    return mask


def resample_band(
    band: BandRaster,
    *,
    target_shape: tuple[int, int],
    target_transform,
    target_crs,
    resampling: Resampling = Resampling.bilinear,
) -> BandRaster:
    """Explicitly resample one band; categorical layers preserve their native data type."""
    is_categorical = band.name.upper() == "SCL"
    target_dtype = band.data.dtype if is_categorical else np.float32
    destination = np.zeros(target_shape, dtype=target_dtype)
    reproject(
        source=band.data,
        destination=destination,
        src_transform=band.transform,
        src_crs=band.crs,
        dst_transform=target_transform,
        dst_crs=target_crs,
        src_nodata=band.nodata,
        dst_nodata=band.nodata,
        resampling=resampling,
    )
    profile = band.profile.copy()
    profile.update(
        height=target_shape[0],
        width=target_shape[1],
        transform=target_transform,
        crs=target_crs,
        dtype=destination.dtype.name,
    )
    return replace(band, data=destination, transform=target_transform, crs=target_crs, profile=profile)


def align_scene(
    scene: Sentinel2Scene,
    *,
    target_band: str = "B04",
    resampling: Resampling = Resampling.bilinear,
) -> Sentinel2Scene:
    """Align all bands to target_band's grid, ensuring categorical layers (SCL) use nearest-neighbour."""
    if target_band not in scene.bands:
        raise SceneValidationError(f"Target band {target_band} is not present.")
    target = scene.bands[target_band]
    aligned = {}
    for name, band in scene.bands.items():
        if band.data.shape == target.data.shape and band.transform == target.transform and band.crs == target.crs:
            aligned[name] = band
        else:
            # SCL is a discrete thematic classification layer and MUST use nearest-neighbor resampling
            # to preserve valid integer class IDs and prevent artificial interpolated classes.
            band_resampling = Resampling.nearest if name.upper() == "SCL" else resampling
            aligned[name] = resample_band(
                band,
                target_shape=target.data.shape,
                target_transform=target.transform,
                target_crs=target.crs,
                resampling=band_resampling,
            )
    return Sentinel2Scene(aligned)
