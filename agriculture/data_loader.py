from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

from .config import REQUIRED_BANDS

_BAND_PATTERN = re.compile(r"(?<![A-Z0-9])(B(?:0[2348]|11)|SCL)(?![A-Z0-9])", re.IGNORECASE)


class SceneValidationError(ValueError):
    """Raised when a Sentinel-2 scene cannot be safely interpreted."""


@dataclass(frozen=True)
class BandRaster:
    name: str
    path: Path
    data: np.ndarray
    transform: Affine
    crs: CRS | None
    nodata: float | int | None
    resolution: tuple[float, float]
    profile: dict


@dataclass(frozen=True)
class Sentinel2Scene:
    bands: dict[str, BandRaster]

    @property
    def shape(self) -> tuple[int, int]:
        return next(iter(self.bands.values())).data.shape

    @property
    def crs(self) -> CRS | None:
        return next(iter(self.bands.values())).crs

    @property
    def transform(self) -> Affine:
        return next(iter(self.bands.values())).transform


def identify_band(path: str | Path) -> str | None:
    """Identify a documented Sentinel-2 band token from a filename."""
    matches = _BAND_PATTERN.findall(Path(path).name.upper())
    return matches[0].upper() if matches else None


def _validate_scene_metadata(reference: BandRaster, current: BandRaster) -> None:
    if current.data.ndim != 2 or not current.data.size:
        raise SceneValidationError(f"Band {current.name} has an invalid raster shape.")
    if current.crs != reference.crs:
        raise SceneValidationError(
            f"Band {current.name} CRS does not match {reference.name}; "
            "reproject explicitly before use."
        )


def load_band(path: str | Path, band_name: str | None = None) -> BandRaster:
    path = Path(path)
    name = (band_name or identify_band(path))
    if not name:
        raise SceneValidationError(f"Could not identify a Sentinel-2 band from {path.name}.")
    with rasterio.open(path) as src:
        if src.count != 1:
            raise SceneValidationError(f"{path.name} has {src.count} bands; expected one named spectral band.")
        data = src.read(1)
        profile = src.profile.copy()
        return BandRaster(
            name=name,
            path=path,
            data=data,
            transform=src.transform,
            crs=src.crs,
            nodata=src.nodata,
            resolution=(abs(src.transform.a), abs(src.transform.e)),
            profile=profile,
        )


def load_scene(directory: str | Path, required_bands: tuple[str, ...] = REQUIRED_BANDS) -> Sentinel2Scene:
    """Load one-file-per-band GeoTIFF/JP2-like rasters without treating RGB as bands."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Sentinel-2 scene directory not found: {directory}")
    candidates = [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in {".tif", ".tiff", ".jp2"}]
    paths: dict[str, Path] = {}
    for path in candidates:
        band = identify_band(path)
        if band in required_bands or band == "SCL":
            if band in paths:
                raise SceneValidationError(f"Multiple files identify as {band}: {paths[band]} and {path}")
            paths[band] = path
    missing = sorted(set(required_bands) - paths.keys())
    if missing:
        raise SceneValidationError(f"Missing required Sentinel-2 bands: {', '.join(missing)}")
    loaded = {name: load_band(paths[name], name) for name in required_bands}
    reference = loaded[required_bands[0]]
    for band in loaded.values():
        _validate_scene_metadata(reference, band)
    return Sentinel2Scene(loaded)
