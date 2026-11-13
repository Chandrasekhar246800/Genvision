from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio


def write_index_geotiff(index: np.ma.MaskedArray, reference_profile: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    profile = reference_profile.copy()
    profile.update(driver="GTiff", count=1, dtype="float32", nodata=-9999.0, compress="deflate")
    output = np.ma.filled(index, -9999.0).astype("float32")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(output, 1)


def save_index_preview(index: np.ma.MaskedArray, output_path: str | Path, *, title: str) -> None:
    output_path = Path(output_path)
    values = np.ma.masked_invalid(index)
    plt.imsave(output_path, values, cmap="RdYlGn", vmin=-1, vmax=1)

