from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import REQUIRED_BANDS
from .data_loader import SceneValidationError, load_scene
from .preprocessing import align_scene, scale_reflectance
from .spectral_indices import ndmi, ndvi, ndwi, validate_index_range


def validate_scene(scene_directory: str | Path) -> dict:
    scene = align_scene(load_scene(scene_directory))
    bands = {}
    for name in REQUIRED_BANDS:
        bands[name], _ = scale_reflectance(
            scene.bands[name].data,
            scale_factor=10000.0,
            nodata=scene.bands[name].nodata,
        )
    results = {
        "shape": list(scene.shape),
        "crs": scene.crs.to_string() if scene.crs else None,
        "bands": list(bands),
    }
    for name, index in {
        "ndvi": ndvi(bands["B08"], bands["B04"]),
        "ndwi": ndwi(bands["B03"], bands["B08"]),
        "ndmi": ndmi(bands["B08"], bands["B11"]),
    }.items():
        validate_index_range(index)
        results[name] = {"valid_pixels": int(index.count()), "masked_pixels": int(index.mask.sum())}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a Sentinel-2 Level-2A scene.")
    parser.add_argument("scene_directory", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(validate_scene(args.scene_directory), indent=2))
    except (FileNotFoundError, SceneValidationError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
