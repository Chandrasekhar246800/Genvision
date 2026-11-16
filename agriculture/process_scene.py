from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import transform
from rasterio.windows import Window

from .cloud_mask import (
    scl_cloud_mask,
    scl_pure_cloud_mask,
    scl_shadow_mask,
)
from .config import (
    ALL_SCENE_BANDS,
    DEFAULT_BOA_OFFSET,
    DEFAULT_SCALE_FACTOR,
    PULIVENDULA_COORDINATES,
    RESULTS_ROOT,
    SCL_COLORMAP,
    resolve_dataset_dir,
)
from .data_loader import load_scene
from .preprocessing import align_scene, scale_reflectance
from .spectral_indices import ndmi, ndvi, ndwi, validate_index_range
from .visualization import (
    save_cloud_free_agricultural_roi,
    save_index_visualization,
    save_obstruction_map,
    save_rgb_composite,
    save_scl_visualization,
    write_index_geotiff,
    write_mask_geotiff,
)


def compute_pulivendula_window(
    dataset_dir: Path,
    target_lat: float = PULIVENDULA_COORDINATES["lat"],
    target_lon: float = PULIVENDULA_COORDINATES["lon"],
    roi_pixels: int = 1000,
) -> tuple[Window, Affine, dict[str, Any]]:
    """Compute the 1000x1000 pixel window centered at target lat/lon on the 10m reference grid."""
    ref_band_path = dataset_dir / "B04.jp2"
    with rasterio.open(ref_band_path) as src:
        tile_transform = src.transform
        tile_crs = src.crs
        tile_width = src.width
        tile_height = src.height
        tile_bounds = src.bounds

    xs, ys = transform("EPSG:4326", tile_crs.to_string(), [target_lon], [target_lat])
    xc, yc = xs[0], ys[0]

    col_exact = (xc - tile_bounds.left) / 10.0
    row_exact = (tile_bounds.top - yc) / 10.0

    # Align to even column/row offsets so 20m sub-sampling snaps perfectly (2:1 ratio)
    col_start = int(round((col_exact - (roi_pixels / 2.0)) / 2.0) * 2)
    row_start = int(round((row_exact - (roi_pixels / 2.0)) / 2.0) * 2)

    # Guard bounds
    col_start = max(0, min(tile_width - roi_pixels, col_start))
    row_start = max(0, min(tile_height - roi_pixels, row_start))

    win_10m = Window(col_start, row_start, roi_pixels, roi_pixels)
    roi_transform = rasterio.windows.transform(win_10m, tile_transform)
    roi_bounds = rasterio.windows.bounds(win_10m, tile_transform)

    # Compute actual corner lat/lons
    lons, lats = transform(
        tile_crs.to_string(),
        "EPSG:4326",
        [roi_bounds[0], roi_bounds[2], (roi_bounds[0] + roi_bounds[2]) / 2],
        [roi_bounds[1], roi_bounds[3], (roi_bounds[1] + roi_bounds[3]) / 2],
    )

    metadata = {
        "target_requested": {"lat": target_lat, "lon": target_lon},
        "target_projected_center": {"x": xc, "y": yc},
        "roi_actual_center": {"lat": lats[2], "lon": lons[2]},
        "roi_bounds_projected": {
            "left": roi_bounds[0],
            "bottom": roi_bounds[1],
            "right": roi_bounds[2],
            "top": roi_bounds[3],
        },
        "roi_bounds_wgs84": {
            "min_lon": lons[0],
            "min_lat": lats[0],
            "max_lon": lons[1],
            "max_lat": lats[1],
        },
        "window_10m": {
            "col_off": col_start,
            "row_off": row_start,
            "width": roi_pixels,
            "height": roi_pixels,
        },
        "crs": tile_crs.to_string(),
        "pixel_size_meters": 10.0,
        "roi_size_km": 10.0,
    }

    return win_10m, roi_transform, metadata


def compute_masked_stats(arr: np.ma.MaskedArray) -> dict[str, float | None]:
    valid = arr.compressed()
    if valid.size == 0:
        return {"min": None, "max": None, "mean": None, "std": None, "count": 0}
    return {
        "min": float(valid.min()),
        "max": float(valid.max()),
        "mean": float(valid.mean()),
        "std": float(valid.std()),
        "median": float(np.median(valid)),
        "p5": float(np.percentile(valid, 5)),
        "p95": float(np.percentile(valid, 95)),
        "valid_count": int(valid.size),
        "total_count": int(arr.size),
        "valid_percent": float((valid.size / arr.size) * 100),
    }


def process_pulivendula_scene(
    scene_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    dataset_dir = resolve_dataset_dir(scene_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Sentinel-2 dataset directory not found at: {dataset_dir}")

    dest_dir = Path(output_dir) if output_dir else RESULTS_ROOT / "pulivendula_scene_01"
    dest_dir.mkdir(parents=True, exist_ok=True)

    dir_metadata = dest_dir / "metadata"
    dir_indices = dest_dir / "indices"
    dir_masks = dest_dir / "masks"
    dir_composites = dest_dir / "composites"
    dir_visualizations = dest_dir / "visualizations"
    dir_statistics = dest_dir / "statistics"

    for d in (dir_metadata, dir_indices, dir_masks, dir_composites, dir_visualizations, dir_statistics):
        d.mkdir(parents=True, exist_ok=True)

    print(f"=== PROCESSING SENTINEL-2 DATASET: {dataset_dir.name} ===")
    print(f"Destination: {dest_dir}")

    # 1. Compute 10km x 10km ROI window
    win_10m, roi_transform, roi_meta = compute_pulivendula_window(dataset_dir)
    print(f"Extracted 10m Window: col={win_10m.col_off}, row={win_10m.row_off}, {win_10m.width}x{win_10m.height}")

    # 2. Load bands for the ROI window (efficient windowed reading)
    print("Loading spectral bands and SCL window...")
    scene = load_scene(dataset_dir, required_bands=ALL_SCENE_BANDS, window_10m=win_10m)

    # 3. Align bands (B11 -> bilinear, SCL -> nearest-neighbor)
    print("Aligning bands to 10m reference grid...")
    aligned_scene = align_scene(scene, target_band="B04")

    # Verify spatial alignment
    ref_shape = (1000, 1000)
    for b_name, b_obj in aligned_scene.bands.items():
        assert b_obj.data.shape == ref_shape, f"Band {b_name} shape mismatch: {b_obj.data.shape}"
        assert b_obj.transform == roi_transform, f"Band {b_name} transform mismatch"

    # Reference GeoTIFF profile
    ref_profile = aligned_scene.bands["B04"].profile.copy()
    ref_profile.update(
        height=1000,
        width=1000,
        transform=roi_transform,
        crs=aligned_scene.crs,
    )

    # 4. Extract aligned raw arrays
    b02_raw = aligned_scene.bands["B02"].data
    b03_raw = aligned_scene.bands["B03"].data
    b04_raw = aligned_scene.bands["B04"].data
    b08_raw = aligned_scene.bands["B08"].data
    b11_raw = aligned_scene.bands["B11"].data
    scl_aligned = aligned_scene.bands["SCL"].data

    # Verify SCL contains strictly valid integer classes
    assert scl_aligned.dtype == np.uint8, f"SCL dtype is {scl_aligned.dtype}, expected uint8"
    unique_scl = set(np.unique(scl_aligned))
    assert unique_scl.issubset(set(range(12))), f"Invalid SCL classes detected: {unique_scl}"

    # 5. Radiometric scaling with -1000 offset (PB 04.00+)
    print("Scaling reflectance with offset -1000.0 DN...")
    b02_refl, _ = scale_reflectance(b02_raw, scale_factor=DEFAULT_SCALE_FACTOR, offset=DEFAULT_BOA_OFFSET)
    b03_refl, _ = scale_reflectance(b03_raw, scale_factor=DEFAULT_SCALE_FACTOR, offset=DEFAULT_BOA_OFFSET)
    b04_refl, _ = scale_reflectance(b04_raw, scale_factor=DEFAULT_SCALE_FACTOR, offset=DEFAULT_BOA_OFFSET)
    b08_refl, _ = scale_reflectance(b08_raw, scale_factor=DEFAULT_SCALE_FACTOR, offset=DEFAULT_BOA_OFFSET)
    b11_refl, _ = scale_reflectance(b11_raw, scale_factor=DEFAULT_SCALE_FACTOR, offset=DEFAULT_BOA_OFFSET)

    # 6. Generate masks
    print("Generating cloud, shadow, and obstruction masks...")
    cloud_mask = scl_pure_cloud_mask(scl_aligned)
    shadow_mask = scl_shadow_mask(scl_aligned)
    combined_obstruction_mask = scl_cloud_mask(scl_aligned)

    total_pixels = scl_aligned.size
    cloud_pixels = int(np.sum(cloud_mask))
    shadow_pixels = int(np.sum(shadow_mask))
    obstructed_pixels = int(np.sum(combined_obstruction_mask))
    clear_pixels = total_pixels - obstructed_pixels

    # SCL breakdown
    scl_u, scl_c = np.unique(scl_aligned, return_counts=True)
    scl_counts = {int(k): int(v) for k, v in zip(scl_u, scl_c)}
    scl_stats = {
        "total_pixels": total_pixels,
        "classes": {
            int(val): {
                "name": SCL_COLORMAP.get(val, ("#888888", f"Class {val}"))[1],
                "count": count,
                "percentage": float((count / total_pixels) * 100),
            }
            for val, count in scl_counts.items()
        },
        "obstruction_summary": {
            "cloud_pixels": cloud_pixels,
            "cloud_percent": float((cloud_pixels / total_pixels) * 100),
            "shadow_pixels": shadow_pixels,
            "shadow_percent": float((shadow_pixels / total_pixels) * 100),
            "total_obstructed_pixels": obstructed_pixels,
            "total_obstructed_percent": float((obstructed_pixels / total_pixels) * 100),
            "clear_pixels": clear_pixels,
            "clear_percent": float((clear_pixels / total_pixels) * 100),
        },
    }

    # 7. Compute spectral indices (both masked and unmasked)
    print("Computing NDVI, NDWI, NDMI...")
    ndvi_m = ndvi(b08_refl, b04_refl, mask=combined_obstruction_mask)
    ndvi_u = ndvi(b08_refl, b04_refl)

    ndwi_m = ndwi(b03_refl, b08_refl, mask=combined_obstruction_mask)
    ndwi_u = ndwi(b03_refl, b08_refl)

    ndmi_m = ndmi(b08_refl, b11_refl, mask=combined_obstruction_mask)
    ndmi_u = ndmi(b08_refl, b11_refl)

    # Validate index ranges
    validate_index_range(ndvi_m)
    validate_index_range(ndvi_u)
    validate_index_range(ndwi_m)
    validate_index_range(ndwi_u)
    validate_index_range(ndmi_m)
    validate_index_range(ndmi_u)

    index_stats = {
        "ndvi_masked_scientifically_valid": compute_masked_stats(ndvi_m),
        "ndvi_unmasked_all_pixels": compute_masked_stats(ndvi_u),
        "ndwi_masked_scientifically_valid": compute_masked_stats(ndwi_m),
        "ndwi_unmasked_all_pixels": compute_masked_stats(ndwi_u),
        "ndmi_masked_scientifically_valid": compute_masked_stats(ndmi_m),
        "ndmi_unmasked_all_pixels": compute_masked_stats(ndmi_u),
    }

    # 8. Write GeoTIFF files
    print("Writing GeoTIFF products...")
    write_index_geotiff(ndvi_m, ref_profile, dir_indices / "ndvi.tif")
    write_index_geotiff(ndwi_m, ref_profile, dir_indices / "ndwi.tif")
    write_index_geotiff(ndmi_m, ref_profile, dir_indices / "ndmi.tif")

    write_index_geotiff(ndvi_u, ref_profile, dir_indices / "ndvi_unmasked.tif")
    write_index_geotiff(ndwi_u, ref_profile, dir_indices / "ndwi_unmasked.tif")
    write_index_geotiff(ndmi_u, ref_profile, dir_indices / "ndmi_unmasked.tif")

    write_mask_geotiff(cloud_mask.astype(np.uint8), ref_profile, dir_masks / "cloud_mask.tif")
    write_mask_geotiff(shadow_mask.astype(np.uint8), ref_profile, dir_masks / "shadow_mask.tif")
    write_mask_geotiff(combined_obstruction_mask.astype(np.uint8), ref_profile, dir_masks / "combined_obstruction_mask.tif")
    write_mask_geotiff(scl_aligned, ref_profile, dir_masks / "scl_aligned.tif", dtype="uint8")

    # 9. Write Composites & Visualizations
    print("Rendering RGB composites and thematic visualizations...")
    save_rgb_composite(
        b04_raw, b03_raw, b02_raw,
        dir_composites / "true_color.png",
        title="Sentinel-2C Level-2A True-Color Composite (B04 - B03 - B02)",
    )
    save_rgb_composite(
        b08_raw, b04_raw, b03_raw,
        dir_composites / "false_color.png",
        title="Sentinel-2C False-Color NIR Composite (B08 - B04 - B03)",
        subtitle="Healthy Crop Canopies Appear in Vibrant Red · Pulivendula ROI (10 km × 10 km)",
    )

    save_index_visualization(
        ndvi_m,
        dir_visualizations / "ndvi.png",
        title="Normalized Difference Vegetation Index (NDVI) — Pulivendula Agricultural ROI",
        cmap="RdYlGn",
        vmin=-0.2,
        vmax=0.85,
        label="NDVI (Clear-Sky)",
    )
    save_index_visualization(
        ndwi_m,
        dir_visualizations / "ndwi.png",
        title="Normalized Difference Water Index (NDWI - McFeeters) — Pulivendula ROI",
        cmap="Blues_r",
        vmin=-0.8,
        vmax=0.3,
        label="NDWI (Water > 0)",
    )
    save_index_visualization(
        ndmi_m,
        dir_visualizations / "ndmi.png",
        title="Normalized Difference Moisture Index (NDMI) — Canopy Moisture Stress",
        cmap="YlGnBu",
        vmin=-0.4,
        vmax=0.6,
        label="NDMI (Canopy Moisture)",
    )
    save_scl_visualization(
        scl_aligned,
        dir_visualizations / "scl_classification.png",
        class_counts=scl_counts,
        title="Sentinel-2 Scene Classification (SCL) — Pulivendula Agricultural ROI",
    )
    save_obstruction_map(
        cloud_mask,
        shadow_mask,
        dir_visualizations / "obstruction_map.png",
        title="Cloud and Cloud-Shadow Obstruction Distribution — Pulivendula ROI",
    )
    save_cloud_free_agricultural_roi(
        b04_raw, b03_raw, b02_raw, ndvi_m,
        dir_visualizations / "cloud_free_agricultural_roi.png",
        title="Cloud-Free Agricultural Canopy Health (NDVI ≥ 0.2)",
    )

    # 10. Write JSON metadata and statistics
    elapsed_seconds = time.perf_counter() - start_time
    summary = {
        "status": "PROCESSING SUCCESSFUL",
        "dataset": {
            "name": dataset_dir.name,
            "satellite": "Sentinel-2C",
            "tile": "43PHS",
            "acquisition_date": "14 October 2026",
            "processing_level": "Level-2A",
            "crs": "EPSG:32643",
        },
        "roi": {
            "name": "Pulivendula Agricultural Study Window",
            "dimensions": [1000, 1000],
            "resolution_meters": 10.0,
            "area_sq_km": 100.0,
            "center": PULIVENDULA_COORDINATES,
            "wgs84_bounds": roi_meta["roi_bounds_wgs84"],
            "utm_bounds": roi_meta["roi_bounds_projected"],
        },
        "reflectance_calibration": {
            "scale_factor": DEFAULT_SCALE_FACTOR,
            "offset_dn": DEFAULT_BOA_OFFSET,
            "formula": "Reflectance = max((DN - 1000) / 10000.0, 0.0)",
            "baseline": "ESA Processing Baseline 04.00+",
        },
        "statistics": {
            "total_pixels": total_pixels,
            "valid_clear_pixels": clear_pixels,
            "clear_percentage": float((clear_pixels / total_pixels) * 100),
            "cloud_pixels": cloud_pixels,
            "cloud_percentage": float((cloud_pixels / total_pixels) * 100),
            "shadow_pixels": shadow_pixels,
            "shadow_percentage": float((shadow_pixels / total_pixels) * 100),
            "total_obstructed_pixels": obstructed_pixels,
            "total_obstructed_percentage": float((obstructed_pixels / total_pixels) * 100),
            "ndvi_mean_clear": index_stats["ndvi_masked_scientifically_valid"]["mean"],
            "ndvi_max_clear": index_stats["ndvi_masked_scientifically_valid"]["max"],
            "ndvi_min_clear": index_stats["ndvi_masked_scientifically_valid"]["min"],
            "ndwi_mean_clear": index_stats["ndwi_masked_scientifically_valid"]["mean"],
            "ndmi_mean_clear": index_stats["ndmi_masked_scientifically_valid"]["mean"],
        },
        "performance": {
            "elapsed_seconds": round(elapsed_seconds, 2),
            "windowed_processing": True,
        },
    }

    with open(dir_metadata / "roi_metadata.json", "w", encoding="utf-8") as f:
        json.dump(roi_meta, f, indent=2)

    with open(dir_statistics / "scl_statistics.json", "w", encoding="utf-8") as f:
        json.dump(scl_stats, f, indent=2)

    with open(dir_statistics / "index_statistics.json", "w", encoding="utf-8") as f:
        json.dump(index_stats, f, indent=2)

    with open(dir_statistics / "processing_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nProcessing complete in {elapsed_seconds:.2f} seconds!")
    print(f"Summary: Clear {clear_pixels:,} ({clear_pixels/total_pixels*100:.1f}%), "
          f"Cloud {cloud_pixels:,} ({cloud_pixels/total_pixels*100:.1f}%), "
          f"Shadow {shadow_pixels:,} ({shadow_pixels/total_pixels*100:.1f}%)")
    print(f"Mean NDVI (Clear sky): {index_stats['ndvi_masked_scientifically_valid']['mean']:.4f}")
    return summary


if __name__ == "__main__":
    process_pulivendula_scene()
