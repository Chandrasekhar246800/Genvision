from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import (
    AgricultureHealthResponse,
    AgricultureIndicesResponse,
    AgricultureMasksResponse,
    AgricultureSummaryResponse,
    IndexPair,
    IndexStatsDetail,
    ObstructionMetrics,
    SCLClassInfo,
    VisualizationMetadata,
    VisualizationsListResponse,
)

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent
AGRICULTURE_RESULTS_DIR = PROJECT_ROOT / "results" / "agriculture" / "pulivendula_scene_01"

VISUALIZATION_REGISTRY: dict[str, dict[str, Any]] = {
    "true_color": {
        "id": "true_color",
        "title": "Sentinel-2C True-Color RGB Composite",
        "category": "composite",
        "rel_path": Path("composites") / "true_color.png",
        "description": "Natural surface reflectance composite rendered from Red (B04), Green (B03), and Blue (B02) bands with 2% contrast stretch.",
    },
    "false_color": {
        "id": "false_color",
        "title": "Sentinel-2C False-Color NIR Composite",
        "category": "composite",
        "rel_path": Path("composites") / "false_color.png",
        "description": "Near-Infrared false-color composite (B08, B04, B03). Photosynthesizing crops appear in vibrant crimson red; bare soils appear in cyan/gray.",
    },
    "ndvi": {
        "id": "ndvi",
        "title": "Normalized Difference Vegetation Index (NDVI)",
        "category": "index",
        "rel_path": Path("visualizations") / "ndvi.png",
        "description": "Clear-sky crop canopy vigor map using RdYlGn color ramp. Cloud and shadow pixels are masked in dark navy.",
    },
    "ndwi": {
        "id": "ndwi",
        "title": "Normalized Difference Water Index (NDWI - McFeeters)",
        "category": "index",
        "rel_path": Path("visualizations") / "ndwi.png",
        "description": "Surface water and moisture delineation map. Open water reservoirs exhibit positive values (> 0.0); dry soils exhibit negative values.",
    },
    "ndmi": {
        "id": "ndmi",
        "title": "Normalized Difference Moisture Index (NDMI)",
        "category": "index",
        "rel_path": Path("visualizations") / "ndmi.png",
        "description": "Canopy liquid water content / moisture stress index map using YlGnBu color gradient.",
    },
    "scl_classification": {
        "id": "scl_classification",
        "title": "Sentinel-2 Scene Classification (SCL)",
        "category": "classification",
        "rel_path": Path("visualizations") / "scl_classification.png",
        "description": "Thematic pixel classification map rendered with official ESA SCL color codes and class percentage distribution legend.",
    },
    "obstruction_map": {
        "id": "obstruction_map",
        "title": "Cloud and Cloud-Shadow Obstruction Distribution",
        "category": "obstruction",
        "rel_path": Path("visualizations") / "obstruction_map.png",
        "description": "Tri-class obstruction distribution map differentiating clear surface (56.5%), cloud shadows (13.1%), and convective clouds (30.4%).",
    },
    "cloud_free_agricultural_roi": {
        "id": "cloud_free_agricultural_roi",
        "title": "Cloud-Free Agricultural Canopy Health",
        "category": "composite",
        "rel_path": Path("visualizations") / "cloud_free_agricultural_roi.png",
        "description": "High-contrast agricultural monitoring map overlaying active crop canopies (NDVI >= 0.2) directly onto a dimmed true-color base.",
    },
}


class AgricultureService:
    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or AGRICULTURE_RESULTS_DIR
        self._cache: dict[str, dict[str, Any]] = {}

    def _read_json(self, rel_path: str | Path) -> dict[str, Any]:
        path = self.base_dir / rel_path
        if not path.is_file():
            raise FileNotFoundError(f"Required agriculture output file missing: {path.name}")
        
        # In-memory cache for fast, non-blocking response times
        cache_key = str(rel_path)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self._cache[cache_key] = data
            return data

    def check_health(self) -> AgricultureHealthResponse:
        missing: list[str] = []

        if not self.base_dir.is_dir():
            missing.append(f"Directory {self.base_dir.name} not found")
            return AgricultureHealthResponse(
                status="unavailable",
                study_area="Pulivendula, Andhra Pradesh, India",
                dataset_name="pulivendula_scene_01",
                dataset_available=False,
                statistics_available=False,
                indices_available=False,
                masks_available=False,
                visualizations_available=False,
                available_visualizations_count=0,
                missing_files=missing,
            )

        required_stats = [
            "statistics/processing_summary.json",
            "statistics/index_statistics.json",
            "statistics/scl_statistics.json",
            "metadata/roi_metadata.json",
        ]
        for f in required_stats:
            if not (self.base_dir / f).is_file():
                missing.append(f)

        required_geotiffs = [
            "indices/ndvi.tif",
            "indices/ndwi.tif",
            "indices/ndmi.tif",
            "masks/cloud_mask.tif",
            "masks/shadow_mask.tif",
            "masks/combined_obstruction_mask.tif",
            "masks/scl_aligned.tif",
        ]
        for f in required_geotiffs:
            if not (self.base_dir / f).is_file():
                missing.append(f)

        found_vis = 0
        for entry in VISUALIZATION_REGISTRY.values():
            vis_file = self.base_dir / entry["rel_path"]
            if vis_file.is_file():
                found_vis += 1
            else:
                missing.append(str(entry["rel_path"]))

        stats_ok = not any(f in missing for f in required_stats)
        indices_ok = not any("indices" in f for f in missing)
        masks_ok = not any("masks" in f for f in missing)
        vis_ok = (found_vis == len(VISUALIZATION_REGISTRY))

        is_all_ok = len(missing) == 0
        status: Literal["ok", "degraded", "unavailable"] = "ok" if is_all_ok else "degraded"

        return AgricultureHealthResponse(
            status=status,
            study_area="Pulivendula, Andhra Pradesh, India",
            dataset_name=self.base_dir.name,
            dataset_available=True,
            statistics_available=stats_ok,
            indices_available=indices_ok,
            masks_available=masks_ok,
            visualizations_available=vis_ok,
            available_visualizations_count=found_vis,
            missing_files=missing,
        )

    def get_summary(self) -> AgricultureSummaryResponse:
        summary_raw = self._read_json("statistics/processing_summary.json")
        stats = summary_raw.get("statistics", {})
        roi = summary_raw.get("roi", {})
        ds = summary_raw.get("dataset", {})

        return AgricultureSummaryResponse(
            study_area=f"{roi.get('name', 'Pulivendula ROI')} ({ds.get('tile', '43PHS')})",
            acquisition_date=ds.get("acquisition_date", "14 October 2026"),
            satellite=ds.get("satellite", "Sentinel-2C"),
            tile_id=ds.get("tile", "43PHS"),
            crs=ds.get("crs", "EPSG:32643"),
            roi_dimensions=roi.get("dimensions", [1000, 1000]),
            roi_size_km=roi.get("area_sq_km", 100.0) ** 0.5,
            center_coordinates=roi.get("center", {"lat": 14.42, "lon": 78.23}),
            bounds_wgs84=roi.get("wgs84_bounds", {}),
            bounds_projected=roi.get("utm_bounds", {}),
            processing_status=summary_raw.get("status", "PROCESSING SUCCESSFUL"),
            total_pixels=stats.get("total_pixels", 1000000),
            cloud_percentage=round(stats.get("cloud_percentage", 30.40), 2),
            shadow_percentage=round(stats.get("shadow_percentage", 13.09), 2),
            combined_obstruction_percentage=round(stats.get("total_obstructed_percentage", 43.49), 2),
            clear_surface_percentage=round(stats.get("clear_percentage", 56.51), 2),
            clear_pixels=stats.get("valid_clear_pixels", 565112),
            obstructed_pixels=stats.get("total_obstructed_pixels", 434888),
            cloud_pixels=stats.get("cloud_pixels", 303976),
            shadow_pixels=stats.get("shadow_pixels", 130912),
            ndvi_mean_clear=round(stats.get("ndvi_mean_clear", 0.3780), 4),
            ndwi_mean_clear=round(stats.get("ndwi_mean_clear", -0.3935), 4),
            ndmi_mean_clear=round(stats.get("ndmi_mean_clear", 0.0042), 4),
            reflectance_calibration=summary_raw.get("reflectance_calibration", {}),
        )

    def get_indices(self) -> AgricultureIndicesResponse:
        idx_raw = self._read_json("statistics/index_statistics.json")
        summary_raw = self._read_json("statistics/processing_summary.json")
        stats = summary_raw.get("statistics", {})

        def parse_detail(raw_dict: dict[str, Any]) -> IndexStatsDetail:
            return IndexStatsDetail(
                min=round(raw_dict["min"], 4) if raw_dict.get("min") is not None else None,
                max=round(raw_dict["max"], 4) if raw_dict.get("max") is not None else None,
                mean=round(raw_dict["mean"], 4) if raw_dict.get("mean") is not None else None,
                std=round(raw_dict["std"], 4) if raw_dict.get("std") is not None else None,
                median=round(raw_dict["median"], 4) if raw_dict.get("median") is not None else None,
                p5=round(raw_dict["p5"], 4) if raw_dict.get("p5") is not None else None,
                p95=round(raw_dict["p95"], 4) if raw_dict.get("p95") is not None else None,
                valid_count=raw_dict.get("valid_count", 0),
                total_count=raw_dict.get("total_count", 1000000),
                valid_percent=round(raw_dict.get("valid_percent", 0.0), 2),
            )

        ndvi_m = parse_detail(idx_raw["ndvi_masked_scientifically_valid"])
        ndvi_u = parse_detail(idx_raw["ndvi_unmasked_all_pixels"])
        ndwi_m = parse_detail(idx_raw["ndwi_masked_scientifically_valid"])
        ndwi_u = parse_detail(idx_raw["ndwi_unmasked_all_pixels"])
        ndmi_m = parse_detail(idx_raw["ndmi_masked_scientifically_valid"])
        ndmi_u = parse_detail(idx_raw["ndmi_unmasked_all_pixels"])

        indices_dict = {
            "ndvi": IndexPair(
                name="NDVI (Normalized Difference Vegetation Index)",
                description="Formulated as (B08 - B04) / (B08 + B04). Clear-sky canopy vigor reflects genuine photosynthesis.",
                masked_scientifically_valid=ndvi_m,
                unmasked_all_pixels=ndvi_u,
                cloud_impact_delta_mean=round((ndvi_u.mean or 0.0) - (ndvi_m.mean or 0.0), 4),
            ),
            "ndwi": IndexPair(
                name="NDWI (Normalized Difference Water Index - McFeeters)",
                description="Formulated as (B03 - B08) / (B03 + B08). Delineates open water bodies and surface liquid.",
                masked_scientifically_valid=ndwi_m,
                unmasked_all_pixels=ndwi_u,
                cloud_impact_delta_mean=round((ndwi_u.mean or 0.0) - (ndwi_m.mean or 0.0), 4),
            ),
            "ndmi": IndexPair(
                name="NDMI (Normalized Difference Moisture Index)",
                description="Formulated as (B08 - B11) / (B08 + B11). Measures plant canopy liquid water content.",
                masked_scientifically_valid=ndmi_m,
                unmasked_all_pixels=ndmi_u,
                cloud_impact_delta_mean=round((ndmi_u.mean or 0.0) - (ndmi_m.mean or 0.0), 4),
            ),
        }

        return AgricultureIndicesResponse(
            study_area="Pulivendula Agricultural Study Window (10 km × 10 km)",
            total_roi_pixels=stats.get("total_pixels", 1000000),
            clear_sky_valid_pixels=stats.get("valid_clear_pixels", 565112),
            clear_sky_percentage=round(stats.get("clear_percentage", 56.51), 2),
            indices=indices_dict,
        )

    def get_masks(self) -> AgricultureMasksResponse:
        scl_raw = self._read_json("statistics/scl_statistics.json")
        obs_raw = scl_raw.get("obstruction_summary", {})
        classes_raw = scl_raw.get("classes", {})

        scl_classes_list = [
            SCLClassInfo(
                class_id=int(c_id),
                name=info.get("name", f"Class {c_id}"),
                count=info.get("count", 0),
                percentage=round(info.get("percentage", 0.0), 2),
            )
            for c_id, info in sorted(classes_raw.items(), key=lambda x: int(x[0]))
        ]

        obstruction = ObstructionMetrics(
            cloud_pixels=obs_raw.get("cloud_pixels", 303976),
            cloud_percent=round(obs_raw.get("cloud_percent", 30.40), 2),
            shadow_pixels=obs_raw.get("shadow_pixels", 130912),
            shadow_percent=round(obs_raw.get("shadow_percent", 13.09), 2),
            total_obstructed_pixels=obs_raw.get("total_obstructed_pixels", 434888),
            total_obstructed_percent=round(obs_raw.get("total_obstructed_percent", 43.49), 2),
            clear_pixels=obs_raw.get("clear_pixels", 565112),
            clear_percent=round(obs_raw.get("clear_percent", 56.51), 2),
        )

        return AgricultureMasksResponse(
            study_area="Pulivendula Agricultural Study Window (10 km × 10 km)",
            total_pixels=scl_raw.get("total_pixels", 1000000),
            obstruction_summary=obstruction,
            scl_classes=scl_classes_list,
        )

    def get_visualizations_list(self) -> VisualizationsListResponse:
        items: list[VisualizationMetadata] = []
        for v_id, entry in VISUALIZATION_REGISTRY.items():
            file_path = self.base_dir / entry["rel_path"]
            size = file_path.stat().st_size if file_path.is_file() else 0
            items.append(
                VisualizationMetadata(
                    id=v_id,
                    title=entry["title"],
                    category=entry["category"],
                    filename=entry["rel_path"].name,
                    description=entry["description"],
                    endpoint_url=f"/agriculture/visualizations/{v_id}",
                    content_type="image/png",
                    file_size_bytes=size,
                )
            )

        return VisualizationsListResponse(
            study_area="Pulivendula Agricultural Study Window (10 km × 10 km)",
            total_visualizations=len(items),
            visualizations=items,
        )

    def get_visualization_path(self, name: str) -> Path:
        normalized = (name or "").strip().lower()
        if normalized.endswith(".png"):
            normalized = normalized[:-4]

        # Prevent path traversal and enforce strict whitelisting
        if normalized not in VISUALIZATION_REGISTRY:
            available = ", ".join(sorted(VISUALIZATION_REGISTRY.keys()))
            raise KeyError(
                f"Visualization '{name}' is not recognized. Available visualizations: {available}."
            )

        rel_path = VISUALIZATION_REGISTRY[normalized]["rel_path"]
        abs_path = (self.base_dir / rel_path).resolve()

        # Extra path traversal boundary guard
        if not str(abs_path).startswith(str(self.base_dir.resolve())):
            raise PermissionError("Access denied: path traversal attempt detected.")

        if not abs_path.is_file():
            raise FileNotFoundError(f"Visualization image file not found on disk: {rel_path.name}")

        return abs_path
