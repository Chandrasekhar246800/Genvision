from typing import Any, Literal

from pydantic import BaseModel, Field


# ============================================================================
# Existing Model & Evaluation Schemas (Preserved for ML Inference)
# ============================================================================


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_name: str
    available_models: list[str] = []


class ModelInfo(BaseModel):
    id: str
    name: str
    architecture: str
    checkpoint: str
    checkpoint_size_bytes: int
    verified: bool
    loaded: bool
    device: str = "cpu"


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


class ModelMetricSummary(BaseModel):
    image_count: int
    l1_loss_mean: float
    l1_loss_median: float
    l1_loss_std: float
    l1_loss_min: float
    l1_loss_max: float
    psnr_mean: float
    psnr_median: float
    psnr_std: float
    psnr_min: float
    psnr_max: float
    ssim_mean: float
    ssim_median: float
    ssim_std: float
    ssim_min: float
    ssim_max: float


class MetricComparison(BaseModel):
    l1_loss: float
    psnr: float
    ssim: float


class EvaluationMetadata(BaseModel):
    test_count: int
    evaluated_count: int
    device: str
    preprocessing: str
    prediction_clamp: str
    l1_definition: str
    psnr_definition: str
    ssim_definition: str
    test_indices: list[int]
    errors: list[str] = []


class MetricsResponse(BaseModel):
    evaluation: EvaluationMetadata
    models: dict[str, ModelMetricSummary]
    comparison: dict[str, MetricComparison]


# ============================================================================
# Agriculture API Schemas (Phase 3.3)
# ============================================================================


class AgricultureHealthResponse(BaseModel):
    status: Literal["ok", "degraded", "unavailable"]
    study_area: str
    dataset_name: str
    dataset_available: bool
    statistics_available: bool
    indices_available: bool
    masks_available: bool
    visualizations_available: bool
    available_visualizations_count: int
    missing_files: list[str] = []


class AgricultureSummaryResponse(BaseModel):
    study_area: str
    acquisition_date: str
    satellite: str
    tile_id: str
    crs: str
    roi_dimensions: list[int]
    roi_size_km: float
    center_coordinates: dict[str, float]
    bounds_wgs84: dict[str, float]
    bounds_projected: dict[str, float]
    processing_status: str
    total_pixels: int
    cloud_percentage: float
    shadow_percentage: float
    combined_obstruction_percentage: float
    clear_surface_percentage: float
    clear_pixels: int
    obstructed_pixels: int
    cloud_pixels: int
    shadow_pixels: int
    ndvi_mean_clear: float
    ndwi_mean_clear: float
    ndmi_mean_clear: float
    reflectance_calibration: dict[str, Any]


class IndexStatsDetail(BaseModel):
    min: float | None
    max: float | None
    mean: float | None
    std: float | None
    median: float | None
    p5: float | None
    p95: float | None
    valid_count: int
    total_count: int
    valid_percent: float


class IndexPair(BaseModel):
    name: str
    description: str
    masked_scientifically_valid: IndexStatsDetail
    unmasked_all_pixels: IndexStatsDetail
    cloud_impact_delta_mean: float


class AgricultureIndicesResponse(BaseModel):
    study_area: str
    total_roi_pixels: int
    clear_sky_valid_pixels: int
    clear_sky_percentage: float
    indices: dict[str, IndexPair]


class SCLClassInfo(BaseModel):
    class_id: int
    name: str
    count: int
    percentage: float


class ObstructionMetrics(BaseModel):
    cloud_pixels: int
    cloud_percent: float
    shadow_pixels: int
    shadow_percent: float
    total_obstructed_pixels: int
    total_obstructed_percent: float
    clear_pixels: int
    clear_percent: float


class AgricultureMasksResponse(BaseModel):
    study_area: str
    total_pixels: int
    obstruction_summary: ObstructionMetrics
    scl_classes: list[SCLClassInfo]


class VisualizationMetadata(BaseModel):
    id: str
    title: str
    category: Literal["composite", "index", "classification", "obstruction"]
    filename: str
    description: str
    endpoint_url: str
    content_type: str = "image/png"
    file_size_bytes: int


class VisualizationsListResponse(BaseModel):
    study_area: str
    total_visualizations: int
    visualizations: list[VisualizationMetadata]
