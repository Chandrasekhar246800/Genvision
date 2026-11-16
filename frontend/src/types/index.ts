export type ModelName = "U-Net" | "Attention U-Net";

export type ImageAsset = {
  name: string;
  type: string;
  size: number;
  previewUrl?: string;
};

export type BackendHealth = {
  status: "ok" | "degraded";
  model_loaded: boolean;
  model_name: string;
  available_models?: string[];
};

export type BackendModel = {
  id: string;
  name: string;
  architecture: string;
  checkpoint: string;
  checkpoint_size_bytes: number;
  verified: boolean;
  loaded: boolean;
  device?: string;
};

export type ReconstructionResult = {
  blob: Blob;
  modelName: string;
  modelId?: string;
  inferenceTimeMs: number | null;
};

export type ModelMetricSummary = {
  image_count: number;
  l1_loss_mean: number;
  l1_loss_median: number;
  l1_loss_std: number;
  l1_loss_min: number;
  l1_loss_max: number;
  psnr_mean: number;
  psnr_median: number;
  psnr_std: number;
  psnr_min: number;
  psnr_max: number;
  ssim_mean: number;
  ssim_median: number;
  ssim_std: number;
  ssim_min: number;
  ssim_max: number;
};

export type MetricComparison = {
  l1_loss: number;
  psnr: number;
  ssim: number;
};

export type EvaluationMetrics = {
  evaluation: {
    test_count: number;
    evaluated_count: number;
    device: string;
    preprocessing: string;
    prediction_clamp: string;
    l1_definition: string;
    psnr_definition: string;
    ssim_definition: string;
    test_indices: number[];
    errors: string[];
  };
  models: {
    unet: ModelMetricSummary;
    attention_unet: ModelMetricSummary;
  };
  comparison: {
    attention_minus_unet: MetricComparison;
  };
};

export type AgricultureHealth = {
  status: "ok" | "degraded" | "unavailable";
  study_area: string;
  dataset_name: string;
  dataset_available: boolean;
  statistics_available: boolean;
  indices_available: boolean;
  masks_available: boolean;
  visualizations_available: boolean;
  available_visualizations_count: number;
  missing_files: string[];
};

export type AgricultureSummary = {
  study_area: string;
  acquisition_date: string;
  satellite: string;
  tile_id: string;
  crs: string;
  roi_dimensions: [number, number];
  roi_size_km: number;
  center_coordinates: {
    lat: number;
    lon: number;
  };
  bounds_wgs84: {
    west?: number;
    south?: number;
    east?: number;
    north?: number;
  };
  bounds_projected: {
    min_x?: number;
    min_y?: number;
    max_x?: number;
    max_y?: number;
  };
  processing_status: string;
  total_pixels: number;
  cloud_percentage: number;
  shadow_percentage: number;
  combined_obstruction_percentage: number;
  clear_surface_percentage: number;
  clear_pixels: number;
  obstructed_pixels: number;
  cloud_pixels: number;
  shadow_pixels: number;
  ndvi_mean_clear: number;
  ndwi_mean_clear: number;
  ndmi_mean_clear: number;
  reflectance_calibration: {
    scale_factor?: number;
    boa_quantification_value?: number;
    radiometric_offset_verified?: boolean;
    [key: string]: unknown;
  };
};

export type IndexStatsDetail = {
  min: number | null;
  max: number | null;
  mean: number | null;
  std: number | null;
  median: number | null;
  p5: number | null;
  p95: number | null;
  valid_count: number;
  total_count: number;
  valid_percent: number;
};

export type IndexPair = {
  name: string;
  description: string;
  masked_scientifically_valid: IndexStatsDetail;
  unmasked_all_pixels: IndexStatsDetail;
  cloud_impact_delta_mean: number;
};

export type AgricultureIndices = {
  study_area: string;
  total_roi_pixels: number;
  clear_sky_valid_pixels: number;
  clear_sky_percentage: number;
  indices: {
    ndvi: IndexPair;
    ndwi: IndexPair;
    ndmi: IndexPair;
    [key: string]: IndexPair;
  };
};

export type SCLClassInfo = {
  class_id: number;
  name: string;
  count: number;
  percentage: number;
};

export type ObstructionMetrics = {
  cloud_pixels: number;
  cloud_percent: number;
  shadow_pixels: number;
  shadow_percent: number;
  total_obstructed_pixels: number;
  total_obstructed_percent: number;
  clear_pixels: number;
  clear_percent: number;
};

export type AgricultureMasks = {
  study_area: string;
  total_pixels: number;
  obstruction_summary: ObstructionMetrics;
  scl_classes: SCLClassInfo[];
};

export type VisualizationCategory = "composite" | "index" | "classification" | "obstruction";

export type VisualizationMetadata = {
  id: string;
  title: string;
  category: VisualizationCategory;
  filename: string;
  description: string;
  endpoint_url: string;
  content_type: string;
  file_size_bytes: number;
};

export type VisualizationsList = {
  study_area: string;
  total_visualizations: number;
  visualizations: VisualizationMetadata[];
};
