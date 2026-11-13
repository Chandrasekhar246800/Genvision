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
};

export type BackendModel = {
  name: string;
  architecture: string;
  checkpoint: string;
  checkpoint_size_bytes: number;
  verified: boolean;
  loaded: boolean;
};

export type ReconstructionResult = {
  blob: Blob;
  modelName: string;
  inferenceTimeMs: number | null;
};
