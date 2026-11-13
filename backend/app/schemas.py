from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_name: str


class ModelInfo(BaseModel):
    name: str
    architecture: str
    checkpoint: str
    checkpoint_size_bytes: int
    verified: bool
    loaded: bool


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
