import io
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent
EVALUATION_METRICS_PATH = PROJECT_ROOT / "results" / "evaluation" / "aggregate_metrics.json"

MAX_FILE_SIZE = 12 * 1024 * 1024
IMAGE_SIZE = (256, 256)
ACCEPTED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg"}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attention_unet import AttentionUNet  # noqa: E402
from unet import UNet  # noqa: E402

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "unet": {
        "id": "unet",
        "name": "U-Net",
        "architecture": "U-Net",
        "checkpoint_rel": "models/unet_controlled_best.pth",
        "model_class": UNet,
        "description": "Standard convolutional encoder-decoder with skip connections.",
    },
    "attention_unet": {
        "id": "attention_unet",
        "name": "Attention U-Net",
        "architecture": "Attention U-Net",
        "checkpoint_rel": "models/attention_controlled_best.pth",
        "model_class": AttentionUNet,
        "description": "Attention-gated encoder-decoder network.",
    },
}
DEFAULT_MODEL_ID = "unet"


def validate_checkpoint(path: Path) -> tuple[bool, str | None]:
    """Validate that a checkpoint exists, is a regular file, and has non-zero size."""
    if not path.exists():
        return False, f"Model checkpoint does not exist: {path.name}"
    if not path.is_file():
        return False, f"Model checkpoint is not a valid file: {path.name}"
    if path.stat().st_size == 0:
        return False, f"Model checkpoint is invalid (0 bytes / empty): {path.name}"
    return True, None


class ModelService:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models: dict[str, nn.Module] = {}
        self.load_errors: dict[str, str] = {}
        self.transform = transforms.Compose(
            [
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
            ]
        )

    @property
    def loaded(self) -> bool:
        """Returns True if at least the default model is loaded."""
        return DEFAULT_MODEL_ID in self.models

    @property
    def default_model_name(self) -> str:
        return MODEL_CONFIGS[DEFAULT_MODEL_ID]["name"]

    def load(self) -> None:
        """Independently load each configured model into the registry."""
        for model_id, config in MODEL_CONFIGS.items():
            ckpt_path = PROJECT_ROOT / config["checkpoint_rel"]
            is_valid, error_msg = validate_checkpoint(ckpt_path)
            if not is_valid:
                self.load_errors[model_id] = error_msg or "Invalid checkpoint"
                continue

            try:
                model = config["model_class"]().to(self.device)
                state_dict = torch.load(
                    ckpt_path,
                    map_location=self.device,
                    weights_only=True,
                )
                model.load_state_dict(state_dict)
                model.eval()
                self.models[model_id] = model
                self.load_errors.pop(model_id, None)
            except Exception as error:
                self.load_errors[model_id] = (
                    f"Model checkpoint could not be loaded: {type(error).__name__}: {error}"
                )

    def get_model_metadata(self, model_id: str) -> dict[str, Any]:
        config = MODEL_CONFIGS[model_id]
        ckpt_path = PROJECT_ROOT / config["checkpoint_rel"]
        is_valid, _ = validate_checkpoint(ckpt_path)
        size = ckpt_path.stat().st_size if ckpt_path.is_file() else 0
        is_loaded = model_id in self.models

        return {
            "id": model_id,
            "name": config["name"],
            "architecture": config["architecture"],
            "checkpoint": config["checkpoint_rel"],
            "checkpoint_size_bytes": size,
            "verified": is_valid and (model_id not in self.load_errors) and is_loaded,
            "loaded": is_loaded,
            "device": str(self.device),
        }

    def get_all_metadata(self) -> list[dict[str, Any]]:
        return [self.get_model_metadata(m_id) for m_id in MODEL_CONFIGS]

    def predict(self, content: bytes, model_id: str = DEFAULT_MODEL_ID) -> tuple[bytes, float, str]:
        if model_id not in MODEL_CONFIGS:
            valid_ids = ", ".join(MODEL_CONFIGS.keys())
            raise ValueError(f"Unsupported model '{model_id}'. Available models: {valid_ids}.")

        if model_id not in self.models:
            error_reason = self.load_errors.get(model_id, "Model failed to load.")
            raise RuntimeError(f"The model '{model_id}' is not available: {error_reason}")

        model = self.models[model_id]
        model_name = MODEL_CONFIGS[model_id]["name"]

        try:
            with Image.open(io.BytesIO(content)) as image:
                image.verify()
            with Image.open(io.BytesIO(content)) as image:
                if image.format not in {"PNG", "JPEG"}:
                    raise ValueError("The uploaded file is not a valid PNG or JPEG image.")
                rgb_image = image.convert("RGB")
        except (OSError, UnidentifiedImageError, ValueError) as error:
            raise ValueError("The uploaded file is not a valid PNG or JPEG image.") from error

        input_tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)
        start = time.perf_counter()
        with torch.inference_mode():
            output = model(input_tensor).squeeze(0).cpu().clamp(0, 1)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if not torch.isfinite(output).all():
            raise RuntimeError(f"Model '{model_name}' inference produced a non-finite output.")

        output_array = (output.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
        output_image = Image.fromarray(output_array, mode="RGB")
        result = io.BytesIO()
        output_image.save(result, format="PNG")
        return result.getvalue(), elapsed_ms, model_name

    def get_metrics(self) -> dict[str, Any]:
        if not EVALUATION_METRICS_PATH.is_file():
            raise FileNotFoundError("Evaluation metrics artifact not found.")
        with open(EVALUATION_METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
