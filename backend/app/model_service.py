import io
import sys
import time
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "unet_controlled_best.pth"
MODEL_NAME = "U-Net"
MAX_FILE_SIZE = 12 * 1024 * 1024
IMAGE_SIZE = (256, 256)
ACCEPTED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg"}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unet import UNet  # noqa: E402


class ModelService:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.model: UNet | None = None
        self.load_error: str | None = None
        self.transform = transforms.Compose(
            [
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
            ]
        )

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if not CHECKPOINT_PATH.is_file():
            self.load_error = "The configured model checkpoint is unavailable."
            return
        if CHECKPOINT_PATH.stat().st_size == 0:
            self.load_error = "The configured model checkpoint is empty."
            return

        try:
            model = UNet().to(self.device)
            state_dict = torch.load(
                CHECKPOINT_PATH,
                map_location=self.device,
                weights_only=True,
            )
            model.load_state_dict(state_dict)
            model.eval()
            self.model = model
            self.load_error = None
        except Exception as error:
            self.model = None
            self.load_error = f"Model checkpoint could not be loaded: {type(error).__name__}."

    def metadata(self) -> dict:
        size = CHECKPOINT_PATH.stat().st_size if CHECKPOINT_PATH.is_file() else 0
        return {
            "name": MODEL_NAME,
            "architecture": "U-Net",
            "checkpoint": "models/unet_controlled_best.pth",
            "checkpoint_size_bytes": size,
            "verified": self.load_error is None and size > 0,
            "loaded": self.loaded,
        }

    def predict(self, content: bytes) -> tuple[bytes, float]:
        if self.model is None:
            raise RuntimeError("The inference model is not available.")

        try:
            with Image.open(io.BytesIO(content)) as image:
                image.verify()
            with Image.open(io.BytesIO(content)) as image:
                rgb_image = image.convert("RGB")
        except (OSError, UnidentifiedImageError, ValueError) as error:
            raise ValueError("The uploaded file is not a valid PNG or JPEG image.") from error

        input_tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)
        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model(input_tensor).squeeze(0).cpu().clamp(0, 1)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if not torch.isfinite(output).all():
            raise RuntimeError("Model inference produced a non-finite output.")

        output_array = (output.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
        output_image = Image.fromarray(output_array, mode="RGB")
        result = io.BytesIO()
        output_image.save(result, format="PNG")
        return result.getvalue(), elapsed_ms
