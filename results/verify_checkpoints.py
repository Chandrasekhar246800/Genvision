"""Non-destructive checkpoint verification for the GenVision ML artifacts."""

import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from attention_unet import AttentionUNet
from unet import UNet


MODEL_DIR = ROOT / "models"
INPUT_SHAPE = (1, 3, 256, 256)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_for_checkpoint(name: str):
    if "attention" in name:
        return "Attention U-Net", AttentionUNet
    return "U-Net", UNet


def verify(path: Path) -> dict:
    architecture, model_class = model_for_checkpoint(path.name)
    result = {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "architecture": architecture,
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "status": "invalid",
        "load_status": "not_attempted",
        "forward_status": "not_attempted",
        "input_shape": list(INPUT_SHAPE),
        "output_shape": None,
        "output_finite": None,
        "error": None,
    }

    if result["size_bytes"] == 0:
        result["load_status"] = "failed_empty_file"
        result["error"] = "Checkpoint file is empty."
        return result

    try:
        model = model_class()
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        result["load_status"] = "passed"

        with torch.no_grad():
            output = model(torch.rand(INPUT_SHAPE))

        result["forward_status"] = "passed"
        result["output_shape"] = list(output.shape)
        result["output_finite"] = bool(torch.isfinite(output).all().item())
        expected = list(INPUT_SHAPE)
        expected[1] = 3
        if list(output.shape) != expected:
            result["error"] = f"Expected output shape {expected}, got {list(output.shape)}."
        elif not result["output_finite"]:
            result["error"] = "Forward output contains NaN or infinite values."
        else:
            result["status"] = "verified"
    except Exception as error:  # noqa: BLE001 - report the exact verification failure
        result["load_status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"

    return result


def main() -> None:
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

    results = [verify(path) for path in sorted(MODEL_DIR.glob("*.pth"))]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
