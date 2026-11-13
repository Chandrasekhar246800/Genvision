from pathlib import Path
import torch

from unet import UNet
from attention_unet import AttentionUNet


def check_checkpoint(path, model_class):
    print("\n" + "=" * 60)
    print(f"Checking: {path}")
    print("=" * 60)

    checkpoint_path = Path(path)

    if not checkpoint_path.exists():
        print("FAIL: File does not exist.")
        return

    if checkpoint_path.stat().st_size == 0:
        print("FAIL: Checkpoint is empty (0 bytes).")
        return

    try:
        model = model_class()

        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )

        model.load_state_dict(state_dict)
        model.eval()

        with torch.no_grad():
            dummy_input = torch.rand(1, 3, 256, 256)
            output = model(dummy_input)

        print("PASS: Checkpoint loaded successfully.")
        print(f"Input shape : {tuple(dummy_input.shape)}")
        print(f"Output shape: {tuple(output.shape)}")
        print(f"File size   : {checkpoint_path.stat().st_size:,} bytes")

    except Exception as error:
        print("FAIL: Checkpoint could not be loaded.")
        print(f"Error: {error}")


if __name__ == "__main__":
    check_checkpoint("models/unet.pth", UNet)

    check_checkpoint(
        "models/attention_unet.pth",
        AttentionUNet
    )

    check_checkpoint(
        "models/best_attention_unet.pth",
        AttentionUNet
    )

    check_checkpoint(
        "models/unet_baseline.pth",
        UNet
    )