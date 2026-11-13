import torch

from dataset import CloudDataset


def main():

    dataset = CloudDataset(
        "dataset/cloudy",
        "dataset/clean"
    )

    print("\nTesting dataset integrity...")

    assert len(dataset) > 0, (
        "Dataset is empty."
    )

    cloudy, clean = dataset[0]

    print("\nFirst image pair:")
    print(f"Cloudy shape: {cloudy.shape}")
    print(f"Clean shape : {clean.shape}")

    print(f"Cloudy dtype: {cloudy.dtype}")
    print(f"Clean dtype : {clean.dtype}")

    print(
        f"Cloudy range: "
        f"{cloudy.min().item():.4f} - "
        f"{cloudy.max().item():.4f}"
    )

    print(
        f"Clean range : "
        f"{clean.min().item():.4f} - "
        f"{clean.max().item():.4f}"
    )

    assert cloudy.shape == (3, 256, 256)
    assert clean.shape == (3, 256, 256)

    assert cloudy.dtype == torch.float32
    assert clean.dtype == torch.float32

    assert torch.isfinite(cloudy).all()
    assert torch.isfinite(clean).all()

    assert cloudy.min() >= 0
    assert cloudy.max() <= 1

    assert clean.min() >= 0
    assert clean.max() <= 1

    print("\nPASS: Dataset integrity checks succeeded.")


if __name__ == "__main__":
    main()