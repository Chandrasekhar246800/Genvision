import json
import random
from pathlib import Path

SEED = 42

CLOUDY_DIR = Path("dataset/cloudy")
CLEAN_DIR = Path("dataset/clean")
OUTPUT_FILE = Path("results/data_split.json")

SUPPORTED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"
}


def get_images(directory):
    return sorted(
        p.name
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def main():
    if not CLOUDY_DIR.is_dir() or not CLEAN_DIR.is_dir():
        raise FileNotFoundError("Cloudy or clean dataset directory is missing.")

    cloudy_files = get_images(CLOUDY_DIR)
    clean_files = get_images(CLEAN_DIR)

    if cloudy_files != clean_files:
        raise ValueError(
            "Cloudy and clean filenames do not match exactly. "
            "Check the dataset before creating the split."
        )

    files = cloudy_files
    total = len(files)

    if total != 500:
        print(f"WARNING: Expected 500 pairs but found {total}.")

    if total < 3:
        raise ValueError("At least three image pairs are required.")

    indices = list(range(total))
    random.Random(SEED).shuffle(indices)

    train_end = int(0.70 * total)
    val_end = int(0.85 * total)

    split = {
        "seed": SEED,
        "total_images": total,
        "train": indices[:train_end],
        "validation": indices[train_end:val_end],
        "test": indices[val_end:],
        "filenames": files
    }

    # Verify that every image belongs to exactly one subset.
    train_set = set(split["train"])
    val_set = set(split["validation"])
    test_set = set(split["test"])

    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)

    assert len(train_set | val_set | test_set) == total

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)

    print("=" * 50)
    print("DATASET SPLIT CREATED")
    print("=" * 50)
    print(f"Total images : {total}")
    print(f"Training     : {len(split['train'])}")
    print(f"Validation   : {len(split['validation'])}")
    print(f"Testing      : {len(split['test'])}")
    print(f"Random seed  : {SEED}")
    print(f"Saved to     : {OUTPUT_FILE}")
    print("=" * 50)


if __name__ == "__main__":
    main()