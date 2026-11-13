import json
from pathlib import Path

SPLIT_FILE = Path("results/data_split.json")


def main():
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"Split file not found: {SPLIT_FILE}"
        )

    with SPLIT_FILE.open("r", encoding="utf-8") as f:
        split = json.load(f)

    train = split["train"]
    validation = split["validation"]
    test = split["test"]
    filenames = split["filenames"]
    total = split["total_images"]

    train_set = set(train)
    val_set = set(validation)
    test_set = set(test)

    assert len(train_set) == len(train), "Duplicate training indices."
    assert len(val_set) == len(validation), "Duplicate validation indices."
    assert len(test_set) == len(test), "Duplicate test indices."

    assert train_set.isdisjoint(val_set), "Train/validation overlap."
    assert train_set.isdisjoint(test_set), "Train/test overlap."
    assert val_set.isdisjoint(test_set), "Validation/test overlap."

    assert len(train_set | val_set | test_set) == total
    assert len(filenames) == total

    all_indices = train_set | val_set | test_set

    assert all(
        isinstance(i, int) and 0 <= i < total
        for i in all_indices
    ), "Invalid dataset index."

    print("=" * 50)
    print("DATA SPLIT VERIFICATION")
    print("=" * 50)
    print(f"Training images   : {len(train)}")
    print(f"Validation images : {len(validation)}")
    print(f"Testing images    : {len(test)}")
    print(f"Total images      : {total}")
    print(f"Random seed       : {split['seed']}")
    print("Overlapping indices: None")
    print("Missing indices    : None")
    print("RESULT: PASS")
    print("=" * 50)


if __name__ == "__main__":
    main()