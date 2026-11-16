from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "agriculture" / "data"
RESULTS_ROOT = PROJECT_ROOT / "results" / "agriculture"

# Default band tuples
REQUIRED_BANDS = ("B02", "B03", "B04", "B08", "B11")
ALL_SCENE_BANDS = ("B02", "B03", "B04", "B08", "B11", "SCL")
SCL_BAND = "SCL"

# Radiometric scaling constants (Sentinel-2 Level-2A Processing Baseline 04.00+)
DEFAULT_SCALE_FACTOR = 10000.0
DEFAULT_BOA_OFFSET = -1000.0

# Scene Classification Layer (SCL) class definitions
DEFAULT_MASKED_SCL_CLASSES = frozenset({0, 1, 3, 8, 9, 10})
CLOUD_SCL_CLASSES = frozenset({8, 9, 10})
SHADOW_SCL_CLASSES = frozenset({3})
INVALID_SCL_CLASSES = frozenset({0, 1})
CLEAR_LAND_CLASSES = frozenset({4, 5})
WATER_SCL_CLASSES = frozenset({6})

SCL_COLORMAP = {
    0: ("#000000", "No Data"),
    1: ("#ff0000", "Saturated / Defective"),
    2: ("#2f2f2f", "Dark Area"),
    3: ("#663300", "Cloud Shadow"),
    4: ("#009900", "Vegetation"),
    5: ("#ffff00", "Bare Soil / Urban"),
    6: ("#0000ff", "Water"),
    7: ("#808080", "Unclassified"),
    8: ("#b3b3b3", "Cloud Medium Probability"),
    9: ("#ffffff", "Cloud High Probability"),
    10: ("#00ffff", "Thin Cirrus"),
    11: ("#ffccff", "Snow / Ice"),
}

# Study Area Definition (Pulivendula, Andhra Pradesh, India)
PULIVENDULA_COORDINATES = {"lat": 14.42, "lon": 78.23}
ROI_SIZE_METERS = 10000.0  # 10 km x 10 km

BAND_REFERENCE = {
    "B02": {"name": "Blue", "center_nm": 490, "range_nm": "458-523", "resolution_m": 10},
    "B03": {"name": "Green", "center_nm": 560, "range_nm": "543-578", "resolution_m": 10},
    "B04": {"name": "Red", "center_nm": 665, "range_nm": "650-680", "resolution_m": 10},
    "B08": {"name": "Near infrared", "center_nm": 842, "range_nm": "785-900", "resolution_m": 10},
    "B11": {"name": "Short-wave infrared", "center_nm": 1610, "range_nm": "1565-1655", "resolution_m": 20},
    "SCL": {"name": "Scene Classification Layer", "center_nm": None, "range_nm": None, "resolution_m": 20},
}


def resolve_dataset_dir(preferred_path: str | Path | None = None) -> Path:
    """Resolve the genuine Sentinel-2 dataset directory, supporting sentinal2/sentinel2 variations."""
    if preferred_path:
        p = Path(preferred_path)
        if p.is_dir():
            return p

    candidates = [
        PROJECT_ROOT / "dataset" / "sentinal2" / "pulivendula_scene_01",
        PROJECT_ROOT / "dataset" / "sentinel2" / "pulivendula_scene_01",
        PROJECT_ROOT / "datasets" / "sentinal2" / "pulivendula_scene_01",
        PROJECT_ROOT / "datasets" / "sentinel2" / "pulivendula_scene_01",
        DATA_ROOT,
    ]
    for c in candidates:
        if c.is_dir() and any(c.glob("*.jp2")) or any(c.glob("*.tif")):
            return c
    return candidates[0]
