from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "agriculture" / "data"
RESULTS_ROOT = PROJECT_ROOT / "results" / "agriculture"

REQUIRED_BANDS = ("B02", "B03", "B04", "B08", "B11")
SCL_BAND = "SCL"
DEFAULT_SCALE_FACTOR = 10000.0
DEFAULT_MASKED_SCL_CLASSES = frozenset({0, 1, 3, 8, 9, 10})

BAND_REFERENCE = {
    "B02": {"name": "Blue", "center_nm": 490, "range_nm": "458-523", "resolution_m": 10},
    "B03": {"name": "Green", "center_nm": 560, "range_nm": "543-578", "resolution_m": 10},
    "B04": {"name": "Red", "center_nm": 665, "range_nm": "650-680", "resolution_m": 10},
    "B08": {"name": "Near infrared", "center_nm": 842, "range_nm": "785-900", "resolution_m": 10},
    "B11": {"name": "Short-wave infrared", "center_nm": 1610, "range_nm": "1565-1655", "resolution_m": 20},
}
