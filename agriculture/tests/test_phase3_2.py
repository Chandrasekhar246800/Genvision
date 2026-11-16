from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.windows import Window

from agriculture.cloud_mask import (
    apply_cloud_mask,
    scl_cloud_mask,
    scl_pure_cloud_mask,
    scl_shadow_mask,
)
from agriculture.config import ALL_SCENE_BANDS, DEFAULT_BOA_OFFSET, DEFAULT_SCALE_FACTOR
from agriculture.data_loader import SceneValidationError, load_band, load_scene
from agriculture.preprocessing import align_scene, scale_reflectance
from agriculture.spectral_indices import ndmi, ndvi, ndwi, validate_index_range


class Phase32Tests(unittest.TestCase):
    def _write_raster(
        self,
        directory: Path,
        name: str,
        data: np.ndarray,
        *,
        resolution: int = 10,
        crs: str = "EPSG:32643",
        dtype: str = "uint16",
    ) -> Path:
        path = directory / f"{name}.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=data.shape[1],
            height=data.shape[0],
            count=1,
            dtype=dtype,
            crs=crs,
            transform=from_origin(800000, 1700000, resolution, resolution),
            nodata=0,
        ) as dst:
            dst.write(data.astype(dtype), 1)
        return path

    def test_scl_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            td = Path(temp)
            for b in ("B02", "B03", "B04", "B08"):
                self._write_raster(td, b, np.ones((4, 4), dtype="uint16"), resolution=10)
            self._write_raster(td, "B11", np.ones((2, 2), dtype="uint16"), resolution=20)
            self._write_raster(td, "SCL", np.full((2, 2), 4, dtype="uint8"), resolution=20, dtype="uint8")

            # Default load_scene loads 5 bands
            scene_default = load_scene(td)
            self.assertEqual(len(scene_default.bands), 5)
            self.assertNotIn("SCL", scene_default.bands)

            # Explicit required_bands with SCL loads 6 bands
            scene_with_scl = load_scene(td, required_bands=ALL_SCENE_BANDS)
            self.assertEqual(len(scene_with_scl.bands), 6)
            self.assertIn("SCL", scene_with_scl.bands)
            self.assertEqual(scene_with_scl.bands["SCL"].data.dtype, np.uint8)

    def test_categorical_nearest_neighbour_resampling(self):
        with tempfile.TemporaryDirectory() as temp:
            td = Path(temp)
            for b in ("B02", "B03", "B04", "B08"):
                self._write_raster(td, b, np.ones((4, 4), dtype="uint16"), resolution=10)
            self._write_raster(td, "B11", np.array([[2000, 4000], [3000, 5000]], dtype="uint16"), resolution=20)
            # SCL with distinct classes: 4 (vegetation) and 9 (cloud)
            self._write_raster(td, "SCL", np.array([[4, 9], [4, 9]], dtype="uint8"), resolution=20, dtype="uint8")

            scene = load_scene(td, required_bands=ALL_SCENE_BANDS)
            aligned = align_scene(scene)

            # Check that SCL is resampled to (4, 4)
            scl_data = aligned.bands["SCL"].data
            self.assertEqual(scl_data.shape, (4, 4))
            # Must remain uint8
            self.assertEqual(scl_data.dtype, np.uint8)
            # Nearest neighbor must ONLY contain original classes {4, 9}, never intermediate values
            unique_classes = set(np.unique(scl_data))
            self.assertEqual(unique_classes, {4, 9})

            # Check B11 continuous interpolation
            b11_data = aligned.bands["B11"].data
            self.assertEqual(b11_data.shape, (4, 4))
            self.assertEqual(b11_data.dtype, np.float32)

    def test_reflectance_offset_application(self):
        # Raw DN array
        dn_data = np.array([500, 1000, 2000, 11000], dtype=np.uint16)
        scaled, meta = scale_reflectance(
            dn_data,
            scale_factor=DEFAULT_SCALE_FACTOR,
            offset=DEFAULT_BOA_OFFSET,
            clamp_min_zero=True,
        )
        self.assertEqual(meta["scale_factor"], 10000.0)
        self.assertEqual(meta["offset"], -1000.0)
        self.assertTrue(meta["clamped_min_zero"])

        # DN 1000 -> 0.00
        self.assertAlmostEqual(scaled[1], 0.0, places=5)
        # DN 2000 -> 0.10
        self.assertAlmostEqual(scaled[2], 0.10, places=5)
        # DN 11000 -> 1.00
        self.assertAlmostEqual(scaled[3], 1.00, places=5)
        # DN 500 -> (500 - 1000) = -500 -> clamped to 0.0
        self.assertAlmostEqual(scaled[0], 0.0, places=5)

    def test_roi_extraction_window(self):
        with tempfile.TemporaryDirectory() as temp:
            td = Path(temp)
            # Create a 20x20 10m raster and 10x10 20m raster
            for b in ("B02", "B03", "B04", "B08"):
                self._write_raster(td, b, np.arange(400, dtype="uint16").reshape(20, 20), resolution=10)
            self._write_raster(td, "B11", np.arange(100, dtype="uint16").reshape(10, 10), resolution=20)
            self._write_raster(td, "SCL", np.full((10, 10), 4, dtype="uint8"), resolution=20, dtype="uint8")

            # 10m window: col_off=4, row_off=6, width=8, height=8
            win_10m = Window(4, 6, 8, 8)
            scene = load_scene(td, required_bands=ALL_SCENE_BANDS, window_10m=win_10m)

            # 10m bands should be 8x8
            self.assertEqual(scene.bands["B04"].data.shape, (8, 8))
            self.assertEqual(scene.bands["B08"].data.shape, (8, 8))
            # 20m bands should be 4x4
            self.assertEqual(scene.bands["B11"].data.shape, (4, 4))
            self.assertEqual(scene.bands["SCL"].data.shape, (4, 4))

    def test_cloud_masking_distinctions(self):
        scl = np.array([0, 1, 3, 4, 5, 6, 8, 9, 10], dtype="uint8")
        pure_cloud = scl_pure_cloud_mask(scl)
        # Only 8, 9, 10
        np.testing.assert_array_equal(pure_cloud, [False, False, False, False, False, False, True, True, True])

        shadow = scl_shadow_mask(scl)
        # Only 3
        np.testing.assert_array_equal(shadow, [False, False, True, False, False, False, False, False, False])

        combined = scl_cloud_mask(scl)
        # 0, 1, 3, 8, 9, 10
        np.testing.assert_array_equal(combined, [True, True, True, False, False, False, True, True, True])

        # Test apply_cloud_mask
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype="float32")
        masked = apply_cloud_mask(data, combined)
        self.assertEqual(masked.count(), 3)  # indices 3, 4, 5 (values 4.0, 5.0, 6.0)

    def test_index_calculations_after_alignment(self):
        nir = np.array([[0.4, 0.5], [0.6, 0.7]], dtype="float32")
        red = np.array([[0.1, 0.1], [0.2, 0.2]], dtype="float32")
        green = np.array([[0.1, 0.1], [0.15, 0.15]], dtype="float32")
        swir = np.array([[0.2, 0.25], [0.3, 0.35]], dtype="float32")

        c_mask = np.array([[False, True], [False, False]])

        v_ndvi = ndvi(nir, red, mask=c_mask)
        v_ndwi = ndwi(green, nir, mask=c_mask)
        v_ndmi = ndmi(nir, swir, mask=c_mask)

        validate_index_range(v_ndvi)
        validate_index_range(v_ndwi)
        validate_index_range(v_ndmi)

        # Check mask propagation
        self.assertTrue(v_ndvi.mask[0, 1])
        self.assertTrue(v_ndwi.mask[0, 1])
        self.assertTrue(v_ndmi.mask[0, 1])

        # Check numerical values on [0, 0]
        # NDVI: (0.4 - 0.1) / (0.4 + 0.1) = 0.3 / 0.5 = 0.6
        self.assertAlmostEqual(v_ndvi[0, 0], 0.6, places=4)
        # NDWI: (0.1 - 0.4) / (0.1 + 0.4) = -0.3 / 0.5 = -0.6
        self.assertAlmostEqual(v_ndwi[0, 0], -0.6, places=4)
        # NDMI: (0.4 - 0.2) / (0.4 + 0.2) = 0.2 / 0.6 = 0.3333
        self.assertAlmostEqual(v_ndmi[0, 0], 1.0 / 3.0, places=4)


if __name__ == "__main__":
    unittest.main()
