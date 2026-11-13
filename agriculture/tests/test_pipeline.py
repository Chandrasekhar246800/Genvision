from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from agriculture.cloud_mask import apply_cloud_mask, scl_cloud_mask
from agriculture.data_loader import SceneValidationError, load_scene
from agriculture.preprocessing import align_scene
from agriculture.spectral_indices import ndmi, ndvi, ndwi, validate_index_range


class SpectralIndexTests(unittest.TestCase):
    def test_known_ndvi(self):
        np.testing.assert_allclose(ndvi(np.array([[0.8]]), np.array([[0.2]])).filled(), [[0.6]])

    def test_known_ndwi(self):
        np.testing.assert_allclose(ndwi(np.array([[0.3]]), np.array([[0.7]])).filled(), [[-0.4]])

    def test_known_ndmi(self):
        np.testing.assert_allclose(ndmi(np.array([[0.6]]), np.array([[0.2]])).filled(), [[0.5]], atol=1e-6)

    def test_zero_denominator_is_masked(self):
        self.assertTrue(ndvi(np.array([[0.0]]), np.array([[0.0]])).mask[0, 0])

    def test_nodata_propagates_and_range_is_validated(self):
        nir = np.ma.array([[0.8, 0.4]], mask=[[False, True]])
        result = ndvi(nir, np.array([[0.2, 0.2]]))
        self.assertTrue(result.mask[0, 1])
        validate_index_range(result)
        with self.assertRaises(ValueError):
            validate_index_range(np.ma.array([[2.0]]))


class RasterPipelineTests(unittest.TestCase):
    def _write_band(self, directory: Path, name: str, data: np.ndarray, *, resolution: int = 10, crs="EPSG:32631"):
        path = directory / f"T_SAMPLE_{name}_{resolution}m.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=data.shape[1],
            height=data.shape[0],
            count=1,
            dtype=data.dtype,
            crs=crs,
            transform=from_origin(500000, 4600000, resolution, resolution),
            nodata=0,
        ) as dst:
            dst.write(data, 1)

    def test_missing_band_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for name in ("B02", "B03", "B04", "B08"):
                self._write_band(directory, name, np.ones((2, 2), dtype="uint16"))
            with self.assertRaises(SceneValidationError):
                load_scene(directory)

    def test_misaligned_resolution_requires_explicit_alignment(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for name in ("B02", "B03", "B04", "B08"):
                self._write_band(directory, name, np.ones((2, 2), dtype="uint16"))
            self._write_band(directory, "B11", np.ones((1, 1), dtype="uint16"), resolution=20)
            scene = load_scene(directory)
            self.assertEqual(scene.bands["B11"].data.shape, (1, 1))
            aligned = align_scene(scene)
            self.assertEqual(aligned.bands["B11"].data.shape, (2, 2))

    def test_crs_preservation(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for name in ("B02", "B03", "B04", "B08", "B11"):
                self._write_band(directory, name, np.ones((2, 2), dtype="uint16"))
            scene = load_scene(directory)
            self.assertEqual(scene.crs, CRS.from_epsg(32631))
            self.assertEqual(align_scene(scene).crs, CRS.from_epsg(32631))

    def test_cloud_mask_classes_and_application(self):
        scl = np.array([[3, 4, 8, 9, 10, 5]], dtype="uint8")
        mask = scl_cloud_mask(scl)
        np.testing.assert_array_equal(mask, [[True, False, True, True, True, False]])
        masked = apply_cloud_mask(np.ones_like(scl, dtype="float32"), mask)
        self.assertEqual(masked.count(), 2)


if __name__ == "__main__":
    unittest.main()
