# Dataset Suitability Report for Genuine Agricultural Spectral Analysis

**Project:** GenVision Cloud Removal in Satellite Imagery  
**Inspection date:** 2026-11-01  
**Scope:** Read-only inspection of local datasets, image assets, preprocessing code, model definitions, checkpoints, and available documentation.

## Executive conclusion

The local cloud-removal dataset is an RGB image dataset, not a multispectral dataset. It contains 500 cloudy/clean pairs (1,000 PNG files total), with each file stored as a 512 x 512, 8-bit RGB PNG. The files contain red, green, and blue image channels, but no near-infrared (NIR), short-wave infrared (SWIR), wavelength metadata, reflectance calibration, or band descriptions.

Therefore:

- Genuine NDVI is **not currently possible** because NDVI requires calibrated NIR and red reflectance.
- Standard NDWI is **not currently possible**. The common McFeeters formulation requires green and NIR; the Gao formulation requires NIR and SWIR. The local data has neither NIR nor SWIR.
- The RGB files must not be relabelled or reused as NIR/SWIR bands.
- The existing U-Net checkpoint cannot process multispectral input without an architecture and checkpoint change. Its first convolution accepts 3 channels and its output layer produces 3 channels.

The current assets can support RGB cloud-removal research and ordinary RGB visualization, but not scientifically valid vegetation or water indices.

## Inspected local assets

### Dataset directories

| Asset | Confirmed finding |
|---|---|
| `dataset/cloudy/` | 500 PNG files |
| `dataset/clean/` | 500 PNG files |
| Pairing | Existing `dataset.py` validates matching filenames |
| Total image files | 1,000 |
| On-disk dimensions | 512 x 512 for all dataset files |
| File format | PNG |
| Channel mode | RGB for all 1,000 dataset files |
| Channel names exposed by Pillow | `R`, `G`, `B` |
| Per-pixel storage | 8-bit integer RGB |
| Dataset preprocessing | `convert("RGB")`, resize to 256 x 256, `ToTensor()` to [0, 1] |

`results/data_split.json` confirms a 500-image split with seed 42: 350 training indices, 75 validation indices, and 75 test indices.

### Other local image assets

The repository-wide image scan found 1,042 image-like assets outside dependency and virtual-environment directories:

- 1,040 readable PNG files.
- 1,030 RGB images.
- 10 RGBA images, all presentation, plot, logo, or UI-related assets rather than spectral data.
- 2 unreadable legacy result files: `results/baseline_prediction.png` and `results/loss_curve.png`.
- No TIFF, JPEG, JP2, BMP, WebP, GeoTIFF, NetCDF, or other multispectral raster was found.
- Readable dataset and prediction images contained no EXIF, wavelength, CRS, geotransform, sensor, or band metadata. A small number of generated plots contained only ordinary `Software` and `dpi` PNG metadata.

The RGBA assets represent transparency, not a fourth measured spectral band.

## Available and missing spectral information

### Confirmed available

- **Red:** RGB `R` channel is present.
- **Green:** RGB `G` channel is present.
- **Blue:** RGB `B` channel is present.
- **NIR:** Not present.
- **SWIR:** Not present.
- **Multispectral metadata:** Not present.
- **Reflectance calibration:** Not present.
- **Geospatial metadata:** Not present in the inspected image files.

The RGB channels are confirmed image channels, not confirmed physically calibrated reflectance bands. Their existence alone does not establish a sensor bandpass, wavelength response, or radiometric unit.

### Confirmed missing

- A measured NIR band, typically needed around the vegetation-sensitive near-infrared wavelengths.
- A measured SWIR band, needed by SWIR-based water-index variants.
- Band wavelength definitions and band ordering documentation.
- Sensor, acquisition date, solar geometry, and radiometric calibration metadata.
- CRS, affine transform, and geospatial footprint metadata.

## NDVI suitability

NDVI is conventionally calculated as:

```text
NDVI = (NIR - Red) / (NIR + Red)
```

The local files have an RGB red channel but no NIR measurement or wavelength metadata. Consequently, a genuine NDVI cannot be calculated from this dataset. Substituting blue, green, or a transformed RGB value for NIR would be an unsupported proxy and is explicitly not performed.

**Status: NOT POSSIBLE with the inspected local data.**

## NDWI suitability

There are multiple established NDWI definitions. The McFeeters form uses green and NIR, while the Gao form uses NIR and SWIR. The local RGB data has green but lacks NIR and SWIR.

Consequently, a genuine standard NDWI cannot be calculated. A green-blue or other RGB-only ratio would be a different RGB index, not NDWI, and is not implemented.

**Status: NOT POSSIBLE with the inspected local data.**

## Existing cloud-removal U-Net compatibility

The inspected `unet.py` defines:

- First convolution: `nn.Conv2d(3, 64, ...)`
- Final convolution: `nn.Conv2d(64, 3, ...)`
- Expected evaluation input used by the project: `(1, 3, 256, 256)`

The verified `models/unet_controlled_best.pth` checkpoint has matching first-layer weights shaped `(64, 3, 3, 3)` and final-layer weights shaped `(3, 64, 1, 1)`.

This means the existing model can process three-channel RGB tensors only. A 4-band RGB+NIR or 13-band Sentinel-2 tensor would fail at the first convolution. Supporting multispectral imagery would require, at minimum, a changed input interface, a changed output/interface decision, a new compatible checkpoint, and a documented preprocessing strategy. Those changes were not made.

The backend also explicitly converts inputs to RGB and emits RGB PNG output, so it is not a multispectral inference path.

## Recommended future datasets

These are recommendations for a future, separate multispectral workstream; none were downloaded or added.

1. **Sentinel-2 MSI Level-2A surface reflectance**
   - Recommended primary option for agriculture.
   - Provides 13 multispectral bands, including visible, red-edge, NIR, and SWIR bands, at multiple spatial resolutions.
   - Supports genuine NDVI and multiple NDWI variants when bands are correctly selected, resampled, and cloud-masked.
   - Official mission documentation: [Copernicus Sentinel-2 mission](https://sentiwiki.copernicus.eu/web/s2-mission).
   - A future dataset should preserve band names, wavelengths, scale factors, acquisition dates, cloud masks, CRS, and geotransforms.

2. **Landsat 8/9 Collection 2 Level-2 surface reflectance**
   - Suitable for longer historical agricultural time series.
   - Includes visible, NIR, and SWIR observations with established surface-reflectance products and quality masks.
   - Use official USGS documentation and product metadata when selecting bands and scale factors: [USGS Landsat missions](https://www.usgs.gov/landsat-missions).

3. **USDA NAIP four-band imagery**
   - Useful for high-resolution agricultural mapping where coverage and acquisition dates meet the project needs.
   - Available four-band products may include RGB plus NIR; coverage and product specifications must be checked for each acquisition.
   - Official portal: [USDA NAIP Hub](https://naip-usdaonline.hub.arcgis.com/pages/naip).

For cloud-removal training specifically, the preferred future source should contain temporally or spatially matched cloudy and cloud-free multispectral observations, a reliable cloud/quality mask, and consistent radiometric processing. A generic multispectral classification dataset is not automatically suitable as a paired cloud-removal dataset.

## Limitations and confidence

- **Confirmed by file inspection:** file counts, PNG formats, dimensions, RGB/RGBA modes, channel names, metadata keys, and absence of local multispectral raster formats.
- **Confirmed by source inspection:** RGB conversion and 3-channel preprocessing in `dataset.py` and the backend; 3-channel U-Net input/output in `unet.py`; matching checkpoint tensor shapes.
- **Not inferred:** no claim is made that the RGB colors correspond to calibrated sensor reflectance or a particular satellite sensor.
- The two unreadable result files were not treated as spectral evidence and were not modified.
- Absence of metadata means sensor wavelengths, radiance/reflectance units, and geographic context cannot be recovered from these files.

## Recommended next implementation steps

1. Select an official multispectral source, preferably Sentinel-2 Level-2A for the first agricultural experiment.
2. Define a band schema and retain original band names, wavelengths, resolutions, scale factors, QA/cloud masks, CRS, and transforms.
3. Build a read-only ingestion and validation step that verifies band count, dtype, nodata handling, alignment, and reflectance ranges.
4. Create an explicitly multispectral dataset split without mixing it with the current RGB split.
5. Decide whether the research objective is multispectral cloud removal, index reconstruction, or both.
6. Design and train a new multispectral-compatible model/checkpoint; do not reuse the current 3-channel checkpoint as if it supported NIR.
7. Validate NDVI/NDWI calculations against known band formulas and quality masks before exposing them in the application.
8. Keep the current RGB U-Net and its results unchanged as a separate baseline.

