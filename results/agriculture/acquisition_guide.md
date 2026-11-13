# Sentinel-2 Data Acquisition Guide

## Recommended product

Use Sentinel-2 MSI Level-2A surface reflectance for the first agricultural experiment. The required files are B02, B03, B04, B08, B11 and, when cloud masking is needed, SCL. Keep the complete product metadata alongside the extracted bands.

## Access options

### Copernicus Data Space Ecosystem

The Copernicus Data Space Ecosystem is the authoritative access route for Sentinel data. It supports browser and API access. Some API operations require a free account and OAuth2 authentication. Do not put credentials in this repository. Configure credentials through environment variables or the provider's documented login flow, and never commit tokens.

Reference: <https://dataspace.copernicus.eu/>.

### STAC catalogs

STAC is suitable for discovering scenes and assets by date, footprint, cloud cover, and product metadata. A STAC client should select an item, inspect its asset URLs and band metadata, then download only the required assets. Do not infer band identity from an RGB preview.

Useful public discovery services include the Microsoft Planetary Computer STAC API and Earth Search. Their access policies and asset signing requirements should be checked at implementation time.

### Cloud-optimized GeoTIFF sources

Cloud-optimized GeoTIFF assets can allow HTTP range reads of only required windows or bands. Confirm that the asset is surface reflectance, that its band metadata is present, and that the licensing and terms permit the intended use. A COG URL is not sufficient evidence of Sentinel-2 band identity by itself.

## Initial sample status

No Sentinel-2 scene was downloaded in Phase 5B. The environment did not contain a local multispectral scene, and no credentials were assumed or invented. The loader and validation tests use temporary tiny GeoTIFFs created only during testing; they are mathematical/metadata fixtures, not satellite data and are not retained as project data.

## Acquisition checklist

1. Select a Level-2A product and record product ID, tile ID, acquisition time, processing baseline, and license.
2. Obtain B02, B03, B04, B08, B11 and SCL from the same product/tile.
3. Verify that every asset is from the same acquisition and CRS/grid family.
4. Read `QUANTIFICATION_VALUE`, any band offsets, NoData, and SCL class definitions from product metadata.
5. Preserve source files or immutable URLs outside Git if they are large.
6. Validate dimensions, CRS, transforms, bounds, dtype, and finite reflectance after loading.
7. Explicitly resample 20 m B11 to the chosen analysis grid, documenting method and target resolution.
8. Apply SCL masking before calculating indices.
9. Save processing metadata with every output GeoTIFF.

## SCL masking assumptions

The isolated cloud-mask helper masks SCL classes:

- 0: No data
- 1: Saturated/defective
- 3: Cloud shadows
- 8: Cloud medium probability
- 9: Cloud high probability
- 10: Thin cirrus

Class 2 (dark features/shadows) and class 11 (snow/ice) are not masked by default. This is a conservative starting policy, not perfect cloud detection. Scene-specific quality review remains required.

