# Sentinel-2 MSI Level-2A Band Reference

This reference applies to Sentinel-2 MSI Level-2A surface-reflectance products. The exact product metadata and processing baseline must be retained with every scene; filenames and scale factors can vary by distribution.

| Band | Common name | Approx. spectral range (nm) | Center (nm) | Native resolution | Agriculture use |
|---|---|---:|---:|---:|---|
| B02 | Blue | 458–523 | 490 | 10 m | Water/atmospheric sensitivity, RGB composites |
| B03 | Green | 543–578 | 560 | 10 m | Visible vegetation and green-NIR NDWI |
| B04 | Red | 650–680 | 665 | 10 m | Chlorophyll absorption and NDVI |
| B08 | NIR | 785–900 | 842 | 10 m | Vegetation structure and NDVI |
| B11 | SWIR 1 | 1565–1655 | 1610 | 20 m | Moisture-sensitive SWIR index such as NDMI |

These are nominal MSI band characteristics. Product metadata remains authoritative for a specific acquisition.

## Reflectance scaling

Level-2A image values are commonly delivered as quantized surface reflectance with a quantification value of 10,000, so a valid reflectance conversion is generally:

```text
reflectance = (digital_value + RADIO_ADD_OFFSET) / QUANTIFICATION_VALUE
```

The pipeline does not silently assume this for arbitrary files. `agriculture.preprocessing.scale_reflectance` requires an explicit scale factor and optional offset. Callers must obtain the values from the product metadata (for example, `QUANTIFICATION_VALUE` and any applicable band offset) and record them in processing metadata.

## Resolution and alignment

B02, B03, B04, and B08 are native 10 m bands. B11 is native 20 m. A 20 m B11 raster must not be treated as pixel-aligned with 10 m data. The loader preserves each source grid and `align_scene` performs an explicit resampling step to the selected target grid. The current helper defaults to bilinear resampling and records that choice through the caller's processing metadata; categorical SCL data must use nearest-neighbor resampling instead.

## Metadata to preserve

For reproducible analysis, retain:

- Product and tile identifiers.
- Acquisition date and processing baseline.
- Band name and native resolution.
- Quantification value and radiometric offsets.
- NoData value and masks.
- CRS, affine transform, bounds, width, and height.
- SCL/quality-mask provenance.

## Index formulas implemented

```text
NDVI = (B08 - B04) / (B08 + B04)
NDWI = (B03 - B08) / (B03 + B08)  # Green-NIR / McFeeters
NDMI = (B08 - B11) / (B08 + B11)
```

Division by zero and invalid/NoData pixels are masked. The pipeline does not confuse the Green-NIR NDWI with SWIR-based moisture indices.

Reference: [Copernicus Sentinel-2 mission](https://sentiwiki.copernicus.eu/web/s2-mission).

