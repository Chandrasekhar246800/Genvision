from __future__ import annotations

import os
from pathlib import Path

# Ensure Matplotlib temp dir is set cleanly in Windows sandbox
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(Path(os.environ.get("TEMP", ".")) / "matplotlib_genvision")

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from .config import SCL_COLORMAP


def write_index_geotiff(
    index: np.ma.MaskedArray | np.ndarray,
    reference_profile: dict,
    output_path: str | Path,
    nodata: float = -9999.0,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = reference_profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=nodata,
        compress="deflate",
    )
    if isinstance(index, np.ma.MaskedArray):
        output = np.ma.filled(index, nodata).astype("float32")
    else:
        output = index.astype("float32")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(output, 1)


def write_mask_geotiff(
    mask: np.ndarray,
    reference_profile: dict,
    output_path: str | Path,
    dtype: str = "uint8",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = reference_profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype=dtype,
        nodata=None,
        compress="deflate",
    )
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mask.astype(dtype), 1)


def stretch_channel(arr: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    """Apply linear percentile contrast stretch to normalize an image channel to [0, 1]."""
    valid = arr[np.isfinite(arr) & (arr > 0)]
    if valid.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    vmin = float(np.percentile(valid, p_low))
    vmax = float(np.percentile(valid, p_high))
    if vmax <= vmin:
        vmax = vmin + 1e-4
    stretched = np.clip((arr.astype(np.float32) - vmin) / (vmax - vmin), 0.0, 1.0)
    return stretched


def save_rgb_composite(
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    output_path: str | Path,
    *,
    title: str,
    subtitle: str = "Pulivendula ROI (10 km × 10 km) · Sentinel-2C Level-2A",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    r_s = stretch_channel(r)
    g_s = stretch_channel(g)
    b_s = stretch_channel(b)
    rgb = np.stack([r_s, g_s, b_s], axis=-1)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=200, facecolor="#080d18")
    ax.imshow(rgb, extent=[0, 10, 0, 10])
    ax.set_title(title, color="#f1f5f9", fontsize=14, pad=12, weight="bold")
    ax.set_xlabel("Easting Distance (km) — UTM Zone 43N (EPSG:32643)", color="#94a3b8", fontsize=10, labelpad=8)
    ax.set_ylabel("Northing Distance (km)", color="#94a3b8", fontsize=10, labelpad=8)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#334155")

    fig.text(0.5, 0.02, subtitle, ha="center", color="#64748b", fontsize=9)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_index_visualization(
    index: np.ma.MaskedArray,
    output_path: str | Path,
    *,
    title: str,
    cmap: str = "RdYlGn",
    vmin: float = -0.2,
    vmax: float = 0.8,
    label: str = "Index Value",
    mask_color: str = "#152235",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.5, 9), dpi=200, facecolor="#080d18")
    current_cmap = matplotlib.colormaps[cmap].copy()
    current_cmap.set_bad(color=mask_color)

    im = ax.imshow(index, cmap=current_cmap, vmin=vmin, vmax=vmax, extent=[0, 10, 0, 10])
    ax.set_title(title, color="#f1f5f9", fontsize=14, pad=12, weight="bold")
    ax.set_xlabel("Easting (km) — UTM Zone 43N", color="#94a3b8", fontsize=10, labelpad=8)
    ax.set_ylabel("Northing (km)", color="#94a3b8", fontsize=10, labelpad=8)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#334155")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label, color="#94a3b8", fontsize=10, labelpad=8)
    cbar.ax.yaxis.set_tick_params(color="#94a3b8")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#94a3b8", size=9)
    cbar.outline.set_edgecolor("#334155")

    fig.text(0.5, 0.02, "Pulivendula ROI (10 km × 10 km) · Cloud/Shadow Obstructed Pixels Masked in Dark Navy", ha="center", color="#64748b", fontsize=9)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_scl_visualization(
    scl: np.ndarray,
    output_path: str | Path,
    *,
    class_counts: dict[int, int],
    title: str = "Sentinel-2 Scene Classification Layer (SCL) — Pulivendula ROI",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 9), dpi=200, facecolor="#080d18")

    # Build custom listed colormap for present classes
    unique_vals = sorted(np.unique(scl))
    colors = [SCL_COLORMAP.get(val, ("#888888", f"Class {val}"))[0] for val in unique_vals]
    cmap = mcolors.ListedColormap(colors)
    bounds = np.array(unique_vals + [unique_vals[-1] + 1]) - 0.5
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    ax.imshow(scl, cmap=cmap, norm=norm, extent=[0, 10, 0, 10], interpolation="nearest")
    ax.set_title(title, color="#f1f5f9", fontsize=14, pad=12, weight="bold")
    ax.set_xlabel("Easting (km) — UTM Zone 43N", color="#94a3b8", fontsize=10, labelpad=8)
    ax.set_ylabel("Northing (km)", color="#94a3b8", fontsize=10, labelpad=8)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#334155")

    # Build legend
    total_pix = scl.size
    patches = []
    for val in unique_vals:
        color, name = SCL_COLORMAP.get(val, ("#888888", f"Class {val}"))
        cnt = class_counts.get(val, 0)
        pct = (cnt / total_pix) * 100
        patches.append(mpatches.Patch(color=color, label=f"{name} ({pct:.1f}%)"))

    leg = ax.legend(
        handles=patches,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        facecolor="#111b2b",
        edgecolor="#334155",
        fontsize=8.5,
        labelcolor="#e2edf7",
        title="SCL Classes",
        title_fontsize=9.5,
    )
    leg.get_title().set_color("#55d6f2")

    fig.text(0.5, 0.02, "Pulivendula ROI (10 km × 10 km) · Nearest-Neighbor 10m Resampling", ha="center", color="#64748b", fontsize=9)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_obstruction_map(
    cloud_mask: np.ndarray,
    shadow_mask: np.ndarray,
    output_path: str | Path,
    *,
    title: str = "Cloud & Cloud-Shadow Obstruction Map — Pulivendula ROI",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 0 = Clear, 1 = Cloud Shadow, 2 = Cloud
    obs = np.zeros(cloud_mask.shape, dtype=np.uint8)
    obs[shadow_mask] = 1
    obs[cloud_mask] = 2

    cmap = mcolors.ListedColormap(["#1e3a29", "#d97706", "#f8fafc"])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(10.5, 9), dpi=200, facecolor="#080d18")
    ax.imshow(obs, cmap=cmap, norm=norm, extent=[0, 10, 0, 10], interpolation="nearest")
    ax.set_title(title, color="#f1f5f9", fontsize=14, pad=12, weight="bold")
    ax.set_xlabel("Easting (km) — UTM Zone 43N", color="#94a3b8", fontsize=10, labelpad=8)
    ax.set_ylabel("Northing (km)", color="#94a3b8", fontsize=10, labelpad=8)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#334155")

    total = obs.size
    clear_pct = (np.sum(obs == 0) / total) * 100
    shadow_pct = (np.sum(obs == 1) / total) * 100
    cloud_pct = (np.sum(obs == 2) / total) * 100

    patches = [
        mpatches.Patch(color="#1e3a29", label=f"Clear Surface ({clear_pct:.1f}%)"),
        mpatches.Patch(color="#d97706", label=f"Cloud Shadow ({shadow_pct:.1f}%)"),
        mpatches.Patch(color="#f8fafc", label=f"Cloud (Med/High/Cirrus) ({cloud_pct:.1f}%)"),
    ]
    leg = ax.legend(
        handles=patches,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        facecolor="#111b2b",
        edgecolor="#334155",
        fontsize=9,
        labelcolor="#e2edf7",
        title="Obstruction Status",
        title_fontsize=10,
    )
    leg.get_title().set_color("#55d6f2")

    fig.text(0.5, 0.02, f"Total Obstructed Area: {cloud_pct + shadow_pct:.1f}% · Clear Ground: {clear_pct:.1f}%", ha="center", color="#64748b", fontsize=9)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_cloud_free_agricultural_roi(
    rgb_r: np.ndarray,
    rgb_g: np.ndarray,
    rgb_b: np.ndarray,
    ndvi_arr: np.ma.MaskedArray,
    output_path: str | Path,
    *,
    title: str = "Cloud-Free Agricultural Vegetation Canopy Analysis",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Base background: dim true color
    r_s = stretch_channel(rgb_r) * 0.4
    g_s = stretch_channel(rgb_g) * 0.4
    b_s = stretch_channel(rgb_b) * 0.4
    base_rgb = np.stack([r_s, g_s, b_s], axis=-1)

    fig, ax = plt.subplots(figsize=(10.5, 9), dpi=200, facecolor="#080d18")
    ax.imshow(base_rgb, extent=[0, 10, 0, 10])

    # Overlay NDVI on clear vegetation pixels only
    veg_cmap = matplotlib.colormaps["YlGn"].copy()
    veg_cmap.set_bad(alpha=0.0)

    # Filter to NDVI >= 0.2
    active_veg = np.ma.masked_where(ndvi_arr.mask | (ndvi_arr < 0.2), ndvi_arr)
    im = ax.imshow(active_veg, cmap=veg_cmap, vmin=0.2, vmax=0.8, extent=[0, 10, 0, 10], alpha=0.92)

    ax.set_title(title, color="#f1f5f9", fontsize=14, pad=12, weight="bold")
    ax.set_xlabel("Easting (km) — UTM Zone 43N", color="#94a3b8", fontsize=10, labelpad=8)
    ax.set_ylabel("Northing (km)", color="#94a3b8", fontsize=10, labelpad=8)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#334155")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Vegetation Vigor (NDVI ≥ 0.2)", color="#94a3b8", fontsize=10, labelpad=8)
    cbar.ax.yaxis.set_tick_params(color="#94a3b8")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#94a3b8", size=9)
    cbar.outline.set_edgecolor("#334155")

    fig.text(0.5, 0.02, "Active Crop Canopy Highlighted Over True-Color Base · Obstructed Clouds Excluded", ha="center", color="#64748b", fontsize=9)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
