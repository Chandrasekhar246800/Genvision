"use client";

import { useEffect, useRef, useState } from "react";
import {
  Download,
  Expand,
  ExternalLink,
  Eye,
  Info,
  Layers,
  Minimize2,
  RefreshCw,
  RotateCcw,
  Sparkles,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { getAgricultureVisualizationUrl } from "../../lib/api";
import type { VisualizationMetadata } from "../../types";
import { Card } from "../app-shell";

type TabDefinition = {
  id: string;
  label: string;
  shortLabel: string;
  category: "composite" | "index" | "classification" | "obstruction";
  description: string;
  scientificContext: string;
  bandsUsed: string;
};

const VISUALIZATION_TABS: TabDefinition[] = [
  {
    id: "true_color",
    label: "True-Color RGB",
    shortLabel: "True Color",
    category: "composite",
    description: "Natural surface reflectance composite rendered from Red (B04), Green (B03), and Blue (B02) bands with 2% contrast stretch.",
    scientificContext: "Represents visual human perception of the agricultural landscape. Clouds appear bright white with high visible reflectance; shadows appear dark gray; vegetated fields exhibit subtle natural green hues.",
    bandsUsed: "B04 (Red 665nm), B03 (Green 560nm), B02 (Blue 490nm) · 10m GSD",
  },
  {
    id: "false_color",
    label: "False-Color NIR",
    shortLabel: "False Color",
    category: "composite",
    description: "Near-Infrared false-color composite (B08, B04, B03) delineating cellular chlorophyll density.",
    scientificContext: "Photosynthesizing crops reflect strongly in NIR (B08) and absorb Red (B04), rendering healthy vegetation in vivid crimson red. Bare soils and fallow fields appear cyan/slate; water absorbs NIR completely.",
    bandsUsed: "B08 (NIR 842nm), B04 (Red 665nm), B03 (Green 560nm) · 10m GSD",
  },
  {
    id: "ndvi",
    label: "NDVI Vegetation Index",
    shortLabel: "NDVI",
    category: "index",
    description: "Normalized Difference Vegetation Index (B08 - B04)/(B08 + B04) displaying crop canopy vigor.",
    scientificContext: "Calculated with strict cloud and shadow masking. RdYlGn color ramp: Deep green indicates dense, active crops (NDVI > 0.6); yellow/orange signifies moderate/stressed biomass; navy mask covers clouds and shadows.",
    bandsUsed: "B08 (NIR) & B04 (Red) · Clear-Sky Mean: 0.3780 (Max: 0.8657)",
  },
  {
    id: "ndwi",
    label: "NDWI Water Index",
    shortLabel: "NDWI",
    category: "index",
    description: "Normalized Difference Water Index (B03 - B08)/(B03 + B08 - McFeeters) delineating surface water and soil wetness.",
    scientificContext: "Water bodies have positive values (NDWI > 0.0); dry soils and agricultural canopies exhibit strongly negative values. Mean clear-sky value is -0.3935, reflecting dry seasonal conditions in Pulivendula.",
    bandsUsed: "B03 (Green) & B08 (NIR) · Clear-Sky Mean: -0.3935",
  },
  {
    id: "ndmi",
    label: "NDMI Moisture Index",
    shortLabel: "NDMI",
    category: "index",
    description: "Normalized Difference Moisture Index (B08 - B11)/(B08 + B11 - Gao) assessing canopy internal water content.",
    scientificContext: "Contrasts NIR chlorophyll scattering against SWIR liquid water absorption. Healthy hydrated canopies exhibit positive NDMI; dry fallow fields drop below 0. Clear-sky mean is +0.0042.",
    bandsUsed: "B08 (NIR 10m) & B11 (SWIR-1 20m aligned) · Clear-Sky Mean: +0.0042",
  },
  {
    id: "scl_classification",
    label: "SCL Classification",
    shortLabel: "SCL Map",
    category: "classification",
    description: "Sentinel-2 Level-2A Scene Classification Layer (SCL) pixel classification map.",
    scientificContext: "Discrete 12-class categorical classification from Sen2Cor. In this scene: Vegetation (15.96%), Bare Soil/Urban (38.59%), Cloud Shadow (13.09%), Medium Cloud (13.26%), High Cloud (17.14%).",
    bandsUsed: "SCL 20m resampled to 10m CRS: EPSG:32643",
  },
  {
    id: "obstruction_map",
    label: "Obstruction Map",
    shortLabel: "Obstruction",
    category: "obstruction",
    description: "Spatial distribution of atmospheric obstruction: Cloud and cloud-shadow mask.",
    scientificContext: "Delineates areas unsuitable for optical vegetation analysis without cloud removal. Total obstructed area is 43.49% (434,888 pixels), composed of 30.40% cloud cover and 13.09% cloud shadow.",
    bandsUsed: "SCL Classes 3 (Shadow), 8 (Med Cloud), 9 (High Cloud)",
  },
  {
    id: "cloud_free_agricultural_roi",
    label: "Canopy Health Composite",
    shortLabel: "Canopy ROI",
    category: "composite",
    description: "Analytical clear-sky canopy health composite with cloud obstruction masked out.",
    scientificContext: "Scientific analytical composite visualizing valid agricultural pixels after atmospheric filtering. (Note: This is an analytical data mask, not an AI-reconstructed synthetic scene).",
    bandsUsed: "Multispectral False-Color masked with combined SCL obstruction",
  },
];

export function SatelliteImageViewer({
  catalog,
}: {
  catalog?: VisualizationMetadata[];
}) {
  const [activeTab, setActiveTab] = useState<string>("true_color");
  const [zoom, setZoom] = useState<number>(1);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const [imageLoading, setImageLoading] = useState<boolean>(true);
  const [imageError, setImageError] = useState<boolean>(false);

  const viewerContainerRef = useRef<HTMLDivElement>(null);

  const currentTab = VISUALIZATION_TABS.find((t) => t.id === activeTab) || VISUALIZATION_TABS[0];
  const imageUrl = getAgricultureVisualizationUrl(activeTab);

  // Sync with catalog metadata if available
  const meta = catalog?.find((c) => c.id === activeTab);

  useEffect(() => {
    setImageLoading(true);
    setImageError(false);
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, [activeTab]);

  const handleZoomIn = () => setZoom((z) => Math.min(3, +(z + 0.25).toFixed(2)));
  const handleZoomOut = () => setZoom((z) => Math.max(0.75, +(z - 0.25).toFixed(2)));
  const handleResetZoom = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  const toggleFullscreen = async () => {
    if (!viewerContainerRef.current) return;
    try {
      if (!document.fullscreenElement) {
        await viewerContainerRef.current.requestFullscreen();
        setIsFullscreen(true);
      } else {
        await document.exitFullscreen();
        setIsFullscreen(false);
      }
    } catch {
      // ignore fullscreen rejection
    }
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (zoom <= 1) return;
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    setPan({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y,
    });
  };

  const handleMouseUp = () => setIsDragging(false);

  return (
    <Card className="p-6 bg-slate-900/70 border border-slate-800 space-y-5">
      {/* Header and Title */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-mono font-medium uppercase tracking-wider text-cyan-400 flex items-center gap-1.5">
              <Eye size={13} /> SATELLITE VISUALIZATION VIEWER
            </span>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
              1000 × 1000 px · 10m GSD
            </span>
          </div>
          <h3 className="text-xl font-bold text-white mt-1 flex items-center gap-2">
            {currentTab.label}
          </h3>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-2">
          <div className="flex items-center bg-slate-800/80 rounded-lg border border-slate-700 p-0.5">
            <button
              onClick={handleZoomOut}
              disabled={zoom <= 0.75}
              className="p-1.5 text-slate-300 hover:text-white disabled:text-slate-600 disabled:cursor-not-allowed transition"
              title="Zoom Out"
              aria-label="Zoom Out"
            >
              <ZoomOut size={15} />
            </button>
            <span className="px-2 text-[11px] font-mono text-slate-300 min-w-[45px] text-center">
              {Math.round(zoom * 100)}%
            </span>
            <button
              onClick={handleZoomIn}
              disabled={zoom >= 3}
              className="p-1.5 text-slate-300 hover:text-white disabled:text-slate-600 disabled:cursor-not-allowed transition"
              title="Zoom In"
              aria-label="Zoom In"
            >
              <ZoomIn size={15} />
            </button>
            <button
              onClick={handleResetZoom}
              className="p-1.5 text-slate-300 hover:text-white border-l border-slate-700 transition"
              title="Reset Zoom"
              aria-label="Reset Zoom"
            >
              <RotateCcw size={13} />
            </button>
          </div>

          <button
            onClick={toggleFullscreen}
            className="p-2 rounded-lg bg-slate-800/80 border border-slate-700 text-slate-300 hover:text-white transition"
            title={isFullscreen ? "Exit Fullscreen" : "Fullscreen View"}
            aria-label="Toggle Fullscreen"
          >
            {isFullscreen ? <Minimize2 size={15} /> : <Expand size={15} />}
          </button>

          <a
            href={imageUrl}
            target="_blank"
            rel="noopener noreferrer"
            download={`${activeTab}.png`}
            className="inline-flex items-center gap-1.5 px-3 py-2 rounded-lg bg-cyan-950/60 border border-cyan-500/30 text-cyan-300 hover:bg-cyan-900/60 text-xs font-mono transition"
            title="Open Full Resolution Image"
          >
            <Download size={13} />
            <span className="hidden sm:inline">Export PNG</span>
          </a>
        </div>
      </div>

      {/* Tabs Navigation */}
      <div className="flex gap-1.5 overflow-x-auto pb-2 border-b border-slate-800 scrollbar-thin">
        {VISUALIZATION_TABS.map((tab) => {
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-3 py-2 rounded-lg text-xs font-medium whitespace-nowrap transition cursor-pointer flex items-center gap-2 ${
                isActive
                  ? "bg-cyan-500/15 text-cyan-300 border border-cyan-500/40 shadow-sm"
                  : "bg-slate-800/40 text-slate-400 hover:text-slate-200 hover:bg-slate-800/80 border border-transparent"
              }`}
            >
              <span className={`w-2 h-2 rounded-full ${isActive ? "bg-cyan-400 animate-pulse" : "bg-slate-600"}`} />
              <span>{tab.shortLabel}</span>
            </button>
          );
        })}
      </div>

      {/* Main Image Viewport */}
      <div
        ref={viewerContainerRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        className={`relative min-h-[460px] md:min-h-[540px] rounded-xl overflow-hidden bg-slate-950 border border-slate-800/80 flex items-center justify-center select-none ${
          zoom > 1 ? "cursor-grab active:cursor-grabbing" : "cursor-default"
        }`}
      >
        {/* Loading Spinner */}
        {imageLoading && (
          <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-slate-950/80 backdrop-blur-xs gap-3">
            <RefreshCw className="text-cyan-400 animate-spin" size={32} />
            <span className="text-xs font-mono text-slate-400">Loading Sentinel-2 high-res raster...</span>
          </div>
        )}

        {/* Error Fallback */}
        {imageError ? (
          <div className="text-center p-8 space-y-3 z-10">
            <Info className="mx-auto text-amber-400" size={36} />
            <p className="text-sm text-slate-300 font-medium">Visualization raster could not be rendered.</p>
            <p className="text-xs text-slate-500 font-mono">Endpoint: /agriculture/visualizations/{activeTab}</p>
            <button
              onClick={() => {
                setImageLoading(true);
                setImageError(false);
              }}
              className="px-3 py-1.5 rounded bg-slate-800 hover:bg-slate-700 text-xs text-slate-200 transition"
            >
              Reload Raster
            </button>
          </div>
        ) : (
          <div
            style={{
              transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
              transformOrigin: "center center",
              transition: isDragging ? "none" : "transform 0.15s ease-out",
            }}
            className="w-full h-full flex items-center justify-center"
          >
            <img
              src={imageUrl}
              alt={currentTab.label}
              onLoad={() => setImageLoading(false)}
              onError={() => {
                setImageLoading(false);
                setImageError(true);
              }}
              className="max-h-[540px] w-auto max-w-full object-contain rounded"
              draggable={false}
            />
          </div>
        )}

        {/* Floating Overlay Badge */}
        <div className="absolute bottom-3 left-3 z-10 flex items-center gap-2">
          <div className="px-3 py-1.5 rounded-md bg-slate-950/85 backdrop-blur-md border border-slate-800 text-[11px] font-mono text-slate-300 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-400" />
            <span>{currentTab.label}</span>
            <span className="text-slate-500">|</span>
            <span className="text-slate-400">{meta?.file_size_bytes ? `${(meta.file_size_bytes / 1024 / 1024).toFixed(2)} MB` : "PNG"}</span>
          </div>
        </div>
      </div>

      {/* Scientific Context Description Banner */}
      <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-800/90 space-y-2">
        <div className="flex items-center justify-between text-xs font-mono">
          <span className="text-cyan-400 font-medium flex items-center gap-1.5">
            <Info size={14} /> SPECTRAL BAND CONFIGURATION
          </span>
          <span className="text-slate-500">{currentTab.bandsUsed}</span>
        </div>
        <p className="text-xs text-slate-300 leading-relaxed">{currentTab.description}</p>
        <p className="text-xs text-slate-400 leading-relaxed font-sans">{currentTab.scientificContext}</p>
      </div>
    </Card>
  );
}
