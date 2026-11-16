"use client";

import { useState } from "react";
import { AlertCircle, Columns2, Eye, Info, MoveHorizontal, Sliders } from "lucide-react";
import { getAgricultureVisualizationUrl } from "../../lib/api";
import { Card } from "../app-shell";

export function SatelliteImageCompare() {
  const [sliderPosition, setSliderPosition] = useState<number>(50);
  const [viewMode, setViewMode] = useState<"slider" | "side-by-side">("slider");
  const [isHovered, setIsHovered] = useState<boolean>(false);

  const trueColorUrl = getAgricultureVisualizationUrl("true_color");
  const clearCanopyUrl = getAgricultureVisualizationUrl("cloud_free_agricultural_roi");

  const handleSliderMove = (e: React.MouseEvent<HTMLDivElement> | React.TouchEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clientX = "touches" in e ? e.touches[0].clientX : e.clientX;
    const offset = clientX - rect.left;
    const percent = Math.max(0, Math.min(100, (offset / rect.width) * 100));
    setSliderPosition(percent);
  };

  return (
    <Card className="p-6 bg-slate-900/70 border border-slate-800 space-y-6">
      {/* Header and Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <span className="text-[10px] font-mono font-medium uppercase tracking-wider text-cyan-400 flex items-center gap-1.5">
            <Eye size={13} /> SATELLITE COMPARISON WORKSPACE
          </span>
          <h3 className="text-lg font-bold text-white mt-1">
            Cloud-Contaminated Scene vs. Clear-Sky Agricultural Canopy Health
          </h3>
        </div>

        {/* View Mode Toggle */}
        <div className="flex items-center gap-2">
          <div className="flex items-center bg-slate-800/80 rounded-lg border border-slate-700 p-0.5 text-xs font-mono">
            <button
              onClick={() => setViewMode("slider")}
              className={`px-3 py-1.5 rounded-md flex items-center gap-1.5 transition cursor-pointer ${
                viewMode === "slider" ? "bg-cyan-500/20 text-cyan-300 font-medium" : "text-slate-400 hover:text-white"
              }`}
            >
              <Sliders size={13} /> Slider Split
            </button>
            <button
              onClick={() => setViewMode("side-by-side")}
              className={`px-3 py-1.5 rounded-md flex items-center gap-1.5 transition cursor-pointer ${
                viewMode === "side-by-side" ? "bg-cyan-500/20 text-cyan-300 font-medium" : "text-slate-400 hover:text-white"
              }`}
            >
              <Columns2 size={13} /> Side-by-Side
            </button>
          </div>
        </div>
      </div>

      {/* Critical Scientific Compliance Alert */}
      <div className="p-3.5 rounded-lg bg-indigo-950/30 border border-indigo-500/30 flex items-start gap-3">
        <Info size={17} className="text-indigo-400 shrink-0 mt-0.5" />
        <div className="text-xs space-y-1">
          <p className="font-semibold text-indigo-200">
            Scientific Boundary Notice: Analytical Masking vs. AI Image Reconstruction
          </p>
          <p className="text-slate-300 leading-relaxed font-sans">
            The <strong>Analytical Canopy Composite</strong> on the right represents genuine, calibrated surface reflectance with cloud and shadow pixels filtered out using Sentinel-2 Level-2A Scene Classification (SCL). It is <strong>NOT</strong> an AI-generated or reconstructed satellite scene. GenVision’s deep-learning reconstruction framework (U-Net and Attention U-Net) is evaluated separately in the Cloud Removal Studio and Analytics benchmarks.
          </p>
        </div>
      </div>

      {/* Comparison Viewport */}
      {viewMode === "slider" ? (
        <div
          onMouseMove={(e) => isHovered && handleSliderMove(e)}
          onTouchMove={handleSliderMove}
          onMouseDown={() => setIsHovered(true)}
          onMouseUp={() => setIsHovered(false)}
          onMouseEnter={() => setIsHovered(true)}
          onMouseLeave={() => setIsHovered(false)}
          className="relative min-h-[460px] md:min-h-[520px] rounded-xl overflow-hidden bg-slate-950 border border-slate-800 select-none cursor-ew-resize"
        >
          {/* Background: Analytical Canopy Health */}
          <img
            src={clearCanopyUrl}
            alt="Analytical clear-sky agricultural canopy health"
            className="absolute inset-0 w-full h-full object-contain pointer-events-none"
            draggable={false}
          />
          <div className="absolute top-3 right-3 z-10 px-3 py-1.5 rounded-md bg-slate-950/85 backdrop-blur-md border border-slate-700 text-[11px] font-mono text-emerald-300 flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-400" />
            <span>CANOPY HEALTH COMPOSITE (CLEAR-SKY)</span>
          </div>

          {/* Foreground: True-Color RGB Clipped */}
          <div
            style={{ clipPath: `inset(0 ${100 - sliderPosition}% 0 0)` }}
            className="absolute inset-0 w-full h-full pointer-events-none"
          >
            <img
              src={trueColorUrl}
              alt="Raw cloud-contaminated Sentinel-2 scene"
              className="absolute inset-0 w-full h-full object-contain"
              draggable={false}
            />
            <div className="absolute top-3 left-3 z-10 px-3 py-1.5 rounded-md bg-slate-950/85 backdrop-blur-md border border-slate-700 text-[11px] font-mono text-cyan-300 flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-cyan-400" />
              <span>TRUE-COLOR RGB (43.49% OBSTRUCTED)</span>
            </div>
          </div>

          {/* Draggable Divider Handle */}
          <div
            style={{ left: `${sliderPosition}%` }}
            className="absolute top-0 bottom-0 w-0.5 bg-cyan-400 shadow-[0_0_12px_rgba(56,189,248,0.8)] pointer-events-none"
          >
            <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-8 h-8 rounded-full bg-slate-950 border-2 border-cyan-400 flex items-center justify-center text-cyan-300 shadow-xl">
              <MoveHorizontal size={14} />
            </div>
          </div>

          {/* Bottom Slider Position Helper */}
          <div className="absolute bottom-3 left-1/2 -translate-x-1/2 z-10 px-3 py-1 rounded-full bg-slate-950/85 border border-slate-800 text-[10px] font-mono text-slate-400 pointer-events-none">
            Slide to compare: {Math.round(sliderPosition)}%
          </div>
        </div>
      ) : (
        /* Side-by-Side View */
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="rounded-xl overflow-hidden bg-slate-950 border border-slate-800 space-y-2 p-3">
            <div className="flex justify-between items-center px-1 text-xs font-mono">
              <span className="text-cyan-300 font-medium">True-Color RGB</span>
              <span className="text-slate-400">43.49% Obstructed</span>
            </div>
            <div className="relative min-h-[360px] md:min-h-[420px] rounded-lg overflow-hidden bg-slate-900/50 flex items-center justify-center">
              <img
                src={trueColorUrl}
                alt="True Color RGB Composite"
                className="max-h-[420px] w-auto max-w-full object-contain rounded"
              />
            </div>
            <p className="text-[11px] text-slate-400 px-1 font-sans">
              Natural visual scene with visible cloud formations and cast shadows over the agricultural landscape.
            </p>
          </div>

          <div className="rounded-xl overflow-hidden bg-slate-950 border border-slate-800 space-y-2 p-3">
            <div className="flex justify-between items-center px-1 text-xs font-mono">
              <span className="text-emerald-300 font-medium">Canopy Health Composite</span>
              <span className="text-emerald-400">56.51% Clear Valid</span>
            </div>
            <div className="relative min-h-[360px] md:min-h-[420px] rounded-lg overflow-hidden bg-slate-900/50 flex items-center justify-center">
              <img
                src={clearCanopyUrl}
                alt="Analytical Clear-Sky Agricultural Canopy Health"
                className="max-h-[420px] w-auto max-w-full object-contain rounded"
              />
            </div>
            <p className="text-[11px] text-slate-400 px-1 font-sans">
              Valid agricultural pixels rendered in false-color NIR with cloud and shadow pixels analytically masked out.
            </p>
          </div>
        </div>
      )}
    </Card>
  );
}
