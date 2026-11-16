"use client";

import { Calendar, Compass, Globe2, Layers, MapPin, Maximize2, Satellite, ShieldCheck } from "lucide-react";
import type { AgricultureSummary } from "../../types";
import { Card } from "../app-shell";

export function AgricultureOverview({ summary }: { summary: AgricultureSummary }) {
  const centerLat = summary.center_coordinates?.lat ? summary.center_coordinates.lat.toFixed(4) : "14.4172";
  const centerLon = summary.center_coordinates?.lon ? summary.center_coordinates.lon.toFixed(4) : "78.2341";

  return (
    <div className="space-y-4">
      {/* Top Banner Card */}
      <Card className="p-6 bg-gradient-to-r from-slate-900/90 via-slate-900/80 to-slate-950 border border-cyan-500/20">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-mono font-medium tracking-wide bg-cyan-950/80 text-cyan-300 border border-cyan-500/30">
                <Satellite size={12} className="text-cyan-400" />
                {summary.satellite} LEVEL-2A MULTISPECTRAL
              </span>
              <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[10px] font-mono font-medium tracking-wide bg-emerald-950/80 text-emerald-300 border border-emerald-500/30">
                <ShieldCheck size={12} className="text-emerald-400" />
                OFFICIALLY CALIBRATED (10m GSD)
              </span>
            </div>
            <h2 className="text-xl md:text-2xl font-bold tracking-tight text-white flex items-center gap-2.5">
              <span>{summary.study_area || "Pulivendula Agricultural Study Window"}</span>
              <span className="text-xs font-mono text-cyan-400/80 font-normal px-2 py-0.5 rounded bg-slate-800/80 border border-slate-700">
                Tile {summary.tile_id}
              </span>
            </h2>
            <p className="text-xs text-slate-400 max-w-3xl leading-relaxed">
              Operational Earth Observation analysis over agricultural regions of Pulivendula, Andhra Pradesh. Processed at 10-metre spatial resolution across Sentinel-2 Level-2A surface reflectance bands (B02, B03, B04, B08, B11) and SCL cloud masking.
            </p>
          </div>

          <div className="flex flex-wrap lg:flex-nowrap gap-3 items-center">
            <div className="px-4 py-2.5 rounded-lg bg-slate-800/60 border border-slate-700/60 text-right">
              <span className="block text-[10px] font-mono text-slate-400 uppercase tracking-wider">Acquisition Date</span>
              <span className="text-sm font-semibold text-slate-200 flex items-center justify-end gap-1.5 mt-0.5">
                <Calendar size={13} className="text-cyan-400" />
                {summary.acquisition_date}
              </span>
            </div>
            <div className="px-4 py-2.5 rounded-lg bg-slate-800/60 border border-slate-700/60 text-right">
              <span className="block text-[10px] font-mono text-slate-400 uppercase tracking-wider">Projected CRS</span>
              <span className="text-sm font-semibold text-slate-200 flex items-center justify-end gap-1.5 mt-0.5">
                <Compass size={13} className="text-cyan-400" />
                {summary.crs}
              </span>
            </div>
          </div>
        </div>
      </Card>

      {/* Geospatial Specification Matrix */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        <div className="p-3.5 rounded-lg bg-slate-900/60 border border-slate-800/90 flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-[10px] font-mono uppercase tracking-wider">Study Area</span>
            <MapPin size={13} className="text-cyan-400" />
          </div>
          <span className="text-xs font-semibold text-slate-200 truncate">Pulivendula</span>
          <span className="text-[10px] text-slate-500 font-mono mt-1">Andhra Pradesh, IN</span>
        </div>

        <div className="p-3.5 rounded-lg bg-slate-900/60 border border-slate-800/90 flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-[10px] font-mono uppercase tracking-wider">Satellite</span>
            <Satellite size={13} className="text-emerald-400" />
          </div>
          <span className="text-xs font-semibold text-slate-200">{summary.satellite}</span>
          <span className="text-[10px] text-slate-500 font-mono mt-1">ESA Copernicus L2A</span>
        </div>

        <div className="p-3.5 rounded-lg bg-slate-900/60 border border-slate-800/90 flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-[10px] font-mono uppercase tracking-wider">ROI Dimensions</span>
            <Maximize2 size={13} className="text-indigo-400" />
          </div>
          <span className="text-xs font-semibold text-slate-200">1000 × 1000 px</span>
          <span className="text-[10px] text-slate-500 font-mono mt-1">Area: {summary.roi_size_km?.toFixed(0) || "100"} km²</span>
        </div>

        <div className="p-3.5 rounded-lg bg-slate-900/60 border border-slate-800/90 flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-[10px] font-mono uppercase tracking-wider">Spatial GSD</span>
            <Layers size={13} className="text-cyan-400" />
          </div>
          <span className="text-xs font-semibold text-slate-200">10 Metres</span>
          <span className="text-[10px] text-slate-500 font-mono mt-1">Resampled (B11, SCL)</span>
        </div>

        <div className="p-3.5 rounded-lg bg-slate-900/60 border border-slate-800/90 flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-[10px] font-mono uppercase tracking-wider">Center Point</span>
            <Globe2 size={13} className="text-amber-400" />
          </div>
          <span className="text-xs font-semibold text-slate-200 font-mono">{centerLat}°N</span>
          <span className="text-[10px] text-slate-500 font-mono mt-1">{centerLon}°E</span>
        </div>

        <div className="p-3.5 rounded-lg bg-slate-900/60 border border-slate-800/90 flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-[10px] font-mono uppercase tracking-wider">Tile Code</span>
            <Compass size={13} className="text-purple-400" />
          </div>
          <span className="text-xs font-semibold text-slate-200 font-mono">{summary.tile_id}</span>
          <span className="text-[10px] text-slate-500 font-mono mt-1">S2MSI2A Baseline</span>
        </div>
      </div>
    </div>
  );
}
