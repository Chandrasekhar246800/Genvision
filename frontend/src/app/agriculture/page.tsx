"use client";

import { useCallback, useEffect, useState } from "react";
import {
  Activity,
  AlertCircle,
  BarChart3,
  CheckCircle2,
  Database,
  ExternalLink,
  Layers,
  Leaf,
  RefreshCw,
  Satellite,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import { AppShell } from "../../components/app-shell";
import {
  AgricultureErrorState,
  AgricultureLoadingSkeleton,
} from "../../components/agriculture/AgricultureLoadingState";
import { AgricultureMetricCards } from "../../components/agriculture/AgricultureMetricCard";
import { AgricultureOverview } from "../../components/agriculture/AgricultureOverview";
import { CloudImpactComparison } from "../../components/agriculture/CloudImpactComparison";
import { ObstructionChart } from "../../components/agriculture/ObstructionChart";
import { SatelliteImageCompare } from "../../components/agriculture/SatelliteImageCompare";
import { SatelliteImageViewer } from "../../components/agriculture/SatelliteImageViewer";
import {
  getAgricultureHealth,
  getAgricultureIndices,
  getAgricultureMasks,
  getAgricultureSummary,
  getAgricultureVisualizations,
} from "../../lib/api";
import type {
  AgricultureHealth,
  AgricultureIndices,
  AgricultureMasks,
  AgricultureSummary,
  VisualizationsList,
} from "../../types";

export default function AgriculturePage() {
  const [health, setHealth] = useState<AgricultureHealth | null>(null);
  const [summary, setSummary] = useState<AgricultureSummary | null>(null);
  const [indices, setIndices] = useState<AgricultureIndices | null>(null);
  const [masks, setMasks] = useState<AgricultureMasks | null>(null);
  const [visualizations, setVisualizations] = useState<VisualizationsList | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [lastRefreshed, setLastRefreshed] = useState<Date | null>(null);

  const fetchAgricultureData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [h, s, idx, m, v] = await Promise.all([
        getAgricultureHealth(),
        getAgricultureSummary(),
        getAgricultureIndices(),
        getAgricultureMasks(),
        getAgricultureVisualizations(),
      ]);
      setHealth(h);
      setSummary(s);
      setIndices(idx);
      setMasks(m);
      setVisualizations(v);
      setLastRefreshed(new Date());
    } catch (err) {
      console.error("Agriculture API Fetch Error:", err);
      setError(
        err instanceof Error
          ? err.message
          : "Failed to connect to the agricultural intelligence service. Please check that the FastAPI backend is running."
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAgricultureData();
  }, [fetchAgricultureData]);

  return (
    <AppShell
      title="Agricultural Intelligence"
      description="Operational Sentinel-2C Level-2A Earth Observation analysis over agricultural regions of Pulivendula, Andhra Pradesh. Multispectral vegetation vigor, water deficit, canopy moisture, and atmospheric obstruction analysis."
    >
      <div className="space-y-6">
        {/* Status Toolbar */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-3 rounded-xl bg-slate-900/60 border border-slate-800 text-xs font-mono">
          <div className="flex items-center gap-3">
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-cyan-950/80 text-cyan-300 border border-cyan-500/30 font-medium">
              <Satellite size={13} className="text-cyan-400" />
              EARTH OBSERVATION PIPELINE
            </span>
            {loading ? (
              <span className="text-slate-400 flex items-center gap-1.5">
                <RefreshCw size={12} className="animate-spin text-cyan-400" />
                Querying FastAPI endpoints...
              </span>
            ) : error ? (
              <span className="text-red-400 flex items-center gap-1.5">
                <AlertCircle size={13} />
                Service Offline
              </span>
            ) : (
              <span className="text-emerald-400 flex items-center gap-1.5">
                <CheckCircle2 size={13} />
                Backend Operational · {summary?.satellite} ({summary?.tile_id})
              </span>
            )}
          </div>

          <div className="flex items-center gap-3 text-slate-400">
            {!loading && !error && (
              <>
                {lastRefreshed && (
                  <span className="hidden md:inline">
                    Refreshed: {lastRefreshed.toLocaleTimeString()}
                  </span>
                )}
                <button
                  onClick={fetchAgricultureData}
                  className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 transition cursor-pointer flex items-center gap-1"
                  title="Reload data from backend"
                >
                  <RefreshCw size={12} /> Sync API
                </button>
              </>
            )}
          </div>
        </div>

        {/* Content Body */}
        {loading ? (
          <AgricultureLoadingSkeleton />
        ) : error ? (
          <AgricultureErrorState error={error} onRetry={fetchAgricultureData} />
        ) : summary && indices && masks ? (
          <div className="space-y-6">
            {/* 1. Overview Section */}
            <AgricultureOverview summary={summary} />

            {/* 2. Key Agricultural KPI Cards */}
            <AgricultureMetricCards summary={summary} />

            {/* 3. Interactive Visualization Raster Viewer */}
            <SatelliteImageViewer catalog={visualizations?.visualizations} />

            {/* 4. Spectral Contamination & Cloud Impact */}
            <CloudImpactComparison data={indices} />

            {/* 5. Atmospheric Obstruction & SCL Land Cover */}
            <ObstructionChart masks={masks} />

            {/* 6. Satellite Comparison: True Color vs Clear Canopy */}
            <SatelliteImageCompare />

            {/* 7. Methodology & Scientific Provenance Footer */}
            <div className="p-6 rounded-xl bg-slate-900/50 border border-slate-800 space-y-3 text-xs text-slate-400">
              <div className="flex items-center gap-2 text-slate-200 font-semibold font-mono text-sm">
                <Database size={15} className="text-cyan-400" />
                <span>Geospatial Processing Methodology & Citation</span>
              </div>
              <p className="leading-relaxed">
                Multispectral observations are sourced from ESA Copernicus Sentinel-2C Level-2A (S2MSI2A, Datatake 43PHS, acquired 14 October 2026). Bottom-of-Atmosphere (BOA) surface reflectance values were scaled by 0.0001 with verified radiometric offset calibration. NDVI, NDWI (McFeeters), and NDMI (Gao) were computed from aligned 10m raster bands B02, B03, B04, B08, and bilinear resampled B11. Cloud and shadow pixels were identified and masked using the aligned Scene Classification Layer (SCL).
              </p>
              <div className="pt-2 flex flex-wrap gap-4 text-[11px] font-mono text-slate-500">
                <span>Product Baseline: S2MSI2A</span>
                <span>CRS: EPSG:32643 (WGS 84 / UTM Zone 43N)</span>
                <span>Spatial GSD: 10.0m</span>
                <span>ROI Bounding Box: 196320 - 206320 E, 1590480 - 1600480 N</span>
              </div>
            </div>
          </div>
        ) : null}
      </div>
    </AppShell>
  );
}
