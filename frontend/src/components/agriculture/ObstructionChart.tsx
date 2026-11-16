"use client";

import { Pie, PieChart, Cell, ResponsiveContainer, Tooltip } from "recharts";
import { Cloud, Eye, Layers, ShieldAlert } from "lucide-react";
import type { AgricultureMasks } from "../../types";
import { Card } from "../app-shell";

export function ObstructionChart({ masks }: { masks: AgricultureMasks }) {
  const obs = masks.obstruction_summary;

  const donutData = [
    { name: "Clear Surface", value: obs.clear_percent, pixels: obs.clear_pixels, color: "#10b981" },
    { name: "Cloud Cover", value: obs.cloud_percent, pixels: obs.cloud_pixels, color: "#38bdf8" },
    { name: "Cloud Shadow", value: obs.shadow_percent, pixels: obs.shadow_pixels, color: "#818cf8" },
  ];

  // SCL color map aligned with official ESA color palette
  const sclColors: Record<number, string> = {
    3: "#475569", // Cloud Shadow (dark gray)
    4: "#10b981", // Vegetation (green)
    5: "#d97706", // Bare soil / Urban (amber)
    6: "#0284c7", // Water (blue)
    7: "#64748b", // Unclassified (slate)
    8: "#94a3b8", // Cloud medium probability (light slate)
    9: "#38bdf8", // Cloud high probability (cyan/white)
  };

  return (
    <Card className="p-6 bg-slate-900/70 border border-slate-800 space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-4">
        <div>
          <span className="text-[10px] font-mono font-medium uppercase tracking-wider text-cyan-400 flex items-center gap-1.5">
            <Layers size={13} /> ATMOSPHERIC OBSTRUCTION & SCL CLASSIFICATION
          </span>
          <h3 className="text-lg font-bold text-white mt-1">Surface Visibility & Obstruction Analysis</h3>
        </div>
        <div className="text-right">
          <span className="text-xs font-mono text-slate-400">Total Analyzed: </span>
          <span className="text-xs font-mono font-semibold text-white">
            {masks.total_pixels.toLocaleString()} px (100 km²)
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
        {/* Donut Chart */}
        <div className="lg:col-span-5 flex flex-col items-center justify-center">
          <div className="w-full h-56 relative flex items-center justify-center">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={donutData}
                  cx="50%"
                  cy="50%"
                  innerRadius={65}
                  outerRadius={95}
                  paddingAngle={3}
                  dataKey="value"
                >
                  {donutData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} stroke="#0f172a" strokeWidth={2} />
                  ))}
                </Pie>
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload as typeof donutData[0];
                      return (
                        <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-700 text-xs font-mono space-y-1 shadow-xl">
                          <p className="font-semibold text-white">{data.name}</p>
                          <p className="text-slate-300">
                            {data.value.toFixed(2)}% ({data.pixels.toLocaleString()} pixels)
                          </p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
            {/* Center label */}
            <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
              <span className="text-xs font-mono text-slate-400">Clear Surface</span>
              <span className="text-2xl font-bold font-mono text-emerald-400">
                {obs.clear_percent.toFixed(1)}%
              </span>
            </div>
          </div>

          {/* Donut Legend */}
          <div className="grid grid-cols-3 gap-2 w-full mt-2 pt-2 border-t border-slate-800 text-center font-mono">
            <div className="p-2 rounded bg-slate-950/40 border border-slate-800/60">
              <span className="inline-block w-2 h-2 rounded-full bg-emerald-500 mr-1.5" />
              <span className="text-[10px] text-slate-400 block">Clear</span>
              <span className="text-xs font-bold text-white">{obs.clear_percent.toFixed(2)}%</span>
            </div>
            <div className="p-2 rounded bg-slate-950/40 border border-slate-800/60">
              <span className="inline-block w-2 h-2 rounded-full bg-sky-400 mr-1.5" />
              <span className="text-[10px] text-slate-400 block">Cloud</span>
              <span className="text-xs font-bold text-white">{obs.cloud_percent.toFixed(2)}%</span>
            </div>
            <div className="p-2 rounded bg-slate-950/40 border border-slate-800/60">
              <span className="inline-block w-2 h-2 rounded-full bg-indigo-400 mr-1.5" />
              <span className="text-[10px] text-slate-400 block">Shadow</span>
              <span className="text-xs font-bold text-white">{obs.shadow_percent.toFixed(2)}%</span>
            </div>
          </div>
        </div>

        {/* Detailed Breakdown & SCL Classes */}
        <div className="lg:col-span-7 space-y-4">
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3.5 rounded-lg bg-emerald-950/20 border border-emerald-500/25">
              <div className="flex items-center gap-2 text-emerald-400 text-xs font-mono font-medium">
                <Eye size={14} /> VALID OBSERVATION AREA
              </div>
              <div className="text-xl font-bold font-mono text-white mt-1">
                {obs.clear_percent.toFixed(2)}%
              </div>
              <p className="text-[11px] text-slate-400 mt-0.5">
                {obs.clear_pixels.toLocaleString()} pixels available for vegetation health retrieval.
              </p>
            </div>

            <div className="p-3.5 rounded-lg bg-amber-950/20 border border-amber-500/25">
              <div className="flex items-center gap-2 text-amber-400 text-xs font-mono font-medium">
                <ShieldAlert size={14} /> TOTAL OCCLUSION
              </div>
              <div className="text-xl font-bold font-mono text-white mt-1">
                {obs.total_obstructed_percent.toFixed(2)}%
              </div>
              <p className="text-[11px] text-slate-400 mt-0.5">
                {obs.total_obstructed_pixels.toLocaleString()} pixels obscured by cloud and shadow.
              </p>
            </div>
          </div>

          {/* SCL Histogram Bars */}
          <div className="space-y-2 pt-2">
            <span className="text-[11px] font-mono text-slate-400 uppercase tracking-wider block">
              Scene Classification Layer (SCL) Land Cover Breakdown
            </span>
            <div className="space-y-1.5">
              {masks.scl_classes.map((cls) => {
                const color = sclColors[cls.class_id] || "#64748b";
                return (
                  <div key={cls.class_id} className="space-y-1">
                    <div className="flex justify-between text-xs font-mono text-slate-300">
                      <span className="flex items-center gap-2">
                        <span className="w-2.5 h-2.5 rounded-xs" style={{ backgroundColor: color }} />
                        <span>Class {cls.class_id}: {cls.name}</span>
                      </span>
                      <span className="text-slate-400">
                        {cls.percentage.toFixed(2)}% ({cls.count.toLocaleString()} px)
                      </span>
                    </div>
                    <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-500"
                        style={{ width: `${Math.min(100, cls.percentage)}%`, backgroundColor: color }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
}
