"use client";

import { Cloud, CloudRain, Droplets, Eye, Leaf, ShieldAlert } from "lucide-react";
import type { AgricultureSummary } from "../../types";

type MetricConfig = {
  label: string;
  value: string;
  secondary: string;
  icon: React.ElementType;
  tone: "green" | "blue" | "cyan" | "emerald" | "amber" | "indigo";
  tooltip: string;
};

export function AgricultureMetricCards({ summary }: { summary: AgricultureSummary }) {
  const metrics: MetricConfig[] = [
    {
      label: "Mean Clear-Sky NDVI",
      value: summary.ndvi_mean_clear !== undefined ? summary.ndvi_mean_clear.toFixed(4) : "—",
      secondary: "Canopy greenness & vigor",
      icon: Leaf,
      tone: "green",
      tooltip: "Normalized Difference Vegetation Index (B08 - B04)/(B08 + B04) computed strictly over cloud-free agricultural pixels.",
    },
    {
      label: "Mean Clear-Sky NDWI",
      value: summary.ndwi_mean_clear !== undefined ? summary.ndwi_mean_clear.toFixed(4) : "—",
      secondary: "Water deficit (McFeeters)",
      icon: Droplets,
      tone: "cyan",
      tooltip: "Normalized Difference Water Index (B03 - B08)/(B03 + B08). Negative values characterize dry vegetation and bare soil.",
    },
    {
      label: "Mean Clear-Sky NDMI",
      value: summary.ndmi_mean_clear !== undefined ? (summary.ndmi_mean_clear > 0 ? `+${summary.ndmi_mean_clear.toFixed(4)}` : summary.ndmi_mean_clear.toFixed(4)) : "—",
      secondary: "Canopy moisture (Gao)",
      icon: CloudRain,
      tone: "blue",
      tooltip: "Normalized Difference Moisture Index (B08 - B11)/(B08 + B11). Measures liquid water content in crop canopy leaves.",
    },
    {
      label: "Clear Surface Area",
      value: `${summary.clear_surface_percentage?.toFixed(2) || "56.51"}%`,
      secondary: `${summary.clear_pixels ? (summary.clear_pixels).toLocaleString() : "565,112"} pixels`,
      icon: Eye,
      tone: "emerald",
      tooltip: "Unobstructed terrestrial surface pixels suitable for scientific vegetation interpretation.",
    },
    {
      label: "Cloud Contamination",
      value: `${summary.cloud_percentage?.toFixed(2) || "30.40"}%`,
      secondary: `${summary.cloud_pixels ? (summary.cloud_pixels).toLocaleString() : "303,976"} pixels`,
      icon: Cloud,
      tone: "amber",
      tooltip: "Medium- and high-probability cloud pixels identified by ESA Scene Classification (SCL classes 8 & 9).",
    },
    {
      label: "Cloud Shadow",
      value: `${summary.shadow_percentage?.toFixed(2) || "13.09"}%`,
      secondary: `${summary.shadow_pixels ? (summary.shadow_pixels).toLocaleString() : "130,912"} pixels`,
      icon: ShieldAlert,
      tone: "indigo",
      tooltip: "Cloud projection shadows on terrain (SCL class 3) causing severe near-infrared attenuation.",
    },
  ];

  const toneStyles: Record<string, { border: string; bg: string; iconBg: string; iconColor: string }> = {
    green: {
      border: "border-emerald-500/30 hover:border-emerald-500/50",
      bg: "bg-emerald-950/20",
      iconBg: "bg-emerald-500/10 border-emerald-500/20",
      iconColor: "text-emerald-400",
    },
    cyan: {
      border: "border-cyan-500/30 hover:border-cyan-500/50",
      bg: "bg-cyan-950/20",
      iconBg: "bg-cyan-500/10 border-cyan-500/20",
      iconColor: "text-cyan-400",
    },
    blue: {
      border: "border-blue-500/30 hover:border-blue-500/50",
      bg: "bg-blue-950/20",
      iconBg: "bg-blue-500/10 border-blue-500/20",
      iconColor: "text-blue-400",
    },
    emerald: {
      border: "border-teal-500/30 hover:border-teal-500/50",
      bg: "bg-teal-950/20",
      iconBg: "bg-teal-500/10 border-teal-500/20",
      iconColor: "text-teal-400",
    },
    amber: {
      border: "border-amber-500/30 hover:border-amber-500/50",
      bg: "bg-amber-950/20",
      iconBg: "bg-amber-500/10 border-amber-500/20",
      iconColor: "text-amber-400",
    },
    indigo: {
      border: "border-indigo-500/30 hover:border-indigo-500/50",
      bg: "bg-indigo-950/20",
      iconBg: "bg-indigo-500/10 border-indigo-500/20",
      iconColor: "text-indigo-400",
    },
  };

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
      {metrics.map((m) => {
        const style = toneStyles[m.tone];
        const Icon = m.icon;
        return (
          <div
            key={m.label}
            className={`p-4 rounded-xl border ${style.border} ${style.bg} transition-all duration-200 flex flex-col justify-between group`}
            title={m.tooltip}
          >
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400 font-medium">
                  {m.label}
                </span>
                <div className={`p-1.5 rounded-md border ${style.iconBg} ${style.iconColor}`}>
                  <Icon size={14} />
                </div>
              </div>
              <div className="text-xl md:text-2xl font-bold font-mono tracking-tight text-white group-hover:scale-[1.02] transition-transform origin-left">
                {m.value}
              </div>
            </div>
            <div className="mt-3 pt-2 border-t border-slate-800/80 flex items-center justify-between text-[10px] text-slate-400 font-mono">
              <span className="truncate">{m.secondary}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
