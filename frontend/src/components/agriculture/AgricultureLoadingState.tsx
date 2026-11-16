"use client";

import { AlertCircle, RefreshCw, Satellite } from "lucide-react";
import { Card } from "../app-shell";

export function AgricultureLoadingSkeleton() {
  return (
    <div className="space-y-6 animate-pulse" aria-label="Loading agricultural intelligence data">
      {/* Top Banner Skeleton */}
      <Card className="p-6 bg-slate-900/60 border border-slate-800">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="space-y-2">
            <div className="h-4 w-32 bg-slate-800 rounded" />
            <div className="h-7 w-64 bg-slate-800 rounded" />
            <div className="h-4 w-96 bg-slate-800 rounded" />
          </div>
          <div className="flex gap-2">
            <div className="h-9 w-28 bg-slate-800 rounded-md" />
            <div className="h-9 w-28 bg-slate-800 rounded-md" />
          </div>
        </div>
      </Card>

      {/* Overview Grid Skeleton */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        {Array.from({ length: 6 }).map((_, i) => (
          <Card key={i} className="p-4 bg-slate-900/40 border border-slate-800/80 space-y-2">
            <div className="h-3 w-16 bg-slate-800 rounded" />
            <div className="h-5 w-24 bg-slate-800 rounded" />
            <div className="h-3 w-20 bg-slate-800/60 rounded" />
          </Card>
        ))}
      </div>

      {/* KPI Cards Skeleton */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        {Array.from({ length: 6 }).map((_, i) => (
          <Card key={i} className="p-4 bg-slate-900/50 border border-slate-800 space-y-2">
            <div className="h-6 w-6 bg-slate-800 rounded-full" />
            <div className="h-3 w-20 bg-slate-800 rounded" />
            <div className="h-7 w-24 bg-slate-800 rounded" />
            <div className="h-3 w-28 bg-slate-800/60 rounded" />
          </Card>
        ))}
      </div>

      {/* Viewer Skeleton */}
      <Card className="p-6 bg-slate-900/60 border border-slate-800 space-y-4">
        <div className="flex justify-between items-center">
          <div className="h-5 w-48 bg-slate-800 rounded" />
          <div className="h-8 w-64 bg-slate-800 rounded-md" />
        </div>
        <div className="h-96 w-full bg-slate-800/50 rounded-lg flex items-center justify-center">
          <Satellite className="text-slate-700 animate-spin" size={40} />
        </div>
      </Card>
    </div>
  );
}

export function AgricultureErrorState({
  error,
  onRetry,
}: {
  error: string;
  onRetry: () => void;
}) {
  return (
    <Card className="p-8 border border-red-500/30 bg-red-950/20 text-center max-w-2xl mx-auto my-12 space-y-4">
      <div className="w-12 h-12 rounded-full bg-red-500/20 text-red-400 flex items-center justify-center mx-auto">
        <AlertCircle size={24} />
      </div>
      <div className="space-y-2">
        <h3 className="text-lg font-semibold text-slate-100">Agricultural Data Connection Failed</h3>
        <p className="text-sm text-slate-400 max-w-md mx-auto">{error}</p>
        <p className="text-xs text-slate-500">
          Ensure the FastAPI backend is running (`uvicorn backend.app.main:app --port 8000`) and the Phase 3.2 processing outputs are available.
        </p>
      </div>
      <div>
        <button
          onClick={onRetry}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-medium transition cursor-pointer"
        >
          <RefreshCw size={14} /> Retry Connection
        </button>
      </div>
    </Card>
  );
}
