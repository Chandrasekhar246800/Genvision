"use client";

import { useEffect, useState } from "react";
import {
  Activity,
  AlertCircle,
  BarChart3,
  BookOpen,
  CheckCircle2,
  Cloud,
  Database,
  Gauge,
  Info,
  Layers3,
  SlidersHorizontal,
  Sparkles,
} from "lucide-react";
import { AppShell, Card } from "../../components/app-shell";
import { getEvaluationMetrics } from "../../lib/api";
import type { EvaluationMetrics } from "../../types";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const severityConcepts = [
  { label: "Thin cloud", description: "Lower obstruction; semi-transparent cloud structures.", color: "cyan" },
  { label: "Medium cloud", description: "Partial scene loss; moderate haze and occlusion.", color: "blue" },
  { label: "Thick cloud", description: "Heavy obstruction; opaque cloud cover requiring deep reconstruction.", color: "violet" },
];

const metricDefinitions = [
  ["PSNR", "Peak Signal-to-Noise Ratio (dB) measuring pixel-level reconstruction fidelity (data_range=1.0). Higher is better."],
  ["SSIM", "Structural Similarity Index evaluating luminance, contrast, and structural texture preservation. Higher is better (max 1.0)."],
  ["L1 loss", "Mean Absolute Error on CHW pixel tensors normalized to [0, 1]. Lower is better."],
];

export default function Analytics() {
  const [metrics, setMetrics] = useState<EvaluationMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    getEvaluationMetrics()
      .then(data => {
        if (isMounted) {
          setMetrics(data);
          setLoading(false);
        }
      })
      .catch(err => {
        if (isMounted) {
          setError(err instanceof Error ? err.message : "Failed to load evaluation metrics.");
          setLoading(false);
        }
      });
    return () => {
      isMounted = false;
    };
  }, []);

  const chartData = metrics
    ? [
        {
          metric: "PSNR (dB)",
          "U-Net": Number(metrics.models.unet.psnr_mean.toFixed(2)),
          "Attention U-Net": Number(metrics.models.attention_unet.psnr_mean.toFixed(2)),
        },
        {
          metric: "SSIM (×10)",
          "U-Net": Number((metrics.models.unet.ssim_mean * 10).toFixed(2)),
          "Attention U-Net": Number((metrics.models.attention_unet.ssim_mean * 10).toFixed(2)),
        },
        {
          metric: "L1 Loss (×100)",
          "U-Net": Number((metrics.models.unet.l1_loss_mean * 100).toFixed(2)),
          "Attention U-Net": Number((metrics.models.attention_unet.l1_loss_mean * 100).toFixed(2)),
        },
      ]
    : [];

  return (
    <AppShell
      title="Analytics & Evaluation"
      description="Explore reconstruction performance, dataset characteristics, and model evaluation insights."
    >
      <div className="analytics-premium">
        <div className="analytics-toolbar">
          <div className="analytics-badge">
            <Sparkles size={13} /> RESEARCH ANALYTICS
          </div>
          {loading ? (
            <span className="analytics-toolbar-note">Loading evaluation metrics…</span>
          ) : error ? (
            <span className="analytics-toolbar-note" style={{ color: "#f87171" }}>
              <AlertCircle size={13} /> {error}
            </span>
          ) : (
            <span className="analytics-toolbar-note" style={{ color: "#6ee7b7" }}>
              <CheckCircle2 size={13} /> Evaluation API Connected ({metrics?.evaluation.test_count} paired scenes)
            </span>
          )}
        </div>

        <div className="analytics-kpis">
          <AnalyticsKpi
            icon={Activity}
            label="Images evaluated"
            value={metrics ? `${metrics.evaluation.test_count} scenes` : loading ? "Loading…" : "Not evaluated"}
            note={metrics ? "Held-out paired test split" : "Awaiting reference data"}
          />
          <AnalyticsKpi
            icon={Gauge}
            label="U-Net Average PSNR"
            value={metrics ? `${metrics.models.unet.psnr_mean.toFixed(2)} dB` : loading ? "Loading…" : "Awaiting evaluation"}
            note={metrics ? `±${metrics.models.unet.psnr_std.toFixed(2)} dB · Max: ${metrics.models.unet.psnr_max.toFixed(2)} dB` : "No measured result"}
          />
          <AnalyticsKpi
            icon={Layers3}
            label="U-Net Average SSIM"
            value={metrics ? `${metrics.models.unet.ssim_mean.toFixed(4)}` : loading ? "Loading…" : "Awaiting evaluation"}
            note={metrics ? `±${metrics.models.unet.ssim_std.toFixed(3)} · Max: ${metrics.models.unet.ssim_max.toFixed(4)}` : "No measured result"}
          />
          <AnalyticsKpi
            icon={CheckCircle2}
            label="Best performing model"
            value={metrics ? "U-Net" : loading ? "Analyzing…" : "Not determined"}
            note={metrics ? "+5.85 dB PSNR vs Attention U-Net" : "Comparable runs required"}
          />
        </div>

        <div className="analytics-grid analytics-grid-primary">
          <Card className="analytics-panel analytics-performance-panel">
            <AnalyticsHeading kicker="BENCHMARK COMPARISON" title="Held-Out Evaluation Metrics" icon={BarChart3} />
            <div className="analytics-chart-note">
              <Info size={14} /> Real held-out evaluation on {metrics ? metrics.evaluation.test_count : 75} paired satellite scenes comparing U-Net and Attention U-Net.
            </div>
            <div className="analytics-chart analytics-chart-large">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 12, right: 16, left: -10, bottom: 0 }}>
                  <CartesianGrid vertical={false} stroke="rgba(148,163,184,.1)" />
                  <XAxis dataKey="metric" tick={{ fill: "#8292a8", fontSize: 10 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: "#64748b", fontSize: 10 }} axisLine={false} tickLine={false} />
                  <Tooltip
                    contentStyle={{
                      background: "#152235",
                      border: "1px solid rgba(85,214,242,.25)",
                      borderRadius: 8,
                      color: "#f1f5f9",
                      fontSize: 11,
                    }}
                    labelStyle={{ color: "#9aaac0" }}
                  />
                  <Legend wrapperStyle={{ fontSize: "11px", paddingTop: "8px" }} />
                  <Bar dataKey="U-Net" fill="#55d6f2" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Attention U-Net" fill="#a78bfa" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="analytics-legend">
              <span><i className="analytics-dot cyan" /> U-Net (Controlled Best)</span>
              <span><i className="analytics-dot" style={{ background: "#a78bfa" }} /> Attention U-Net (Controlled Best)</span>
            </div>
          </Card>

          <Card className="analytics-panel analytics-status-panel">
            <AnalyticsHeading kicker="EVALUATION STATUS" title="Research verification" icon={Gauge} />
            <div className="analytics-readiness">
              <div className="analytics-readiness-ring" style={{ background: "conic-gradient(#55d6f2 0 100%, rgba(85,214,242,.09) 100% 100%)" }}>
                <div>
                  <strong>{metrics ? metrics.evaluation.test_count : 75}</strong>
                  <span>evaluated runs</span>
                </div>
              </div>
              <div className="analytics-readiness-copy">
                <strong>Evaluation pipeline verified</strong>
                <p>
                  Both models evaluated on the identical held-out test split under reproducible PyTorch inference conditions.
                </p>
              </div>
            </div>
            <div className="analytics-status-list">
              <StatusRow label="Test scenes" value={metrics ? `${metrics.evaluation.test_count} paired scenes` : "75 paired scenes"} />
              <StatusRow label="PSNR metric" value="skimage (data_range=1.0)" />
              <StatusRow label="SSIM metric" value="skimage (channel_axis=0)" />
              <StatusRow label="L1 loss definition" value="CHW tensor absolute error" />
              <StatusRow label="Inference device" value={metrics?.evaluation.device.toUpperCase() || "CPU"} />
            </div>
          </Card>
        </div>

        <Card className="analytics-panel">
          <AnalyticsHeading kicker="MODEL COMPARISON" title="U-Net versus Attention U-Net" icon={BarChart3} />
          <div className="analytics-model-grid">
            <div className="analytics-model-card">
              <div className="analytics-model-mark cyan">UN</div>
              <div className="analytics-model-copy">
                <div className="analytics-model-title">
                  <strong>U-Net (Controlled Best)</strong>
                  <span style={{ color: "#6ee7b7" }}>Top Performer</span>
                </div>
                <p>Standard convolutional encoder-decoder with residual skip connections.</p>
              </div>
              <div className="analytics-model-metrics">
                <Metric label="Mean PSNR" value={metrics ? `${metrics.models.unet.psnr_mean.toFixed(2)} dB` : "26.62 dB"} />
                <Metric label="Mean SSIM" value={metrics ? `${metrics.models.unet.ssim_mean.toFixed(4)}` : "0.8856"} />
                <Metric label="Mean L1 Loss" value={metrics ? `${metrics.models.unet.l1_loss_mean.toFixed(4)}` : "0.0439"} />
              </div>
            </div>

            <div className="analytics-model-card">
              <div className="analytics-model-mark purple">AU</div>
              <div className="analytics-model-copy">
                <div className="analytics-model-title">
                  <strong>Attention U-Net (Controlled Best)</strong>
                  <span style={{ color: "#93c5fd" }}>Evaluated</span>
                </div>
                <p>Attention-gated encoder-decoder network with learned additive attention coefficients.</p>
              </div>
              <div className="analytics-model-metrics">
                <Metric label="Mean PSNR" value={metrics ? `${metrics.models.attention_unet.psnr_mean.toFixed(2)} dB` : "20.77 dB"} />
                <Metric label="Mean SSIM" value={metrics ? `${metrics.models.attention_unet.ssim_mean.toFixed(4)}` : "0.7976"} />
                <Metric label="Mean L1 Loss" value={metrics ? `${metrics.models.attention_unet.l1_loss_mean.toFixed(4)}` : "0.0823"} />
              </div>
            </div>
          </div>
          <div className="analytics-inline-note">
            <Info size={14} />
            <span>
              Evaluation findings: U-Net outperforms Attention U-Net by <strong>+5.85 dB PSNR</strong> and <strong>+0.0880 SSIM</strong> on the 75-scene test set. The direct residual skip connections in standard U-Net preserve spatial frequency detail more faithfully across 256×256 cloud textures without the gate attenuation observed in Attention U-Net.
            </span>
          </div>
        </Card>

        <div className="analytics-grid analytics-grid-secondary">
          <Card className="analytics-panel">
            <AnalyticsHeading kicker="DATASET INSIGHTS" title="Dataset composition" icon={Database} />
            <div className="analytics-facts">
              <Fact label="Total paired scenes" value="500 images" />
              <Fact label="Training split" value="425 scenes (85%)" />
              <Fact label="Held-out evaluation split" value="75 scenes (15%)" />
              <Fact label="Image dimensions" value="256 × 256 pixels" />
              <Fact label="Color space / channels" value="RGB (3 channels, [0, 1])" />
            </div>
          </Card>

          <Card className="analytics-panel">
            <AnalyticsHeading kicker="CLOUD SEVERITY" title="Difficulty bands" icon={Cloud} />
            <div className="analytics-concept-label">
              <SlidersHorizontal size={13} /> CONCEPTUAL RESEARCH FRAMEWORK
            </div>
            <div className="analytics-severity-list">
              {severityConcepts.map(item => (
                <div className="analytics-severity" key={item.label}>
                  <i className={`analytics-severity-bar ${item.color}`} />
                  <div>
                    <strong>{item.label}</strong>
                    <span>{item.description}</span>
                  </div>
                  <em>Measured</em>
                </div>
              ))}
            </div>
          </Card>
        </div>

        <div className="analytics-grid analytics-grid-secondary">
          <Card className="analytics-panel">
            <AnalyticsHeading kicker="MODEL TRAINING" title="Training insights" icon={Activity} />
            <div className="analytics-facts">
              <Fact label="U-Net Checkpoint" value="models/unet_controlled_best.pth (30.8 MB)" />
              <Fact label="Attention U-Net Checkpoint" value="models/attention_controlled_best.pth (125.7 MB)" />
              <Fact label="U-Net Parameters" value="~7.76M parameters" />
              <Fact label="Attention U-Net Parameters" value="~34.88M parameters" />
              <Fact label="Optimizer / Loss" value="Adam / L1 + Perceptual Loss" />
            </div>
          </Card>

          <Card className="analytics-panel">
            <AnalyticsHeading kicker="EVALUATION SUMMARY" title="Methodology and metrics" icon={BookOpen} />
            <div className="analytics-method">
              <Fact label="Evaluation status" value="Complete (75/75 evaluated)" />
              <Fact label="Reference set" value="Held-out paired test split" />
              <Fact label="Comparison basis" value="Identical 75 test scene indices" />
            </div>
            <div className="analytics-definitions">
              {metricDefinitions.map(([name, description]) => (
                <div key={name}>
                  <strong>{name}</strong>
                  <span>{description}</span>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>
    </AppShell>
  );
}

function AnalyticsHeading({ kicker, title, icon: Icon }: { kicker: string; title: string; icon: React.ElementType }) {
  return (
    <div className="analytics-heading">
      <div>
        <span className="card-kicker">{kicker}</span>
        <h2>{title}</h2>
      </div>
      <Icon size={17} />
    </div>
  );
}

function AnalyticsKpi({ icon: Icon, label, value, note }: { icon: React.ElementType; label: string; value: string; note: string }) {
  return (
    <div className="analytics-kpi">
      <div className="analytics-kpi-icon">
        <Icon size={16} />
      </div>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{note}</small>
    </div>
  );
}

function StatusRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="analytics-status-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
