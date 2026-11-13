"use client";

import { Activity, BarChart3, BookOpen, CheckCircle2, Cloud, Database, Gauge, Info, Layers3, LineChart as LineChartIcon, LockKeyhole, SlidersHorizontal, Sparkles } from "lucide-react";
import { AppShell, Card } from "../../components/app-shell";
import { activity, models } from "../../lib/mock-data";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const severityConcepts = [
  { label: "Thin cloud", description: "Lower obstruction; useful for testing visibility-aware reconstruction.", color: "cyan" },
  { label: "Medium cloud", description: "Partial scene loss; a planned difficulty band for evaluation.", color: "blue" },
  { label: "Thick cloud", description: "High obstruction; requires labelled paired references.", color: "violet" },
];

const metricDefinitions = [
  ["PSNR", "Pixel-level reconstruction quality, measured against a cloud-free reference."],
  ["SSIM", "Structural similarity between reconstructed and reference images."],
  ["L1 loss", "Average absolute pixel difference during model evaluation."],
];

export default function Analytics() {
  return <AppShell title="Analytics & Evaluation" description="Explore reconstruction performance, dataset characteristics, and model evaluation insights.">
    <div className="analytics-premium">
      <div className="analytics-toolbar">
        <div className="analytics-badge"><Sparkles size={13} /> RESEARCH ANALYTICS</div>
        <span className="analytics-toolbar-note"><LockKeyhole size={13} /> Evaluation API not connected</span>
      </div>

      <div className="analytics-kpis">
        <AnalyticsKpi icon={Activity} label="Images evaluated" value="Not evaluated" note="Awaiting reference data" />
        <AnalyticsKpi icon={Gauge} label="Average PSNR" value="Awaiting evaluation" note="No measured result" />
        <AnalyticsKpi icon={Layers3} label="Average SSIM" value="Awaiting evaluation" note="No measured result" />
        <AnalyticsKpi icon={CheckCircle2} label="Best performing model" value="Not determined" note="Comparable runs required" />
      </div>

      <div className="analytics-grid analytics-grid-primary">
        <Card className="analytics-panel analytics-performance-panel">
          <AnalyticsHeading kicker="DEMONSTRATION ACTIVITY" title="Reconstruction activity overview" icon={LineChartIcon} />
          <div className="analytics-chart-note"><Info size={14} /> This chart shows illustrative frontend activity only. It is not a reconstruction-quality or evaluation metric.</div>
          <div className="analytics-chart analytics-chart-large">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={activity} margin={{ top: 10, right: 12, left: -18, bottom: 0 }}>
                <CartesianGrid vertical={false} stroke="rgba(148,163,184,.1)" />
                <XAxis dataKey="day" tick={{ fill: "#8292a8", fontSize: 9 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: "#64748b", fontSize: 9 }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={{ background: "#152235", border: "1px solid rgba(85,214,242,.25)", borderRadius: 8, color: "#f1f5f9", fontSize: 11 }} labelStyle={{ color: "#9aaac0" }} formatter={(value) => [`${value} demo runs`, "Activity"]} />
                <Line type="monotone" dataKey="value" name="Demo activity" stroke="#55d6f2" strokeWidth={2} dot={{ fill: "#55d6f2", r: 3, strokeWidth: 0 }} activeDot={{ r: 5 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="analytics-legend"><span><i className="analytics-dot cyan" /> Demo workflow activity</span><span>PSNR / SSIM unavailable</span></div>
        </Card>

        <Card className="analytics-panel analytics-status-panel">
          <AnalyticsHeading kicker="EVALUATION STATUS" title="Research readiness" icon={Gauge} />
          <div className="analytics-readiness">
            <div className="analytics-readiness-ring"><div><strong>0</strong><span>evaluated runs</span></div></div>
            <div className="analytics-readiness-copy"><strong>Evaluation pipeline pending</strong><p>Models are available in demo mode, but no genuine paired-reference evaluation has been connected.</p></div>
          </div>
          <div className="analytics-status-list">
            <StatusRow label="Reference images" value="Not connected" />
            <StatusRow label="Metric computation" value="Unavailable" />
            <StatusRow label="Training logs" value="Not connected" />
          </div>
        </Card>
      </div>

      <Card className="analytics-panel">
        <AnalyticsHeading kicker="MODEL COMPARISON" title="U-Net and Attention U-Net" icon={BarChart3} />
        <div className="analytics-model-grid">
          {models.map(model => <div className="analytics-model-card" key={model.name}>
            <div className={`analytics-model-mark ${model.color}`}>{model.short}</div>
            <div className="analytics-model-copy"><div className="analytics-model-title"><strong>{model.name}</strong><span>{model.status}</span></div><p>{model.description}</p></div>
            <div className="analytics-model-metrics"><Metric label="PSNR" value="Not evaluated" /><Metric label="SSIM" value="Not evaluated" /><Metric label="Validation loss" value="No logs" /></div>
          </div>)}
        </div>
        <div className="analytics-inline-note"><Info size={14} /> Model superiority cannot be determined until both architectures are evaluated on the same held-out paired dataset.</div>
      </Card>

      <div className="analytics-grid analytics-grid-secondary">
        <Card className="analytics-panel">
          <AnalyticsHeading kicker="DATASET INSIGHTS" title="Dataset composition" icon={Database} />
          <div className="analytics-empty-visual"><Database size={22} /><strong>Dataset manifest not connected</strong><span>Image counts, splits, dimensions, and formats will appear when the dataset registry is available.</span></div>
          <div className="analytics-facts"><Fact label="Total images" value="Not available" /><Fact label="Train / validation / test" value="Not available" /><Fact label="Image dimensions" value="Not available" /><Fact label="Formats" value="Not available" /></div>
        </Card>
        <Card className="analytics-panel">
          <AnalyticsHeading kicker="CLOUD SEVERITY" title="Difficulty bands" icon={Cloud} />
          <div className="analytics-concept-label"><SlidersHorizontal size={13} /> CONCEPTUAL RESEARCH FRAMEWORK</div>
          <div className="analytics-severity-list">{severityConcepts.map(item => <div className="analytics-severity" key={item.label}><i className={`analytics-severity-bar ${item.color}`} /><div><strong>{item.label}</strong><span>{item.description}</span></div><em>Unmeasured</em></div>)}</div>
        </Card>
      </div>

      <div className="analytics-grid analytics-grid-secondary">
        <Card className="analytics-panel">
          <AnalyticsHeading kicker="MODEL TRAINING" title="Training insights" icon={Activity} />
          <div className="analytics-empty-visual analytics-empty-training"><Activity size={22} /><strong>Training history is not connected</strong><span>Loss curves, epoch progression, learning rate, and best checkpoints require experiment logs.</span><button type="button" disabled>Awaiting training logs</button></div>
        </Card>
        <Card className="analytics-panel">
          <AnalyticsHeading kicker="EVALUATION SUMMARY" title="Methodology and metrics" icon={BookOpen} />
          <div className="analytics-method"><Fact label="Evaluation status" value="Not started" /><Fact label="Reference set" value="Required" /><Fact label="Comparison basis" value="Same held-out split" /></div>
          <div className="analytics-definitions">{metricDefinitions.map(([name, description]) => <div key={name}><strong>{name}</strong><span>{description}</span></div>)}</div>
        </Card>
      </div>
    </div>
  </AppShell>;
}

function AnalyticsHeading({ kicker, title, icon: Icon }: { kicker: string; title: string; icon: React.ElementType }) {
  return <div className="analytics-heading"><div><span className="card-kicker">{kicker}</span><h2>{title}</h2></div><Icon size={17} /></div>;
}

function AnalyticsKpi({ icon: Icon, label, value, note }: { icon: React.ElementType; label: string; value: string; note: string }) {
  return <div className="analytics-kpi"><div className="analytics-kpi-icon"><Icon size={16} /></div><span>{label}</span><strong>{value}</strong><small>{note}</small></div>;
}

function StatusRow({ label, value }: { label: string; value: string }) {
  return <div className="analytics-status-row"><span>{label}</span><strong>{value}</strong></div>;
}

function Metric({ label, value }: { label: string; value: string }) {
  return <div><span>{label}</span><strong>{value}</strong></div>;
}

function Fact({ label, value }: { label: string; value: string }) {
  return <div><span>{label}</span><strong>{value}</strong></div>;
}
