"use client";

import { useRef, useState } from "react";
import { BarChart3, CheckCircle2, Cloud, Droplets, Expand, Eye, Info, Leaf, LockKeyhole, Map, Mountain, Sprout, ZoomIn, ZoomOut } from "lucide-react";
import { AppShell, Card } from "../../components/app-shell";

const applications = [
  { icon: Leaf, title: "Crop monitoring", text: "Observe agricultural land over time when cloud-obscured scenes limit continuity.", tone: "green" },
  { icon: Sprout, title: "Vegetation assessment", text: "Prepare clearer imagery for future vegetation-focused research workflows.", tone: "cyan" },
  { icon: Droplets, title: "Water resource monitoring", text: "Support research into water-related features once appropriate spectral data is available.", tone: "blue" },
  { icon: Map, title: "Land use analysis", text: "Study field structure, surrounding terrain, and seasonal observation opportunities.", tone: "violet" },
];

export default function Agriculture() {
  const [zoom, setZoom] = useState(1);
  const viewer = useRef<HTMLDivElement>(null);

  async function fullscreen() {
    if (!viewer.current) return;
    if (document.fullscreenElement) await document.exitFullscreen();
    else await viewer.current.requestFullscreen();
  }

  return <AppShell title="Agricultural Intelligence" description="Explore how reconstructed satellite imagery can support vegetation monitoring, crop research, and environmental observation.">
    <div className="agri-premium">
      <div className="agri-toolbar"><span className="agri-earth-badge"><Eye size={13} /> EARTH OBSERVATION</span><span className="agri-status"><i /> Research concept · RGB prototype</span></div>

      <Card className="agri-scene-card">
        <div className="agri-scene-copy"><span className="card-kicker">01 / SATELLITE SCENE</span><h2>A clearer view of agricultural terrain.</h2><p>Cloud-free image reconstruction can help researchers maintain visual continuity across agricultural observation studies. This prototype currently presents RGB imagery and illustrative research concepts.</p><div className="agri-scene-meta"><span><strong>Scene type</strong> Agricultural terrain</span><span><strong>Input</strong> RGB imagery</span><span><strong>Status</strong> Illustrative visualization</span></div></div>
        <div ref={viewer} className="agri-image-viewer"><img src="/images/hero/satellite-scene.png" alt="Illustrative RGB satellite scene for agricultural research" style={{ transform: `scale(${zoom})` }} /><div className="agri-image-overlay"><span>AGRICULTURAL RESEARCH</span><span>RGB · ILLUSTRATIVE</span></div><div className="agri-viewer-tools"><button onClick={() => setZoom(value => Math.max(1, value - .25))} disabled={zoom === 1} aria-label="Zoom out"><ZoomOut size={14} /></button><button onClick={() => setZoom(value => Math.min(2, value + .25))} disabled={zoom === 2} aria-label="Zoom in"><ZoomIn size={14} /></button><button onClick={fullscreen} aria-label="View satellite scene fullscreen"><Expand size={14} /></button></div></div>
      </Card>

      <div className="agri-section-heading"><div><span className="card-kicker">02 / VEGETATION MONITORING</span><h2>Research indicators, not fabricated measurements.</h2></div><p>These indicators describe future analysis paths. No vegetation score is calculated from the current RGB prototype.</p></div>
      <div className="agri-indicators">
        <ResearchIndicator icon={Leaf} title="Vegetation health" status="PLANNED" text="Potentially assess vegetation condition using validated multispectral indices and reference data." tone="green" />
        <ResearchIndicator icon={BarChart3} title="Vegetation coverage" status="RESEARCH CONCEPT" text="Explore field coverage and seasonal change after suitable labelled imagery is connected." tone="cyan" />
        <ResearchIndicator icon={Mountain} title="Land surface observation" status="CONCEPTUAL" text="Use reconstructed scenes to study terrain context and improve observation continuity." tone="violet" />
      </div>

      <Card className="agri-crop-health"><div className="agri-section-heading"><div><span className="card-kicker">02A / CROP HEALTH MONITORING</span><h2>From visual continuity to seasonal research.</h2></div><span className="agri-concept-pill"><LockKeyhole size={12} /> PLANNED CAPABILITY</span></div><div className="agri-crop-grid"><CropTopic icon={Sprout} title="Crop growth observation" text="Compare field appearance across observation windows when cloud-free imagery is available." /><CropTopic icon={Leaf} title="Agricultural land assessment" text="Study field structure and surrounding land context without claiming crop classification." /><CropTopic icon={Cloud} title="Potential crop stress" text="Investigate stress signals only after validated spectral data and labelled references are connected." /><CropTopic icon={BarChart3} title="Seasonal monitoring" text="Build a consistent time series for research, with reconstruction uncertainty kept visible." /></div></Card>

      <div className="agri-analysis-grid">
        <IndexPanel type="NDVI" icon={Leaf} title="Vegetation greenness research" description="NDVI uses near-infrared and red spectral information to assess vegetation greenness." />
        <IndexPanel type="NDWI" icon={Droplets} title="Water observation research" description="NDWI can help identify water content or water-related features, depending on the formulation and bands used." />
      </div>

      <Card className="agri-applications"><div className="agri-section-heading"><div><span className="card-kicker">03 / AGRICULTURAL APPLICATIONS</span><h2>Where clearer observation could help.</h2></div><span className="agri-concept-pill"><LockKeyhole size={12} /> RESEARCH APPLICATIONS</span></div><div className="agri-application-grid">{applications.map(({ icon: Icon, title, text, tone }) => <div className={`agri-application ${tone}`} key={title}><Icon size={18} /><strong>{title}</strong><p>{text}</p></div>)}</div></Card>

      <Card className="agri-comparison-card"><div className="agri-section-heading"><div><span className="card-kicker">04 / RESEARCH MOTIVATION</span><h2>Cloud-obscured input versus clean reference.</h2></div></div><div className="agri-comparison"><div><img src="/images/hero/hero-cloudy.png" alt="Cloud-obscured satellite image input" /><span>CLOUDY INPUT</span></div><div><img src="/images/hero/hero-clean.png" alt="Clean reference satellite image" /><span>CLEAN REFERENCE · DATASET PAIR</span></div></div><div className="agri-disclaimer"><Info size={14} /> The clean reference is a paired dataset image, not a generated model output. It is shown only to explain the research motivation.</div></Card>

      <Card className="agri-insights"><div className="agri-insight-main"><span className="card-kicker">05 / RESEARCH INSIGHTS</span><h2>From clearer scenes to stronger observation studies.</h2><p>Cloud obstruction can interrupt agricultural time series and hide field-level context. Reconstruction research may help make those scenes more usable for visual analysis, but reconstructed pixels require validation before scientific interpretation.</p></div><div className="agri-insight-list"><Insight icon={Cloud} title="Cloud obstruction" text="Missing visibility can break continuity across observation windows." /><Insight icon={CheckCircle2} title="Validation first" text="Reference pairs and held-out evaluation are required before claims." /><Insight icon={LockKeyhole} title="Spectral limitation" text="RGB imagery cannot provide genuine NDVI or NDWI calculations." /></div></Card>
    </div>
  </AppShell>;
}

function ResearchIndicator({ icon: Icon, title, status, text, tone }: { icon: React.ElementType; title: string; status: string; text: string; tone: string }) {
  return <div className={`agri-indicator ${tone}`}><div className="agri-indicator-icon"><Icon size={17} /></div><div><span>{status}</span><strong>{title}</strong><p>{text}</p></div></div>;
}

function IndexPanel({ type, icon: Icon, title, description }: { type: string; icon: React.ElementType; title: string; description: string }) {
  return <Card className="agri-index-panel"><div className="agri-index-head"><div className={`agri-index-icon ${type.toLowerCase()}`}><Icon size={17} /></div><div><span className="card-kicker">{type} · PLANNED</span><h2>{title}</h2></div></div><p>{description}</p><div className="agri-spectrum"><span /><span /><span /><span /><span /></div><div className="agri-spectrum-labels"><small>Low intensity</small><small>Higher intensity</small></div><div className="agri-index-note"><LockKeyhole size={13} /> Requires multispectral red / near-infrared bands. No {type} values are calculated here.</div></Card>;
}

function Insight({ icon: Icon, title, text }: { icon: React.ElementType; title: string; text: string }) {
  return <div><Icon size={16} /><span><strong>{title}</strong><small>{text}</small></span></div>;
}

function CropTopic({ icon: Icon, title, text }: { icon: React.ElementType; title: string; text: string }) {
  return <div className="agri-crop-topic"><Icon size={15} /><div><strong>{title}</strong><span>{text}</span></div></div>;
}
