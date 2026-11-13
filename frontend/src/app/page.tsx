"use client";

import Image from "next/image";
import Link from "next/link";
import { MotionConfig, motion } from "framer-motion";
import { ArrowDownRight, ArrowRight, Crosshair, Eye, Layers3, Menu, ShieldCheck, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";

const reveal = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.6, ease: "easeOut" as const } },
};

export default function Home() {
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [heroComparisonPosition, setHeroComparisonPosition] = useState(50);
  const [heroComparisonDragging, setHeroComparisonDragging] = useState(false);
  const heroFrameRef = useRef<HTMLDivElement>(null);
  const motionProps = { initial: "hidden", whileInView: "visible", viewport: { once: true, amount: 0.2 }, variants: reveal };
  const updateHeroComparison = (clientX: number, element: HTMLElement) => {
    const bounds = element.getBoundingClientRect();
    const position = ((clientX - bounds.left) / bounds.width) * 100;
    setHeroComparisonPosition(Math.min(100, Math.max(0, position)));
  };
  const adjustHeroComparison = (offset: number) => {
    setHeroComparisonPosition(position => Math.min(100, Math.max(0, position + offset)));
  };
  useEffect(() => {
    if (!heroComparisonDragging) return;
    const handlePointerMove = (event: PointerEvent) => {
      const frame = heroFrameRef.current;
      if (frame) updateHeroComparison(event.clientX, frame);
    };
    const stopDragging = () => setHeroComparisonDragging(false);
    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", stopDragging);
    window.addEventListener("pointercancel", stopDragging);
    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", stopDragging);
      window.removeEventListener("pointercancel", stopDragging);
    };
  }, [heroComparisonDragging]);

  return <MotionConfig reducedMotion="user"><main className="landing landing-premium">
    <nav className={`landing-nav container ${mobileNavOpen ? "mobile-open" : ""}`}>
      <Link href="/" className="brand"><span className="brand-mark"><Image src="/genvision.png" alt="GenVision" width={27} height={27} priority /></span><span>Gen<span>Vision</span></span></Link>
      <div className="landing-links">
        {["technology", "features", "research"].map((item) => <a href={`#${item}`} key={item} onClick={() => setMobileNavOpen(false)}>{item === "features" ? "Capabilities" : item[0].toUpperCase() + item.slice(1)}</a>)}
      </div>
      <Link href="/studio" className="button button-primary">Launch Studio <ArrowRight size={16} /></Link>
      <button className="landing-menu-button" onClick={() => setMobileNavOpen(value => !value)} aria-label={mobileNavOpen ? "Close navigation" : "Open navigation"} aria-expanded={mobileNavOpen}>{mobileNavOpen ? <X size={19} /> : <Menu size={19} />}</button>
    </nav>

    <section className="hero hero-cinematic container">
      <motion.div className="hero-copy" initial="hidden" animate="visible" variants={{ hidden: {}, visible: { transition: { staggerChildren: .09 } } }}>
        <motion.div className="eyebrow" variants={reveal}><span className="pulse-dot" /> AI-POWERED EARTH OBSERVATION</motion.div>
        <motion.h1 variants={reveal}>SEE WHAT LIES<br /><em>BEYOND THE CLOUDS.</em></motion.h1>
        <motion.p variants={reveal}>GenVision explores intelligent reconstruction of cloud-obscured satellite imagery using deep learning, opening new possibilities for Earth observation research.</motion.p>
        <motion.div className="hero-actions" variants={reveal}><Link href="/studio" className="button button-primary button-large">Explore GenVision Studio <ArrowRight size={17} /></Link><a href="#technology" className="button button-ghost button-large">Discover the Technology <ArrowDownRight size={16} /></a></motion.div>
        <motion.div className="hero-meta" variants={reveal}><span><Layers3 size={14} /> Deep learning reconstruction</span><i /> <span><Eye size={14} /> Satellite imagery</span><i /> <span><ShieldCheck size={14} /> Research prototype</span></motion.div>
      </motion.div>
      <motion.div className="hero-cinematic-visual" initial={{ opacity: 0, clipPath: "inset(0 0 0 18%)" }} animate={{ opacity: 1, clipPath: "inset(0 0 0 0%)" }} transition={{ duration: .9, delay: .2 }}>
        <div ref={heroFrameRef} className="cinematic-frame" onPointerMove={event => { if (event.buttons === 1 || event.currentTarget.hasPointerCapture(event.pointerId)) updateHeroComparison(event.clientX, event.currentTarget); }}>
          <Image src="/images/hero/hero-cloudy.png" alt="Cloud-covered satellite image from a paired dataset" fill priority sizes="(max-width: 700px) 100vw, 55vw" className="cinematic-image" />
          <div className="cinematic-clean-layer" style={{ clipPath: `inset(0 0 0 ${heroComparisonPosition}%)` }}><Image src="/images/hero/hero-clean.png" alt="Clean reference satellite image from the same paired dataset scene" fill sizes="(max-width: 700px) 100vw, 55vw" className="cinematic-image" /></div>
          <div className="cinematic-tint" /><div className="cinematic-grid" />
          <button className="cinematic-divider" style={{ left: `${heroComparisonPosition}%` }} type="button" aria-label={`Image comparison divider at ${Math.round(heroComparisonPosition)} percent. Use left and right arrow keys to adjust.`} aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(heroComparisonPosition)} role="slider" onPointerDown={event => { setHeroComparisonDragging(true); event.currentTarget.setPointerCapture(event.pointerId); updateHeroComparison(event.clientX, event.currentTarget.parentElement as HTMLElement); }} onKeyDown={event => { if (event.key === "ArrowLeft") { event.preventDefault(); adjustHeroComparison(-2); } if (event.key === "ArrowRight") { event.preventDefault(); adjustHeroComparison(2); } }}><span /></button>
          <span className="frame-corner corner-tl" /><span className="frame-corner corner-tr" /><span className="frame-corner corner-bl" /><span className="frame-corner corner-br" /><span className="cinematic-label label-earth">EARTH OBSERVATION</span><span className="cinematic-label label-research">AI RECONSTRUCTION RESEARCH</span><span className="cinematic-label label-input">CLOUD-COVERED INPUT</span><span className="cinematic-label label-clean">CLEAN REFERENCE</span><div className="cinematic-coordinate">PAIRED DATASET REFERENCE<br /><span>ILLUSTRATIVE DATASET COMPARISON</span></div>
        </div>
      </motion.div>
    </section>

    <motion.section className="section landing-section capabilities-editorial container" id="features" {...motionProps}>
      <div className="section-heading capabilities-heading"><div><div className="eyebrow">01 / PLATFORM CAPABILITIES</div><h2>Intelligence beyond <span>visibility.</span></h2></div><p>An integrated research platform exploring satellite image reconstruction, deep-learning architectures, and the possibilities of clearer Earth observation.</p></div>
      <div className="capabilities-editorial-grid">
        <article className="capability-feature-main">
          <div className="capability-image-frame"><Image src="/images/hero/hero-cloudy.png" alt="Cloud-covered input from a paired satellite image dataset" fill sizes="(max-width: 700px) 100vw, 58vw" className="capability-image capability-cloudy-image" /><div className="capability-clean-layer"><Image src="/images/hero/hero-clean.png" alt="Clean reference from the same paired satellite image dataset" fill sizes="(max-width: 700px) 100vw, 58vw" className="capability-image capability-clean-image" /></div><div className="capability-image-grid" /><div className="capability-static-divider" /><span className="capability-image-label">CLOUD-COVERED INPUT</span><span className="capability-clean-label">CLEAN REFERENCE</span><span className="capability-image-note">PAIRED DATASET COMPARISON</span><span className="capability-corner capability-corner-tl" /><span className="capability-corner capability-corner-br" /></div>
          <div className="capability-feature-copy"><div><div className="capability-index">01 — CLOUD-OBSCURED IMAGE RECONSTRUCTION</div><h3>Reconstruct what<br /><span>clouds conceal.</span></h3><p>Explore deep-learning approaches to reconstruct satellite imagery affected by cloud coverage.</p></div><span className="capability-status capability-status-large"><i /> PROTOTYPE</span></div>
        </article>
        <div className="capability-secondary">
          <article className="capability-feature-side"><div className="capability-index">02 — ATTENTION-GUIDED RECONSTRUCTION</div><div className="attention-s-flow" aria-label="Attention U-Net conceptual six-stage architecture"><div className="attention-s-stage attention-s-input"><span className="attention-s-media"><Image src="/images/hero/hero-cloudy.png" alt="" fill sizes="100px" /></span><b>CLOUDY<br />INPUT</b></div><i className="attention-s-arrow attention-s-arrow-1">→</i><div className="attention-s-stage attention-s-encoder"><span className="attention-s-media"><Image src="/images/hero/hero-cloudy.png" alt="" fill sizes="100px" /></span><b>ENCODER</b></div><i className="attention-s-arrow attention-s-arrow-2">→</i><div className="attention-s-stage attention-s-gate"><span className="attention-s-media">          <Image src="/images/hero/hero-cloudy.png" alt="" fill sizes="100px" /></span><b>ATTENTION<br />GATE</b></div><i className="attention-s-turn attention-s-turn-down">↓</i><div className="attention-s-stage attention-s-output"><span className="attention-s-media"><Image src="/images/hero/hero-clean.png" alt="" fill sizes="100px" /></span><b>CLEAN<br />OUTPUT</b></div><i className="attention-s-arrow attention-s-arrow-back">←</i><div className="attention-s-stage attention-s-decoder"><span className="attention-s-media"><Image src="/images/hero/hero-clean.png" alt="" fill sizes="100px" /></span><b>DECODER</b></div><i className="attention-s-arrow attention-s-arrow-bottleneck">←</i><div className="attention-s-stage attention-s-bottleneck"><span className="attention-s-media"><Image src="/images/hero/hero-cloudy.png" alt="" fill sizes="100px" /></span><b>BOTTLENECK</b></div><span className="attention-s-caption">ATTENTION U-NET / CONCEPTUAL FLOW</span></div><div className="capability-side-copy"><div><h3>Focus the model<br /><span>where it matters.</span></h3><p>Investigate attention mechanisms that help reconstruction models prioritize relevant spatial features.</p></div><span className="capability-status"><i /> PROTOTYPE</span></div></article>
          <article className="capability-feature-side severity-feature"><div className="capability-index">03 — CLOUD SEVERITY INTELLIGENCE</div><div className="severity-visual"><div><span className="severity-cloud thin" /><small>THIN</small></div><div><span className="severity-cloud medium" /><small>MEDIUM</small></div><div><span className="severity-cloud thick" /><small>THICK</small></div></div><div className="capability-side-copy"><div><h3>Read the<br /><span>obscuration.</span></h3><p>Planned cloud severity estimation to support reconstruction strategies for varying cloud conditions.</p></div><span className="capability-status roadmap-status"><i /> RESEARCH ROADMAP</span></div></article>
        </div>
        <article className="capability-feature-bottom"><div className="agri-mini-visual"><Image src="/images/hero/capability-terrain.png" alt="Illustrative satellite terrain visualization for future agricultural research" fill sizes="(max-width: 700px) 100vw, 35vw" className="capability-image" /><div className="agri-field-lines" /><span>RGB SCENE / RESEARCH CONCEPT</span></div><div className="capability-bottom-copy"><div><div className="capability-index">04 — AGRICULTURAL INSIGHTS</div><h3>From reconstruction<br /><span>to application.</span></h3><p>Explore downstream applications of reconstructed satellite imagery for agricultural and environmental research.</p></div><span className="capability-status planned-status"><i /> PLANNED</span></div></article>
      </div>
    </motion.section>

    <motion.section className="technology technology-premium deep-learning-showcase container" id="technology" {...motionProps}>
      <div className="deep-learning-header"><div><div className="eyebrow">02 / BUILT ON DEEP LEARNING</div><h2>Intelligence <span>beneath the surface.</span></h2></div><p>GenVision explores encoder-decoder deep-learning architectures to reconstruct cloud-obscured satellite imagery while preserving important spatial information.</p></div>
      <div className="architecture-showcases">
        <article className="architecture-showcase">
          <div className="architecture-showcase-copy"><div className="architecture-kicker">01 / U-NET</div><h3>Reconstruct through <span>context.</span></h3><p>U-Net uses an encoder-decoder architecture with skip connections to combine contextual information and spatial details during image reconstruction.</p></div>
          <div className="architecture-visual architecture-unet"><div className="architecture-caption">ENCODER <span aria-hidden="true" /> BOTTLENECK <span aria-hidden="true" /> DECODER</div><div className="s-architecture-flow" aria-label="U-Net five-stage architecture"><div className="s-architecture-stage s-stage-input"><span className="s-stage-media"><Image src="/images/hero/hero-cloudy.png" alt="Cloudy satellite input" fill sizes="120px" /></span><b>CLOUDY<br />INPUT</b></div><i className="s-flow-arrow s-arrow-one" aria-hidden="true" /><div className="s-architecture-stage s-stage-encoder"><span className="s-stage-media s-feature-map"><Image src="/images/hero/hero-cloudy.png" alt="Encoder feature map" fill sizes="120px" /></span><b>ENCODER</b></div><i className="s-flow-arrow s-arrow-two" aria-hidden="true" /><div className="s-architecture-stage s-stage-bottleneck"><span className="s-stage-media s-feature-map"><Image src="/images/hero/hero-cloudy.png" alt="Deep bottleneck feature representation" fill sizes="120px" /></span><b>BOTTLENECK</b></div><i className="s-flow-arrow s-arrow-down" aria-hidden="true" /><div className="s-architecture-stage s-stage-output"><span className="s-stage-media"><Image src="/images/hero/hero-clean.png" alt="Clean satellite output" fill sizes="120px" /></span><b>CLEAN<br />OUTPUT</b></div><i className="s-flow-arrow s-arrow-back" aria-hidden="true" /><div className="s-architecture-stage s-stage-decoder"><span className="s-stage-media s-feature-map"><Image src="/images/hero/hero-clean.png" alt="Decoder feature map" fill sizes="120px" /></span><b>DECODER</b></div><span className="s-flow-note">SIMPLE ENCODER-DECODER FLOW</span></div></div></article>
        <article className="architecture-showcase architecture-showcase-attention">
          <div className="architecture-showcase-copy"><div className="architecture-kicker">02 / ATTENTION U-NET</div><h3>Focus on what <span>matters.</span></h3><p>Attention U-Net incorporates attention gates to help the network emphasize relevant spatial features during reconstruction.</p></div>
          <div className="architecture-visual architecture-attention"><div className="architecture-caption">ATTENTION-GUIDED FEATURE SELECTION</div><div className="s-architecture-flow" aria-label="Attention U-Net six-stage architecture"><div className="s-architecture-stage s-stage-input"><span className="s-stage-media"><Image src="/images/hero/hero-cloudy.png" alt="Cloudy satellite input" fill sizes="120px" /></span><b>CLOUDY<br />INPUT</b></div><i className="s-flow-arrow s-arrow-one" aria-hidden="true" /><div className="s-architecture-stage s-stage-encoder"><span className="s-stage-media s-feature-map"><Image src="/images/hero/hero-cloudy.png" alt="Encoder feature map" fill sizes="120px" /></span><b>ENCODER</b></div><i className="s-flow-arrow s-arrow-two" aria-hidden="true" /><div className="s-architecture-stage s-stage-gate"><span className="s-stage-media s-feature-map"><Image src="/images/hero/hero-cloudy.png" alt="Attention feature selection map" fill sizes="120px" /></span><b>ATTENTION<br />GATE</b></div><i className="s-flow-arrow s-arrow-down" aria-hidden="true" /><div className="s-architecture-stage s-stage-output"><span className="s-stage-media"><Image src="/images/hero/hero-clean.png" alt="Clean satellite output" fill sizes="120px" /></span><b>CLEAN<br />OUTPUT</b></div><i className="s-flow-arrow s-arrow-back" aria-hidden="true" /><div className="s-architecture-stage s-stage-decoder"><span className="s-stage-media s-feature-map"><Image src="/images/hero/hero-clean.png" alt="Decoder feature map" fill sizes="120px" /></span><b>DECODER</b></div><i className="s-flow-arrow s-arrow-bottleneck-back" aria-hidden="true" /><div className="s-architecture-stage s-stage-bottleneck"><span className="s-stage-media s-feature-map"><Image src="/images/hero/hero-cloudy.png" alt="Deep bottleneck feature representation" fill sizes="120px" /></span><b>BOTTLENECK</b></div><span className="s-flow-note">ATTENTION-GUIDED ENCODER-DECODER FLOW</span></div></div></article>
      </div>
      <div className="model-comparison"><div className="comparison-title">TWO ARCHITECTURES. ONE RESEARCH GOAL.</div><div className="comparison-grid"><span>DIMENSION</span><b>U-NET</b><b>ATTENTION U-NET</b><span>Architecture</span><span>Encoder-decoder</span><span>Attention-enhanced encoder-decoder</span><span>Feature handling</span><span>Skip connections</span><span>Attention-guided skip connections</span><span>Main focus</span><span>Spatial reconstruction</span><span>Relevant feature selection</span></div></div>
      <div className="research-roadmap-preview"><div><div className="comparison-title">BEYOND CURRENT ARCHITECTURES</div><p>Planned research directions, not currently implemented capabilities.</p></div><div className="roadmap-preview-items"><span><b>01</b> Vision Transformer Feature Extraction</span><span><b>02</b> Cloud Severity Estimation</span><span><b>03</b> Severity-Aware Diffusion Reconstruction</span></div><em>RESEARCH ROADMAP</em></div>
    </motion.section>

    <motion.section className="research-section container" id="research" {...motionProps}><div className="research-copy"><div className="eyebrow">03 / RESEARCH IDENTITY</div><h2>Reconstructing<br /><span>what clouds conceal.</span></h2><p>Cloud coverage can obscure critical surface information in optical satellite imagery. GenVision investigates deep-learning reconstruction methods while recognizing that heavily obscured regions may require inferred information rather than direct observation.</p><div className="research-note"><Crosshair size={16} /><span>Reconstructed imagery should be evaluated carefully and should not be treated as a substitute for verified ground observations.</span></div></div><div className="research-visual"><div className="research-grid" /><div className="research-crosshair"><Crosshair size={28} /></div><div className="research-label label-top">OBSERVATION / INFERENCE</div><div className="research-label label-bottom">RESEARCH PROTOTYPE · 2026</div><div className="research-line line-one" /><div className="research-line line-two" /></div></motion.section>

    <motion.section className="cta container cta-premium" {...motionProps}><div className="cta-inner"><div><div className="eyebrow">START EXPLORING</div><h2>Ready to explore<br /><span>GenVision?</span></h2><p>Experience the reconstruction workflow and explore the next generation of satellite imagery research.</p></div><div className="cta-actions"><Link href="/studio" className="button button-primary button-large">Open GenVision Studio <ArrowRight size={17} /></Link><Link href="/models" className="button button-ghost button-large">Explore Models <ArrowRight size={16} /></Link></div></div></motion.section>

    <footer className="footer footer-premium container"><div className="footer-brand"><Link href="/" className="brand"><span className="brand-mark"><Image src="/genvision.png" alt="GenVision" width={27} height={27} /></span><span>Gen<span>Vision</span></span></Link><p>AI-powered satellite image reconstruction research platform.</p></div><nav className="footer-nav" aria-label="Footer navigation"><Link href="/">Home</Link><Link href="/studio">Studio</Link><Link href="/dashboard">Dashboard</Link><Link href="/models">Models</Link><Link href="/agriculture">Agriculture</Link></nav><div className="footer-bottom"><span>© 2026 GenVision. Research Prototype.</span><span className="mono">DEMO MODE / NO PRODUCTION CLAIMS</span></div></footer>
  </main></MotionConfig>;
}
