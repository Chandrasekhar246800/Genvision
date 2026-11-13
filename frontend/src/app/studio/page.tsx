"use client";

import { ChangeEvent, DragEvent, useEffect, useRef, useState } from "react";
import { AlertCircle, Check, Download, FileImage, Fullscreen, Info, Play, RotateCcw, Upload, X, ZoomIn, ZoomOut } from "lucide-react";
import { AppShell, Card } from "../../components/app-shell";
import { getBackendHealth, reconstructImage } from "../../lib/api";

type Dimensions = { width: number; height: number };
const MAX_FILE_SIZE = 12 * 1024 * 1024;
const ACCEPTED_TYPES = ["image/png", "image/jpeg", "image/jpg", "image/tiff"];

export default function Studio() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [resultPreview, setResultPreview] = useState("");
  const [dimensions, setDimensions] = useState<Dimensions | null>(null);
  const [selected] = useState("U-Net");
  const [processing, setProcessing] = useState(false);
  const [done, setDone] = useState(false);
  const [error, setError] = useState("");
  const [inferenceTimeMs, setInferenceTimeMs] = useState<number | null>(null);
  const [resultModel, setResultModel] = useState("");
  const [backendStatus, setBackendStatus] = useState<"checking" | "connected" | "unavailable">("checking");
  const [dragActive, setDragActive] = useState(false);
  const [slider, setSlider] = useState(50);
  const [zoom, setZoom] = useState(1);
  const input = useRef<HTMLInputElement>(null);
  const comparisonFrame = useRef<HTMLDivElement>(null);
  const previewUrl = useRef("");
  const resultUrl = useRef("");

  useEffect(() => {
    void getBackendHealth()
      .then(health => setBackendStatus(health.status === "ok" && health.model_loaded ? "connected" : "unavailable"))
      .catch(() => setBackendStatus("unavailable"));
    return () => {
      if (previewUrl.current) URL.revokeObjectURL(previewUrl.current);
      if (resultUrl.current) URL.revokeObjectURL(resultUrl.current);
    };
  }, []);

  function clearPreviewUrl() {
    if (previewUrl.current) {
      URL.revokeObjectURL(previewUrl.current);
      previewUrl.current = "";
    }
  }

  function clearResultUrl() {
    if (resultUrl.current) {
      URL.revokeObjectURL(resultUrl.current);
      resultUrl.current = "";
    }
    setResultPreview("");
    setInferenceTimeMs(null);
    setResultModel("");
  }

  async function choose(next: File) {
    setError("");
    setDone(false);
    clearResultUrl();
    if (!ACCEPTED_TYPES.includes(next.type) && !/\.(png|jpe?g|tiff?)$/i.test(next.name)) {
      setError("Please choose a PNG, JPG, JPEG, or TIFF image.");
      return;
    }
    if (next.size > MAX_FILE_SIZE) {
      setError("Images must be smaller than 12 MB.");
      return;
    }
    const url = URL.createObjectURL(next);
    const image = new Image();
    image.onload = () => {
      clearPreviewUrl();
      previewUrl.current = url;
      setFile(next);
      setPreview(url);
      setDimensions({ width: image.naturalWidth, height: image.naturalHeight });
      setSlider(50);
      setZoom(1);
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      setError("This image cannot be decoded by your browser. TIFF preview depends on browser support.");
    };
    image.src = url;
  }

  function onDrop(event: DragEvent) {
    event.preventDefault();
    setDragActive(false);
    const dropped = event.dataTransfer.files[0];
    if (dropped) void choose(dropped);
  }

  function reset() {
    clearPreviewUrl();
    clearResultUrl();
    setFile(null);
    setPreview("");
    setDimensions(null);
    setProcessing(false);
    setDone(false);
    setError("");
    setSlider(50);
    setZoom(1);
    if (input.current) input.current.value = "";
  }

  async function start() {
    if (!file || processing || backendStatus !== "connected") return;
    setProcessing(true);
    setDone(false);
    setError("");
    clearResultUrl();
    try {
      const result = await reconstructImage(file);
      const url = URL.createObjectURL(result.blob);
      resultUrl.current = url;
      setResultPreview(url);
      setInferenceTimeMs(result.inferenceTimeMs);
      setResultModel(result.modelName);
      setDone(true);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "The image reconstruction failed.");
    } finally {
      setProcessing(false);
    }
  }

  function onFile(event: ChangeEvent<HTMLInputElement>) {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) void choose(selectedFile);
  }

  function downloadResult() {
    if (!resultPreview) return;
    const anchor = document.createElement("a");
    anchor.href = resultPreview;
    anchor.download = "genvision_reconstructed.png";
    anchor.click();
  }

  function updateSliderFromPointer(clientX: number) {
    const frame = comparisonFrame.current;
    if (!frame) return;
    const bounds = frame.getBoundingClientRect();
    const next = ((clientX - bounds.left) / bounds.width) * 100;
    setSlider(Math.min(90, Math.max(10, next)));
  }

  function adjustSlider(amount: number) {
    setSlider(value => Math.min(90, Math.max(10, value + amount)));
  }

  async function toggleFullscreen() {
    const viewer = comparisonFrame.current?.parentElement;
    if (!viewer) return;
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await viewer.requestFullscreen();
    } catch {
      setError("Fullscreen viewing is not available in this browser.");
    }
  }

  const stages = ["Image validation", "Preprocessing", "Model execution", "Result generation"];
  const activeStage = processing ? 2 : done ? 3 : file ? 1 : 0;

  return <AppShell title="Cloud Removal Studio" description="Reconstruct cloud-obscured satellite imagery using deep-learning models.">
    <div className="studio-layout studio-premium">
      <section className="studio-main">
        <Card className="upload-card studio-upload-card"><div className="card-heading"><div><span className="card-kicker">01 / INPUT SCENE</span><h2>Upload satellite image</h2></div><div className="studio-heading-actions"><span className="studio-prototype-badge"><i /> PROTOTYPE</span>{file && <button className="studio-reset-link" onClick={reset}><RotateCcw size={13} /> New reconstruction</button>}</div></div>
          {preview ? <div className="preview-wrap studio-preview"><img src={preview} alt="Uploaded satellite scene" /><div className="preview-overlay"><span><Check size={14} /> Original image ready</span><button onClick={reset} aria-label="Remove image"><X size={15} /></button></div></div> : <label className={`dropzone studio-dropzone ${dragActive ? "drag-active" : ""}`} onDrop={onDrop} onDragEnter={event => { event.preventDefault(); setDragActive(true); }} onDragOver={event => event.preventDefault()} onDragLeave={event => { if (event.currentTarget === event.target) setDragActive(false); }}><input ref={input} type="file" accept=".png,.jpg,.jpeg,.tif,.tiff,image/png,image/jpeg,image/tiff" hidden onChange={onFile} /><span className="upload-icon"><Upload size={22} /></span><strong>{dragActive ? "Release to inspect this scene" : "Upload Satellite Image"}</strong><span>Drag and drop your scene here, or <u>browse files</u></span><small>PNG · JPG · JPEG · TIFF where browser decoding is supported · Max 12 MB</small></label>}
          {error && <div className="error-message"><AlertCircle size={15} />{error}</div>}
          {file && <div className="file-detail studio-file-detail"><FileImage size={16} /><span>{file.name}</span><small>{dimensions ? `${dimensions.width} × ${dimensions.height}px · ` : ""}{(file.size / 1024 / 1024).toFixed(2)} MB · {file.type.split("/").pop()?.toUpperCase() || "IMAGE"}</small><button onClick={reset} aria-label="Remove uploaded image"><X size={14} /></button></div>}
        </Card>
        <Card className="aside-card studio-comparison-card"><div className="card-heading"><div><span className="card-kicker">02 / IMAGE VIEWER</span><h2>Original versus reconstruction</h2></div><div className="comparison-tools"><button className="icon-button" onClick={() => setZoom(value => Math.max(1, value - .25))} disabled={zoom === 1} aria-label="Zoom out"><ZoomOut size={15} /></button><button className="icon-button" onClick={() => setZoom(value => Math.min(2, value + .25))} disabled={zoom === 2} aria-label="Zoom in"><ZoomIn size={15} /></button><button className="icon-button" onClick={() => setZoom(1)} disabled={zoom === 1} aria-label="Fit image"><RotateCcw size={15} /></button><button className="icon-button" onClick={toggleFullscreen} disabled={!preview} aria-label="View image fullscreen"><Fullscreen size={15} /></button></div></div>
          <div className={`comparison studio-comparison ${preview ? "has-image" : ""}`}>{preview ? <div ref={comparisonFrame} className="comparison-stage studio-image-frame" style={{ transform: `scale(${zoom})` }} onPointerDown={event => { event.currentTarget.setPointerCapture(event.pointerId); updateSliderFromPointer(event.clientX); }} onPointerMove={event => { if (event.currentTarget.hasPointerCapture(event.pointerId)) updateSliderFromPointer(event.clientX); }} onPointerUp={event => { if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId); }}><img src={preview} alt="Original uploaded satellite scene" /><div className="demo-pane" style={{ clipPath: `inset(0 0 0 ${slider}%)` }}>{resultPreview ? <img src={resultPreview} alt="Genuine reconstructed satellite scene" /> : <div className="empty-comparison"><FileImage size={25} /><strong>{processing ? "Reconstructing image…" : "Reconstruction result will appear here"}</strong><span>{processing ? "Waiting for the FastAPI model response" : "Run reconstruction to compare outputs"}</span></div>}</div><div className="studio-image-tint" /><span className="compare-label original-label">ORIGINAL</span><span className="compare-label demo-result-label">{resultPreview ? "RECONSTRUCTED OUTPUT" : "OUTPUT PENDING"}</span><div className="compare-divider" style={{ left: `${slider}%` }} onKeyDown={event => { if (event.key === "ArrowLeft") { event.preventDefault(); adjustSlider(-2); } if (event.key === "ArrowRight") { event.preventDefault(); adjustSlider(2); } }} role="slider" tabIndex={0} aria-label="Image comparison position" aria-valuemin={10} aria-valuemax={90} aria-valuenow={Math.round(slider)}><span /></div></div> : <div className="empty-comparison"><FileImage size={25} /><strong>Your result will appear here</strong><span>Upload an image to begin comparison</span></div>}</div>
          {preview && <label className="comparison-slider">Compare original / reconstruction <input type="range" min="10" max="90" value={slider} onChange={event => setSlider(Number(event.target.value))} /> <span>{slider}%</span></label>}
          <div className="comparison-footer"><span>{resultPreview ? "GENUINE PYTORCH OUTPUT · ORIGINAL PRESERVED" : "RECONSTRUCTION OUTPUT REQUIRES BACKEND INFERENCE"}</span>{done && <button className="download-button" onClick={downloadResult}><Download size={14} /> Download reconstructed PNG</button>}</div>
        </Card>
      </section>
      <aside className="studio-aside">
        <Card className="studio-config-card"><div className="card-heading"><div><span className="card-kicker">03 / CONFIGURATION</span><h2>Reconstruction configuration</h2></div><span className={`demo-pill ${backendStatus === "connected" ? "connected-pill" : ""}`}><i /> {backendStatus === "connected" ? "BACKEND CONNECTED" : backendStatus === "checking" ? "CHECKING BACKEND" : "BACKEND UNAVAILABLE"}</span></div><div className="studio-model-list"><button className="model-option studio-model-option selected" type="button" aria-pressed="true"><span className="studio-radio checked" /><div><strong>U-Net</strong><p>Verified encoder-decoder checkpoint for genuine reconstruction.</p><span><i /> Controlled best checkpoint</span></div></button></div><div className="studio-config-meta"><div><span>Input image</span><strong>{file ? file.name : "Waiting for upload"}</strong></div><div><span>Mode</span><strong>FastAPI inference</strong></div><div><span>Inference backend</span><strong className={backendStatus === "connected" ? "" : "studio-warning"}>{backendStatus === "connected" ? "Connected · U-Net ready" : backendStatus === "checking" ? "Checking connection…" : "Unavailable"}</strong></div></div></Card>
        <Card className="process-card studio-process-card"><div className="process-row"><div><span className="card-kicker">04 / RECONSTRUCTION</span><h2>Run reconstruction</h2><p>{processing ? "Sending image to the FastAPI U-Net inference service…" : done ? `Genuine PyTorch inference complete${inferenceTimeMs !== null ? ` in ${inferenceTimeMs.toFixed(2)} ms` : ""}.` : backendStatus === "connected" ? "The selected U-Net checkpoint will generate the reconstructed output." : "Connect the FastAPI backend to enable genuine inference."}</p></div><button className="button button-primary process-button" disabled={!file || processing || backendStatus !== "connected"} onClick={() => void start()}>{processing ? "Reconstructing…" : done ? <><Check size={15} /> Complete</> : <><Play size={14} fill="currentColor" /> Reconstruct</>}</button></div><div className="workflow-stages">{stages.map((stage, index) => <div className={`workflow-stage ${index < activeStage ? "complete" : ""} ${index === activeStage ? "active" : ""}`} key={stage}><span>{index < activeStage ? <Check size={11} /> : `0${index + 1}`}</span><small>{stage}</small></div>)}</div>{processing && <div className="progress-wrap"><div className="progress-label"><span>Reconstructing image…</span><span>In progress</span></div><div className="progress-track"><span className="progress-indeterminate" /></div></div>}</Card>
        <Card className="info-card studio-results-card"><div className="card-heading"><div><span className="card-kicker">05 / RESULT DETAILS</span><h2>Image information</h2></div>{done && <span className="studio-complete-badge"><Check size={13} /> Ready</span>}</div><div className="detail-list studio-result-details"><div><span>Architecture</span><strong>{resultModel || selected}</strong></div><div><span>Filename</span><strong>{file?.name || "—"}</strong></div><div><span>Dimensions</span><strong>{dimensions ? `${dimensions.width} × ${dimensions.height}` : "—"}</strong></div><div><span>File size / format</span><strong>{file ? `${(file.size / 1024 / 1024).toFixed(2)} MB · ${file.type.split("/").pop()?.toUpperCase() || "IMAGE"}` : "—"}</strong></div><div><span>Inference time</span><strong>{inferenceTimeMs !== null ? `${inferenceTimeMs.toFixed(2)} ms` : "—"}</strong></div></div><div className="notice"><Info size={14} /><span>Output is generated by the selected PyTorch checkpoint. Evaluation metrics remain unavailable without a paired reference in this workflow.</span></div></Card>
      </aside>
    </div>
  </AppShell>;
}
