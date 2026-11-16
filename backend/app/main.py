from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from .agriculture_service import AgricultureService
from .model_service import (
    ACCEPTED_CONTENT_TYPES,
    DEFAULT_MODEL_ID,
    MAX_FILE_SIZE,
    MODEL_CONFIGS,
    ModelService,
)
from .schemas import (
    AgricultureHealthResponse,
    AgricultureIndicesResponse,
    AgricultureMasksResponse,
    AgricultureSummaryResponse,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    VisualizationsListResponse,
)

SUPPORTED_MODEL_IDS = list(MODEL_CONFIGS.keys())


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize ML Inference Model Service
    service = ModelService()
    service.load()
    app.state.model_service = service

    # Initialize Agriculture Service
    agri_service = AgricultureService()
    app.state.agriculture_service = agri_service

    yield

    app.state.model_service = None
    app.state.agriculture_service = None


app = FastAPI(
    title="GenVision Cloud Removal & Agriculture API",
    version="3.3.0",
    description="Multi-model PyTorch cloud reconstruction and genuine Sentinel-2 agricultural analytics API.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=[
        "X-GenVision-Model",
        "X-GenVision-Model-Id",
        "X-Inference-Time-Ms",
        "X-Output-Type",
        "X-GenVision-Study-Area",
        "X-GenVision-Product",
    ],
)


def service_for(request: Request) -> ModelService:
    service = getattr(request.app.state, "model_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Inference service is unavailable.")
    return service


def agriculture_service_for(request: Request) -> AgricultureService:
    service = getattr(request.app.state, "agriculture_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Agriculture service is unavailable.")
    return service


# ============================================================================
# Existing Model & Inference Endpoints (Preserved)
# ============================================================================


@app.get("/health", response_model=HealthResponse, tags=["Inference"])
def health(request: Request) -> HealthResponse:
    service = service_for(request)
    loaded_ids = list(service.models.keys())
    return HealthResponse(
        status="ok" if service.loaded else "degraded",
        model_loaded=service.loaded,
        model_name=", ".join(MODEL_CONFIGS[m_id]["name"] for m_id in loaded_ids) or service.default_model_name,
        available_models=loaded_ids,
    )


@app.get("/models", response_model=ModelsResponse, tags=["Inference"])
def models(request: Request) -> ModelsResponse:
    service = service_for(request)
    return ModelsResponse(models=[ModelInfo(**item) for item in service.get_all_metadata()])


@app.get("/metrics", tags=["Evaluation"])
def metrics(request: Request) -> dict:
    service = service_for(request)
    try:
        return service.get_metrics()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail="Evaluation metrics are currently unavailable.") from error


@app.post("/predict", response_class=Response, tags=["Inference"])
async def predict(
    request: Request,
    model: str = Query(default=DEFAULT_MODEL_ID, description="Target model ID ('unet' or 'attention_unet')"),
    file: UploadFile = File(...),
) -> Response:
    service = service_for(request)
    model_id = (model or DEFAULT_MODEL_ID).strip().lower()

    if model_id not in MODEL_CONFIGS:
        available = ", ".join(SUPPORTED_MODEL_IDS)
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model ID '{model}'. Available models: {available}.",
        )

    if model_id not in service.models:
        reason = service.load_errors.get(model_id, "Model failed to load.")
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model_id}' is unavailable: {reason}",
        )

    filename = (file.filename or "").lower()
    if file.content_type not in ACCEPTED_CONTENT_TYPES or (
        filename and not any(filename.endswith(ext) for ext in {".png", ".jpg", ".jpeg"})
    ):
        raise HTTPException(status_code=415, detail="Only PNG and JPEG images are supported.")

    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit() and int(content_length) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Uploaded file exceeds the 12 MB limit.")

    content = await file.read(MAX_FILE_SIZE + 1)
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Uploaded file exceeds the 12 MB limit.")
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        output, elapsed_ms, model_name = service.predict(content, model_id=model_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return Response(
        content=output,
        media_type="image/png",
        headers={
            "X-GenVision-Model": model_name,
            "X-GenVision-Model-Id": model_id,
            "X-Inference-Time-Ms": f"{elapsed_ms:.2f}",
            "X-Output-Type": "genuine-pytorch-inference",
        },
    )


# ============================================================================
# Agriculture API Endpoints (Phase 3.3)
# ============================================================================


@app.get("/agriculture/health", response_model=AgricultureHealthResponse, tags=["Agriculture"])
def agriculture_health(request: Request) -> AgricultureHealthResponse:
    service = agriculture_service_for(request)
    return service.check_health()


@app.get("/agriculture/summary", response_model=AgricultureSummaryResponse, tags=["Agriculture"])
def agriculture_summary(request: Request) -> AgricultureSummaryResponse:
    service = agriculture_service_for(request)
    try:
        return service.get_summary()
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=f"Agriculture summary data is unavailable: {error}",
        ) from error


@app.get("/agriculture/indices", response_model=AgricultureIndicesResponse, tags=["Agriculture"])
def agriculture_indices(request: Request) -> AgricultureIndicesResponse:
    service = agriculture_service_for(request)
    try:
        return service.get_indices()
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=f"Agriculture spectral index data is unavailable: {error}",
        ) from error


@app.get("/agriculture/masks", response_model=AgricultureMasksResponse, tags=["Agriculture"])
def agriculture_masks(request: Request) -> AgricultureMasksResponse:
    service = agriculture_service_for(request)
    try:
        return service.get_masks()
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=f"Agriculture mask data is unavailable: {error}",
        ) from error


@app.get("/agriculture/visualizations", response_model=VisualizationsListResponse, tags=["Agriculture"])
def agriculture_visualizations(request: Request) -> VisualizationsListResponse:
    service = agriculture_service_for(request)
    return service.get_visualizations_list()


@app.get("/agriculture/visualizations/{name}", response_class=FileResponse, tags=["Agriculture"])
def get_agriculture_visualization(name: str, request: Request) -> FileResponse:
    service = agriculture_service_for(request)
    try:
        image_path = service.get_visualization_path(name)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error

    return FileResponse(
        path=str(image_path),
        media_type="image/png",
        filename=image_path.name,
        headers={
            "X-GenVision-Study-Area": "Pulivendula, Andhra Pradesh, India",
            "X-GenVision-Product": "Sentinel-2C Level-2A",
            "Cache-Control": "public, max-age=3600",
        },
    )
