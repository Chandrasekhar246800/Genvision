from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .model_service import (
    ACCEPTED_CONTENT_TYPES,
    MAX_FILE_SIZE,
    MODEL_NAME,
    ModelService,
)
from .schemas import HealthResponse, ModelInfo, ModelsResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = ModelService()
    service.load()
    app.state.model_service = service
    yield
    app.state.model_service = None


app = FastAPI(
    title="GenVision Cloud Removal API",
    version="1.0.0",
    description="Genuine U-Net inference for cloud-obscured RGB satellite imagery.",
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
        "X-Inference-Time-Ms",
        "X-Output-Type",
    ],
)


def service_for(request: Request) -> ModelService:
    service = getattr(request.app.state, "model_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Inference service is unavailable.")
    return service


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    service = service_for(request)
    return HealthResponse(
        status="ok" if service.loaded else "degraded",
        model_loaded=service.loaded,
        model_name=MODEL_NAME,
    )


@app.get("/models", response_model=ModelsResponse)
def models(request: Request) -> ModelsResponse:
    service = service_for(request)
    return ModelsResponse(models=[ModelInfo(**service.metadata())])


@app.post("/predict", response_class=Response)
async def predict(request: Request, file: UploadFile = File(...)) -> Response:
    service = service_for(request)
    if service.load_error is not None:
        raise HTTPException(status_code=503, detail="Inference model is unavailable.")
    if file.content_type not in ACCEPTED_CONTENT_TYPES:
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
        output, elapsed_ms = service.predict(content)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return Response(
        content=output,
        media_type="image/png",
        headers={
            "X-GenVision-Model": MODEL_NAME,
            "X-Inference-Time-Ms": f"{elapsed_ms:.2f}",
            "X-Output-Type": "genuine-pytorch-inference",
        },
    )
