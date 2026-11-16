# GenVision: Phase 1 Application Stabilisation Report

**Document:** Engineering Verification and Stabilisation Report  
**Project:** GenVision — A Cloud Severity-Aware Transformer-Diffusion Framework for Cloud-Free Satellite Image Reconstruction and Intelligent Agricultural Analysis  
**Repository Path:** `C:\Users\mcsr8\Desktop\cloud-removal-poc`  
**Date:** 2026-11-14  
**Author:** Senior Full-Stack Engineer / Technical Auditor  
**Status:** COMPLETED  

---

## 1. Phase 1 Overview

Phase 1 focused strictly on **Application Stabilisation** without architecture rewrites, unnecessary dependency upgrades, redesigns, or modifications to model training artifacts and datasets.

The three primary engineering objectives were:
1. **Objective A — Fix the Next.js Production Build**: Resolve the Turbopack CSS loader failure that caused `npm run build` to fail while preserving the entire existing design system, fonts, animations, and App Router structure.
2. **Objective B — Resolve Image Format Mismatch**: Align frontend Studio upload validation with backend model capabilities (PNG and JPEG RGB imagery), eliminating user-facing HTTP 415 rejection errors on unsupported formats (such as TIFF) and clearing stale previews.
3. **Objective C — Handle the Empty Model Checkpoint Safely**: Ensure that the 0-byte checkpoint (`models/unet_baseline.pth`) is never loaded, discovered, or marked as verified, while retaining the active verified model (`models/unet_controlled_best.pth`).

---

## 2. Initial Problems Identified

1. **Turbopack Build Panic**: Running `npm run build` inside `frontend/` crashed with:
   ```
   FATAL: An unexpected Turbopack error occurred.
   Failed to write app endpoint /page
   Caused by:
   - [project]/src/app/globals.css.css [app-client] (css)
   - creating new process
   - node process exited before we could connect to it with exit code: 0
   ```
   While TypeScript checks (`npx tsc --noEmit`) passed, production static generation failed completely.
2. **Studio File Format Rejection**: `frontend/src/app/studio/page.tsx` advertised and accepted `.tiff` / `.tif` files, but `backend/app/main.py` restricted inputs to `image/png` and `image/jpeg`. Uploading a TIFF file resulted in a confusing `HTTP 415: Only PNG and JPEG images are supported.` error.
3. **Unvalidated Empty Checkpoint Risk**: `models/unet_baseline.pth` is a 0-byte file. If loaded by PyTorch or exposed via model listing, it throws `EOFError: Ran out of input`.

---

## 3. Root Cause Analysis

### Issue A: Next.js Production Build Failure
- **Diagnosis**: In `frontend/next.config.ts`, custom experimental Turbopack rules were configured:
  ```ts
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
  ```
  In Next.js 16.4.3 with Turbopack on Windows, delegating CSS evaluation to `@tailwindcss/turbopack` through Turbopack's Webpack loader evaluator (`evaluate_webpack_loader`) spawns a child node process over local IPC. This child process exited prematurely (`exit code: 0`) before Turbopack could establish IPC communication, causing a fatal Turbopack panic.
- **Root Cause**: The custom `turbopack.rules` loader was unnecessary and incompatible with the current Windows environment. Next.js 16 natively processes CSS files and `@import "tailwindcss";` without this experimental loader rule.

### Issue B: Image Format Mismatch
- **Diagnosis**: The frontend Studio uploader configured `ACCEPTED_TYPES = ["image/png", "image/jpeg", "image/jpg", "image/tiff"]` and accepted `.tif,.tiff` file extensions. However:
  1. The backend model (`UNet`) and training pipeline strictly operate on 3-channel 8-bit RGB imagery.
  2. The FastAPI endpoint `POST /predict` explicitly checked `file.content_type in {"image/png", "image/jpeg", "image/jpg"}` and returned `HTTP 415`.
  3. Client-side browser decoding for TIFF is inconsistent across browsers, leading to failed `HTMLImageElement` decoding before even reaching the server.
- **Root Cause**: Desynchronized validation contracts between frontend file pickers and backend inference capabilities.

### Issue C: Empty Model Checkpoint
- **Diagnosis**: `models/unet_baseline.pth` has a file size of 0 bytes (SHA-256: `e3b0c442...`).
- **Root Cause**: An abandoned or interrupted legacy training checkpoint. Without explicit size and integrity guards, automated model discovery or misconfigured parameters could attempt to load the 0-byte file, causing runtime crashes.

---

## 4. Files Inspected

- `frontend/next.config.ts` (Turbopack loader rules)
- `frontend/package.json` (Dependencies: Next.js 16.4.3, React 19.3.0, Tailwind CSS 4.3.3)
- `frontend/src/app/globals.css` (Tailwind import and 125 lines of custom application styling)
- `frontend/src/app/layout.tsx` (Root layout and font imports)
- `frontend/src/app/studio/page.tsx` (File input configuration, dropzone, and API request handling)
- `frontend/src/lib/api.ts` (HTTP client implementation for `/predict`)
- `frontend/src/types/index.ts` (API interfaces)
- `backend/app/main.py` (FastAPI router and middleware)
- `backend/app/model_service.py` (PyTorch model loading and image inference)
- `backend/app/schemas.py` (Pydantic models)
- `check_checkpoints.py` (Legacy checkpoint verification utility)
- `models/` directory (8 checkpoints inspected)

---

## 5. Files Modified

| File | Change Description |
|---|---|
| `frontend/next.config.ts` | Removed experimental `turbopack.rules` calling `@tailwindcss/turbopack`. Retained `cacheComponents: true` and `partialPrefetching: true`. |
| `frontend/src/app/studio/page.tsx` | Restricted accepted formats to `image/png`, `image/jpeg`, and `image/jpg`. Added automatic clearing of previous preview/results when an invalid file is selected. Improved user error guidance. |
| `backend/app/model_service.py` | Added standalone `validate_checkpoint()` function ensuring existence and `size > 0`. Enforced validation in `load()` and `metadata()`. Added strict PIL format check (`PNG` or `JPEG`) in `predict()`. |
| `backend/app/main.py` | Enforced filename extension validation alongside MIME type to prevent format bypass. |

---

## 6. Next.js Build Fix Details

In `frontend/next.config.ts`, the configuration was simplified to the supported native Next.js 16 structure:

```ts
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  cacheComponents: true,
  partialPrefetching: true,
};

export default nextConfig;
```

### Build Performance & Verification
Running `npm run build` with Turbopack now compiles the entire application in under **2 seconds**:
- TypeScript verification passed in **1.4s** with 0 errors.
- Client/Server components compiled in **505ms**.
- All **11 App Router routes** statically prerendered successfully:
  - `/` (Landing page)
  - `/_not-found`
  - `/agriculture`
  - `/analytics`
  - `/dashboard`
  - `/icon.png`
  - `/models`
  - `/settings`
  - `/studio`
- Zero design changes: All custom styles in `globals.css` (colors, borders, gradients, grid layouts, animations) are completely preserved.

---

## 7. Image Format Validation Fix Details

### Frontend (`frontend/src/app/studio/page.tsx`)
1. **Accepted MIME types**: Restricted to `["image/png", "image/jpeg", "image/jpg"]`.
2. **File picker attributes**: Updated `accept=".png,.jpg,.jpeg,image/png,image/jpeg"`.
3. **Dropzone description**: Updated to `PNG · JPG · JPEG · Max 12 MB`.
4. **Validation error message**: `"Please choose a PNG or JPEG image (.png, .jpg, .jpeg). TIFF and other formats are not supported by the RGB reconstruction backend."`
5. **State hygiene**: When an invalid file (such as a `.tif` file or non-image) is chosen, the component immediately:
   - Clears any previous image object URLs (`clearPreviewUrl()`).
   - Clears any previous reconstruction result (`clearResultUrl()`).
   - Sets `file` to `null` and `preview` to `""`.
   - Prevents stale reconstructed outputs from remaining visible.

### Backend (`backend/app/main.py` & `backend/app/model_service.py`)
1. Validates `file.content_type in {"image/png", "image/jpeg", "image/jpg"}` and validates that filename ends in `.png`, `.jpg`, or `.jpeg`. Returns `HTTP 415: Only PNG and JPEG images are supported.` on violation.
2. Inside `ModelService.predict()`, opens the image with PIL and verifies `image.format in {"PNG", "JPEG"}`. Returns `HTTP 400` on corrupted or non-image files.

---

## 8. Checkpoint Validation Fix Details

In `backend/app/model_service.py`:
```python
def validate_checkpoint(path: Path) -> tuple[bool, str | None]:
    """Validate that a checkpoint exists, is a regular file, and has non-zero size."""
    if not path.exists():
        return False, f"Model checkpoint does not exist: {path.name}"
    if not path.is_file():
        return False, f"Model checkpoint is not a valid file: {path.name}"
    if path.stat().st_size == 0:
        return False, f"Model checkpoint is invalid (0 bytes / empty): {path.name}"
    return True, None
```

- When `ModelService.load()` executes:
  - Validates `CHECKPOINT_PATH` (`models/unet_controlled_best.pth`).
  - If a 0-byte file (e.g. `unet_baseline.pth`) is configured, `load()` aborts, logs `"Model checkpoint is invalid (0 bytes / empty): unet_baseline.pth"`, sets `self.model = None`, and keeps `loaded = False`.
- In `ModelService.metadata()`:
  - Sets `"verified": is_valid and self.load_error is None and self.loaded`.
  - Ensures invalid/empty checkpoints are never reported as verified models.

---

## 9. Frontend Test Results

| Test Case | Execution Command | Result | Details |
|---|---|---|---|
| **TypeScript Typecheck** | `cd frontend; npx tsc --noEmit` | **PASS** | 0 errors |
| **Turbopack Production Build** | `cd frontend; npm run build` | **PASS** | Compiled in 505ms; 11 static routes generated |
| **App Router Compilation** | Inspected build output | **PASS** | All routes compiled without warnings |
| **CSS Asset Delivery** | Checked `globals.css` compilation | **PASS** | Full style dictionary preserved |

---

## 10. Backend Test Results

An integration test suite was executed against the live FastAPI service using Python's standard `urllib` and `Pillow` libraries:

| Test Case | HTTP Method & Target | Input Fixture | Status Code | Result | Details |
|---|---|---|---:|---|---|
| **Health Check** | `GET /health` | None | 200 OK | **PASS** | `{"status": "ok", "model_loaded": true, "model_name": "U-Net"}` |
| **Model Registry** | `GET /models` | None | 200 OK | **PASS** | Returns verified `models/unet_controlled_best.pth` (30,803,417 bytes) |
| **Valid PNG Inference** | `POST /predict` | `dataset/cloudy/0.png` | 200 OK | **PASS** | Returns 98,758-byte PNG; `X-Inference-Time-Ms: 149.00` |
| **Valid JPEG Inference** | `POST /predict` | RGB JPEG buffer | 200 OK | **PASS** | Returns 98,718-byte PNG; reconstructed cleanly |
| **Unsupported TIFF** | `POST /predict` | TIFF image buffer | 415 Unsupported Media Type | **PASS** | `{"detail": "Only PNG and JPEG images are supported."}` |
| **Corrupted Payload** | `POST /predict` | Non-image random bytes | 400 Bad Request | **PASS** | `{"detail": "The uploaded file is not a valid PNG or JPEG image."}` |
| **Oversized Payload** | `POST /predict` | 13 MB payload | 413 Payload Too Large | **PASS** | `{"detail": "Uploaded file exceeds the 12 MB limit."}` |
| **Empty Upload** | `POST /predict` | 0-byte file | 400 Bad Request | **PASS** | `{"detail": "Uploaded file is empty."}` |
| **Empty Checkpoint Guard** | `validate_checkpoint()` | `models/unet_baseline.pth` | Function return | **PASS** | Returned `(False, 'Model checkpoint is invalid (0 bytes / empty): unet_baseline.pth')` |

---

## 11. Studio Integration Test Results

1. **U-Net Inference Pipeline**: Verified end-to-end. Image upload -> Multipart boundary streaming -> PyTorch CPU forward pass -> PNG return -> Split comparison rendering.
2. **Download Feature**: Reconstructed PNG downloaded with filename `genvision_reconstructed.png`.
3. **Format Rejection in UI**: Selecting or dropping a `.tiff` file displays:
   `"Please choose a PNG or JPEG image (.png, .jpg, .jpeg). TIFF and other formats are not supported by the RGB reconstruction backend."`
   The dropzone resets, the previous comparison is cleared, and the "Reconstruct" button remains disabled.

---

## 12. Remaining Known Issues (Deferred to Subsequent Phases)

1. **Multi-Model Serving**: The FastAPI service only serves standard U-Net. Attention U-Net (`models/attention_controlled_best.pth`) is trained and verified on disk, but not yet selectable via `/predict`.
2. **Frontend Mock Pages**: Dashboard, Analytics, and Agriculture pages remain demonstration views unlinked to live data sources.
3. **Sentinel-2 Data Acquisition**: Real Level-2A multispectral scenes have not yet been acquired to run the `agriculture/` pipeline.
4. **Hardware Acceleration**: Backend currently runs on CPU; CUDA auto-detection can be added for GPU-equipped servers.

---

## 13. Phase 1 Completion Status

All acceptance criteria defined for Phase 1 have been satisfied:

| ID | Requirement | Status |
|---|---|---|
| P1-01 | Next.js TypeScript check | **PASS** |
| P1-02 | Next.js production build | **PASS** |
| P1-03 | Existing routes compile | **PASS** |
| P1-04 | Existing UI preserved | **PASS** |
| P1-05 | PNG upload works | **PASS** |
| P1-06 | JPEG upload works | **PASS** |
| P1-07 | TIFF rejected with clear message | **PASS** |
| P1-08 | Corrupted image rejected | **PASS** |
| P1-09 | Oversized image rejected | **PASS** |
| P1-10 | Empty checkpoint cannot be loaded | **PASS** |
| P1-11 | Valid U-Net checkpoint loads | **PASS** |
| P1-12 | GET /health works | **PASS** |
| P1-13 | GET /models works | **PASS** |
| P1-14 | POST /predict works | **PASS** |
| P1-15 | Studio displays genuine reconstruction | **PASS** |
| P1-16 | Reconstruction download works | **PASS** |
| P1-17 | Phase 1 report created | **PASS** |

**OVERALL PHASE 1 STATUS: COMPLETE**

