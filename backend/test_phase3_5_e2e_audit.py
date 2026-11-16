"""
GenVision Phase 3.5 Complete End-to-End Audit & Performance Benchmark Suite
Executes comprehensive testing across all backend endpoints, ML models,
agricultural data integrity, security boundaries, and latency profiling.
"""

import io
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from PIL import Image

import uvicorn
from app.main import app

PORT = 8092
BASE_URL = f"http://127.0.0.1:{PORT}"


def start_server():
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


def create_test_image(size=(256, 256), color=(73, 109, 137), fmt="PNG") -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def send_multipart_request(url: str, filename: str, content: bytes, content_type: str = "image/png"):
    boundary = "----WebKitFormBoundaryE2EAuditSuite"
    body = io.BytesIO()
    body.write(f"--{boundary}\r\n".encode("utf-8"))
    body.write(
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode("utf-8")
    )
    body.write(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
    body.write(content)
    body.write(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    body_bytes = body.getvalue()

    req = urllib.request.Request(
        url,
        data=body_bytes,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body_bytes)),
        },
        method="POST",
    )
    return req


def timed_get(url: str, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        data = resp.read()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return resp.status, dict(resp.headers), data, elapsed_ms


def timed_post_multipart(url: str, filename: str, content: bytes, content_type: str = "image/png"):
    req = send_multipart_request(url, filename, content, content_type)
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        data = resp.read()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return resp.status, dict(resp.headers), data, elapsed_ms


def run_full_audit():
    print("=" * 80)
    print("=== GENVISION PHASE 3.5: COMPLETE SYSTEM INTEGRATION AUDIT & BENCHMARK ===")
    print("=" * 80)

    audit_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backend": {},
        "models": {},
        "agriculture": {},
        "security": {},
        "performance": {},
    }

    png_magic = b"\x89PNG\r\n\x1a\n"

    # -------------------------------------------------------------------------
    # SECTION 1: Backend System & Model Registry Endpoints
    # -------------------------------------------------------------------------
    print("\n[SECTION 1: Core ML Backend Endpoints]")

    # 1.1: GET /health
    status, headers, data, elapsed = timed_get(f"{BASE_URL}/health")
    assert status == 200
    health = json.loads(data.decode())
    assert health["status"] == "ok"
    assert health["model_loaded"] is True
    assert set(health["available_models"]) == {"unet", "attention_unet"}
    print(f"  1.1 GET /health: 200 OK | {elapsed:.2f} ms | Models: {health['available_models']}")
    audit_results["performance"]["get_health_ms"] = round(elapsed, 2)

    # 1.2: GET /models
    status, headers, data, elapsed = timed_get(f"{BASE_URL}/models")
    assert status == 200
    models_resp = json.loads(data.decode())["models"]
    assert len(models_resp) == 2
    for m in models_resp:
        assert m["verified"] is True
        assert m["checkpoint_size_bytes"] > 0
        print(f"  1.2 GET /models: {m['id']} ({m['name']}) -> {m['checkpoint_size_bytes']:,} bytes | Loaded={m['loaded']}")
    audit_results["performance"]["get_models_ms"] = round(elapsed, 2)

    # 1.3: GET /metrics
    status, headers, data, elapsed = timed_get(f"{BASE_URL}/metrics")
    assert status == 200
    metrics_resp = json.loads(data.decode())
    test_count = metrics_resp["evaluation"]["test_count"]
    unet_psnr = metrics_resp["models"]["unet"]["psnr_mean"]
    attn_psnr = metrics_resp["models"]["attention_unet"]["psnr_mean"]
    assert test_count == 75
    assert unet_psnr > 25.0
    assert attn_psnr > 20.0
    print(f"  1.3 GET /metrics: 200 OK | {elapsed:.2f} ms | Evaluated: {test_count} paired scenes")
    print(f"      U-Net PSNR: {unet_psnr:.4f} dB, SSIM: {metrics_resp['models']['unet']['ssim_mean']:.4f}")
    print(f"      Attention U-Net PSNR: {attn_psnr:.4f} dB, SSIM: {metrics_resp['models']['attention_unet']['ssim_mean']:.4f}")
    audit_results["performance"]["get_metrics_ms"] = round(elapsed, 2)

    # -------------------------------------------------------------------------
    # SECTION 2: Machine Learning Inference & Checkpoint Verification
    # -------------------------------------------------------------------------
    print("\n[SECTION 2: PyTorch Model Inference & Timing]")
    test_img = create_test_image((256, 256), color=(60, 120, 180), fmt="PNG")

    # 2.1: U-Net Inference
    unet_runs = []
    for _ in range(3):
        status, headers, out_bytes, elapsed = timed_post_multipart(
            f"{BASE_URL}/predict?model=unet", "sample.png", test_img
        )
        assert status == 200
        assert out_bytes.startswith(png_magic)
        out_img = Image.open(io.BytesIO(out_bytes))
        assert out_img.size == (256, 256)
        reported_inf = float(headers.get("x-inference-time-ms", headers.get("X-Inference-Time-Ms", "0")))
        unet_runs.append((elapsed, reported_inf))

    avg_unet_wall = sum(r[0] for r in unet_runs) / len(unet_runs)
    avg_unet_inf = sum(r[1] for r in unet_runs) / len(unet_runs)
    print(f"  2.1 POST /predict?model=unet: 200 OK | Wall: {avg_unet_wall:.2f} ms | PyTorch CPU Inference: {avg_unet_inf:.2f} ms | Output: 256x256 PNG")
    audit_results["performance"]["unet_wall_ms"] = round(avg_unet_wall, 2)
    audit_results["performance"]["unet_inference_ms"] = round(avg_unet_inf, 2)

    # 2.2: Attention U-Net Inference
    attn_runs = []
    for _ in range(3):
        status, headers, out_bytes, elapsed = timed_post_multipart(
            f"{BASE_URL}/predict?model=attention_unet", "sample.png", test_img
        )
        assert status == 200
        assert out_bytes.startswith(png_magic)
        out_img = Image.open(io.BytesIO(out_bytes))
        assert out_img.size == (256, 256)
        reported_inf = float(headers.get("x-inference-time-ms", headers.get("X-Inference-Time-Ms", "0")))
        attn_runs.append((elapsed, reported_inf))

    avg_attn_wall = sum(r[0] for r in attn_runs) / len(attn_runs)
    avg_attn_inf = sum(r[1] for r in attn_runs) / len(attn_runs)
    print(f"  2.2 POST /predict?model=attention_unet: 200 OK | Wall: {avg_attn_wall:.2f} ms | PyTorch CPU Inference: {avg_attn_inf:.2f} ms | Output: 256x256 PNG")
    audit_results["performance"]["attention_unet_wall_ms"] = round(avg_attn_wall, 2)
    audit_results["performance"]["attention_unet_inference_ms"] = round(avg_attn_inf, 2)

    # 2.3: JPEG Input Acceptance Test
    jpeg_img = create_test_image((256, 256), color=(80, 140, 100), fmt="JPEG")
    status, headers, out_bytes, elapsed = timed_post_multipart(
        f"{BASE_URL}/predict?model=unet", "sample.jpg", jpeg_img, content_type="image/jpeg"
    )
    assert status == 200
    assert out_bytes.startswith(png_magic)
    print(f"  2.3 POST /predict with JPEG input: 200 OK | Output converted & reconstructed as PNG")

    # -------------------------------------------------------------------------
    # SECTION 3: Agriculture API & Remote Sensing Scientific Audit
    # -------------------------------------------------------------------------
    print("\n[SECTION 3: Agriculture API & Scientific Data Audit]")

    # 3.1: GET /agriculture/health
    status, headers, data, elapsed = timed_get(f"{BASE_URL}/agriculture/health")
    assert status == 200
    agri_health = json.loads(data.decode())
    assert agri_health["status"] == "ok"
    assert agri_health["dataset_available"] is True
    assert agri_health["statistics_available"] is True
    assert agri_health["indices_available"] is True
    assert agri_health["masks_available"] is True
    assert agri_health["visualizations_available"] is True
    assert agri_health["available_visualizations_count"] == 8
    print(f"  3.1 GET /agriculture/health: 200 OK | {elapsed:.2f} ms | File availability 100% verified")
    audit_results["performance"]["agri_health_ms"] = round(elapsed, 2)

    # 3.2: GET /agriculture/summary
    status, headers, data, elapsed = timed_get(f"{BASE_URL}/agriculture/summary")
    assert status == 200
    summary = json.loads(data.decode())
    
    # Ground truth scientific verification
    assert summary["satellite"] == "Sentinel-2C"
    assert summary["tile_id"] == "43PHS"
    assert summary["roi_dimensions"] == [1000, 1000]
    assert summary["total_pixels"] == 1000000
    assert summary["clear_pixels"] == 565112
    assert summary["clear_surface_percentage"] == 56.51
    assert summary["obstructed_pixels"] == 434888
    assert summary["combined_obstruction_percentage"] == 43.49
    assert summary["cloud_pixels"] == 303976
    assert summary["cloud_percentage"] == 30.40
    assert summary["shadow_pixels"] == 130912
    assert summary["shadow_percentage"] == 13.09
    assert summary["ndvi_mean_clear"] == 0.3780
    assert summary["ndwi_mean_clear"] == -0.3935
    assert summary["ndmi_mean_clear"] == 0.0042
    print(f"  3.2 GET /agriculture/summary: 200 OK | {elapsed:.2f} ms")
    print(f"      Satellite: {summary['satellite']} | Tile: {summary['tile_id']} | Area: 100 km² (1000x1000 px)")
    print(f"      Clear: {summary['clear_surface_percentage']}% ({summary['clear_pixels']:,} px) | Cloud: {summary['cloud_percentage']}% | Shadow: {summary['shadow_percentage']}%")
    print(f"      Mean Indices: NDVI={summary['ndvi_mean_clear']}, NDWI={summary['ndwi_mean_clear']}, NDMI={summary['ndmi_mean_clear']}")
    audit_results["performance"]["agri_summary_ms"] = round(elapsed, 2)

    # 3.3: GET /agriculture/indices
    status, headers, data, elapsed = timed_get(f"{BASE_URL}/agriculture/indices")
    assert status == 200
    idx_resp = json.loads(data.decode())["indices"]
    
    ndvi = idx_resp["ndvi"]
    assert ndvi["masked_scientifically_valid"]["valid_count"] == 565112
    assert ndvi["unmasked_all_pixels"]["valid_count"] == 1000000
    assert ndvi["masked_scientifically_valid"]["mean"] == 0.3780
    assert ndvi["unmasked_all_pixels"]["mean"] == 0.2939
    assert ndvi["cloud_impact_delta_mean"] == -0.0841

    ndwi = idx_resp["ndwi"]
    assert ndwi["masked_scientifically_valid"]["mean"] == -0.3935
    assert ndwi["unmasked_all_pixels"]["mean"] == -0.2905

    ndmi = idx_resp["ndmi"]
    assert ndmi["masked_scientifically_valid"]["mean"] == 0.0042
    assert ndmi["unmasked_all_pixels"]["mean"] == 0.0480
    print(f"  3.3 GET /agriculture/indices: 200 OK | {elapsed:.2f} ms")
    print(f"      NDVI Masked: {ndvi['masked_scientifically_valid']['mean']} vs Unmasked: {ndvi['unmasked_all_pixels']['mean']} (Δ = {ndvi['cloud_impact_delta_mean']})")
    audit_results["performance"]["agri_indices_ms"] = round(elapsed, 2)

    # 3.4: GET /agriculture/masks & SCL Class Verification
    status, headers, data, elapsed = timed_get(f"{BASE_URL}/agriculture/masks")
    assert status == 200
    masks_resp = json.loads(data.decode())
    scl_classes = masks_resp["scl_classes"]
    assert len(scl_classes) == 7
    
    # Verify Class 7 (Unclassified) is accurately tracked and NOT claimed as clear vegetation
    class_7 = next(c for c in scl_classes if c["class_id"] == 7)
    assert class_7["name"] == "Unclassified"
    assert class_7["count"] == 17588
    assert class_7["percentage"] == 1.76
    
    veg_class = next(c for c in scl_classes if c["class_id"] == 4)
    assert veg_class["name"] == "Vegetation"
    assert veg_class["count"] == 159608
    assert veg_class["percentage"] == 15.96
    print(f"  3.4 GET /agriculture/masks: 200 OK | {elapsed:.2f} ms")
    print(f"      SCL Class 4 (Vegetation): {veg_class['percentage']}% ({veg_class['count']:,} px)")
    print(f"      SCL Class 7 (Unclassified): {class_7['percentage']}% ({class_7['count']:,} px) -> Correctly isolated from clear land")
    audit_results["performance"]["agri_masks_ms"] = round(elapsed, 2)

    # 3.5: GET /agriculture/visualizations (Catalog)
    status, headers, data, elapsed = timed_get(f"{BASE_URL}/agriculture/visualizations")
    assert status == 200
    viz_list = json.loads(data.decode())["visualizations"]
    assert len(viz_list) == 8
    print(f"  3.5 GET /agriculture/visualizations: 200 OK | {elapsed:.2f} ms | Catalog: 8 products")
    audit_results["performance"]["agri_visualizations_catalog_ms"] = round(elapsed, 2)

    # 3.6: Streaming all 8 High-Resolution Visualizations
    print("\n[SECTION 4: High-Resolution Raster Streaming Benchmark]")
    stream_benchmarks = {}
    for item in viz_list:
        v_id = item["id"]
        status, headers, img_data, elapsed = timed_get(f"{BASE_URL}/agriculture/visualizations/{v_id}")
        assert status == 200
        assert headers.get("content-type") == "image/png"
        assert img_data.startswith(png_magic)
        img = Image.open(io.BytesIO(img_data))
        stream_benchmarks[v_id] = {
            "size_bytes": len(img_data),
            "dimensions": f"{img.size[0]}x{img.size[1]}",
            "latency_ms": round(elapsed, 2),
        }
        print(f"  4.{len(stream_benchmarks)} /visualizations/{v_id:28s} | {elapsed:6.2f} ms | {len(img_data):,} bytes | {img.size[0]}x{img.size[1]}")

    audit_results["performance"]["raster_streaming"] = stream_benchmarks

    # -------------------------------------------------------------------------
    # SECTION 5: Security Boundaries & Attack Simulation
    # -------------------------------------------------------------------------
    print("\n[SECTION 5: Security & Defensive Boundaries]")

    # 5.1: Path Traversal Defenses
    traversal_payloads = [
        "../../etc/passwd",
        "..\\..\\windows\\system32\\calc.exe",
        "..%2F..%2F..%2Fetc%2Fshadow",
        "true_color.png/../../secret",
        "null%00byte",
        "nonexistent_image_product",
    ]
    for payload in traversal_payloads:
        req = urllib.request.Request(f"{BASE_URL}/agriculture/visualizations/{payload}")
        try:
            with urllib.request.urlopen(req) as resp:
                assert False, f"Expected rejection for traversal payload: {payload}"
        except urllib.error.HTTPError as err:
            assert err.code in (404, 403, 400), f"Unexpected status code {err.code} on payload {payload}"
            print(f"  ✓ Traversal '{payload:35s}' -> Blocked with HTTP {err.code}")

    # 5.2: File Upload Defense (POST /predict)
    # Empty buffer
    try:
        req = send_multipart_request(f"{BASE_URL}/predict?model=unet", "empty.png", b"")
        with urllib.request.urlopen(req) as resp:
            assert False, "Empty buffer should be rejected"
    except urllib.error.HTTPError as err:
        assert err.code == 400
        print("  ✓ Empty file upload -> Rejected with HTTP 400")

    # Corrupted / Non-image bytes
    try:
        req = send_multipart_request(f"{BASE_URL}/predict?model=unet", "fake.png", b"NOT_A_REAL_IMAGE_DATA_123456789")
        with urllib.request.urlopen(req) as resp:
            assert False, "Corrupted image should be rejected"
    except urllib.error.HTTPError as err:
        assert err.code == 400
        print("  ✓ Corrupted non-image upload -> Rejected with HTTP 400")

    # Unsupported Model ID
    try:
        req = send_multipart_request(f"{BASE_URL}/predict?model=unknown_gpt5", "test.png", test_img)
        with urllib.request.urlopen(req) as resp:
            assert False, "Unsupported model should be rejected"
    except urllib.error.HTTPError as err:
        assert err.code == 400
        print("  ✓ Unsupported model ID -> Rejected with HTTP 400")

    # Oversized File (> 12 MB)
    try:
        oversized = b"0" * (13 * 1024 * 1024)
        req = send_multipart_request(f"{BASE_URL}/predict?model=unet", "huge.png", oversized)
        with urllib.request.urlopen(req) as resp:
            assert False, "Oversized file should be rejected"
    except urllib.error.HTTPError as err:
        assert err.code == 413
        print("  ✓ Oversized upload (>12 MB) -> Rejected with HTTP 413 Payload Too Large")

    # 5.3: CORS Policy Inspection
    req = urllib.request.Request(f"{BASE_URL}/agriculture/summary", headers={"Origin": "http://localhost:3000"})
    with urllib.request.urlopen(req) as resp:
        cors_origin = resp.headers.get("access-control-allow-origin", resp.headers.get("Access-Control-Allow-Origin"))
        assert cors_origin in ("*", "http://localhost:3000")
        print(f"  ✓ CORS configured: Access-Control-Allow-Origin = {cors_origin}")

    print("\n" + "=" * 80)
    print("=== ALL END-TO-END AUDIT SUITES COMPLETED WITH ZERO ERRORS ===")
    print("=" * 80)

    # Save structured audit results
    results_path = os.path.join("results", "agriculture", "audit_benchmarks.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)
    print(f"Benchmark telemetry saved to {results_path}")


if __name__ == "__main__":
    t = threading.Thread(target=start_server, daemon=True)
    t.start()
    time.sleep(3.0)
    try:
        run_full_audit()
    except Exception as e:
        print(f"\nAUDIT FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    exit(0)
