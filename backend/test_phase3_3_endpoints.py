"""
GenVision Phase 3.3 Agriculture API Integration Test Suite
Verifies all 6 new /agriculture endpoints and validates regression on existing ML endpoints.
"""

import io
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from PIL import Image

import uvicorn
from app.main import app

PORT = 8090
BASE_URL = f"http://127.0.0.1:{PORT}"


def start_server():
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


def create_test_image_bytes() -> bytes:
    img = Image.new("RGB", (256, 256), color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def send_multipart_predict(url: str, filename: str, content: bytes, content_type: str = "image/png"):
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
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


def run_tests():
    print("=" * 70)
    print("=== STARTING GENVISION PHASE 3.3 AGRICULTURE API TEST SUITE ===")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. Agriculture Health Check
    # -------------------------------------------------------------
    print("\n--- 1. Testing GET /agriculture/health ---")
    req = urllib.request.Request(f"{BASE_URL}/agriculture/health")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        health_data = json.loads(resp.read().decode())
        print(f"Health status: {health_data['status']}")
        print(f"Study area: {health_data['study_area']}")
        print(f"Dataset available: {health_data['dataset_available']}")
        print(f"Statistics available: {health_data['statistics_available']}")
        print(f"Visualizations available: {health_data['visualizations_available']} ({health_data['available_visualizations_count']} found)")
        print(f"Indices available: {health_data['indices_available']}")
        print(f"Masks available: {health_data['masks_available']}")
        print(f"Missing files: {health_data['missing_files']}")

        assert health_data["status"] == "ok"
        assert health_data["dataset_available"] is True
        assert health_data["statistics_available"] is True
        assert health_data["indices_available"] is True
        assert health_data["masks_available"] is True
        assert health_data["visualizations_available"] is True
        assert health_data["available_visualizations_count"] == 8
        assert len(health_data["missing_files"]) == 0
        print("PASS: /agriculture/health verified completely.")

    # -------------------------------------------------------------
    # 2. Agriculture Summary Check
    # -------------------------------------------------------------
    print("\n--- 2. Testing GET /agriculture/summary ---")
    req = urllib.request.Request(f"{BASE_URL}/agriculture/summary")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        headers = dict(resp.headers)
        assert "X-GenVision-Study-Area" in headers or "x-genvision-study-area" in headers
        assert "X-GenVision-Product" in headers or "x-genvision-product" in headers
        
        summary = json.loads(resp.read().decode())
        print(f"Study area: {summary['study_area']}")
        print(f"Satellite: {summary['satellite']}, Tile: {summary['tile_id']}")
        print(f"Acquisition date: {summary['acquisition_date']}, CRS: {summary['crs']}")
        print(f"ROI dimensions: {summary['roi_dimensions']}, Size: {summary['roi_size_km']} km")
        print(f"Total pixels: {summary['total_pixels']:,}")
        print(f"Clear pixels: {summary['clear_pixels']:,} ({summary['clear_surface_percentage']}%)")
        print(f"Obstructed pixels: {summary['obstructed_pixels']:,} ({summary['combined_obstruction_percentage']}%)")
        print(f"Cloud pixels: {summary['cloud_pixels']:,} ({summary['cloud_percentage']}%)")
        print(f"Shadow pixels: {summary['shadow_pixels']:,} ({summary['shadow_percentage']}%)")
        print(f"Mean Clear-Sky NDVI: {summary['ndvi_mean_clear']:.4f}")
        print(f"Mean Clear-Sky NDWI: {summary['ndwi_mean_clear']:.4f}")
        print(f"Mean Clear-Sky NDMI: {summary['ndmi_mean_clear']:.4f}")

        assert "Pulivendula" in summary["study_area"]
        assert summary["satellite"] == "Sentinel-2C"
        assert summary["tile_id"] == "43PHS"
        assert summary["crs"] == "EPSG:32643"
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
        assert abs(summary["ndvi_mean_clear"] - 0.3780) < 0.001
        assert abs(summary["ndwi_mean_clear"] - (-0.3935)) < 0.001
        assert abs(summary["ndmi_mean_clear"] - 0.0042) < 0.001
        print("PASS: /agriculture/summary verified completely.")

    # -------------------------------------------------------------
    # 3. Agriculture Indices Check
    # -------------------------------------------------------------
    print("\n--- 3. Testing GET /agriculture/indices ---")
    req = urllib.request.Request(f"{BASE_URL}/agriculture/indices")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        indices_resp = json.loads(resp.read().decode())
        indices = indices_resp["indices"]
        print(f"Indices returned: {list(indices.keys())}")
        assert "ndvi" in indices and "ndwi" in indices and "ndmi" in indices

        ndvi = indices["ndvi"]
        ndvi_m = ndvi["masked_scientifically_valid"]
        ndvi_u = ndvi["unmasked_all_pixels"]
        print("NDVI Masked (Clear-Sky):", ndvi_m)
        print("NDVI Unmasked (Whole ROI):", ndvi_u)
        print(f"NDVI Cloud Impact Delta Mean: {ndvi['cloud_impact_delta_mean']}")

        assert ndvi_m["valid_count"] == 565112
        assert ndvi_u["valid_count"] == 1000000
        assert abs(ndvi_m["mean"] - 0.3780) < 0.001
        assert abs(ndvi_m["max"] - 0.8657) < 0.001
        assert abs(ndvi_m["min"] - (-0.1705)) < 0.001
        assert abs(ndvi_u["mean"] - 0.2939) < 0.001
        assert abs(ndvi["cloud_impact_delta_mean"] - (-0.0841)) < 0.001

        ndwi = indices["ndwi"]
        assert ndwi["masked_scientifically_valid"]["valid_count"] == 565112
        assert abs(ndwi["masked_scientifically_valid"]["mean"] - (-0.3935)) < 0.001

        ndmi = indices["ndmi"]
        assert ndmi["masked_scientifically_valid"]["valid_count"] == 565112
        assert abs(ndmi["masked_scientifically_valid"]["mean"] - 0.0042) < 0.001
        print("PASS: /agriculture/indices verified completely.")

    # -------------------------------------------------------------
    # 4. Agriculture Masks Check
    # -------------------------------------------------------------
    print("\n--- 4. Testing GET /agriculture/masks ---")
    req = urllib.request.Request(f"{BASE_URL}/agriculture/masks")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        masks_resp = json.loads(resp.read().decode())
        obstructions = masks_resp["obstruction_summary"]
        print("Obstructions metrics:")
        print(f"  Cloud: {obstructions['cloud_percent']}% ({obstructions['cloud_pixels']:,} px)")
        print(f"  Shadow: {obstructions['shadow_percent']}% ({obstructions['shadow_pixels']:,} px)")
        print(f"  Combined Obstruction: {obstructions['total_obstructed_percent']}% ({obstructions['total_obstructed_pixels']:,} px)")
        print(f"  Clear Surface: {obstructions['clear_percent']}% ({obstructions['clear_pixels']:,} px)")

        assert obstructions["cloud_pixels"] == 303976
        assert obstructions["cloud_percent"] == 30.40
        assert obstructions["shadow_pixels"] == 130912
        assert obstructions["shadow_percent"] == 13.09
        assert obstructions["total_obstructed_pixels"] == 434888
        assert obstructions["total_obstructed_percent"] == 43.49
        assert obstructions["clear_pixels"] == 565112
        assert obstructions["clear_percent"] == 56.51

        scl = masks_resp["scl_classes"]
        print(f"SCL classes present: {len(scl)}")
        assert len(scl) == 7  # 7 active classes present in scene
        vegetation_class = next(c for c in scl if c["class_id"] == 4)
        print(f"Class 4 (Vegetation): {vegetation_class['percentage']}% ({vegetation_class['count']:,} px)")
        assert vegetation_class["count"] == 159608
        assert vegetation_class["percentage"] == 15.96
        print("PASS: /agriculture/masks verified completely.")

    # -------------------------------------------------------------
    # 5. Agriculture Visualizations List
    # -------------------------------------------------------------
    print("\n--- 5. Testing GET /agriculture/visualizations ---")
    req = urllib.request.Request(f"{BASE_URL}/agriculture/visualizations")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        viz_data = json.loads(resp.read().decode())
        visualizations = viz_data["visualizations"]
        print(f"Found {len(visualizations)} registered visualizations:")
        for v in visualizations:
            print(f"  - [{v['id']}] {v['title']} ({v['category']}) -> {v['endpoint_url']} [{v['file_size_bytes']:,} bytes]")
            assert v["file_size_bytes"] > 0
            assert v["content_type"] == "image/png"
            assert v["endpoint_url"].startswith("/agriculture/visualizations/")

        assert len(visualizations) == 8
        viz_ids = [v["id"] for v in visualizations]
        expected_ids = [
            "true_color", "false_color", "ndvi", "ndwi", "ndmi",
            "scl_classification", "obstruction_map", "cloud_free_agricultural_roi"
        ]
        for eid in expected_ids:
            assert eid in viz_ids
        print("PASS: /agriculture/visualizations list verified.")

    # -------------------------------------------------------------
    # 6. Agriculture Visualization Streaming (All 8 Files)
    # -------------------------------------------------------------
    print("\n--- 6. Testing GET /agriculture/visualizations/{name} for all 8 items ---")
    png_magic = b"\x89PNG\r\n\x1a\n"
    for viz_id in expected_ids:
        req = urllib.request.Request(f"{BASE_URL}/agriculture/visualizations/{viz_id}")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            content_type = resp.headers.get("Content-Type")
            assert content_type == "image/png"
            data = resp.read()
            assert data.startswith(png_magic), f"{viz_id} does not have PNG magic signature"
            assert len(data) > 1000, f"{viz_id} content unexpectedly small ({len(data)} bytes)"
            img = Image.open(io.BytesIO(data))
            assert img.size[0] > 500 and img.size[1] > 500, f"{viz_id} image dimensions unexpectedly small: {img.size}"
            print(f"  ✓ {viz_id}: 200 OK | image/png | {len(data):,} bytes | dimensions: {img.size}")
    print("PASS: All 8 visualizations streamed successfully as valid PNGs.")

    # -------------------------------------------------------------
    # 7. Security & Error Handling (Path Traversal / Invalid Names)
    # -------------------------------------------------------------
    print("\n--- 7. Testing Security & Error Handling on /agriculture/visualizations/{name} ---")
    
    # 7a: Unknown name
    try:
        req = urllib.request.Request(f"{BASE_URL}/agriculture/visualizations/invalid_viz_name")
        with urllib.request.urlopen(req) as resp:
            assert False, "Should have returned 404 for invalid viz name"
    except urllib.error.HTTPError as e:
        assert e.code == 404
        error_body = json.loads(e.read().decode())
        print(f"  ✓ Invalid name correctly returned 404: {error_body['detail']}")

    # 7b: Path traversal attempts
    traversal_tests = [
        "../../etc/passwd",
        "..%2F..%2Fmain.py",
        "nested/subpath",
        "true_color.png;cat",
    ]
    for attempt in traversal_tests:
        try:
            req = urllib.request.Request(f"{BASE_URL}/agriculture/visualizations/{attempt}")
            with urllib.request.urlopen(req) as resp:
                assert False, f"Attempt {attempt} should have been rejected"
        except urllib.error.HTTPError as e:
            assert e.code in (404, 400), f"Attempt {attempt} returned {e.code}"
            print(f"  ✓ Traversal attempt '{attempt}' successfully rejected with HTTP {e.code}")
    print("PASS: Security and error handling verified.")

    # -------------------------------------------------------------
    # 8. ML Endpoints Regression Check
    # -------------------------------------------------------------
    print("\n--- 8. Running Regression Check on Existing ML Endpoints ---")
    
    # 8a: GET /health
    print("Checking GET /health...")
    req = urllib.request.Request(f"{BASE_URL}/health")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        health_data = json.loads(resp.read().decode())
        assert health_data["status"] == "ok"
        assert health_data["model_loaded"] is True
        assert "unet" in health_data["available_models"]
        assert "attention_unet" in health_data["available_models"]
        print("  ✓ GET /health: OK")

    # 8b: GET /models
    print("Checking GET /models...")
    req = urllib.request.Request(f"{BASE_URL}/models")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        models_data = json.loads(resp.read().decode())
        assert len(models_data["models"]) == 2
        print(f"  ✓ GET /models: OK ({len(models_data['models'])} models loaded)")

    # 8c: GET /metrics
    print("Checking GET /metrics...")
    req = urllib.request.Request(f"{BASE_URL}/metrics")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        metrics_data = json.loads(resp.read().decode())
        assert metrics_data["evaluation"]["test_count"] == 75
        assert metrics_data["models"]["unet"]["psnr_mean"] > 26.0
        assert metrics_data["models"]["attention_unet"]["psnr_mean"] > 20.0
        print("  ✓ GET /metrics: OK (75 test scenes)")

    # 8d: POST /predict with U-Net
    print("Checking POST /predict?model=unet...")
    test_img = create_test_image_bytes()
    req = send_multipart_predict(f"{BASE_URL}/predict?model=unet", "test.png", test_img, "image/png")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        assert resp.headers.get("Content-Type") == "image/png"
        assert resp.headers.get("X-GenVision-Model-Id") == "unet"
        out_bytes = resp.read()
        assert out_bytes.startswith(png_magic)
        out_img = Image.open(io.BytesIO(out_bytes))
        assert out_img.size == (256, 256)
        print("  ✓ POST /predict?model=unet: OK (valid 256x256 PNG returned)")

    # 8e: POST /predict with Attention U-Net
    print("Checking POST /predict?model=attention_unet...")
    req = send_multipart_predict(f"{BASE_URL}/predict?model=attention_unet", "test.png", test_img, "image/png")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        assert resp.headers.get("Content-Type") == "image/png"
        assert resp.headers.get("X-GenVision-Model-Id") == "attention_unet"
        out_bytes = resp.read()
        assert out_bytes.startswith(png_magic)
        out_img = Image.open(io.BytesIO(out_bytes))
        assert out_img.size == (256, 256)
        print("  ✓ POST /predict?model=attention_unet: OK (valid 256x256 PNG returned)")

    print("\n" + "=" * 70)
    print("=== ALL PHASE 3.3 TESTS AND REGRESSION CHECKS PASSED SUCCESSFULLY ===")
    print("=" * 70)


if __name__ == "__main__":
    t = threading.Thread(target=start_server, daemon=True)
    t.start()
    # Wait for uvicorn to initialize and load checkpoints
    time.sleep(3.0)
    try:
        run_tests()
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    print("\nTest suite execution completed cleanly.")
    exit(0)
