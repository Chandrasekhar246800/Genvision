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

PORT = 8089
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
    print("=== STARTING PHASE 2 FASTAPI BACKEND VERIFICATION ===")
    test_img = create_test_image_bytes()

    # 1. Health check
    print("\n--- 1. Testing GET /health ---")
    req = urllib.request.Request(f"{BASE_URL}/health")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        health_data = json.loads(resp.read().decode())
        print("Health response:", health_data)
        assert health_data["status"] == "ok"
        assert health_data["model_loaded"] is True
        assert "unet" in health_data["available_models"]
        assert "attention_unet" in health_data["available_models"]
        print("PASS: /health verified")

    # 2. Models check
    print("\n--- 2. Testing GET /models ---")
    req = urllib.request.Request(f"{BASE_URL}/models")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        models_data = json.loads(resp.read().decode())
        print(f"Models returned ({len(models_data['models'])}):")
        for m in models_data["models"]:
            print(f"  - {m['id']}: name={m['name']}, loaded={m['loaded']}, size={m['checkpoint_size_bytes']} bytes")
            assert m["loaded"] is True
            assert m["verified"] is True
            assert m["checkpoint_size_bytes"] > 0
        ids = [m["id"] for m in models_data["models"]]
        assert "unet" in ids
        assert "attention_unet" in ids
        print("PASS: /models verified with both models")

    # 3. Metrics check
    print("\n--- 3. Testing GET /metrics ---")
    req = urllib.request.Request(f"{BASE_URL}/metrics")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        metrics_data = json.loads(resp.read().decode())
        print("Metrics summary:")
        print(f"  Test count: {metrics_data['evaluation']['test_count']}")
        print(f"  UNet PSNR: {metrics_data['models']['unet']['psnr_mean']:.4f} dB, SSIM: {metrics_data['models']['unet']['ssim_mean']:.4f}")
        print(f"  Attn UNet PSNR: {metrics_data['models']['attention_unet']['psnr_mean']:.4f} dB, SSIM: {metrics_data['models']['attention_unet']['ssim_mean']:.4f}")
        assert metrics_data["evaluation"]["test_count"] == 75
        assert metrics_data["models"]["unet"]["psnr_mean"] > 26.0
        assert metrics_data["models"]["attention_unet"]["psnr_mean"] > 20.0
        print("PASS: /metrics verified with held-out evaluation values")

    # 4. Predict with U-Net
    print("\n--- 4. Testing POST /predict?model=unet ---")
    req = send_multipart_predict(f"{BASE_URL}/predict?model=unet", "scene.png", test_img)
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        out_bytes = resp.read()
        model_hdr = resp.headers.get("X-GenVision-Model")
        time_hdr = resp.headers.get("X-Inference-Time-Ms")
        print(f"UNet inference time: {time_hdr} ms, model: {model_hdr}")
        assert model_hdr == "U-Net"
        out_img = Image.open(io.BytesIO(out_bytes))
        assert out_img.format == "PNG"
        assert out_img.size == (256, 256)
        print("PASS: /predict?model=unet generated valid 256x256 PNG")

    # 5. Predict with Attention U-Net
    print("\n--- 5. Testing POST /predict?model=attention_unet ---")
    req = send_multipart_predict(f"{BASE_URL}/predict?model=attention_unet", "scene.png", test_img)
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        out_bytes = resp.read()
        model_hdr = resp.headers.get("X-GenVision-Model")
        time_hdr = resp.headers.get("X-Inference-Time-Ms")
        print(f"Attention UNet inference time: {time_hdr} ms, model: {model_hdr}")
        assert model_hdr == "Attention U-Net"
        out_img = Image.open(io.BytesIO(out_bytes))
        assert out_img.format == "PNG"
        assert out_img.size == (256, 256)
        print("PASS: /predict?model=attention_unet generated valid 256x256 PNG")

    # 6. Predict default (no query param)
    print("\n--- 6. Testing POST /predict (default model) ---")
    req = send_multipart_predict(f"{BASE_URL}/predict", "scene.png", test_img)
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        model_hdr = resp.headers.get("X-GenVision-Model")
        print(f"Default model used: {model_hdr}")
        assert model_hdr == "U-Net"
        print("PASS: default model is correctly U-Net")

    # 7. Predict with invalid model ID
    print("\n--- 7. Testing POST /predict?model=invalid_model (expect 400) ---")
    req = send_multipart_predict(f"{BASE_URL}/predict?model=invalid_model", "scene.png", test_img)
    try:
        with urllib.request.urlopen(req) as resp:
            raise AssertionError("Expected 400, got 200")
    except urllib.error.HTTPError as e:
        assert e.code == 400
        detail = json.loads(e.read().decode())
        print(f"Correctly received 400: {detail}")
        print("PASS: invalid model returns 400")

    # 8. Predict with corrupted image (expect 400)
    print("\n--- 8. Testing POST /predict?model=unet with invalid image data ---")
    req = send_multipart_predict(f"{BASE_URL}/predict?model=unet", "scene.png", b"not-a-valid-png-stream")
    try:
        with urllib.request.urlopen(req) as resp:
            raise AssertionError("Expected 400, got 200")
    except urllib.error.HTTPError as e:
        assert e.code == 400
        detail = json.loads(e.read().decode())
        print(f"Correctly received 400: {detail}")
        print("PASS: corrupted payload returns 400")

    print("\n=== ALL BACKEND ACCEPTANCE CRITERIA VERIFIED SUCCESSFULLY! ===")


if __name__ == "__main__":
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    time.sleep(2.5)  # give server time to load both models into memory
    run_tests()
