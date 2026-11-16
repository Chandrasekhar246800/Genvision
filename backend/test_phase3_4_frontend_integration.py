"""
GenVision Phase 3.4 Frontend Integration & Contract Verification Test
Validates that the FastAPI backend provides exactly the contracts and image payloads
consumed by the Next.js Agriculture Dashboard components.
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

PORT = 8091
BASE_URL = f"http://127.0.0.1:{PORT}"


def start_server():
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


def run_tests():
    print("=" * 75)
    print("=== STARTING PHASE 3.4 FRONTEND-BACKEND INTEGRATION CONTRACT TESTS ===")
    print("=" * 75)

    # 1. Health contract
    print("\n[Contract 1] Testing GET /agriculture/health (Backend Connectivity Badge)")
    req = urllib.request.Request(f"{BASE_URL}/agriculture/health")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        health = json.loads(resp.read().decode())
        assert health["status"] == "ok"
        assert health["dataset_available"] is True
        assert health["available_visualizations_count"] == 8
        print(f"  ✓ Status: {health['status']} | Visualizations available: {health['available_visualizations_count']}")

    # 2. Overview & KPI Contract
    print("\n[Contract 2] Testing GET /agriculture/summary (Overview & KPI Cards)")
    req = urllib.request.Request(f"{BASE_URL}/agriculture/summary")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        summary = json.loads(resp.read().decode())
        
        # Overview checks
        assert summary["satellite"] == "Sentinel-2C"
        assert summary["tile_id"] == "43PHS"
        assert summary["acquisition_date"] == "14 October 2026"
        assert summary["crs"] == "EPSG:32643"
        assert summary["roi_dimensions"] == [1000, 1000]
        assert summary["roi_size_km"] == 10.0
        print(f"  ✓ Overview: {summary['satellite']} (Tile {summary['tile_id']}) acquired on {summary['acquisition_date']} ({summary['roi_size_km']} km²)")

        # KPI checks
        assert abs(summary["ndvi_mean_clear"] - 0.3780) < 0.001
        assert abs(summary["ndwi_mean_clear"] - (-0.3935)) < 0.001
        assert abs(summary["ndmi_mean_clear"] - 0.0042) < 0.001
        assert abs(summary["clear_surface_percentage"] - 56.51) < 0.01
        assert abs(summary["cloud_percentage"] - 30.40) < 0.01
        assert abs(summary["shadow_percentage"] - 13.09) < 0.01
        print(f"  ✓ KPI Cards: NDVI={summary['ndvi_mean_clear']:.4f}, NDWI={summary['ndwi_mean_clear']:.4f}, NDMI={summary['ndmi_mean_clear']:.4f}")
        print(f"  ✓ Coverage: Clear={summary['clear_surface_percentage']:.2f}%, Cloud={summary['cloud_percentage']:.2f}%, Shadow={summary['shadow_percentage']:.2f}%")

    # 3. Indices Contract
    print("\n[Contract 3] Testing GET /agriculture/indices (Cloud Impact Analysis Chart)")
    req = urllib.request.Request(f"{BASE_URL}/agriculture/indices")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        indices = json.loads(resp.read().decode())["indices"]
        assert "ndvi" in indices and "ndwi" in indices and "ndmi" in indices

        ndvi = indices["ndvi"]
        m_mean = ndvi["masked_scientifically_valid"]["mean"]
        u_mean = ndvi["unmasked_all_pixels"]["mean"]
        delta = ndvi["cloud_impact_delta_mean"]
        assert abs(m_mean - 0.3780) < 0.001
        assert abs(u_mean - 0.2939) < 0.001
        assert abs(delta - (-0.0841)) < 0.001
        print(f"  ✓ NDVI: Masked={m_mean:.4f} vs Unmasked={u_mean:.4f} (Delta={delta:.4f})")

        ndwi = indices["ndwi"]
        assert abs(ndwi["masked_scientifically_valid"]["mean"] - (-0.3935)) < 0.001
        assert abs(ndwi["unmasked_all_pixels"]["mean"] - (-0.2905)) < 0.001
        print(f"  ✓ NDWI: Masked={ndwi['masked_scientifically_valid']['mean']:.4f} vs Unmasked={ndwi['unmasked_all_pixels']['mean']:.4f}")

        ndmi = indices["ndmi"]
        assert abs(ndmi["masked_scientifically_valid"]["mean"] - 0.0042) < 0.001
        assert abs(ndmi["unmasked_all_pixels"]["mean"] - 0.0480) < 0.001
        print(f"  ✓ NDMI: Masked={ndmi['masked_scientifically_valid']['mean']:.4f} vs Unmasked={ndmi['unmasked_all_pixels']['mean']:.4f}")

    # 4. Masks Contract
    print("\n[Contract 4] Testing GET /agriculture/masks (Obstruction Donut & SCL Histogram)")
    req = urllib.request.Request(f"{BASE_URL}/agriculture/masks")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        masks = json.loads(resp.read().decode())
        obs = masks["obstruction_summary"]
        assert obs["clear_pixels"] == 565112
        assert obs["cloud_pixels"] == 303976
        assert obs["shadow_pixels"] == 130912
        assert obs["total_obstructed_pixels"] == 434888
        
        scl = masks["scl_classes"]
        assert len(scl) == 7
        print(f"  ✓ Obstruction metrics: Clear={obs['clear_percent']:.2f}% ({obs['clear_pixels']:,} px) | Obstructed={obs['total_obstructed_percent']:.2f}% ({obs['total_obstructed_pixels']:,} px)")
        print(f"  ✓ SCL Classes: {len(scl)} classes represented in histogram")

    # 5. Visualizations Catalog & Streaming
    print("\n[Contract 5] Testing Visualizations List & Streaming (Viewer & Comparison)")
    req = urllib.request.Request(f"{BASE_URL}/agriculture/visualizations")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        catalog = json.loads(resp.read().decode())["visualizations"]
        assert len(catalog) == 8
        print(f"  ✓ Catalog contains {len(catalog)} products")

    expected_ids = [
        "true_color", "false_color", "ndvi", "ndwi", "ndmi",
        "scl_classification", "obstruction_map", "cloud_free_agricultural_roi"
    ]
    png_magic = b"\x89PNG\r\n\x1a\n"

    for viz_id in expected_ids:
        url = f"{BASE_URL}/agriculture/visualizations/{viz_id}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            assert resp.headers.get("Content-Type") == "image/png"
            data = resp.read()
            assert data.startswith(png_magic)
            img = Image.open(io.BytesIO(data))
            assert img.size[0] > 500 and img.size[1] > 500
            print(f"  ✓ Streamed {viz_id}: {len(data):,} bytes ({img.size[0]}x{img.size[1]} px)")

    # 6. Verify Comparison Pair Specifically
    print("\n[Contract 6] Verifying Split Comparison Pair (true_color vs cloud_free_agricultural_roi)")
    for pair_id in ["true_color", "cloud_free_agricultural_roi"]:
        url = f"{BASE_URL}/agriculture/visualizations/{pair_id}"
        with urllib.request.urlopen(url) as resp:
            assert resp.status == 200
            data = resp.read()
            assert len(data) > 1_000_000, f"{pair_id} should be high-resolution composite"
            print(f"  ✓ Comparison image {pair_id} verified ({len(data):,} bytes)")

    print("\n" + "=" * 75)
    print("=== ALL FRONTEND CONTRACT AND DATA INTEGRATION TESTS PASSED ===")
    print("=" * 75)


if __name__ == "__main__":
    t = threading.Thread(target=start_server, daemon=True)
    t.start()
    time.sleep(3.0)
    try:
        run_tests()
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    print("\nVerification completed with zero errors.")
    exit(0)
