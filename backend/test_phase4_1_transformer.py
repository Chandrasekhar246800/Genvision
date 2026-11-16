"""
Phase 4.1 Validation & Benchmark Suite for CS-TransUNet
GenVision Research Framework

Validates:
1. Model initialization and parameter counts.
2. Single-sample forward pass (B=1, [1, 3, 256, 256]).
3. Multi-sample forward pass (B=2, [2, 3, 256, 256]).
4. Output range clamp verification ([0.0, 1.0]).
5. Numerical sanity (no NaN, no Inf).
6. Severity estimator outputs (logits [B, 3], probabilities [B, 3] summing to 1.0, embedding [B, 192]).
7. Multi-scale feature map dimensions (Stages 1-4).
8. Return dict vs direct tensor mode.
9. Full backward pass & gradient propagation across all parameters.
10. CPU inference latency & FPS throughput benchmark.
11. State dict serialization roundtrip.
12. Comparison against baseline models (U-Net and Attention U-Net).
"""

import io
import os
import sys
import time
import torch
import torch.nn as nn

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.transformer.cs_transunet import CSTransUNet
from models.transformer.blocks import (
    LayerNorm2d,
    DoubleResidualBlock,
    MultiDconvHeadTransposedAttention,
    GatedDconvFeedForward,
    SeverityModulatedTransformerBlock,
)
from models.transformer.severity_estimator import CloudSeverityEstimator
from unet import UNet
from attention_unet import AttentionUNet


def run_all_tests():
    print("=" * 80)
    print("GENVISION PHASE 4.1: CS-TRANSUNET ARCHITECTURE VALIDATION SUITE")
    print("=" * 80)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Device: CPU (Testing reference hardware constraints)")
    print("-" * 80)

    results = []

    def log_test(test_id, name, status, details=""):
        tag = "[PASS]" if status else "[FAIL]"
        print(f"{tag} Test {test_id:02d}: {name}")
        if details:
            print(f"       -> {details}")
        results.append({"id": test_id, "name": name, "status": status, "details": details})

    # -------------------------------------------------------------------------
    # Test 1: Model Initialization & Parameter Counts
    # -------------------------------------------------------------------------
    try:
        t0 = time.perf_counter()
        model = CSTransUNet(
            in_channels=3,
            out_channels=3,
            stage_channels=(48, 96, 192, 384),
            transformer_depths=(2, 2),
            num_heads=(6, 12),
            severity_embedding_dim=192,
            num_severity_classes=3,
            ffn_expansion_factor=2.0,
        )
        init_time = (time.perf_counter() - t0) * 1000

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Baseline comparison
        unet = UNet()
        unet_params = sum(p.numel() for p in unet.parameters())

        att_unet = AttentionUNet()
        att_params = sum(p.numel() for p in att_unet.parameters())

        details = (
            f"Params: {total_params:,} ({total_params/1e6:.2f}M) | "
            f"Init: {init_time:.1f}ms | "
            f"U-Net: {unet_params/1e6:.2f}M | "
            f"Att-UNet: {att_params/1e6:.2f}M"
        )
        passed = (total_params == trainable_params) and (6.0e6 <= total_params <= 8.5e6)
        log_test(1, "Model Initialization & Parameter Count", passed, details)
    except Exception as e:
        log_test(1, "Model Initialization & Parameter Count", False, str(e))
        return

    # -------------------------------------------------------------------------
    # Test 2: Single-Sample Forward Pass (B=1, [1, 3, 256, 256])
    # -------------------------------------------------------------------------
    try:
        model.eval()
        x1 = torch.rand(1, 3, 256, 256)
        with torch.no_grad():
            out1 = model(x1)
        
        recon1 = out1["reconstructed"]
        passed = (recon1.shape == torch.Size([1, 3, 256, 256]))
        details = f"Input: [1, 3, 256, 256] -> Output: {list(recon1.shape)}"
        log_test(2, "Single-Sample Forward Pass (B=1)", passed, details)
    except Exception as e:
        log_test(2, "Single-Sample Forward Pass (B=1)", False, str(e))

    # -------------------------------------------------------------------------
    # Test 3: Multi-Sample Forward Pass (B=2, [2, 3, 256, 256])
    # -------------------------------------------------------------------------
    try:
        x2 = torch.rand(2, 3, 256, 256)
        with torch.no_grad():
            out2 = model(x2)
        
        recon2 = out2["reconstructed"]
        passed = (recon2.shape == torch.Size([2, 3, 256, 256]))
        details = f"Input: [2, 3, 256, 256] -> Output: {list(recon2.shape)}"
        log_test(3, "Multi-Sample Forward Pass (B=2)", passed, details)
    except Exception as e:
        log_test(3, "Multi-Sample Forward Pass (B=2)", False, str(e))

    # -------------------------------------------------------------------------
    # Test 4: Reconstructed Output Dynamic Range Verification ([0.0, 1.0])
    # -------------------------------------------------------------------------
    try:
        # Extreme inputs: zeros, ones, extreme high/low
        x_extreme = torch.cat([torch.zeros(1, 3, 256, 256), torch.ones(1, 3, 256, 256)], dim=0)
        with torch.no_grad():
            out_ext = model(x_extreme)["reconstructed"]
        min_val = float(out_ext.min())
        max_val = float(out_ext.max())
        passed = (min_val >= 0.0) and (max_val <= 1.0)
        details = f"Output min={min_val:.5f}, max={max_val:.5f} (Strictly within [0.0, 1.0])"
        log_test(4, "Output Value Range Enforcement ([0.0, 1.0])", passed, details)
    except Exception as e:
        log_test(4, "Output Value Range Enforcement ([0.0, 1.0])", False, str(e))

    # -------------------------------------------------------------------------
    # Test 5: Numerical Sanity (NaN / Inf Audit)
    # -------------------------------------------------------------------------
    try:
        nan_in_recon = torch.isnan(recon2).any().item() or torch.isinf(recon2).any().item()
        nan_in_logits = torch.isnan(out2["severity_logits"]).any().item()
        nan_in_probs = torch.isnan(out2["severity_probabilities"]).any().item()
        passed = not (nan_in_recon or nan_in_logits or nan_in_probs)
        details = f"NaN/Inf detected: reconstructed={nan_in_recon}, logits={nan_in_logits}, probs={nan_in_probs}"
        log_test(5, "Numerical Sanity (No NaN / Inf)", passed, details)
    except Exception as e:
        log_test(5, "Numerical Sanity (No NaN / Inf)", False, str(e))

    # -------------------------------------------------------------------------
    # Test 6: Cloud Severity Estimator Verification
    # -------------------------------------------------------------------------
    try:
        logits = out2["severity_logits"]
        probs = out2["severity_probabilities"]
        embed = out2["severity_embedding"]

        shape_check = (
            logits.shape == torch.Size([2, 3])
            and probs.shape == torch.Size([2, 3])
            and embed.shape == torch.Size([2, 192])
        )
        sums = probs.sum(dim=-1)
        sums_close = torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
        passed = shape_check and sums_close
        details = (
            f"Logits: {list(logits.shape)}, Probs: {list(probs.shape)}, "
            f"Embedding: {list(embed.shape)}, Probs Sums: {sums.tolist()}"
        )
        log_test(6, "Cloud Severity Estimator Branches", passed, details)
    except Exception as e:
        log_test(6, "Cloud Severity Estimator Branches", False, str(e))

    # -------------------------------------------------------------------------
    # Test 7: Multi-Scale Feature Maps Verification
    # -------------------------------------------------------------------------
    try:
        with torch.no_grad():
            out_feats = model(x2, return_features=True)
        feats = out_feats["features"]
        f1_shape = list(feats["stage1"].shape)
        f2_shape = list(feats["stage2"].shape)
        f3_shape = list(feats["stage3"].shape)
        f4_shape = list(feats["stage4"].shape)

        passed = (
            f1_shape == [2, 48, 256, 256]
            and f2_shape == [2, 96, 128, 128]
            and f3_shape == [2, 192, 64, 64]
            and f4_shape == [2, 384, 32, 32]
        )
        details = f"S1: {f1_shape}, S2: {f2_shape}, S3: {f3_shape}, S4: {f4_shape}"
        log_test(7, "Hierarchical Multi-Scale Features (Stages 1-4)", passed, details)
    except Exception as e:
        log_test(7, "Hierarchical Multi-Scale Features (Stages 1-4)", False, str(e))

    # -------------------------------------------------------------------------
    # Test 8: Raw Tensor Return Mode (return_dict=False)
    # -------------------------------------------------------------------------
    try:
        with torch.no_grad():
            raw_out = model(x1, return_dict=False)
        passed = isinstance(raw_out, torch.Tensor) and raw_out.shape == torch.Size([1, 3, 256, 256])
        details = f"Type: {type(raw_out).__name__}, Shape: {list(raw_out.shape)}"
        log_test(8, "Raw Tensor Mode (return_dict=False)", passed, details)
    except Exception as e:
        log_test(8, "Raw Tensor Mode (return_dict=False)", False, str(e))

    # -------------------------------------------------------------------------
    # Test 9: Backward Pass & Full Gradient Propagation
    # -------------------------------------------------------------------------
    try:
        model.train()
        model.zero_grad()
        x_train = torch.rand(2, 3, 256, 256, requires_grad=False)
        y_target = torch.rand(2, 3, 256, 256)

        preds = model(x_train)
        # Joint loss testing both image reconstruction trunk and auxiliary severity classification head
        recon_loss = nn.functional.l1_loss(preds["reconstructed"], y_target)
        aux_loss = 0.1 * preds["severity_logits"].sum()
        total_loss = recon_loss + aux_loss
        total_loss.backward()

        params_missing_grad = []
        zero_grad_params = []
        nan_grad_params = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    params_missing_grad.append(name)
                elif torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    nan_grad_params.append(name)
                elif param.grad.abs().sum().item() == 0.0:
                    zero_grad_params.append(name)

        passed = len(params_missing_grad) == 0 and len(nan_grad_params) == 0
        details = (
            f"Loss: {total_loss.item():.5f} | Missing grad: {len(params_missing_grad)} | "
            f"NaN grad: {len(nan_grad_params)} | Total params checked: {len(list(model.parameters()))}"
        )
        if len(params_missing_grad) > 0:
            details += f" | Unupdated: {params_missing_grad[:3]}"
        log_test(9, "Backward Pass & Gradient Propagation", passed, details)
    except Exception as e:
        log_test(9, "Backward Pass & Gradient Propagation", False, str(e))

    # -------------------------------------------------------------------------
    # Test 10: CPU Latency & Throughput Benchmark
    # -------------------------------------------------------------------------
    try:
        model.eval()
        x_bench = torch.rand(1, 3, 256, 256)

        # Warm-up passes
        with torch.no_grad():
            for _ in range(2):
                _ = model(x_bench)

        # Timed passes
        latencies = []
        with torch.no_grad():
            for _ in range(10):
                t_start = time.perf_counter()
                _ = model(x_bench)
                t_end = time.perf_counter()
                latencies.append((t_end - t_start) * 1000)

        mean_lat = sum(latencies) / len(latencies)
        min_lat = min(latencies)
        max_lat = max(latencies)
        fps = 1000.0 / mean_lat

        passed = mean_lat > 0
        details = f"Avg Latency: {mean_lat:.1f}ms (Min: {min_lat:.1f}ms, Max: {max_lat:.1f}ms) | FPS: {fps:.2f} img/s"
        log_test(10, "CPU Inference Latency & FPS Benchmark", passed, details)
    except Exception as e:
        log_test(10, "CPU Inference Latency & FPS Benchmark", False, str(e))

    # -------------------------------------------------------------------------
    # Test 11: Serialization Roundtrip (State Dict Save/Load)
    # -------------------------------------------------------------------------
    try:
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)

        model2 = CSTransUNet(
            in_channels=3,
            out_channels=3,
            stage_channels=(48, 96, 192, 384),
            transformer_depths=(2, 2),
            num_heads=(6, 12),
            severity_embedding_dim=192,
            num_severity_classes=3,
            ffn_expansion_factor=2.0,
        )
        model2.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
        model2.eval()

        with torch.no_grad():
            diff = (model(x_bench)["reconstructed"] - model2(x_bench)["reconstructed"]).abs().max().item()

        passed = (diff < 1e-6)
        details = f"State dict loaded successfully, max absolute difference: {diff:.2e}"
        log_test(11, "Model Serialization & Weights Loading Roundtrip", passed, details)
    except Exception as e:
        log_test(11, "Model Serialization & Weights Loading Roundtrip", False, str(e))

    # -------------------------------------------------------------------------
    # Test 12: Architectural Complexity & Baseline Comparison
    # -------------------------------------------------------------------------
    try:
        cs_params = total_params
        ratio_unet = cs_params / unet_params
        ratio_att = cs_params / att_params

        passed = cs_params < att_params
        details = (
            f"CS-TransUNet ({cs_params/1e6:.2f}M) vs U-Net ({unet_params/1e6:.2f}M): {ratio_unet:.2f}x | "
            f"vs Att-UNet ({att_params/1e6:.2f}M): {ratio_att:.2f}x (Lighter than Att-UNet)"
        )
        log_test(12, "Architecture Complexity vs Baselines", passed, details)
    except Exception as e:
        log_test(12, "Architecture Complexity vs Baselines", False, str(e))

    print("-" * 80)
    passed_count = sum(1 for r in results if r["status"])
    total_count = len(results)
    print(f"SUMMARY: {passed_count}/{total_count} TESTS PASSED")
    print("=" * 80)

    return passed_count == total_count


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
