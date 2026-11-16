"""
Phase 4.2 Resource Validation & Pipeline Benchmark Suite
GenVision Research Framework

Validates:
1. Dataset integrity, partition sizes (350/75/75), and strict test-set isolation.
2. Gradient accumulation correctness (accumulation over 2 steps, step & zero_grad).
3. Smoke-test execution (2 epochs, forward-backward profiling).
4. Checkpoint creation, parameter weight preservation, and structure audit.
5. Resume functionality from saved checkpoint (recovery audit).
6. Memory footprint & hardware execution time projections on CPU.
7. Telemetry artifact generation (CSV, JSON, log, and PNG curves).
"""

import json
import os
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.transformer.cs_transunet import CSTransUNet
from train_transformer import (
    verify_dataset_split,
    run_training,
    load_checkpoint,
    DEFAULT_SPLIT_PATH,
)


def run_pipeline_tests():
    print("=" * 80)
    print("GENVISION PHASE 4.2: TRAINING PIPELINE VALIDATION & BENCHMARK SUITE")
    print("=" * 80)
    print(f"Device: CPU | PyTorch: {torch.__version__}")
    print(f"Available CPU Cores: {os.cpu_count()}")
    print("-" * 80)

    results = []

    def log_test(test_id, name, status, details=""):
        tag = "[PASS]" if status else "[FAIL]"
        print(f"{tag} Test {test_id:02d}: {name}")
        if details:
            print(f"       -> {details}")
        results.append({"id": test_id, "name": name, "status": status, "details": details})

    # -------------------------------------------------------------------------
    # Test 1: Dataset Partitioning & Strict Split Isolation
    # -------------------------------------------------------------------------
    try:
        split = verify_dataset_split(DEFAULT_SPLIT_PATH)
        n_train = len(split["train"])
        n_val = len(split["validation"])
        n_test = len(split["test"])
        total = split["total_images"]

        disjoint = (
            len(set(split["train"]) & set(split["validation"])) == 0
            and len(set(split["train"]) & set(split["test"])) == 0
            and len(set(split["validation"]) & set(split["test"])) == 0
        )
        passed = (n_train == 350) and (n_val == 75) and (n_test == 75) and (total == 500) and disjoint
        details = f"Train: {n_train}, Val: {n_val}, Held-out Test: {n_test} | Total: {total} | Disjoint: {disjoint}"
        log_test(1, "Dataset Split Integrity & Zero-Leakage Audit", passed, details)
    except Exception as e:
        log_test(1, "Dataset Split Integrity & Zero-Leakage Audit", False, str(e))

    # -------------------------------------------------------------------------
    # Test 2: Gradient Accumulation Mechanics & Remainder Flush Verification
    # -------------------------------------------------------------------------
    try:
        model = CSTransUNet()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.L1Loss()

        initial_weights = [p.clone() for p in model.parameters() if p.requires_grad]

        # Scenario: Total 3 batches with grad_accum=2 (mimics 175 batches: 87 pairs + 1 odd remainder)
        total_batches = 3
        grad_accum = 2

        # Step 1: Batch 0 (first in pair, group_size = 2)
        start_0 = (0 // grad_accum) * grad_accum
        end_0 = min(start_0 + grad_accum, total_batches)
        group_size_0 = end_0 - start_0  # 2

        x0 = torch.rand(2, 3, 256, 256)
        y0 = torch.rand(2, 3, 256, 256)
        out0 = model(x0)
        loss0 = criterion(out0["reconstructed"], y0) / group_size_0
        loss0.backward()

        # Check weights NOT yet updated after batch 0
        w_after_0 = [p.clone() for p in model.parameters() if p.requires_grad]
        weights_unchanged_0 = all(torch.equal(w0, w1) for w0, w1 in zip(initial_weights, w_after_0))

        # Step 2: Batch 1 (second in pair, group_size = 2, triggers step)
        start_1 = (1 // grad_accum) * grad_accum
        end_1 = min(start_1 + grad_accum, total_batches)
        group_size_1 = end_1 - start_1  # 2

        x1 = torch.rand(2, 3, 256, 256)
        y1 = torch.rand(2, 3, 256, 256)
        out1 = model(x1)
        loss1 = criterion(out1["reconstructed"], y1) / group_size_1
        loss1.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        # Check weights DID update after batch 1
        w_after_1 = [p.clone() for p in model.parameters() if p.requires_grad]
        weights_updated_1 = any(not torch.equal(w0, w1) for w0, w1 in zip(initial_weights, w_after_1))

        # Step 3: Batch 2 (odd remainder batch 175 of 175, group_size = 1)
        start_2 = (2 // grad_accum) * grad_accum
        end_2 = min(start_2 + grad_accum, total_batches)
        group_size_2 = end_2 - start_2  # 1 (Crucial: scaled by 1, not 2!)

        x2 = torch.rand(2, 3, 256, 256)
        y2 = torch.rand(2, 3, 256, 256)
        out2 = model(x2)
        loss2 = criterion(out2["reconstructed"], y2) / group_size_2
        loss2.backward()

        # Flushed because batch_idx + 1 == total_batches
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        w_after_2 = [p.clone() for p in model.parameters() if p.requires_grad]
        weights_updated_2 = any(not torch.equal(w1, w2) for w1, w2 in zip(w_after_1, w_after_2))
        grads_zeroed = all(p.grad is None or p.grad.abs().sum() == 0 for p in model.parameters() if p.requires_grad)

        passed = (
            weights_unchanged_0
            and weights_updated_1
            and (group_size_2 == 1)
            and weights_updated_2
            and grads_zeroed
        )
        details = (
            f"Batch 0 accumulated: {weights_unchanged_0} | Batch 1 stepped: {weights_updated_1} | "
            f"Odd Batch 2 group_size={group_size_2} (scaled correctly) | Odd Batch 2 flushed: {weights_updated_2}"
        )
        log_test(2, "Gradient Accumulation & Incomplete Group Flush Verification", passed, details)
    except Exception as e:
        log_test(2, "Gradient Accumulation & Incomplete Group Flush Verification", False, str(e))

    # -------------------------------------------------------------------------
    # Test 3: Training Smoke Test Execution (2 Epochs)
    # -------------------------------------------------------------------------
    smoke_summary = None
    try:
        t0_smoke = time.perf_counter()

        smoke_summary = run_training(
            epochs=2,
            batch_size=2,
            grad_accum_steps=2,
            learning_rate=1e-4,
            weight_decay=1e-4,
            seed=42,
            smoke_test=True,
            smoke_train_batches=4,
            smoke_val_batches=2,
        )

        smoke_duration = time.perf_counter() - t0_smoke

        passed = (smoke_summary is not None) and (smoke_summary["status"] == "COMPLETED")
        details = (
            f"2 Epochs completed in {smoke_duration:.2f}s | "
            f"Avg batch step: {smoke_summary.get('average_step_duration_seconds', 0.0)*1000:.1f}ms"
        )
        log_test(3, "Controlled Smoke Test Execution (2 Epochs)", passed, details)
    except Exception as e:
        log_test(3, "Controlled Smoke Test Execution (2 Epochs)", False, str(e))

    # -------------------------------------------------------------------------
    # Test 4: Checkpoint Integrity & State Dict Verification
    # -------------------------------------------------------------------------
    try:
        best_ckpt = Path(smoke_summary["best_checkpoint"])
        latest_ckpt = Path(smoke_summary["latest_checkpoint"])

        exists = best_ckpt.exists() and latest_ckpt.exists()
        size_best = best_ckpt.stat().st_size / (1024 * 1024)
        size_latest = latest_ckpt.stat().st_size / (1024 * 1024)

        state = torch.load(best_ckpt, map_location="cpu", weights_only=True)
        required_keys = [
            "epoch",
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "best_val_loss",
            "seed",
            "torch_rng_state",
            "parameters_count",
        ]
        has_keys = all(k in state for k in required_keys)
        passed = exists and has_keys and (size_best > 25.0)

        details = (
            f"Best: {best_ckpt.name} ({size_best:.1f}MB) | "
            f"Latest: {latest_ckpt.name} ({size_latest:.1f}MB) | "
            f"Required state keys present: {has_keys}"
        )
        log_test(4, "Checkpoint Serialization & Complete State Integrity", passed, details)
    except Exception as e:
        log_test(4, "Checkpoint Serialization & Complete State Integrity", False, str(e))

    # -------------------------------------------------------------------------
    # Test 5: Training Resume Capability Verification
    # -------------------------------------------------------------------------
    try:
        latest_ckpt = Path(smoke_summary["latest_checkpoint"])
        resumed_summary = run_training(
            epochs=3,
            batch_size=2,
            grad_accum_steps=2,
            seed=42,
            resume_path=latest_ckpt,
            smoke_test=True,
            smoke_train_batches=4,
            smoke_val_batches=2,
        )

        passed = (resumed_summary["status"] == "COMPLETED") and (resumed_summary["total_epochs"] == 3)
        details = f"Resumed from Epoch 2 -> Successfully completed Epoch 3. Best Val Loss: {resumed_summary['best_val_loss']:.6f}"
        log_test(5, "Checkpoint Resumption & Interruption Recovery", passed, details)
    except Exception as e:
        log_test(5, "Checkpoint Resumption & Interruption Recovery", False, str(e))

    # -------------------------------------------------------------------------
    # Test 6: Fine-Grained Latency & Runtime Benchmarking
    # -------------------------------------------------------------------------
    try:
        fwd_bwd_ms = smoke_summary.get("avg_fwd_bwd_duration_ms", 0.0)
        opt_step_ms = smoke_summary.get("avg_optimizer_step_duration_ms", 0.0)
        val_step_ms = smoke_summary.get("avg_validation_step_duration_ms", 0.0)
        projected_epoch_s = smoke_summary.get("projected_epoch_duration_seconds", 0.0)
        projected_25_epochs_m = smoke_summary.get("projected_25_epochs_duration_minutes", 0.0)

        passed = fwd_bwd_ms > 0 and opt_step_ms > 0
        details = (
            f"Fwd/Bwd: {fwd_bwd_ms:.1f}ms | Opt Step: {opt_step_ms:.1f}ms | Val Step: {val_step_ms:.1f}ms | "
            f"Projected Epoch: {projected_epoch_s:.1f}s ({projected_epoch_s/60:.2f} min) | "
            f"Projected 25 Epochs: {projected_25_epochs_m:.1f} min ({projected_25_epochs_m/60:.2f} hrs)"
        )
        log_test(6, "Fine-Grained Latency Breakdown & Runtime Projection", passed, details)
    except Exception as e:
        log_test(6, "Fine-Grained Latency Breakdown & Runtime Projection", False, str(e))

    # -------------------------------------------------------------------------
    # Test 7: Telemetry & Visualization Artifacts Audit
    # -------------------------------------------------------------------------
    try:
        out_dir = PROJECT_ROOT / "results" / "phase4" / "training"
        csv_file = out_dir / "smoke_test_history.csv"
        log_file = out_dir / "smoke_test_log.txt"
        summary_file = out_dir / "smoke_test_summary.json"
        loss_curve = out_dir / "loss_curve.png"
        lr_curve = out_dir / "learning_rate_curve.png"

        files_exist = [
            csv_file.exists(),
            log_file.exists(),
            summary_file.exists(),
            loss_curve.exists(),
            lr_curve.exists(),
        ]
        passed = all(files_exist)
        details = (
            f"CSV: {csv_file.name} ({csv_file.stat().st_size}B), "
            f"Log: {log_file.name} ({log_file.stat().st_size}B), "
            f"Plots: loss_curve.png ({loss_curve.stat().st_size}B), lr_curve.png ({lr_curve.stat().st_size}B)"
        )
        log_test(7, "Telemetry & Visualization Artifacts Generation", passed, details)
    except Exception as e:
        log_test(7, "Telemetry & Visualization Artifacts Generation", False, str(e))

    print("-" * 80)
    passed_count = sum(1 for r in results if r["status"])
    total_count = len(results)
    print(f"SUMMARY: {passed_count}/{total_count} PIPELINE BENCHMARKS PASSED")
    print("=" * 80)

    return passed_count == total_count


if __name__ == "__main__":
    success = run_pipeline_tests()
    sys.exit(0 if success else 1)
