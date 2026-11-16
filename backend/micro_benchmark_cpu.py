"""
GenVision Phase 5.2A: CPU Micro-Benchmark Runner
================================================

Measures empirical hardware performance on the local Intel CPU:
1. Model initialization time.
2. Exact parameter count.
3. Forward pass latency (Batch=1, 256x256).
4. Training step latency (Forward + Backward on Denoiser with batch size 2 and 4).
5. Peak memory usage.
6. DDIM sampling latency (S=5 and S=10 steps on 256x256 image).

Strict Protocol:
- Strictly CPU execution.
- No dataset or test set loading.
- Warm-up runs before timing.
- Mean, median, std, min, max recorded.
"""

import ctypes
import gc
import json
import os
import platform
import sys
import time
from pathlib import Path
import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.diffusion.scheduler import GaussianDiffusionScheduler
from models.diffusion.residual import ResidualProcessor, FrozenTransformerConditioner
from models.diffusion.resdiff_unet import ResDiffUNet
from models.diffusion.ddim_sampler import DDIMSampler
from models.diffusion.residual_gating import ResidualGating


def get_process_memory_mb():
    """Get process memory usage in MB on Windows using ctypes."""
    try:
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return {
                "rss_mb": counters.WorkingSetSize / (1024 * 1024),
                "peak_rss_mb": counters.PeakWorkingSetSize / (1024 * 1024),
                "pagefile_mb": counters.PagefileUsage / (1024 * 1024),
            }
    except Exception:
        pass
    return {"rss_mb": -1.0, "peak_rss_mb": -1.0, "pagefile_mb": -1.0}


def run_benchmark():
    print("=" * 80)
    print("GENVISION PHASE 5.2A: CPU MICRO-BENCHMARK")
    print("=" * 80)

    # 1. Environment & Hardware Metadata
    num_threads = torch.get_num_threads()
    torch_version = torch.__version__
    cpu_info = platform.processor()
    system_os = platform.system() + " " + platform.release()

    print(f"OS:               {system_os}")
    print(f"Processor:        {cpu_info}")
    print(f"PyTorch Version:  {torch_version}")
    print(f"CPU Threads:      {num_threads}")
    print(f"CUDA Available:   {torch.cuda.is_available()}")

    # 2. Model Initialization Time & Parameter Counts
    gc.collect()
    t0 = time.perf_counter()
    denoiser = ResDiffUNet()
    t1 = time.perf_counter()
    init_time_ms = (t1 - t0) * 1000.0

    total_params = sum(p.numel() for p in denoiser.parameters())
    trainable_params = sum(p.numel() for p in denoiser.parameters() if p.requires_grad)
    buffers_count = sum(b.numel() for b in denoiser.buffers())
    fp32_size_mb = (total_params * 4) / (1024 * 1024)

    scheduler = GaussianDiffusionScheduler()
    sampler = DDIMSampler(scheduler, denoiser)
    processor = ResidualProcessor(s_r=4.0)

    print("\n--- Model Architecture & Parameter Verification ---")
    print(f"Initialization Time:     {init_time_ms:.2f} ms")
    print(f"Total Parameters:        {total_params:,}")
    print(f"Trainable Parameters:    {trainable_params:,}")
    print(f"Non-Trainable Buffers:   {buffers_count}")
    print(f"FP32 Weights Size:       {fp32_size_mb:.2f} MB")

    # 3. Forward Pass Latency Benchmark (Batch=1, 256x256)
    print("\n--- Forward Pass Latency Benchmark (ResDiffUNet alone, Batch=1, 256x256) ---")
    denoiser.eval()
    x_in_1 = torch.randn(1, 9, 256, 256)
    t_1 = torch.tensor([500], dtype=torch.long)
    e_1 = torch.randn(1, 192)

    # Warm-up (10 runs)
    for _ in range(10):
        with torch.no_grad():
            _ = denoiser(x_in_1, t_1, e_1)

    # Measurement (50 runs)
    forward_times_b1 = []
    for _ in range(50):
        t_start = time.perf_counter()
        with torch.no_grad():
            _ = denoiser(x_in_1, t_1, e_1)
        t_end = time.perf_counter()
        forward_times_b1.append((t_end - t_start) * 1000.0)

    mean_fwd_b1 = float(np.mean(forward_times_b1))
    median_fwd_b1 = float(np.median(forward_times_b1))
    std_fwd_b1 = float(np.std(forward_times_b1, ddof=1))
    min_fwd_b1 = float(np.min(forward_times_b1))
    max_fwd_b1 = float(np.max(forward_times_b1))

    print(f"  Warm-up runs:          10")
    print(f"  Measured iterations:   50")
    print(f"  Mean Latency:          {mean_fwd_b1:.2f} ms")
    print(f"  Median Latency:        {median_fwd_b1:.2f} ms")
    print(f"  Std Dev:               {std_fwd_b1:.2f} ms")
    print(f"  Min / Max:             [{min_fwd_b1:.2f} ms, {max_fwd_b1:.2f} ms]")

    # 4. Training Step Latency Benchmark (Forward + Loss + Backward)
    print("\n--- Training Step Latency Benchmark (Forward + Loss + Backward) ---")
    denoiser.train()
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=2e-4)
    criterion = torch.nn.L1Loss()

    def benchmark_train_step(batch_size: int, num_iters: int = 20):
        x_in = torch.randn(batch_size, 9, 256, 256)
        t = torch.randint(0, 1000, (batch_size,), dtype=torch.long)
        e = torch.randn(batch_size, 192)
        target_noise = torch.randn(batch_size, 3, 256, 256)

        # Warm-up (5 runs)
        for _ in range(5):
            optimizer.zero_grad()
            out = denoiser(x_in, t, e)
            loss = criterion(out, target_noise)
            loss.backward()
            optimizer.step()

        # Measurement
        step_times = []
        for _ in range(num_iters):
            t_start = time.perf_counter()
            optimizer.zero_grad()
            out = denoiser(x_in, t, e)
            loss = criterion(out, target_noise)
            loss.backward()
            optimizer.step()
            t_end = time.perf_counter()
            step_times.append((t_end - t_start) * 1000.0)

        return {
            "batch_size": batch_size,
            "warmup_runs": 5,
            "measured_runs": num_iters,
            "mean_ms": float(np.mean(step_times)),
            "median_ms": float(np.median(step_times)),
            "std_ms": float(np.std(step_times, ddof=1)),
            "min_ms": float(np.min(step_times)),
            "max_ms": float(np.max(step_times)),
        }

    train_b2 = benchmark_train_step(batch_size=2, num_iters=20)
    train_b4 = benchmark_train_step(batch_size=4, num_iters=15)

    print(f"  Batch Size 2 (20 runs): Mean = {train_b2['mean_ms']:.2f} ms | Median = {train_b2['median_ms']:.2f} ms | Min = {train_b2['min_ms']:.2f} ms")
    print(f"  Batch Size 4 (15 runs): Mean = {train_b4['mean_ms']:.2f} ms | Median = {train_b4['median_ms']:.2f} ms | Min = {train_b4['min_ms']:.2f} ms")

    # Projected Epoch Duration based on measured step times:
    # 350 training images with batch size 4 = 88 steps/epoch
    # 350 training images with batch size 2 = 175 steps/epoch
    proj_epoch_b4_sec = (88 * train_b4['mean_ms']) / 1000.0
    proj_epoch_b2_sec = (175 * train_b2['mean_ms']) / 1000.0
    print(f"  Projected Epoch Duration (Batch=4, 88 steps):  {proj_epoch_b4_sec:.1f} s (~{proj_epoch_b4_sec/60:.2f} min)")
    print(f"  Projected Epoch Duration (Batch=2, 175 steps): {proj_epoch_b2_sec:.1f} s (~{proj_epoch_b2_sec/60:.2f} min)")
    print(f"  Projected 50-Epoch Run (Batch=4):              ~{proj_epoch_b4_sec * 50 / 60:.1f} minutes")

    # 5. DDIM Reverse Sampling Latency (S=5 and S=10 steps)
    print("\n--- DDIM Reverse Sampling Latency Benchmark (1 Scene, 256x256) ---")
    denoiser.eval()
    cloudy_sample = torch.rand(1, 3, 256, 256)
    pred_sample = torch.rand(1, 3, 256, 256)
    cseb_sample = torch.randn(1, 192)

    def benchmark_ddim(num_steps: int, runs: int = 5):
        # Warm-up (1 run)
        _ = sampler.sample(cloudy_sample, pred_sample, cseb_sample, num_steps=num_steps, eta=0.0)

        ddim_times = []
        for _ in range(runs):
            t_start = time.perf_counter()
            _ = sampler.sample(cloudy_sample, pred_sample, cseb_sample, num_steps=num_steps, eta=0.0)
            t_end = time.perf_counter()
            ddim_times.append((t_end - t_start) * 1000.0)

        return {
            "steps": num_steps,
            "mean_ms": float(np.mean(ddim_times)),
            "median_ms": float(np.median(ddim_times)),
            "std_ms": float(np.std(ddim_times, ddof=1)),
            "min_ms": float(np.min(ddim_times)),
            "max_ms": float(np.max(ddim_times)),
        }

    ddim_s5 = benchmark_ddim(num_steps=5, runs=5)
    ddim_s10 = benchmark_ddim(num_steps=10, runs=5)

    print(f"  DDIM S = 5 steps (5 runs):  Mean = {ddim_s5['mean_ms']:.2f} ms | Median = {ddim_s5['median_ms']:.2f} ms")
    print(f"  DDIM S = 10 steps (5 runs): Mean = {ddim_s10['mean_ms']:.2f} ms | Median = {ddim_s10['median_ms']:.2f} ms")

    # End-to-end inference calculation: CS-TransUNet (~128 ms measured) + DDIM (measured)
    total_inf_s10_ms = 128.0 + ddim_s10['mean_ms']
    print(f"  Measured Total End-to-End Latency (CS-TransUNet 128 ms + DDIM S=10): {total_inf_s10_ms:.2f} ms (~{total_inf_s10_ms/1000:.3f} s)")

    # 6. Memory Measurement
    mem_dict = get_process_memory_mb()
    print("\n--- Memory Profiling ---")
    print(f"  Process Resident Memory (Working Set): {mem_dict['rss_mb']:.2f} MB")
    print(f"  Peak Working Set:                     {mem_dict['peak_rss_mb']:.2f} MB")
    print(f"  Pagefile Usage:                       {mem_dict['pagefile_mb']:.2f} MB")

    # Compile structured summary dictionary
    summary = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "os": system_os,
            "processor": cpu_info,
            "pytorch_version": torch_version,
            "cpu_threads": num_threads,
            "cuda_available": False,
        },
        "model_architecture": {
            "name": "ResDiffUNet",
            "channels": [32, 64, 128, 256],
            "spatial_input_channels": 9,
            "spatial_output_channels": 3,
            "cond_dim": 320,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "buffers": buffers_count,
            "fp32_size_mb": fp32_size_mb,
            "initialization_time_ms": init_time_ms,
        },
        "forward_latency_b1": {
            "mean_ms": mean_fwd_b1,
            "median_ms": median_fwd_b1,
            "std_ms": std_fwd_b1,
            "min_ms": min_fwd_b1,
            "max_ms": max_fwd_b1,
            "warmup_runs": 10,
            "measured_runs": 50,
        },
        "training_step_latency": {
            "batch_2": train_b2,
            "batch_4": train_b4,
            "projected_epoch_b4_sec": proj_epoch_b4_sec,
            "projected_epoch_b2_sec": proj_epoch_b2_sec,
            "projected_50_epoch_b4_min": proj_epoch_b4_sec * 50 / 60,
        },
        "ddim_reverse_sampling": {
            "s5": ddim_s5,
            "s10": ddim_s10,
            "cs_transunet_reference_ms": 128.0,
            "total_end_to_end_s10_ms": total_inf_s10_ms,
        },
        "memory_profile": mem_dict,
    }

    out_file = PROJECT_ROOT / "results" / "phase5" / "micro_benchmark_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved benchmark results to: {out_file}")
    print("=" * 80)
    print("MICRO-BENCHMARK COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    run_benchmark()
