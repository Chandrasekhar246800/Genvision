import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image
import torchvision.transforms as transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unet import UNet
from attention_unet import AttentionUNet
from models.transformer.cs_transunet import CSTransUNet

def run_audit():
    print("=" * 80)
    print("PHASE 4.3.1 REPRODUCIBILITY AUDIT SCRIPT")
    print("=" * 80)

    # 1. Parameter and Buffer Reconciliation
    print("\n--- 1. Parameter & Buffer Reconciliation ---")
    models = {
        "unet": UNet(),
        "attention_unet": AttentionUNet(),
        "cs_transunet": CSTransUNet(),
    }

    for name, model in models.items():
        total_p = sum(p.numel() for p in model.parameters())
        trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_p = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        buffers = sum(b.numel() for b in model.buffers())
        sd_keys = len(model.state_dict().keys())
        fp32_bytes = total_p * 4
        fp32_mb = fp32_bytes / (1024 * 1024)
        print(f"[{name}]")
        print(f"  Total Params:        {total_p:,}")
        print(f"  Trainable:           {trainable_p:,}")
        print(f"  Non-Trainable:       {non_trainable_p:,}")
        print(f"  Persistent Buffers:  {buffers:,}")
        print(f"  State Dict Keys:     {sd_keys}")
        print(f"  FP32 Param Memory:   {fp32_bytes:,} B ({fp32_mb:.2f} MB)")

    # 2. Checkpoint Disk Audit
    print("\n--- 2. Checkpoint Disk Audit ---")
    ckpts = {
        "unet": Path("models/unet_controlled_best.pth"),
        "attention_unet": Path("models/attention_controlled_best.pth"),
        "cs_transunet": Path("models/transformer_controlled_best.pth"),
    }
    for name, p in ckpts.items():
        size = p.stat().st_size
        size_mb = size / (1024 * 1024)
        ckpt_data = torch.load(p, map_location="cpu", weights_only=False)
        print(f"[{name}] ({p.name})")
        print(f"  Disk Size:           {size:,} bytes ({size_mb:.2f} MB)")
        if isinstance(ckpt_data, dict):
            if "model_state_dict" in ckpt_data:
                print(f"  Format:              Full checkpoint container")
                print(f"  Keys:                {list(ckpt_data.keys())}")
                print(f"  Epoch:               {ckpt_data.get('epoch')}")
                print(f"  Val Loss:            {ckpt_data.get('val_loss')}")
                # check size of optimizer state
                if "optimizer_state_dict" in ckpt_data:
                    opt_state = ckpt_data["optimizer_state_dict"]["state"]
                    n_opt_tensors = sum(len(v) for v in opt_state.values() if isinstance(v, dict))
                    print(f"  Optimizer states:    {len(opt_state)} param entries, {n_opt_tensors} state tensors (AdamW moments)")
            else:
                print(f"  Format:              Direct state_dict (weights/buffers only)")
                print(f"  Keys Count:          {len(ckpt_data.keys())} tensors")
        else:
            print(f"  Format:              Direct tensor/object: {type(ckpt_data)}")

    # 3. Independent Recalculation on 5 Sample Test Pairs
    print("\n--- 3. Independent Recalculation on 5 Sample Test Pairs ---")
    # Load data_split.json
    with open("results/data_split.json", "r", encoding="utf-8") as f:
        split = json.load(f)
    test_indices = split["test"]
    filenames = split["filenames"]

    # Load per_image_metrics.csv
    csv_rows = []
    with open("results/phase4/evaluation/per_image_metrics.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_rows.append(row)

    # Pick 5 deterministically spread test samples
    # e.g., index 0, 18, 37, 56, 74 from test_indices
    sample_indices = [test_indices[0], test_indices[18], test_indices[37], test_indices[56], test_indices[74]]
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Load trained weights
    loaded_models = {}
    for name, p in ckpts.items():
        m = models[name]
        data = torch.load(p, map_location="cpu", weights_only=True)
        sd = data["model_state_dict"] if isinstance(data, dict) and "model_state_dict" in data else data
        m.load_state_dict(sd)
        m.eval()
        loaded_models[name] = m

    all_matched = True
    print(f"Testing {len(sample_indices)} sample scenes...")
    for s_idx in sample_indices:
        fname = filenames[s_idx]
        cloudy_path = Path("dataset/cloudy") / fname
        clean_path = Path("dataset/clean") / fname

        img_cloudy = Image.open(cloudy_path).convert("RGB")
        img_clean = Image.open(clean_path).convert("RGB")

        cloudy_t = transform(img_cloudy).unsqueeze(0)
        clean_t = transform(img_clean)

        for m_name in ["unet", "attention_unet", "cs_transunet"]:
            m = loaded_models[m_name]
            with torch.inference_mode():
                if m_name == "cs_transunet":
                    pred = m(cloudy_t, return_dict=False)
                else:
                    pred = m(cloudy_t)
            pred_t = pred.squeeze(0).clamp(0.0, 1.0)
            clean_clamp = clean_t.clamp(0.0, 1.0)

            # Compute metrics
            l1_val = torch.mean(torch.abs(pred_t - clean_clamp)).item()
            pred_np = pred_t.numpy()
            clean_np = clean_clamp.numpy()
            psnr_val = float(peak_signal_noise_ratio(clean_np, pred_np, data_range=1.0))
            ssim_val = float(structural_similarity(clean_np, pred_np, channel_axis=0, data_range=1.0))

            # Match with CSV
            csv_matches = [r for r in csv_rows if int(r["index"]) == s_idx and r["model"] == m_name]
            if not csv_matches:
                print(f"ERROR: No CSV row for index {s_idx}, model {m_name}")
                all_matched = False
                continue
            csv_row = csv_matches[0]
            csv_psnr = float(csv_row["psnr"])
            csv_ssim = float(csv_row["ssim"])
            csv_l1 = float(csv_row["l1_loss"])

            psnr_diff = abs(psnr_val - csv_psnr)
            ssim_diff = abs(ssim_val - csv_ssim)
            l1_diff = abs(l1_val - csv_l1)

            max_diff = max(psnr_diff, ssim_diff, l1_diff)
            match_status = "EXACT MATCH (< 1e-6)" if max_diff < 1e-6 else f"DIFF {max_diff:.2e}"
            if max_diff >= 1e-5:
                all_matched = False
                print(f"  MISMATCH on {fname} ({m_name}): PSNR diff={psnr_diff}, SSIM diff={ssim_diff}, L1 diff={l1_diff}")
            else:
                print(f"  {fname} [{m_name}]: PSNR={psnr_val:.4f}, SSIM={ssim_val:.4f}, L1={l1_val:.4f} -> {match_status}")

    print(f"\nMetric recalculation status across samples: {'ALL IDENTICAL' if all_matched else 'FAILED'}")

    # 4. Statistical Verification
    print("\n--- 4. Statistical Verification on Full 75-Test Set ---")
    wide_comp = {}
    for m_name in ["unet", "attention_unet", "cs_transunet"]:
        wide_comp[m_name] = {
            "psnr": np.array([float(r["psnr"]) for r in csv_rows if r["model"] == m_name]),
            "ssim": np.array([float(r["ssim"]) for r in csv_rows if r["model"] == m_name]),
            "l1": np.array([float(r["l1_loss"]) for r in csv_rows if r["model"] == m_name]),
        }

    for baseline in ["unet", "attention_unet"]:
        print(f"\nCS-TransUNet vs {baseline}:")
        for metric, lower_better in [("psnr", False), ("ssim", False), ("l1", True)]:
            cs_arr = wide_comp["cs_transunet"][metric]
            base_arr = wide_comp[baseline][metric]
            diff = cs_arr - base_arr

            # Check zero diffs
            zero_count = int(np.sum(diff == 0.0))
            n = len(diff)
            mean_d = float(np.mean(diff))
            median_d = float(np.median(diff))
            std_d = float(np.std(diff, ddof=1))
            sem = std_d / math.sqrt(n)
            t_crit = float(stats.t.ppf(0.975, df=n - 1))
            ci_low, ci_high = mean_d - t_crit * sem, mean_d + t_crit * sem

            t_res = stats.ttest_rel(cs_arr, base_arr)
            # wilcoxon
            w_res = stats.wilcoxon(diff, zero_method="wilcox")

            print(f"  [{metric.upper()}] Mean Diff: {mean_d:+.6f}, Median Diff: {median_d:+.6f}, 95% CI: [{ci_low:+.6f}, {ci_high:+.6f}]")
            print(f"       Zero differences: {zero_count} / {n}")
            print(f"       Paired t-test:   t = {t_res.statistic:.4f}, p = {t_res.pvalue:.4e}")
            print(f"       Wilcoxon:        W = {w_res.statistic:.1f}, p = {w_res.pvalue:.4e}")

            # Multiple comparisons check
            # Bonferroni corrected threshold for 6 tests (3 metrics x 2 baselines) at alpha=0.01:
            bonferroni_alpha_01 = 0.01 / 6
            bonferroni_alpha_05 = 0.05 / 6
            print(f"       Bonferroni p < 0.01/6 ({bonferroni_alpha_01:.4e}): t-test {t_res.pvalue < bonferroni_alpha_01}, Wilcoxon {w_res.pvalue < bonferroni_alpha_01}")

    # 5. CS-TransUNet SSIM Distribution Analysis
    print("\n--- 5. CS-TransUNet SSIM Distribution Analysis ---")
    cs_ssim_all = wide_comp["cs_transunet"]["ssim"]
    min_ssim = float(np.min(cs_ssim_all))
    q25_ssim = float(np.percentile(cs_ssim_all, 25))
    med_ssim = float(np.median(cs_ssim_all))
    q75_ssim = float(np.percentile(cs_ssim_all, 75))
    max_ssim = float(np.max(cs_ssim_all))
    mean_ssim = float(np.mean(cs_ssim_all))
    std_ssim = float(np.std(cs_ssim_all, ddof=1))
    sub_90_count = int(np.sum(cs_ssim_all < 0.90))
    sub_90_pct = (sub_90_count / len(cs_ssim_all)) * 100.0

    print(f"  Count:             {len(cs_ssim_all)}")
    print(f"  Mean ± Std:        {mean_ssim:.6f} ± {std_ssim:.6f}")
    print(f"  Min:               {min_ssim:.6f}")
    print(f"  25th Percentile:   {q25_ssim:.6f}")
    print(f"  Median:            {med_ssim:.6f}")
    print(f"  75th Percentile:   {q75_ssim:.6f}")
    print(f"  Max:               {max_ssim:.6f}")
    print(f"  Scenes < 0.90 SSIM:{sub_90_count} / {len(cs_ssim_all)} ({sub_90_pct:.2f}%)")
    print(f"  Scenes >= 0.90 SSIM: {len(cs_ssim_all) - sub_90_count} / {len(cs_ssim_all)} ({100.0 - sub_90_pct:.2f}%)")

    # List the scenes where CS-TransUNet SSIM < 0.90
    sub_90_scenes = [
        (int(r["index"]), r["filename"], float(r["ssim"]), float(r["psnr"]))
        for r in csv_rows
        if r["model"] == "cs_transunet" and float(r["ssim"]) < 0.90
    ]
    sub_90_scenes.sort(key=lambda x: x[2])
    print("\n  Scenes with SSIM < 0.90:")
    for s in sub_90_scenes:
        print(f"    Index {s[0]:03d} ({s[1]}): SSIM = {s[2]:.4f}, PSNR = {s[3]:.2f} dB")

if __name__ == "__main__":
    run_audit()
