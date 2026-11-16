import json
import math
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.transformer.cs_transunet import CSTransUNet

def compute_split_residuals(split_name, indices, filenames, model, transform):
    print(f"\nProcessing {split_name} split ({len(indices)} images)...")
    residuals_list = []
    
    with torch.inference_mode():
        for idx_num, idx in enumerate(indices):
            fname = filenames[idx]
            cloudy_path = PROJECT_ROOT / "dataset" / "cloudy" / fname
            clean_path = PROJECT_ROOT / "dataset" / "clean" / fname
            
            img_c = Image.open(cloudy_path).convert("RGB")
            img_gt = Image.open(clean_path).convert("RGB")
            
            t_c = transform(img_c).unsqueeze(0)  # (1, 3, 256, 256)
            t_gt = transform(img_gt)             # (3, 256, 256)
            
            # Predict with frozen CS-TransUNet
            pred = model(t_c, return_dict=False).squeeze(0).clamp(0.0, 1.0)
            
            # Residual = clean - pred
            r = (t_gt - pred).numpy() # (3, 256, 256)
            residuals_list.append(r.flatten())
            
            if (idx_num + 1) % 50 == 0 or (idx_num + 1) == len(indices):
                print(f"  Processed {idx_num + 1}/{len(indices)} images...")
                
    all_res = np.concatenate(residuals_list)
    return all_res

def main():
    print("=" * 80)
    print("PHASE 5.1.1: EMPIRICAL RESIDUAL STATISTICS AUDIT (TRAIN & VAL ONLY)")
    print("=" * 80)
    
    # Load data split
    split_path = PROJECT_ROOT / "results" / "data_split.json"
    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)
        
    train_indices = split["train"]
    val_indices = split["validation"]
    test_indices = set(split["test"])
    filenames = split["filenames"]
    
    # Strict assertion: Never touch test indices
    assert len(set(train_indices) & test_indices) == 0, "DATA LEAKAGE: train overlaps test"
    assert len(set(val_indices) & test_indices) == 0, "DATA LEAKAGE: val overlaps test"
    print(f"Verified strict isolation: Test set ({len(test_indices)} scenes) will NOT be accessed.")
    
    # Load frozen CS-TransUNet
    model_path = PROJECT_ROOT / "models" / "transformer_controlled_best.pth"
    print(f"Loading frozen checkpoint: {model_path.name}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    
    model = CSTransUNet()
    model.load_state_dict(state_dict)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # Process Train Split
    train_res = compute_split_residuals("TRAIN", train_indices, filenames, model, transform)
    val_res = compute_split_residuals("VALIDATION", val_indices, filenames, model, transform)
    
    def analyze_stats(name, res):
        mean_val = float(np.mean(res))
        std_val = float(np.std(res))
        mae_val = float(np.mean(np.abs(res)))
        min_val = float(np.min(res))
        max_val = float(np.max(res))
        
        percentiles = [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
        pct_vals = {p: float(np.percentile(res, p)) for p in percentiles}
        abs_pct_vals = {p: float(np.percentile(np.abs(res), p)) for p in percentiles}
        
        # Scaling evaluations
        # Scale 1: r
        out_1 = float(np.mean((res < -1.5) | (res > 1.5)) * 100.0)
        # Scale 4: 4r
        out_4 = float(np.mean((4.0 * res < -1.5) | (4.0 * res > 1.5)) * 100.0)
        # Scale by 1 / (3 * std): maps ~99.7% of Gaussian to [-1, 1]
        s_3std = 1.0 / (3.0 * std_val) if std_val > 0 else 1.0
        out_3std = float(np.mean(((s_3std * res) < -1.5) | ((s_3std * res) > 1.5)) * 100.0)
        # Scale by 1 / (2 * std): maps ~95% of Gaussian to [-1, 1]
        s_2std = 1.0 / (2.0 * std_val) if std_val > 0 else 1.0
        out_2std = float(np.mean(((s_2std * res) < -1.5) | ((s_2std * res) > 1.5)) * 100.0)
        # Scale by 1 / std: unit variance
        s_1std = 1.0 / std_val if std_val > 0 else 1.0
        out_1std = float(np.mean(((s_1std * res) < -1.5) | ((s_1std * res) > 1.5)) * 100.0)
        
        print(f"\n--- Statistics for {name} (N = {len(res):,} elements) ---")
        print(f"  Mean:                  {mean_val:+.6f}")
        print(f"  Std Dev (sigma):       {std_val:.6f}")
        print(f"  Mean Absolute Error:   {mae_val:.6f}")
        print(f"  Min / Max:             [{min_val:+.6f}, {max_val:+.6f}]")
        print(f"  Percentiles (signed):")
        for p in percentiles:
            print(f"    p{p:04.1f}%: {pct_vals[p]:+.6f}")
        print(f"  Percentiles (|residual|):")
        for p in percentiles:
            print(f"    p{p:04.1f}%: {abs_pct_vals[p]:.6f}")
        print(f"\n  Scaling Comparison for {name}:")
        print(f"    Factor 1.0 (raw r):               Out of [-1.5, 1.5] = {out_1:.4f}% | Max |scaled| = {max(abs(min_val), abs(max_val)):.4f}")
        print(f"    Factor 4.0 (4.0 * r):             Out of [-1.5, 1.5] = {out_4:.4f}% | Max |scaled| = {4.0 * max(abs(min_val), abs(max_val)):.4f}")
        print(f"    Factor 1/std ({s_1std:.2f} * r):          Out of [-1.5, 1.5] = {out_1std:.4f}% | Max |scaled| = {s_1std * max(abs(min_val), abs(max_val)):.4f}")
        print(f"    Factor 1/(2*std) ({s_2std:.2f} * r):      Out of [-1.5, 1.5] = {out_2std:.4f}% | Max |scaled| = {s_2std * max(abs(min_val), abs(max_val)):.4f}")
        print(f"    Factor 1/(3*std) ({s_3std:.2f} * r):      Out of [-1.5, 1.5] = {out_3std:.4f}% | Max |scaled| = {s_3std * max(abs(min_val), abs(max_val)):.4f}")
        
        return {
            "mean": mean_val,
            "std": std_val,
            "mae": mae_val,
            "min": min_val,
            "max": max_val,
            "pct": pct_vals,
            "abs_pct": abs_pct_vals,
            "scaling": {
                "factor_1_out_pct": out_1,
                "factor_4_out_pct": out_4,
                "factor_1std": s_1std,
                "factor_1std_out_pct": out_1std,
                "factor_2std": s_2std,
                "factor_2std_out_pct": out_2std,
                "factor_3std": s_3std,
                "factor_3std_out_pct": out_3std,
            }
        }

    train_stats = analyze_stats("TRAINING SET", train_res)
    val_stats = analyze_stats("VALIDATION SET", val_res)
    
    # Save results to json for reporting
    out_json = {
        "train_scenes_count": len(train_indices),
        "val_scenes_count": len(val_indices),
        "test_scenes_isolated": len(test_indices),
        "train_statistics": train_stats,
        "val_statistics": val_stats,
    }
    
    out_path = PROJECT_ROOT / "results" / "phase5" / "residual_statistics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)
    print(f"\nSaved statistical results to: {out_path}")

if __name__ == "__main__":
    main()
