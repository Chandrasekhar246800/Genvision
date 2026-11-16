"""
Validation-Calibrated Residual Gating for GenVision ResDiff-TransUNet
=====================================================================

Implements the residual integration gating mechanism:
    x_final = clamp(x_hat_0 + lambda * predicted_residual, 0.0, 1.0)

Design Rules:
- lambda in [0.0, 1.0] controls the blending strength of residual refinement.
- When lambda = 0.0, x_final is identically equal to the frozen CS-TransUNet prediction.
- Calibration is conducted strictly on validation scenes (never test scenes).
- Primary calibration metric: PSNR (dB).
- Secondary metrics: SSIM and L1 loss.
- Calibration provides an empirical fallback on validation data, but does NOT guarantee
  universal improvement across all individual images or unseen test sets.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
import torch.nn as nn


class ResidualGating:
    """
    Manages residual integration, validation calibration, and baseline fallback.
    """

    def __init__(self, lambda_val: float = 1.0):
        self.set_lambda(lambda_val)

    def set_lambda(self, lambda_val: float):
        """Set the residual gating factor within [0.0, 1.0]."""
        val = float(lambda_val)
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"lambda must be within [0.0, 1.0], got {val}")
        self.lambda_val = val

    def integrate(
        self,
        transformer_prediction: torch.Tensor,
        predicted_residual: torch.Tensor,
        lambda_val: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Integrate predicted residual with deterministic baseline:
        x_final = clamp(x_hat_0 + lambda * r_hat_0, 0.0, 1.0)
        
        Args:
            transformer_prediction: [B, 3, H, W] in [0, 1]
            predicted_residual: [B, 3, H, W] in [-1, 1]
            lambda_val: Optional override for gating factor
            
        Returns:
            Clamped final reconstruction [B, 3, H, W] in [0, 1].
        """
        lam = self.lambda_val if lambda_val is None else float(lambda_val)
        if not (0.0 <= lam <= 1.0):
            raise ValueError(f"lambda must be in [0.0, 1.0], got {lam}")

        if lam == 0.0:
            return transformer_prediction.clamp(0.0, 1.0)

        refined = transformer_prediction + lam * predicted_residual
        return refined.clamp(0.0, 1.0)

    @staticmethod
    def evaluate_metrics(
        pred_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute standard metrics (PSNR, SSIM, L1) for evaluation."""
        pred = pred_tensor.detach().cpu().clamp(0.0, 1.0)
        target = target_tensor.detach().cpu().clamp(0.0, 1.0)

        l1 = torch.mean(torch.abs(pred - target)).item()

        pred_np = pred.numpy()
        target_np = target.numpy()

        psnr = float(peak_signal_noise_ratio(target_np, pred_np, data_range=1.0))
        ssim = float(structural_similarity(target_np, pred_np, channel_axis=0, data_range=1.0))

        return {"psnr": psnr, "ssim": ssim, "l1": l1}

    def calibrate(
        self,
        val_predictions: List[torch.Tensor],
        val_residuals: List[torch.Tensor],
        val_targets: List[torch.Tensor],
        lambda_candidates: Optional[List[float]] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Union[float, Dict[str, Dict[str, float]]]]:
        """
        Calibrate lambda strictly across validation scene predictions.
        
        Args:
            val_predictions: List of transformer predictions [3, H, W]
            val_residuals: List of diffusion-predicted residuals [3, H, W]
            val_targets: List of clean ground-truth targets [3, H, W]
            lambda_candidates: List of candidate lambdas (default: 0.0 to 1.0 in 0.1 steps)
            save_path: Optional path to save calibration metadata JSON
            
        Returns:
            Dictionary with optimal lambda and metric curves.
        """
        if lambda_candidates is None:
            lambda_candidates = [round(x, 2) for x in np.linspace(0.0, 1.0, 11).tolist()]

        n_scenes = len(val_predictions)
        if n_scenes == 0 or len(val_residuals) != n_scenes or len(val_targets) != n_scenes:
            raise ValueError("Mismatched or empty validation lists passed for calibration")

        results_by_lambda = {}
        best_lambda = 0.0
        best_psnr = -float("inf")

        for lam in lambda_candidates:
            psnr_list, ssim_list, l1_list = [], [], []
            for pred_t, res_t, gt_t in zip(val_predictions, val_residuals, val_targets):
                refined_t = self.integrate(pred_t, res_t, lambda_val=lam)
                m = self.evaluate_metrics(refined_t, gt_t)
                psnr_list.append(m["psnr"])
                ssim_list.append(m["ssim"])
                l1_list.append(m["l1"])

            mean_psnr = float(np.mean(psnr_list))
            mean_ssim = float(np.mean(ssim_list))
            mean_l1 = float(np.mean(l1_list))

            results_by_lambda[str(lam)] = {
                "psnr": mean_psnr,
                "ssim": mean_ssim,
                "l1": mean_l1,
            }

            if mean_psnr > best_psnr:
                best_psnr = mean_psnr
                best_lambda = lam

        self.set_lambda(best_lambda)

        calibration_report = {
            "best_lambda": best_lambda,
            "best_val_psnr": best_psnr,
            "baseline_val_psnr": results_by_lambda.get("0.0", {}).get("psnr", None),
            "results_by_lambda": results_by_lambda,
            "evaluated_scenes_count": n_scenes,
            "calibrated_partition": "validation",
        }

        if save_path is not None:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(calibration_report, f, indent=2)

        return calibration_report
