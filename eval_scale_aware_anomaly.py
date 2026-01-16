#!/usr/bin/env python3
# --------------------------------------------------------
# Scale-Aware Anomaly Detection Evaluation Script
# Computes AP, AUROC, FPR95 for the new anomaly detection method
# --------------------------------------------------------

import os
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from tqdm import tqdm
import argparse

from config import get_config
from models import build_model
from data import build_loader
from logger import create_logger
from utils import load_checkpoint, get_rank, init_distributed_mode, get_local_rank


def compute_fpr_at_tpr(y_true, y_scores, tpr_threshold=0.95):
    """
    Compute False Positive Rate at a given True Positive Rate.

    Args:
        y_true: [N] binary labels (0=ID, 1=OOD)
        y_scores: [N] OOD scores (higher = more OOD)
        tpr_threshold: Target TPR (default: 0.95)

    Returns:
        fpr: False Positive Rate at target TPR
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Find threshold where TPR >= tpr_threshold
    idx = np.where(tpr >= tpr_threshold)[0]
    if len(idx) > 0:
        fpr_at_tpr = fpr[idx[0]]
    else:
        fpr_at_tpr = 1.0  # If we can't reach target TPR, FPR is 1.0

    return fpr_at_tpr


def evaluate_scale_aware_anomaly(model, data_loader, config, logger):
    """
    Evaluate Scale-Aware Anomaly Detection performance.

    Args:
        model: ScaleAwareAnomalyDetector model
        data_loader: DataLoader for test set
        config: Configuration object
        logger: Logger object

    Returns:
        metrics: Dictionary with FPR95, AP, AUROC
    """
    model.eval()

    all_ood_scores = []
    all_labels = []
    all_base_logits = []
    all_uncertainty = []
    all_scale_weights = []

    logger.info("Evaluating Scale-Aware Anomaly Detection...")

    with torch.no_grad():
        for idx, (images, ood_masks) in enumerate(tqdm(data_loader, desc="Evaluating")):
            images = images.cuda()
            ood_masks = ood_masks.cuda()

            # Forward pass
            with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                outputs = model(images)

            ood_scores = outputs['ood_scores']      # [B, 1, H, W] - Final anomaly scores
            base_logits = outputs['base_logits']    # [B, 2, H, W] - Base segmentation logits
            uncertainty = outputs['uncertainty']    # [B, 1, H, W] - Base uncertainty H(p(x))
            scale_weights = outputs['scale_weights'] # [B, 1, H, W] - Scale weights w(x)

            # Resize masks to match output size if needed
            if ood_masks.shape[1:] != ood_scores.shape[1:]:
                ood_masks = torch.nn.functional.interpolate(
                    ood_masks.unsqueeze(1).float(),
                    size=ood_scores.shape[1:],
                    mode='nearest'
                ).squeeze(1).long()

            # Flatten to pixel-level
            ood_scores_flat = ood_scores.cpu().numpy().flatten()
            base_logits_flat = base_logits.cpu().numpy().reshape(base_logits.shape[0], -1, base_logits.shape[1]).transpose(0, 2, 1).reshape(base_logits.shape[1], -1)
            uncertainty_flat = uncertainty.cpu().numpy().flatten()
            scale_weights_flat = scale_weights.cpu().numpy().flatten()
            ood_masks_flat = ood_masks.cpu().numpy().flatten()

            all_ood_scores.append(ood_scores_flat)
            all_base_logits.append(base_logits_flat)
            all_uncertainty.append(uncertainty_flat)
            all_scale_weights.append(scale_weights_flat)
            all_labels.append(ood_masks_flat)

    # Concatenate all scores and labels
    all_ood_scores = np.concatenate(all_ood_scores)
    all_base_logits = np.concatenate(all_base_logits, axis=1)
    all_uncertainty = np.concatenate(all_uncertainty)
    all_scale_weights = np.concatenate(all_scale_weights)
    all_labels = np.concatenate(all_labels)

    # Remove ignore pixels (if any)
    valid_mask = all_labels >= 0  # Assuming -1 or negative values are ignore
    all_ood_scores = all_ood_scores[valid_mask]
    all_base_logits = all_base_logits[:, valid_mask]
    all_uncertainty = all_uncertainty[valid_mask]
    all_scale_weights = all_scale_weights[valid_mask]
    all_labels = all_labels[valid_mask]

    # Convert to binary: 0=ID, 1=OOD
    if all_labels.max() > 1:
        # Likely 0/255 format, threshold at 128
        all_labels_binary = (all_labels >= 128).astype(np.int32)
    else:
        # Already 0/1 format
        all_labels_binary = (all_labels > 0).astype(np.int32)

    # Compute metrics for final OOD scores
    logger.info("Computing metrics for Scale-Aware OOD Scores...")

    # AUROC
    try:
        auroc = roc_auc_score(all_labels_binary, all_ood_scores)
    except ValueError:
        logger.warning("Could not compute AUROC (possibly all labels are same class)")
        auroc = 0.0

    # AP (Average Precision)
    try:
        ap = average_precision_score(all_labels_binary, all_ood_scores)
    except ValueError:
        logger.warning("Could not compute AP")
        ap = 0.0

    # FPR at 95% TPR
    fpr95 = compute_fpr_at_tpr(all_labels_binary, all_ood_scores, tpr_threshold=0.95)

    # Also compute metrics for baseline uncertainty (entropy)
    logger.info("Computing metrics for Baseline Uncertainty (Entropy)...")
    try:
        baseline_auroc = roc_auc_score(all_labels_binary, all_uncertainty)
        baseline_ap = average_precision_score(all_labels_binary, all_uncertainty)
        baseline_fpr95 = compute_fpr_at_tpr(all_labels_binary, all_uncertainty, tpr_threshold=0.95)
    except ValueError:
        logger.warning("Could not compute baseline metrics")
        baseline_auroc = baseline_ap = baseline_fpr95 = 0.0

    # Log results
    logger.info("=== Scale-Aware Anomaly Detection Results ===")
    logger.info(".4f")
    logger.info(".4f")
    logger.info(".4f")
    logger.info("")
    logger.info("=== Baseline Entropy Results ===")
    logger.info(".4f")
    logger.info(".4f")
    logger.info(".4f")

    # Statistical analysis of scale weights
    logger.info("")
    logger.info("=== Scale Weight Analysis ===")
    logger.info(".4f")
    logger.info(".4f")
    logger.info(".4f")
    logger.info(".4f")
    logger.info(".4f")

    # Correlation analysis
    from scipy.stats import pearsonr
    try:
        corr_uncertainty_weight, _ = pearsonr(all_uncertainty, all_scale_weights)
        corr_score_weight, _ = pearsonr(all_ood_scores, all_scale_weights)
        logger.info(".4f")
        logger.info(".4f")
    except:
        logger.warning("Could not compute correlations")

    metrics = {
        'auroc': auroc,
        'ap': ap,
        'fpr95': fpr95,
        'baseline_auroc': baseline_auroc,
        'baseline_ap': baseline_ap,
        'baseline_fpr95': baseline_fpr95,
        'scale_weight_mean': np.mean(all_scale_weights),
        'scale_weight_std': np.std(all_scale_weights),
        'scale_weight_min': np.min(all_scale_weights),
        'scale_weight_max': np.max(all_scale_weights),
    }

    return metrics


def main():
    parser = argparse.ArgumentParser("Scale-Aware Anomaly Detection Evaluation", add_help=True)
    parser.add_argument("--cfg", type=str, metavar="FILE", help="path to config file")
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs.",
        default=None,
        nargs="+",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size for evaluation")
    parser.add_argument("--data-path", type=str, help="path to dataset")
    parser.add_argument("--resume", type=str, help="path to checkpoint")
    parser.add_argument(
        "--output",
        default="eval_output",
        type=str,
        metavar="PATH",
        help="output directory",
    )

    args = parser.parse_args()

    # Get config
    config = get_config(args)

    # Override model type
    config.defrost()
    config.MODEL.TYPE = 'scale_aware_anomaly'
    config.freeze()

    # Initialize distributed mode (but we use single GPU for evaluation)
    init_distributed_mode()

    # Create logger
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=get_rank(), name="eval_scale_aware")

    if get_rank() == 0:
        logger.info(config.dump())

    # Build model
    logger.info("Building model...")
    model = build_model(config)
    model.cuda()

    # Load checkpoint
    if args.resume:
        logger.info(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')

        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Handle module prefix for DDP
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict, strict=False)
        logger.info("Checkpoint loaded successfully")

    # Load backbone weights if specified
    if hasattr(config.MODEL, 'BACKBONE_PRETRAIN_PATH') and config.MODEL.BACKBONE_PRETRAIN_PATH:
        logger.info(f"Loading backbone weights from {config.MODEL.BACKBONE_PRETRAIN_PATH}")
        model.load_backbone_weights(config.MODEL.BACKBONE_PRETRAIN_PATH)

    # Build data loader
    logger.info("Building data loader...")
    _, test_loader, _ = build_loader(config)

    # Evaluate
    logger.info("Starting evaluation...")
    metrics = evaluate_scale_aware_anomaly(model, test_loader, config, logger)

    # Save results
    results_file = os.path.join(config.OUTPUT, 'evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write("Scale-Aware Anomaly Detection Evaluation Results\n")
        f.write("=" * 50 + "\n")
        f.write(".4f\n")
        f.write(".4f\n")
        f.write(".4f\n")
        f.write("\nBaseline Entropy Results:\n")
        f.write(".4f\n")
        f.write(".4f\n")
        f.write(".4f\n")
        f.write("\nScale Weight Statistics:\n")
        f.write(".4f\n")
        f.write(".4f\n")
        f.write(".4f\n")
        f.write(".4f\n")

    logger.info(f"Results saved to {results_file}")
    logger.info("Evaluation completed.")


if __name__ == '__main__':
    main()
