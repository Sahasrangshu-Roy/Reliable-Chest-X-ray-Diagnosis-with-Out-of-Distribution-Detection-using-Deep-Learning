#!/usr/bin/env python
import argparse
import os
import torch
import numpy as np

from src.dataset import load_nih_data, load_chexpert_data, create_dataloaders, OODDataset, ALL_LABELS
from src.model import get_densenet_model
from src.ood import (
    compute_msp_and_energy,
    extract_penultimate_features,
    extract_features_and_labels,
    compute_global_mahalanobis,
    fit_multi_cluster_gaussians,
    compute_multi_cluster_mahalanobis
)
from src.evaluate import (
    compute_per_class_auroc,
    evaluate_ood_detection,
    evaluate_risk_mitigation,
    compute_brier_scores
)
from src.drift import (
    simulate_drift_scenario,
    apply_gaussian_noise,
    apply_contrast_reduction,
    apply_blur
)
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score


def main():
    parser = argparse.ArgumentParser(description="Evaluate Chest Disease Classifier & OOD Detection Algorithms")
    parser.add_argument("--nih_dir", type=str, default=".", help="Path to NIH ChestX-ray14 directory")
    parser.add_argument("--chexpert_dir", type=str, default="./chexpert", help="Path to CheXpert dataset directory")
    parser.add_argument("--checkpoint", type=str, default="best_model.pth", help="Path to best model weights")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file '{args.checkpoint}' not found.")
        print("Please train the model first using 'python train.py' or provide a valid path via --checkpoint.")
        return

    # 1. Load Data
    print("\n--- Loading NIH Dataset ---")
    train_df, val_df, test_df, _ = load_nih_data(args.nih_dir)
    train_loader, val_loader, test_loader, eval_transform = create_dataloaders(
        train_df, val_df, test_df, batch_size=args.batch_size
    )

    print("\n--- Loading CheXpert OOD Dataset ---")
    try:
        df_chex = load_chexpert_data(args.chexpert_dir)
        ood_dataset = OODDataset(df_chex["full_path"].tolist(), transform=eval_transform)
        ood_loader  = DataLoader(ood_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        print(f"Loaded {len(ood_dataset)} CheXpert frontal OOD images.")
    except Exception as e:
        print(f"Warning: Could not load CheXpert dataset ({e}). Skipping CheXpert benchmarks.")
        ood_loader = None

    # 2. Load Model
    model = get_densenet_model(num_classes=14, pretrained=False, checkpoint_path=args.checkpoint, device=device)
    model.eval()

    # 3. Validation Performance
    print("\n" + "=" * 65)
    print("VALIDATION SET PER-CLASS AUROC")
    print("=" * 65)
    
    val_logits, val_labels = [], []
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            logits = model(images)
            val_logits.append(logits.cpu().numpy())
            val_labels.append(labels.numpy())

    val_logits = np.concatenate(val_logits, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)
    val_probs  = 1 / (1 + np.exp(-val_logits))

    val_results, val_macro = compute_per_class_auroc(val_labels, val_probs)
    for label, metrics in val_results.items():
        if metrics["auroc"] is not None:
            print(f"  {label:<22} AUROC: {metrics['auroc']:.4f}  (positives: {metrics['positives']})")
    print("-" * 65)
    print(f"  {'Macro AUROC':<22} {val_macro:.4f}")

    if ood_loader is None:
        return

    # 4. OOD Benchmarks (MSP vs Energy Score vs Mahalanobis)
    print("\n" + "=" * 65)
    print("OOD DETECTION BENCHMARK (ID = NIH Val, OOD = CheXpert)")
    print("=" * 65)

    id_msp, id_energy   = compute_msp_and_energy(model, val_loader, device)
    ood_msp, ood_energy = compute_msp_and_energy(model, ood_loader, device)

    eval_msp    = evaluate_ood_detection(id_msp, ood_msp, flip_scores=True)
    eval_energy = evaluate_ood_detection(id_energy, ood_energy, flip_scores=True)

    print("Extracting features for Mahalanobis OOD detection...")
    val_features  = extract_penultimate_features(model, val_loader, device)
    ood_features  = extract_penultimate_features(model, ood_loader, device)
    
    # Train loader ordered (no shuffle) for multi-cluster alignment
    train_loader_ordered = DataLoader(
        train_loader.dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    train_features_aligned, train_labels_aligned = extract_features_and_labels(model, train_loader_ordered, device)

    # Global Mahalanobis
    val_mahal_global = compute_global_mahalanobis(train_features_aligned, val_features)
    ood_mahal_global = compute_global_mahalanobis(train_features_aligned, ood_features)
    eval_global_mahal = evaluate_ood_detection(val_mahal_global, ood_mahal_global, flip_scores=False)

    # Multi-Cluster Mahalanobis
    mus, cov_invs = fit_multi_cluster_gaussians(train_features_aligned, train_labels_aligned)
    val_mahal_multi = compute_multi_cluster_mahalanobis(val_features, mus, cov_invs)
    ood_mahal_multi = compute_multi_cluster_mahalanobis(ood_features, mus, cov_invs)
    eval_multi_mahal = evaluate_ood_detection(val_mahal_multi, ood_mahal_multi, flip_scores=False)

    print(f"{'Method':<30} {'AUROC':>8} {'FPR@95TPR':>12}")
    print("-" * 65)
    print(f"{'MSP':<30} {eval_msp['auroc']:>8.4f} {eval_msp['fpr_at_95tpr']:>12.4f}")
    print(f"{'Energy Score':<30} {eval_energy['auroc']:>8.4f} {eval_energy['fpr_at_95tpr']:>12.4f}")
    print(f"{'Global Mahalanobis':<30} {eval_global_mahal['auroc']:>8.4f} {eval_global_mahal['fpr_at_95tpr']:>12.4f}")
    print(f"{'Multi-Cluster Mahalanobis':<30} {eval_multi_mahal['auroc']:>8.4f} {eval_multi_mahal['fpr_at_95tpr']:>12.4f}")
    print("=" * 65)

    # 5. Risk Mitigation Analysis
    print("\n" + "=" * 65)
    print("RISK MITIGATION & ERROR SUPPRESSION ANALYSIS")
    print("=" * 65)
    risk_results = evaluate_risk_mitigation(val_probs, val_labels, id_energy)
    print(f"Total Val Images:           {len(val_labels)}")
    print(f"Baseline High-Conf Errors:  {risk_results['baseline_hce']} ({risk_results['baseline_hce_rate']:.3f})")
    print(f"{'Suppressed %':>12} {'Coverage':>10} {'HCE Rate':>10} {'HCE Reduction':>15}")
    print("-" * 65)
    for pct, cov, hce, red in zip(risk_results['percentiles'][::2], risk_results['coverages'][::2], risk_results['hce_rates'][::2], risk_results['reductions'][::2]):
        print(f"{pct:>12.0f}% {cov:>10.3f} {hce:>10.3f} {red:>14.1f}%")

    # 6. Drift Simulation
    print("\n" + "=" * 65)
    print("SYNTHETIC DATA DRIFT SIMULATION")
    print("=" * 65)
    severities = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    lower_bound = np.percentile(id_energy, 20)
    upper_bound = np.percentile(id_energy, 80)
    
    test_subset = test_df.sample(min(500, len(test_df)), random_state=42).reset_index(drop=True)
    corrupt_fns = {
        "Gaussian Noise": apply_gaussian_noise,
        "Contrast Reduction": apply_contrast_reduction,
        "Blur": apply_blur
    }
    
    for name, fn in corrupt_fns.items():
        rates = simulate_drift_scenario(model, test_subset, eval_transform, fn, severities, lower_bound, upper_bound, device)
        print(f"  {name:<20} OOD Alert Rates (sev 0.0 -> 1.0): {[round(r, 3) for r in rates]}")

    # 7. Final Test Evaluation
    print("\n" + "=" * 65)
    print("FINAL TEST SET EVALUATION SUMMARY")
    print("=" * 65)

    test_logits, test_labels = [], []
    test_energy_scores = []
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            logits = model(images)
            energy = -torch.logsumexp(logits, dim=1)
            test_logits.append(logits.cpu().numpy())
            test_labels.append(labels.numpy())
            test_energy_scores.extend(energy.cpu().numpy())

    test_logits = np.concatenate(test_logits, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    test_probs  = 1 / (1 + np.exp(-test_logits))
    test_energy_scores = np.array(test_energy_scores)

    _, test_macro_auroc = compute_per_class_auroc(test_labels, test_probs)
    test_ood_eval = evaluate_ood_detection(test_energy_scores, ood_energy, flip_scores=True)
    _, mean_brier = compute_brier_scores(test_labels, test_probs)

    # 30% suppression test
    supp_thresh = np.percentile(id_energy, 30)
    keep_mask_test = test_energy_scores >= supp_thresh
    base_hce_test = risk_results['baseline_hce_rate']
    kept_hce_test = evaluate_risk_mitigation(test_probs[keep_mask_test], test_labels[keep_mask_test], test_energy_scores[keep_mask_test])['baseline_hce_rate']
    red_test = (base_hce_test - kept_hce_test) / (base_hce_test + 1e-8) * 100

    print(f"  Classifier Macro AUROC (Test Set):  {test_macro_auroc:.4f}")
    print(f"  Energy Score OOD AUROC (Test vs OOD): {test_ood_eval['auroc']:.4f}")
    print(f"  Energy Score FPR@95TPR:               {test_ood_eval['fpr_at_95tpr']:.4f}")
    print(f"  HCE Error Reduction @ 30% Supp:       {red_test:.1f}%")
    print(f"  Mean Brier Score Calibration:         {mean_brier:.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
