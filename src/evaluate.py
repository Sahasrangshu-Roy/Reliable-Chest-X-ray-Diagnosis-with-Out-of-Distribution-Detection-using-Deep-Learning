import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from src.dataset import ALL_LABELS


def compute_per_class_auroc(y_true, y_probs, labels=ALL_LABELS):
    """
    Computes per-class AUROC and Macro AUROC.
    
    Returns:
        tuple: (per_class_results_dict, macro_auroc)
    """
    results = {}
    valid_scores = []
    
    for i, label in enumerate(labels):
        pos_count = int(y_true[:, i].sum())
        if pos_count == 0:
            results[label] = {"auroc": None, "positives": 0}
            continue
        auroc = roc_auc_score(y_true[:, i], y_probs[:, i])
        results[label] = {"auroc": auroc, "positives": pos_count}
        valid_scores.append(auroc)
        
    macro_auroc = np.mean(valid_scores) if valid_scores else 0.0
    return results, macro_auroc


def compute_fpr_at_95tpr(labels, scores):
    """
    Calculates False Positive Rate at 95% True Positive Rate.
    
    Args:
        labels (np.ndarray): Binary labels (0 = ID, 1 = OOD).
        scores (np.ndarray): OOD scores (higher = more OOD).
        
    Returns:
        float: FPR value at 95% TPR.
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.argmin(np.abs(tpr - 0.95))
    return fpr[idx]


def evaluate_ood_detection(id_scores, ood_scores, flip_scores=True):
    """
    Evaluates OOD detection performance (AUROC and FPR@95TPR) between ID and OOD.
    
    Args:
        id_scores (np.ndarray): OOD scores for In-Distribution samples.
        ood_scores (np.ndarray): OOD scores for Out-Of-Distribution samples.
        flip_scores (bool): If True, negates scores so higher = more OOD.
        
    Returns:
        dict: {"auroc": float, "fpr_at_95tpr": float}
    """
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    combined_scores = np.concatenate([id_scores, ood_scores])
    
    if flip_scores:
        combined_scores = -combined_scores
        
    auroc = roc_auc_score(labels, combined_scores)
    fpr95 = compute_fpr_at_95tpr(labels, combined_scores)
    
    return {"auroc": auroc, "fpr_at_95tpr": fpr95}


def count_high_conf_errors(probs, labels, conf_thresh=0.7):
    """
    Counts images containing at least one high-confidence error.
    - False Positive: Predicted prob > conf_thresh when GT is 0.
    - False Negative: Predicted prob < (1 - conf_thresh) when GT is 1.
    """
    fp = ((probs > conf_thresh) & (labels == 0)).any(axis=1)
    fn = ((probs < (1 - conf_thresh)) & (labels == 1)).any(axis=1)
    return (fp | fn).sum()


def evaluate_risk_mitigation(val_probs, val_labels, val_energy_scores, conf_thresh=0.7):
    """
    Sweeps Energy Score suppression thresholds to evaluate Coverage vs. High-Confidence Error reduction.
    
    Returns:
        dict: Sweep results containing percentiles, coverages, hce_rates, and reductions.
    """
    total_images = len(val_labels)
    baseline_hce = count_high_conf_errors(val_probs, val_labels, conf_thresh=conf_thresh)
    baseline_hce_rate = baseline_hce / total_images

    percentiles = np.arange(0, 100, 5)
    coverages   = []
    hce_rates   = []
    reductions  = []

    for pct in percentiles:
        threshold = np.percentile(val_energy_scores, pct)
        keep_mask = val_energy_scores >= threshold
        
        if keep_mask.sum() == 0:
            coverages.append(0)
            hce_rates.append(0)
            reductions.append(0)
            continue
        
        coverage = keep_mask.sum() / total_images
        hce = count_high_conf_errors(val_probs[keep_mask], val_labels[keep_mask], conf_thresh=conf_thresh)
        hce_rate = hce / keep_mask.sum()
        reduction = (baseline_hce_rate - hce_rate) / (baseline_hce_rate + 1e-8) * 100
        
        coverages.append(coverage)
        hce_rates.append(hce_rate)
        reductions.append(reduction)

    return {
        "baseline_hce": baseline_hce,
        "baseline_hce_rate": baseline_hce_rate,
        "percentiles": percentiles,
        "coverages": np.array(coverages),
        "hce_rates": np.array(hce_rates),
        "reductions": np.array(reductions)
    }


def compute_brier_scores(y_true, y_probs, labels=ALL_LABELS):
    """
    Computes Brier calibration score for each class and the mean Brier score.
    
    Returns:
        tuple: (brier_dict, mean_brier)
    """
    brier_dict = {}
    valid_scores = []
    
    for i, label in enumerate(labels):
        pos_count = int(y_true[:, i].sum())
        if pos_count == 0:
            continue
        score = brier_score_loss(y_true[:, i], y_probs[:, i])
        brier_dict[label] = score
        valid_scores.append(score)
        
    mean_brier = np.mean(valid_scores) if valid_scores else 0.0
    return brier_dict, mean_brier
