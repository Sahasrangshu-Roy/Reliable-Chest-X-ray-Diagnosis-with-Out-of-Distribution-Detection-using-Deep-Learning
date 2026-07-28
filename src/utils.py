import matplotlib.pyplot as plt
import numpy as np
import torch
from src.dataset import IMAGENET_MEAN, IMAGENET_STD

def denormalize_image(tensor):
    """
    Reverses ImageNet normalization on a PyTorch image tensor for display.
    
    Args:
        tensor (torch.Tensor): Tensor of shape [C, H, W] or [1, C, H, W].
        
    Returns:
        np.ndarray: Denormalized image array of shape [H, W, C] with values in [0, 1].
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    img = tensor.cpu().numpy().transpose((1, 2, 0))
    mean = np.array(IMAGENET_MEAN)
    std  = np.array(IMAGENET_STD)
    img = img * std + mean
    return np.clip(img, 0, 1)


def plot_sample_transformation(original_pil, transformed_tensor, sample_id, sample_label, save_path=None):
    """Plots side-by-side comparison of original image vs transformed tensor."""
    transformed_display = denormalize_image(transformed_tensor)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_pil)
    axes[0].set_title(f"Original\nID: {sample_id}\nLabel: {sample_label}", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(transformed_display)
    axes[1].set_title("Transformed & Augmented\n(224x224)", fontsize=10)
    axes[1].axis("off")

    plt.suptitle("Image Transformation Pipeline Sanity Check", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_ood_comparison(id_tensor, ood_tensor, id_energy_score, ood_energy_score, save_path=None):
    """Plots side-by-side comparison of In-Distribution vs Out-Of-Distribution image and Energy Scores."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(denormalize_image(id_tensor))
    axes[0].set_title(f"In-Distribution (NIH Val)\nEnergy Score: {id_energy_score:.4f}", 
                      fontsize=12, fontweight='bold')
    axes[0].axis('off')
    axes[0].text(0.5, -0.1, "(Highly negative = Confident / Familiar)", 
                 ha='center', va='top', transform=axes[0].transAxes, 
                 fontsize=11, color='green')

    axes[1].imshow(denormalize_image(ood_tensor))
    axes[1].set_title(f"Out-of-Distribution (CheXpert)\nEnergy Score: {ood_energy_score:.4f}", 
                      fontsize=12, fontweight='bold')
    axes[1].axis('off')
    axes[1].text(0.5, -0.1, "(Closer to zero / Positive = Uncertain / Foreign)", 
                 ha='center', va='top', transform=axes[1].transAxes, 
                 fontsize=11, color='red')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_risk_mitigation_curves(results, save_path=None):
    """Plots Coverage vs High Confidence Error Rate and % Error Reduction curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(results["coverages"], results["hce_rates"], 'b-o', markersize=4)
    axes[0].axhline(y=results["baseline_hce_rate"], color='r', linestyle='--',
                    label=f'Baseline: {results["baseline_hce_rate"]:.3f}')
    axes[0].set_xlabel("Coverage (Fraction of Images Predicted On)")
    axes[0].set_ylabel("High-Confidence Error Rate")
    axes[0].set_title("Coverage vs Risk Tradeoff")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(results["percentiles"], results["reductions"], 'g-o', markersize=4)
    axes[1].set_xlabel("% Images Suppressed")
    axes[1].set_ylabel("% Reduction in High-Confidence Errors")
    axes[1].set_title("Suppression vs Error Reduction")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Risk Mitigation via Energy Score Suppression", fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_drift_monitoring_curves(severities, results_dict, save_path=None):
    """Plots OOD alert rate curves across corruption severities for various drift functions."""
    fig, axes = plt.subplots(1, len(results_dict), figsize=(5 * len(results_dict), 5))

    if len(results_dict) == 1:
        axes = [axes]

    for ax, (corrupt_name, ood_rates) in zip(axes, results_dict.items()):
        ax.plot(severities, ood_rates, 'b-o', markersize=5)
        ax.axhline(y=0.2, color='r', linestyle='--', label='Alert Threshold (20%)')
        ax.set_xlabel("Corruption Severity")
        ax.set_ylabel("OOD Rate")
        ax.set_title(corrupt_name)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    plt.suptitle("Drift Monitoring — OOD Rate vs Corruption Severity", fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
