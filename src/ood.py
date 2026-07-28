import torch
import numpy as np
from sklearn.covariance import LedoitWolf, EmpiricalCovariance
from sklearn.decomposition import PCA


def compute_msp_and_energy(model, loader, device):
    """
    Computes Maximum Softmax Probability (MSP) and Energy Scores for a dataloader.
    
    Args:
        model (torch.nn.Module): Trained model.
        loader (DataLoader): PyTorch DataLoader.
        device (torch.device): Device.
        
    Returns:
        tuple: (msp_scores, energy_scores) as numpy arrays.
    """
    all_msp    = []
    all_energy = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device)
            logits = model(images)                 # [B, 14]

            # Sigmoid probabilities for multi-label
            probs  = torch.sigmoid(logits)         # [B, 14]
            msp    = probs.max(dim=1).values       # max probability per image
            all_msp.extend((1 - msp).cpu().numpy())

            # Energy Score = -logsumexp(logits)
            energy = -torch.logsumexp(logits, dim=1)
            all_energy.extend(energy.cpu().numpy())

    return np.array(all_msp), np.array(all_energy)


def extract_penultimate_features(model, loader, device):
    """
    Extracts 1024-dimensional feature vectors from DenseNet-121 penultimate pooling layer.
    
    Returns:
        np.ndarray: [N, 1024] feature matrix.
    """
    features_list = []
    
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device)
            
            # Pass through DenseNet convolutional features up to global average pool
            x = model.features(images)                               # [B, 1024, 7, 7]
            x = torch.nn.functional.relu(x)
            x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))  # [B, 1024, 1, 1]
            x = torch.flatten(x, 1)                                  # [B, 1024]
            
            features_list.append(x.cpu().numpy())
    
    return np.concatenate(features_list, axis=0)


def extract_features_and_labels(model, loader, device):
    """
    Extracts 1024-dim feature vectors aligned with true binary labels.
    
    Returns:
        tuple: (features, labels) as numpy arrays.
    """
    features_list = []
    labels_list   = []
    
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device)
            labels = batch[1]
            
            x = model.features(images)
            x = torch.nn.functional.relu(x)
            x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            
            features_list.append(x.cpu().numpy())
            labels_list.append(labels.numpy())
            
    return np.concatenate(features_list, axis=0), np.concatenate(labels_list, axis=0)


def compute_global_mahalanobis(train_features, test_features):
    """
    Computes global Mahalanobis distance using Ledoit-Wolf covariance shrinkage.
    
    Returns:
        np.ndarray: Mahalanobis distances for test_features.
    """
    lw = LedoitWolf()
    lw.fit(train_features)
    mu = lw.location_
    cov_inv = lw.precision_
    
    diff = test_features - mu
    distances = np.sum(np.dot(diff, cov_inv) * diff, axis=1)
    return distances


def fit_multi_cluster_gaussians(train_features, train_labels, min_samples=10):
    """
    Fits Ledoit-Wolf Gaussians for healthy cluster + 14 individual disease clusters.
    
    Returns:
        tuple: (clusters_mu, clusters_cov_inv)
    """
    healthy_mask = np.sum(train_labels, axis=1) == 0
    clusters_mu = []
    clusters_cov_inv = []

    # Healthy cluster
    if np.sum(healthy_mask) > min_samples:
        lw_healthy = LedoitWolf().fit(train_features[healthy_mask])
        clusters_mu.append(lw_healthy.location_)
        clusters_cov_inv.append(lw_healthy.precision_)

    # Individual disease clusters
    for c in range(train_labels.shape[1]):
        disease_mask = train_labels[:, c] == 1
        if np.sum(disease_mask) > min_samples:
            lw_disease = LedoitWolf().fit(train_features[disease_mask])
            clusters_mu.append(lw_disease.location_)
            clusters_cov_inv.append(lw_disease.precision_)

    return clusters_mu, clusters_cov_inv


def compute_multi_cluster_mahalanobis(features, clusters_mu, clusters_cov_inv):
    """
    Assigns OOD score based on distance to the CLOSEST cluster.
    
    Returns:
        np.ndarray: Minimum Mahalanobis distance per sample.
    """
    all_dists = []
    for mu, cov_inv in zip(clusters_mu, clusters_cov_inv):
        diff = features - mu
        dist = np.sum(np.dot(diff, cov_inv) * diff, axis=1)
        all_dists.append(dist)
    
    all_dists = np.vstack(all_dists)
    return np.min(all_dists, axis=0)
