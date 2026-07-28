import torch
import torch.nn as nn
from torchvision import models
import numpy as np

def get_densenet_model(num_classes=14, pretrained=True, checkpoint_path=None, device="cpu"):
    """
    Initializes DenseNet-121 architecture with a 14-output linear classifier layer.
    
    Args:
        num_classes (int): Number of output classes (14 for NIH ChestX-ray14).
        pretrained (bool): Whether to use ImageNet pretrained weights.
        checkpoint_path (str, optional): Path to saved model .pth weights file.
        device (str or torch.device): Device to send model to ('cuda' or 'cpu').
        
    Returns:
        torch.nn.Module: Configured DenseNet-121 model.
    """
    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained and checkpoint_path is None else None
    model = models.densenet121(weights=weights)
    
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    
    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        
    model = model.to(device)
    return model


def get_weighted_bce_loss(class_weights, cap=50.0, device="cpu"):
    """
    Constructs Weighted Binary Cross-Entropy (BCE) Loss with Logits.
    
    Args:
        class_weights (np.ndarray): Uncapped class imbalance weights.
        cap (float): Maximum cap for pos_weights.
        device (str or torch.device): Target device.
        
    Returns:
        nn.BCEWithLogitsLoss: Weighted loss criterion.
    """
    capped_weights = np.clip(class_weights, a_min=1.0, a_max=cap)
    weight_tensor  = torch.tensor(capped_weights, dtype=torch.float32).to(device)
    criterion      = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
    return criterion


def save_model(model, checkpoint_path):
    """Saves model state dictionary to disk."""
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model state dict saved successfully to: {checkpoint_path}")


def load_model_weights(model, checkpoint_path, device="cpu"):
    """Loads state dict into model."""
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
