import torch
import numpy as np
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from PIL import Image


def apply_gaussian_noise(image, severity):
    """Adds Gaussian noise to normalized PyTorch image tensor (severity 0.0 to 1.0)."""
    noise = torch.randn_like(image) * severity
    return torch.clamp(image + noise, 0, 1)


def apply_contrast_reduction(image, severity):
    """Reduces contrast of PyTorch image tensor (severity 0.0 to 1.0)."""
    factor = 1.0 - severity * 0.8  # factor moves from 1.0 to 0.2
    return TF.adjust_contrast(image, factor)


def apply_blur(image, severity):
    """Applies Gaussian blur to PyTorch image tensor (severity 0.0 to 1.0)."""
    kernel_size = int(severity * 10) * 2 + 1
    kernel_size = max(3, kernel_size)
    return TF.gaussian_blur(image, kernel_size)


class CorruptedDataset(Dataset):
    """Dataset wrapper that applies synthetic corruptions to test images on the fly."""
    def __init__(self, dataframe, transform, corrupt_fn, severity):
        self.df         = dataframe.reset_index(drop=True)
        self.transform  = transform
        self.corrupt_fn = corrupt_fn
        self.severity   = severity

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.corrupt_fn and self.severity > 0:
            image = self.corrupt_fn(image, self.severity)
        return image, row["image_id"]


def simulate_drift_scenario(model, test_subset, eval_transform, corrupt_fn, severities, lower_bound, upper_bound, device, batch_size=32):
    """
    Simulates drift by evaluating OOD detection alert rates across varying corruption severities.
    
    Args:
        model (torch.nn.Module): Trained DenseNet model.
        test_subset (pd.DataFrame): Test dataset subset.
        eval_transform: Evaluation PyTorch transform.
        corrupt_fn: Corruption function.
        severities (list): List of severity levels (e.g. [0.0, 0.1, ..., 1.0]).
        lower_bound (float): 20th percentile energy threshold.
        upper_bound (float): 80th percentile energy threshold.
        device: PyTorch device.
        
    Returns:
        list: OOD alert rates per severity level.
    """
    ood_rates = []
    model.eval()

    for sev in severities:
        dataset = CorruptedDataset(test_subset, eval_transform, corrupt_fn, sev)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        
        scores = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)
                logits = model(images)
                energy = -torch.logsumexp(logits, dim=1)
                scores.extend(energy.cpu().numpy())
                
        scores = np.array(scores)
        # Flag if energy is outside the normal in-distribution range [lower_bound, upper_bound]
        ood_rate = ((scores < lower_bound) | (scores > upper_bound)).mean()
        ood_rates.append(ood_rate)

    return ood_rates
