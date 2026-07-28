import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ── 14 Disease Labels ──────────────────────────────────────
ALL_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
NUM_CLASSES = len(ALL_LABELS)

# ── ImageNet Normalization Constants ────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def binarize_label(label_str):
    """Converts a '|' separated string of labels into a 14-element multi-hot vector."""
    present = set(str(label_str).split("|"))
    return [1 if l in present else 0 for l in ALL_LABELS]


def calculate_class_weights(df, cap=50.0):
    """
    Computes class imbalance weights based on ratio of negative to positive samples.
    
    Args:
        df (pd.DataFrame): Dataframe containing 'label_vector' column.
        cap (float): Maximum cap for class weights to prevent destabilization.
        
    Returns:
        tuple: (raw_weights, capped_weights) as numpy arrays.
    """
    label_matrix = np.array(df["label_vector"].tolist())
    pos_counts   = label_matrix.sum(axis=0)
    neg_counts   = len(df) - pos_counts
    class_weights = neg_counts / (pos_counts + 1e-6)
    capped_weights = np.clip(class_weights, a_min=1.0, a_max=cap)
    return class_weights, capped_weights


def get_transforms():
    """Returns standard train and evaluation transformation pipelines."""
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    
    return train_transform, eval_transform


class ChestXrayDataset(Dataset):
    """PyTorch Dataset for NIH ChestX-ray14 images."""
    def __init__(self, dataframe, transform=None):
        self.df        = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        image  = Image.open(row["filepath"]).convert("RGB")
        label  = torch.tensor(row["label_vector"], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, label, row["image_id"]


class OODDataset(Dataset):
    """PyTorch Dataset for Out-Of-Distribution evaluation (e.g. CheXpert)."""
    def __init__(self, filepaths, transform=None):
        self.filepaths = filepaths
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path  = self.filepaths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(path)


def load_nih_data(base_dir):
    """
    Loads and processes NIH ChestX-ray14 CSV and official train/val/test split files.
    
    Returns:
        tuple: (train_df, val_df, test_df, image_path_map)
    """
    csv_path       = os.path.join(base_dir, "Data_Entry_2017.csv")
    train_val_list = os.path.join(base_dir, "train_val_list.txt")
    test_list      = os.path.join(base_dir, "test_list.txt")

    image_dirs = [
        base_dir,
        os.path.join(base_dir, "images"),
        os.path.join(base_dir, "images_001/images"),
        os.path.join(base_dir, "images_002/images"),
        os.path.join(base_dir, "images_003/images"),
    ]

    df = pd.read_csv(csv_path)
    df = df.drop(columns=["Unnamed: 11"], errors="ignore")
    df = df.rename(columns={
        "Image Index":                 "image_id",
        "Finding Labels":              "labels",
        "Follow-up #":                 "followup",
        "Patient ID":                  "patient_id",
        "Patient Age":                 "patient_age",
        "Patient Gender":              "patient_gender",
        "View Position":               "view_position",
        "OriginalImage[Width":         "img_width",
        "Height]":                     "img_height",
        "OriginalImagePixelSpacing[x": "pixel_spacing_x",
        "y]":                          "pixel_spacing_y",
    })

    # Build image_id -> full path mapping
    image_path_map = {}
    for folder in image_dirs:
        if os.path.exists(folder) and os.path.isdir(folder):
            for fname in os.listdir(folder):
                if fname.endswith(".png") or fname.endswith(".jpg"):
                    image_path_map[fname] = os.path.join(folder, fname)

    df = df[df["image_id"].isin(image_path_map)].reset_index(drop=True)
    df["filepath"] = df["image_id"].map(image_path_map)
    df["label_vector"] = df["labels"].apply(binarize_label)

    # Load splits
    with open(train_val_list) as f:
        train_val_ids = set(f.read().splitlines())
    with open(test_list) as f:
        test_ids = set(f.read().splitlines())

    train_val_df = df[df["image_id"].isin(train_val_ids)].reset_index(drop=True)
    test_df      = df[df["image_id"].isin(test_ids)].reset_index(drop=True)

    # Patient-wise train/val split (80/20)
    unique_patients = train_val_df["patient_id"].unique()
    split_idx       = int(len(unique_patients) * 0.8)
    train_patients  = set(unique_patients[:split_idx])
    val_patients    = set(unique_patients[split_idx:])

    train_df = train_val_df[train_val_df["patient_id"].isin(train_patients)].reset_index(drop=True)
    val_df   = train_val_df[train_val_df["patient_id"].isin(val_patients)].reset_index(drop=True)

    return train_df, val_df, test_df, image_path_map


def load_chexpert_data(chexpert_path):
    """
    Loads frontal images from CheXpert valid CSV for OOD testing.
    
    Returns:
        pd.DataFrame: Processed CheXpert dataframe with valid file paths.
    """
    csv_file = os.path.join(chexpert_path, "valid.csv")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CheXpert valid.csv not found at {csv_file}")
        
    df_chex = pd.read_csv(csv_file)
    df_chex["full_path"] = df_chex["Path"].apply(
        lambda x: os.path.join(chexpert_path, "/".join(x.split("/")[1:]))
    )
    if "Frontal/Lateral" in df_chex.columns:
        df_chex = df_chex[df_chex["Frontal/Lateral"] == "Frontal"].reset_index(drop=True)
        
    df_chex["exists"] = df_chex["full_path"].apply(os.path.exists)
    df_chex = df_chex[df_chex["exists"]].reset_index(drop=True)
    return df_chex


def create_dataloaders(train_df, val_df, test_df, batch_size=32, num_workers=2):
    """Builds PyTorch DataLoaders for train, validation, and test datasets."""
    train_transform, eval_transform = get_transforms()

    train_dataset = ChestXrayDataset(train_df, transform=train_transform)
    val_dataset   = ChestXrayDataset(val_df,   transform=eval_transform)
    test_dataset  = ChestXrayDataset(test_df,  transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, eval_transform
