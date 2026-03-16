import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

IMAGE_DIR = r'C:\skin-classifier\data\raw\images'
CSV_PATH  = r'C:\skin-classifier\data\raw\ham10000_metadata_2026-03-14.csv'

LABEL_MAP = {
    'Nevus': 0,
    'Pigmented benign keratosis': 1,
    'Melanoma, NOS': 2,
    'Basal cell carcinoma': 3,
    'Squamous cell carcinoma, NOS': 4,
    'Dermatofibroma': 5,
    'Solar or actinic keratosis': 6
}

def get_transforms(mode='train'):
    if mode == 'train':
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

class SkinLesionDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df        = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row       = self.df.iloc[idx]
        img_path  = os.path.join(self.image_dir, f"{row['isic_id']}.jpg")
        image     = cv2.imread(img_path)
        image     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']

        label = int(row['label'])
        return image, label

def get_dataloaders(batch_size=32):
    # Load and clean
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=['diagnosis_3'])
    df = df[df['diagnosis_3'].isin(LABEL_MAP.keys())].reset_index(drop=True)
    df['label'] = df['diagnosis_3'].map(LABEL_MAP)

    # Split by lesion_id to avoid data leakage
    unique_lesions = df['lesion_id'].unique()
    train_lesions, val_lesions = train_test_split(
        unique_lesions, test_size=0.2, random_state=42
    )
    train_df = df[df['lesion_id'].isin(train_lesions)].reset_index(drop=True)
    val_df   = df[df['lesion_id'].isin(val_lesions)].reset_index(drop=True)

    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    # Weighted sampler for class imbalance
    class_counts  = train_df['label'].value_counts().to_dict()
    sample_weights = [1.0 / class_counts[l] for l in train_df['label']]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # Datasets
    train_dataset = SkinLesionDataset(train_df, IMAGE_DIR, get_transforms('train'))
    val_dataset   = SkinLesionDataset(val_df,   IMAGE_DIR, get_transforms('val'))

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, val_loader, df