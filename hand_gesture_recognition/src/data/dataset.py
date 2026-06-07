"""
dataset.py — PyTorch Dataset for multi-language sign language gesture data.

Supports:
  - Loading keypoints from JSON / CSV / .npy files
  - Loading raw frames for CNN branch
  - Multi-language batching with optional language labels
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torch_geometric.data import Data, Batch

from .graph_utils import (
    keypoints_to_graph,
    augment_keypoints,
    flip_keypoints,
    normalize_keypoints,
)

LANGUAGE_IDS = {
    'ASL': 0, 'BSL': 1, 'ISL': 2,
    'CSL': 3, 'ArSL': 4, 'FSL': 5,
}

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

IMG_TRANSFORM_TRAIN = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class GestureDataset(Dataset):
    """
    Dataset that returns (graph_data, image_tensor, label, language_id).

    Expected CSV format:
        keypoints_path, image_path, label, language
        data/kp/asl_0001.npy, data/img/asl_0001.jpg, 5, ASL

    Args:
        csv_path:   path to the annotation CSV
        augment:    apply augmentation (train mode)
        use_image:  also load raw image for CNN branch
        languages:  list of language codes to include (None = all)
    """

    def __init__(self, csv_path, augment=False, use_image=True, languages=None):
        self.df = pd.read_csv(csv_path)
        self.augment = augment
        self.use_image = use_image
        self.img_transform = IMG_TRANSFORM_TRAIN if augment else IMG_TRANSFORM

        if languages:
            self.df = self.df[self.df['language'].isin(languages)].reset_index(drop=True)

        self.label_to_idx = {lbl: i for i, lbl in enumerate(sorted(self.df['label'].unique()))}
        self.num_classes = len(self.label_to_idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ── Keypoints → graph ──────────────────────────────────────────────
        kp = np.load(row['keypoints_path'])  # [21, 3]

        if self.augment:
            kp = augment_keypoints(kp)
            if np.random.rand() < 0.3:
                kp = flip_keypoints(kp)

        label = self.label_to_idx[row['label']]
        language = row['language']
        graph = keypoints_to_graph(kp, label=label, language=language)

        # ── Image → tensor ────────────────────────────────────────────────
        image = torch.zeros(3, 224, 224)
        if self.use_image and pd.notna(row.get('image_path')):
            try:
                img = Image.open(row['image_path']).convert('RGB')
                image = self.img_transform(img)
            except Exception:
                pass  # fallback to zeros if image missing

        language_id = LANGUAGE_IDS.get(language, 0)

        return {
            'graph': graph,
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'language_id': torch.tensor(language_id, dtype=torch.long),
        }

    @staticmethod
    def collate_fn(batch):
        graphs = Batch.from_data_list([b['graph'] for b in batch])
        images = torch.stack([b['image'] for b in batch])
        labels = torch.stack([b['label'] for b in batch])
        language_ids = torch.stack([b['language_id'] for b in batch])
        return {
            'graph': graphs,
            'image': images,
            'label': labels,
            'language_id': language_ids,
        }


class KeypointOnlyDataset(Dataset):
    """Lightweight dataset — keypoints only, no image loading. Fast training."""

    def __init__(self, csv_path, augment=False, languages=None):
        self.df = pd.read_csv(csv_path)
        self.augment = augment

        if languages:
            self.df = self.df[self.df['language'].isin(languages)].reset_index(drop=True)

        self.label_to_idx = {lbl: i for i, lbl in enumerate(sorted(self.df['label'].unique()))}
        self.num_classes = len(self.label_to_idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        kp = np.load(row['keypoints_path'])

        if self.augment:
            kp = augment_keypoints(kp)

        label = self.label_to_idx[row['label']]
        graph = keypoints_to_graph(kp, label=label, language=row['language'])
        language_id = LANGUAGE_IDS.get(row['language'], 0)

        return graph, torch.tensor(language_id, dtype=torch.long)

    @staticmethod
    def collate_fn(batch):
        graphs, lang_ids = zip(*batch)
        return Batch.from_data_list(list(graphs)), torch.stack(list(lang_ids))
