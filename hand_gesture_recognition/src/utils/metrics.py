"""
metrics.py — Evaluation metrics for gesture recognition.
"""

import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class ConfusionTracker:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.all_preds = []
        self.all_labels = []

    def update(self, preds, labels):
        self.all_preds.extend(preds.tolist())
        self.all_labels.extend(labels.tolist())

    def reset(self):
        self.all_preds = []
        self.all_labels = []


def compute_metrics(tracker):
    preds = np.array(tracker.all_preds)
    labels = np.array(tracker.all_labels)

    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)

    return {
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'per_class_f1_mean': float(per_class_f1.mean()),
        'per_class_f1_min': float(per_class_f1.min()),
    }


def plot_confusion_matrix(tracker, class_names=None, save_path=None):
    preds = np.array(tracker.all_preds)
    labels = np.array(tracker.all_labels)
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=len(class_names or []) <= 30,
                fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names or 'auto',
                yticklabels=class_names or 'auto')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
