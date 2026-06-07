"""
tests/test_graph_utils.py — Unit tests for graph construction and model.
Run with: pytest tests/
"""

import pytest
import numpy as np
import torch
from torch_geometric.data import Batch

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.graph_utils import (
    build_edge_index, normalize_keypoints, keypoints_to_graph,
    augment_keypoints, flip_keypoints, compute_bone_features
)
from src.models.gcn import GCNBranch, GATBranch
from src.models.cnn import CNNBranch
from src.models.fusion_model import GestureClassifier


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_keypoints():
    return np.random.rand(21, 3).astype(np.float32)


@pytest.fixture
def dummy_batch(dummy_keypoints):
    graphs = [keypoints_to_graph(dummy_keypoints, label=i % 5) for i in range(4)]
    return Batch.from_data_list(graphs)


# ── Graph utils ───────────────────────────────────────────────────────────────

def test_edge_index_shape():
    ei = build_edge_index()
    assert ei.shape[0] == 2
    assert ei.shape[1] > 0

def test_edge_index_bidirectional():
    ei = build_edge_index(bidirectional=True)
    ei_uni = build_edge_index(bidirectional=False)
    assert ei.shape[1] == ei_uni.shape[1] * 2

def test_normalize_keypoints(dummy_keypoints):
    norm = normalize_keypoints(dummy_keypoints)
    assert norm.shape == (21, 3)
    # Wrist should be at origin
    np.testing.assert_allclose(norm[0], 0.0, atol=1e-6)

def test_keypoints_to_graph(dummy_keypoints):
    g = keypoints_to_graph(dummy_keypoints, label=3)
    assert g.x.shape == (21, 3)
    assert g.y.item() == 3
    assert g.edge_index.shape[0] == 2

def test_augment_preserves_shape(dummy_keypoints):
    aug = augment_keypoints(dummy_keypoints)
    assert aug.shape == dummy_keypoints.shape

def test_flip_keypoints(dummy_keypoints):
    flipped = flip_keypoints(dummy_keypoints)
    assert flipped.shape == dummy_keypoints.shape
    np.testing.assert_allclose(flipped[:, 0], -dummy_keypoints[:, 0])

def test_bone_features(dummy_keypoints):
    bones = compute_bone_features(dummy_keypoints)
    assert bones.ndim == 1
    assert len(bones) > 0
    assert (bones >= 0).all()


# ── GCN / GAT ──────────────────────────────────────────────────────────────────

def test_gcn_branch_forward(dummy_batch):
    model = GCNBranch(in_channels=3, out_dim=256)
    model.eval()
    with torch.no_grad():
        out = model(dummy_batch.x, dummy_batch.edge_index, dummy_batch.batch)
    assert out.shape == (4, 256)

def test_gat_branch_forward(dummy_batch):
    model = GATBranch(in_channels=3, out_dim=256)
    model.eval()
    with torch.no_grad():
        out = model(dummy_batch.x, dummy_batch.edge_index, dummy_batch.batch)
    assert out.shape == (4, 256)


# ── CNN branch ────────────────────────────────────────────────────────────────

def test_cnn_branch_forward():
    model = CNNBranch(backbone='mobilenet_v3', out_dim=256, pretrained=False)
    model.eval()
    x = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 256)


# ── Full model ────────────────────────────────────────────────────────────────

def test_gesture_classifier_graph_only(dummy_batch):
    model = GestureClassifier(num_classes=10, use_image=False)
    model.eval()
    lang_ids = torch.zeros(4, dtype=torch.long)
    with torch.no_grad():
        logits, emb = model(dummy_batch, None, lang_ids)
    assert logits.shape == (4, 10)
    assert emb.shape == (4, 256)

def test_gesture_classifier_with_image(dummy_batch):
    model = GestureClassifier(num_classes=10, use_image=True,
                               cnn_backbone='mobilenet_v3')
    model.eval()
    images = torch.rand(4, 3, 224, 224)
    lang_ids = torch.zeros(4, dtype=torch.long)
    with torch.no_grad():
        logits, emb = model(dummy_batch, images, lang_ids)
    assert logits.shape == (4, 10)

def test_gesture_classifier_with_adapters(dummy_batch):
    langs = ['ASL', 'BSL', 'ISL']
    model = GestureClassifier(num_classes=5, use_image=False, languages=langs)
    model.eval()
    lang_ids = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    with torch.no_grad():
        logits, _ = model(dummy_batch, None, lang_ids)
    assert logits.shape == (4, 5)
