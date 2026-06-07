"""
graph_utils.py — Build hand skeleton graphs from MediaPipe keypoints.

Hand joint indices (MediaPipe 21 landmarks):
    0: Wrist
    1-4: Thumb (CMC, MCP, IP, TIP)
    5-8: Index (MCP, PIP, DIP, TIP)
    9-12: Middle (MCP, PIP, DIP, TIP)
    13-16: Ring (MCP, PIP, DIP, TIP)
    17-20: Pinky (MCP, PIP, DIP, TIP)
"""

import numpy as np
import torch
from torch_geometric.data import Data


# Kinematic skeleton edges (parent → child bone connections)
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),             # Palm cross-connections
]


def build_edge_index(edges=None, bidirectional=True):
    """
    Build edge_index tensor for PyTorch Geometric.

    Args:
        edges: list of (src, dst) tuples. Defaults to HAND_EDGES.
        bidirectional: if True, add reverse edges.

    Returns:
        edge_index: LongTensor of shape [2, num_edges]
    """
    if edges is None:
        edges = HAND_EDGES

    src, dst = zip(*edges)
    src, dst = list(src), list(dst)

    if bidirectional:
        src, dst = src + dst, dst + src  # add reverse

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index


def normalize_keypoints(keypoints, ref_joint=0):
    """
    Normalize keypoints relative to reference joint (wrist by default).
    Scale by the max distance from wrist so it's scale-invariant.

    Args:
        keypoints: np.ndarray of shape [21, 3]
        ref_joint: index of reference joint (0 = wrist)

    Returns:
        normalized: np.ndarray of shape [21, 3]
    """
    keypoints = keypoints.copy().astype(np.float32)
    ref = keypoints[ref_joint]
    keypoints -= ref  # translate to origin

    max_dist = np.linalg.norm(keypoints, axis=1).max()
    if max_dist > 1e-6:
        keypoints /= max_dist  # scale invariant

    return keypoints


def keypoints_to_graph(keypoints, label=None, language=None):
    """
    Convert 21 hand keypoints to a PyG Data object.

    Args:
        keypoints: np.ndarray [21, 3] — x, y, z per joint
        label: int class label
        language: str language code ('ASL', 'BSL', etc.)

    Returns:
        torch_geometric.data.Data
    """
    keypoints = normalize_keypoints(keypoints)
    x = torch.tensor(keypoints, dtype=torch.float)  # [21, 3]
    edge_index = build_edge_index()

    data = Data(x=x, edge_index=edge_index)

    if label is not None:
        data.y = torch.tensor([label], dtype=torch.long)

    if language is not None:
        data.language = language

    return data


def augment_keypoints(keypoints, rotate=True, scale=True, jitter=True):
    """
    Data augmentation for keypoints.

    Args:
        keypoints: np.ndarray [21, 3]
        rotate: apply random rotation in XY plane
        scale: apply random scaling
        jitter: add random Gaussian noise

    Returns:
        augmented: np.ndarray [21, 3]
    """
    kp = keypoints.copy().astype(np.float32)

    if rotate:
        angle = np.random.uniform(-30, 30) * np.pi / 180
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot = np.array([[cos_a, -sin_a, 0],
                        [sin_a,  cos_a, 0],
                        [0,      0,     1]], dtype=np.float32)
        kp = kp @ rot.T

    if scale:
        s = np.random.uniform(0.85, 1.15)
        kp *= s

    if jitter:
        kp += np.random.normal(0, 0.01, kp.shape).astype(np.float32)

    return kp


def flip_keypoints(keypoints):
    """Horizontal flip for left-right hand augmentation."""
    kp = keypoints.copy()
    kp[:, 0] = -kp[:, 0]  # mirror x
    return kp


def compute_bone_features(keypoints):
    """
    Compute bone length features (edge features) from keypoints.

    Returns:
        bone_lengths: np.ndarray [num_edges,] — Euclidean length per bone
    """
    kp = normalize_keypoints(keypoints)
    bone_lengths = []
    for src, dst in HAND_EDGES:
        length = np.linalg.norm(kp[dst] - kp[src])
        bone_lengths.append(length)
    return np.array(bone_lengths, dtype=np.float32)
