"""
gcn.py — Graph Convolutional Network and Graph Attention Network
         for hand skeleton gesture recognition.

Models:
  GCNBranch  — 3-layer GCN with batch norm and global pooling
  GATBranch  — Multi-head GAT with residual connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool


class GCNBranch(nn.Module):
    """
    3-layer GCN branch for hand skeleton.

    Input:  PyG batch with x=[N, 3] (x,y,z per joint), edge_index
    Output: graph-level embedding [B, out_dim]

    Architecture:
        GCNConv(3 → 64) → BN → ReLU
        GCNConv(64 → 128) → BN → ReLU
        GCNConv(128 → out_dim) → BN → ReLU
        Global mean+max pool → [B, 2*out_dim]
        Linear → [B, out_dim]
    """

    def __init__(self, in_channels=3, hidden=64, out_dim=256, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden * 2)
        self.conv3 = GCNConv(hidden * 2, out_dim)

        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden * 2)
        self.bn3 = nn.BatchNorm1d(out_dim)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim * 2, out_dim)  # after mean+max concat

    def forward(self, x, edge_index, batch):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 3
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)

        # Global pooling — concat mean + max for richer representation
        mean_pool = global_mean_pool(x, batch)  # [B, out_dim]
        max_pool = global_max_pool(x, batch)    # [B, out_dim]
        pooled = torch.cat([mean_pool, max_pool], dim=-1)  # [B, 2*out_dim]

        return F.relu(self.fc(pooled))  # [B, out_dim]


class GATBranch(nn.Module):
    """
    Multi-head Graph Attention Network branch.

    More expressive than GCN — learns which neighboring joints
    are most important for each gesture.

    Args:
        in_channels: input feature dimension (3 for x,y,z)
        hidden: hidden dimension per head
        out_dim: output embedding dimension
        heads: number of attention heads
        dropout: dropout rate
    """

    def __init__(self, in_channels=3, hidden=32, out_dim=256, heads=4, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden * heads, hidden * 2, heads=heads, dropout=dropout)
        self.conv3 = GATConv(hidden * 2 * heads, out_dim, heads=1, concat=False, dropout=dropout)

        self.bn1 = nn.BatchNorm1d(hidden * heads)
        self.bn2 = nn.BatchNorm1d(hidden * 2 * heads)
        self.bn3 = nn.BatchNorm1d(out_dim)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim * 2, out_dim)

        # Residual projection for skip connections
        self.skip1 = nn.Linear(in_channels, hidden * heads, bias=False)
        self.skip3 = nn.Linear(hidden * 2 * heads, out_dim, bias=False)

    def forward(self, x, edge_index, batch):
        identity = x

        # Layer 1 + residual
        x = self.conv1(x, edge_index)
        x = x + self.skip1(identity)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)

        identity2 = x

        # Layer 3 + residual
        x = self.conv3(x, edge_index)
        x = x + self.skip3(identity2)
        x = self.bn3(x)
        x = F.elu(x)

        # Pooling
        mean_pool = global_mean_pool(x, batch)
        max_pool  = global_max_pool(x, batch)
        pooled = torch.cat([mean_pool, max_pool], dim=-1)

        return F.relu(self.fc(pooled))


class STGCNBlock(nn.Module):
    """
    Spatiotemporal GCN block (Yan et al. 2018) for dynamic gestures.
    Applies graph conv in space + temporal conv over sequence length.

    Input:  [B, C, T, V]  — batch, channels, time, vertices
    Output: [B, C_out, T, V]
    """

    def __init__(self, in_channels, out_channels, num_nodes=21,
                 temporal_kernel=9, stride=1, dropout=0.5):
        super().__init__()
        pad = (temporal_kernel - 1) // 2

        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.tcn = nn.Conv2d(out_channels, out_channels,
                             kernel_size=(temporal_kernel, 1),
                             stride=(stride, 1),
                             padding=(pad, 0))
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
            nn.BatchNorm2d(out_channels),
        ) if in_channels != out_channels or stride != 1 else nn.Identity()

    def forward(self, x, A):
        # x: [B, C, T, V], A: [V, V] adjacency matrix
        B, C, T, V = x.shape
        res = self.residual(x)

        # Spatial: aggregate neighbor features via adjacency
        x = x.permute(0, 3, 1, 2).contiguous().view(B * V, C, T, 1)
        x = x.view(B, V, C, T).permute(0, 2, 3, 1)  # [B, C, T, V]
        x = torch.einsum('bctv,vw->bctw', x, A)

        x = self.gcn(x)
        x = self.tcn(x)
        x = self.bn(x + res)
        return F.relu(self.dropout(x))
