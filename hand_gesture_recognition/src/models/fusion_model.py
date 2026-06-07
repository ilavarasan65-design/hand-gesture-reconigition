"""
fusion_model.py — Hybrid GCN + CNN model with cross-attention fusion
                  and optional per-language adapter layers.

Architecture:
    ┌─────────────────┐   ┌────────────────┐
    │  CNN Branch     │   │  GCN Branch    │
    │ MobileNetV3     │   │ 3-layer GCN/   │
    │ [B, 256]        │   │ GAT [B, 256]   │
    └────────┬────────┘   └───────┬────────┘
             │                   │
             └────────┬──────────┘
                      │
              Cross-Attention Fusion
                  [B, 512] → [B, 256]
                      │
             Language Adapter (opt)
                  [B, 256]
                      │
              Classifier Head
                  [B, num_classes]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gcn import GCNBranch, GATBranch
from .cnn import CNNBranch


class CrossAttentionFusion(nn.Module):
    """
    Fuse two embeddings using cross-attention.
    gcn_feat attends to cnn_feat and vice-versa.

    Input:  gcn [B, D], cnn [B, D]
    Output: fused [B, D]
    """

    def __init__(self, dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, gcn_feat, cnn_feat):
        # Stack as sequence: [B, 2, D]
        seq = torch.stack([gcn_feat, cnn_feat], dim=1)

        # Self-attention over the 2-token sequence
        attended, _ = self.attn(seq, seq, seq)
        seq = self.norm1(seq + self.dropout(attended))

        # Feed-forward
        seq = self.norm2(seq + self.dropout(self.ff(seq)))

        # Aggregate: mean over the 2 tokens
        return seq.mean(dim=1)  # [B, D]


class LanguageAdapter(nn.Module):
    """
    Lightweight language-specific adapter (~4K params per language).
    Applied after the shared fusion embedding.

    Uses a bottleneck: D → D//8 → D with residual.
    """

    def __init__(self, dim=256, bottleneck=32):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.up   = nn.Linear(bottleneck, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        x = F.gelu(self.down(x))
        x = self.up(x)
        return self.norm(x + residual)


class GestureClassifier(nn.Module):
    """
    Full hybrid model for multi-culture sign language recognition.

    Args:
        num_classes:   number of gesture classes
        embed_dim:     shared embedding dimension (GCN + CNN output)
        graph_type:    'gcn' or 'gat'
        cnn_backbone:  backbone name for CNN branch
        use_image:     if False, skip CNN branch (graph-only mode)
        languages:     list of language codes for adapters (or None)
        dropout:       dropout rate
    """

    LANGUAGE_IDS = {
        'ASL': 0, 'BSL': 1, 'ISL': 2,
        'CSL': 3, 'ArSL': 4, 'FSL': 5,
    }

    def __init__(
        self,
        num_classes=100,
        embed_dim=256,
        graph_type='gcn',
        cnn_backbone='mobilenet_v3',
        use_image=True,
        languages=None,
        dropout=0.3,
    ):
        super().__init__()
        self.use_image = use_image
        self.use_adapters = languages is not None

        # ── Graph branch ──────────────────────────────────────────────────
        if graph_type == 'gcn':
            self.graph_branch = GCNBranch(in_channels=3, out_dim=embed_dim)
        elif graph_type == 'gat':
            self.graph_branch = GATBranch(in_channels=3, out_dim=embed_dim)
        else:
            raise ValueError(f'graph_type must be gcn or gat, got {graph_type}')

        # ── CNN branch ────────────────────────────────────────────────────
        if use_image:
            self.cnn_branch = CNNBranch(backbone=cnn_backbone, out_dim=embed_dim)
            self.fusion = CrossAttentionFusion(dim=embed_dim)
        else:
            self.proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
            )

        # ── Language adapters ─────────────────────────────────────────────
        if self.use_adapters and languages:
            self.adapters = nn.ModuleDict({
                lang: LanguageAdapter(embed_dim) for lang in languages
            })
            self.lang_id_to_name = {v: k for k, v in self.LANGUAGE_IDS.items()}

        # ── Classifier head ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, graph_batch, image=None, language_ids=None):
        """
        Args:
            graph_batch:  PyG Batch object (x, edge_index, batch)
            image:        [B, 3, 224, 224] image tensor (optional)
            language_ids: [B] int tensor of language codes

        Returns:
            logits: [B, num_classes]
            embedding: [B, embed_dim] (for metric learning / visualization)
        """
        # Graph features
        gcn_feat = self.graph_branch(
            graph_batch.x,
            graph_batch.edge_index,
            graph_batch.batch
        )  # [B, embed_dim]

        # Fuse with CNN or project alone
        if self.use_image and image is not None:
            cnn_feat = self.cnn_branch(image)
            embedding = self.fusion(gcn_feat, cnn_feat)
        else:
            embedding = self.proj(gcn_feat) if hasattr(self, 'proj') else gcn_feat

        # Apply per-language adapter
        if self.use_adapters and language_ids is not None and hasattr(self, 'adapters'):
            adapted = embedding.clone()
            for lang_id, lang_name in self.lang_id_to_name.items():
                if lang_name in self.adapters:
                    mask = (language_ids == lang_id)
                    if mask.any():
                        adapted[mask] = self.adapters[lang_name](embedding[mask])
            embedding = adapted

        logits = self.classifier(embedding)
        return logits, embedding

    def get_graph_only(self):
        """Return graph-only variant (faster, no CNN)."""
        self.use_image = False
        return self
