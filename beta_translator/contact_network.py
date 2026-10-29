"""Candidate-equivariant temporal convolution network for contact states."""
import numpy as np
import torch
from torch import nn


class TemporalBlock(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.conv = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.mix = nn.Conv1d(width, width, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = self.norm(x.transpose(1, 2)).transpose(1, 2)
        y = self.mix(torch.nn.functional.gelu(self.conv(y)))
        return x + self.dropout(y)


class ContactTCN(nn.Module):
    """Score each hold, free state and wall contact with shared temporal weights.

    Offline/bidirectional: six dilations see 127 frames (5.08 s at 25 FPS).
    No learned embedding depends on a route's hold count or ID ordering.
    """
    def __init__(self, features, width=96, layers=6, dropout=.15):
        super().__init__()
        self.config = dict(features=features, width=width, layers=layers, dropout=dropout)
        self.radius = sum(2**i for i in range(layers))
        self.project = nn.Sequential(nn.Linear(features, width), nn.GELU(), nn.Linear(width, width))
        self.temporal = nn.Sequential(*[TemporalBlock(width, 2**i, dropout) for i in range(layers)])
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, features, candidate_mask=None):
        # Input B,T,H,F; temporal filters never mix candidate identities.
        b, t, h, _ = features.shape
        x = self.project(features).permute(0, 2, 3, 1).reshape(b*h, -1, t)
        x = self.temporal(x).transpose(1, 2)
        scores = self.head(x).reshape(b, h, t).transpose(1, 2)
        if candidate_mask is not None:
            scores = scores.masked_fill(~candidate_mask[:, None], -1e4)
        return scores


@torch.inference_mode()
def predict_logits(model, features, device, chunk=512):
    """Chunk with full receptive-field overlap, preserving every source frame."""
    model.eval()
    t, limbs, holds, _ = features.shape
    output = np.empty((t, limbs, holds), np.float32)
    radius = model.radius
    for limb in range(limbs):
        for start in range(0, t, chunk):
            end = min(t, start+chunk)
            left, right = max(0, start-radius), min(t, end+radius)
            batch = torch.from_numpy(features[left:right, limb].astype(np.float32))[None].to(device)
            logits = model(batch)[0, start-left:end-left].cpu().numpy()
            output[start:end, limb] = logits
    return output
