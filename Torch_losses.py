"""
Loss functions — thin wrappers around torch.nn so call-sites stay identical.
"""

import torch
import torch.nn as nn


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """Numerically-stable cross-entropy. Replaces the custom engine version."""
    def __init__(self):
        super().__init__()

    def __call__(self, logits, target_idx):
        # logits: (C,) tensor, target_idx: int
        return super().forward(logits.unsqueeze(0),
                               torch.tensor([target_idx]))


class MSELoss(nn.MSELoss):
    def __init__(self):
        super().__init__()