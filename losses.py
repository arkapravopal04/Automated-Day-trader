import numpy as np
from engine import Tensor


class CrossEntropyLoss:
    def forward(self, logits, target_idx):
        max_logit = np.max(logits.data)
        shifted = logits - max_logit          
        exp_shifted = shifted.exp()
        log_sum_exp = exp_shifted.sum().log()
        log_prob_target = shifted[target_idx] - log_sum_exp
        loss = log_prob_target * -1.0
        return loss

    def __call__(self, logits, target_idx):
        return self.forward(logits, target_idx)


class MSELoss:
    """Mean-squared-error loss (useful for critic / regression heads)."""

    def forward(self, pred, target):
        diff = pred - target
        return (diff ** 2).mean()

    def __call__(self, pred, target):
        return self.forward(pred, target)