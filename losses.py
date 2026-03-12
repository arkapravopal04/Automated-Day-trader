import numpy as np
from engine import Tensor
import Neural_Nets as nn

class CrossEntropyLoss:
    def __init__(self):
        pass
    def forward(self, logits, target_idx):
        max_logits = np.max(logits.data, axis=1, keepdims=True)
        stable_logits = logits - max_logits
        exp_logits = stable_logits.exp()
        sum_logits = exp_logits.sum()
        probs = exp_logits / sum_logits

        target_prob = probs.data[target_idx]
        target_tensor = Tensor(target_prob, (probs,), 'target_select')

        def _backward():
            grad = np.zeros_like(probs.data)
            grad[target_idx] = target_tensor.grad
            probs.grad += grad

        target_tensor._backward = _backward
        loss = target_tensor.log() * -1
        return loss
    def __call__(self, logits, target_idx):
        return self.forward(logits, target_idx)