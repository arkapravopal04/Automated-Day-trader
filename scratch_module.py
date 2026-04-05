import numpy as np
from engine import Tensor

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def parameters(self):
        params = []
        for attr in self.__dict__.values():
            if isinstance(attr, Tensor):
                params.append(attr)
            elif isinstance(attr, Module):
                params.extend(attr.parameters())
            elif isinstance(attr, (list, tuple)):
                for item in attr:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
                    elif isinstance(item, Tensor):
                        params.append(item)
        return params

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
