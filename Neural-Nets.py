import numpy as np
from engine import Tensor
from module import Module


class Linear(Module):
    def __init__(self, in_features, out_features):
        self.W = Tensor(np.random.randn(in_features, out_features) * np.sqrt(2. / in_features), label='W')
        self.b = Tensor(np.zeros(out_features), label='b')

    def forward(self, x):
        return x.matmul(self.W) + self.b

class SGD:
    def __init__(self, parameters, lr = 0.01):
        self.parameters = parameters
        self.lr = lr
    def step(self):
        for param in self.parameters:
            param.data -= self.lr * param.grad
    def zero_grad(self):
        for param in self.parameters:
            param.grad = np.zeros_like(param.data)