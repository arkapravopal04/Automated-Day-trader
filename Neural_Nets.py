import numpy as np
from engine import Tensor
from module import Module

class Sequential(Module):
    def __init__(self, *layers):
        self.layers = layers
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


class Linear(Module):
    def __init__(self, in_features, out_features):
        self.W = Tensor(np.random.randn(in_features, out_features) * np.sqrt(2. / in_features), label='W')
        self.b = Tensor(np.zeros(out_features), label='b')

    def forward(self, x):
        return x.matmul(self.W) + self.b

class SGD:
    def __init__(self, parameters, lr = 0.01, momentum = 0.9):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.velocities = {id(p): np.zeros_like(p.data) for p in parameters}
    def step(self):
        for param in self.parameters:
            self.velocities[id(param)] = self.momentum * self.velocities[id(param)] - self.lr * param.grad
            param.data += self.velocities[id(param)]
    def zero_grad(self):
        for param in self.parameters:
            param.grad = np.zeros_like(param.data)


class Conv2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, label=''):
        self.in_c = in_channels
        self.out_c = out_channels
        self.kh, self.kw = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding        
        fan_in = in_channels * self.kh * self.kw
        std = np.sqrt(2.0 / fan_in)
  
        self.w = Tensor(np.random.randn(out_channels, fan_in) * std, label=f'{label}_w')
        self.b = Tensor(np.zeros((out_channels, 1)), label=f'{label}_b')

    def __call__(self, x):
        col = x.im2col((self.kh, self.kw), self.stride, self.padding)
        
        res = self.w.matmul(col) + self.b
        
        _, h, w = x.data.shape
        out_h = (h + 2 * self.padding - self.kh) // self.stride + 1
        out_w = (w + 2 * self.padding - self.kw) // self.stride + 1
        
        return res.reshape(self.out_c, out_h, out_w)

    def parameters(self):
        return [self.w, self.b]
    

class Flatten(Module):
    def __call__(self, x):
        return x.flatten()
    def parameters(self):
        return []

class LayerNorm(Module):
    def __init__(self, num_features, eps=1e-5, label=''):
        self.gamma = Tensor(np.ones(num_features), label=f'{label}_gamma')
        self.beta = Tensor(np.zeros(num_features), label=f'{label}_beta')
        self.eps = eps
    def __call__(self , x):
        mean = x.mean(axis = 0, keepdims = True)
        var = ((x - mean) ** 2).mean(axis = 0, keepdims = True)
        x_normalized = (x - mean) / np.sqrt(var + self.eps)
        out = self.gamma * x_normalized + self.beta
        return out
    def parameters(self):
        return [self.gamma, self.beta]
    

class Dropout(Module):
    def __init__(self, p = 0.5):
        self.p = p
    def __call__(self, x):
        if np.random.rand() < self.p:
            return x * 0
        else:
            return x / (1 - self.p)
    def parameters(self):
        return []

class Adam_Optimiser(Module):
    def __init__(self, parameters, lr = 0.001, betas = (0.9, 0.999), eps = 1e-8):
        self.parameters = parameters
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.m = {id(p): np.zeros_like(p.data) for p in parameters}
        self.v = {id(p): np.zeros_like(p.data) for p in parameters}
        self.t = 0
    def step(self):
        self.t += 1
        for param in self.parameters:
            grad = param.grad
            m = self.m[id(param)]
            v = self.v[id(param)]
            beta1, beta2 = self.betas
            
            m[:] = beta1 * m + (1 - beta1) * grad
            v[:] = beta2 * v + (1 - beta2) * (grad ** 2)
            
            m_hat = m / (1 - beta1 ** self.t)
            v_hat = v / (1 - beta2 ** self.t)
            
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
    def zero_grad(self):
        for param in self.parameters:
            param.grad = np.zeros_like(param.data)
