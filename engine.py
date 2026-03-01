import numpy as np
class Tensor:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = np.array(data, dtype=np.float64)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op 
        self.label = label
    def matmul(self, other):
        other = other if isinstance(other , Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), '@')
        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')
        def _backward():
            self_grad = out.grad
            other_grad = out.grad
            
            while self_grad.ndim > self.data.ndim:
                self_grad = self_grad.sum(axis=0)
            for i, dim in enumerate(self.data.shape):
                if dim == 1:
                    self_grad = self_grad.sum(axis=i, keepdims=True)
                    
            self.grad += self_grad
            other_grad = out.grad
            while other_grad.ndim > other.data.ndim:
                other_grad = other_grad.sum(axis=0)
            for i, dim in enumerate(other.data.shape):
                if dim == 1:
                    other_grad = other_grad.sum(axis=i, keepdims=True)
            other.grad += other_grad
        out._backward = _backward
        return out
    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), 'ReLU')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out
    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out
    def sigmoid(self):
        s = 1 / (1 + np.exp(-self.data))
        out = Tensor(s, (self,), 'sigmoid')
        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out
    def __pow__(self , other):
        assert isinstance(other, (int, float))
        out = Tensor(self.data**other , (self,), f'**{other}')
        def _backward():
            self.grad += (other * self.data**(other - 1)) * out.grad
        out._backward = _backward
        return out
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, (self, other), '/')
        def _backward():
            self.grad += (1 / other.data) * out.grad
            other.grad += (-self.data / (other.data**2)) * out.grad
        out._backward = _backward
        return out
    def __neg__(self):
        return self * -1
    def __rmul__(self, other):
        return self * other
    def __radd__(self, other):
        return self + other
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()