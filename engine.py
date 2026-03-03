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
        out._backward  = _backward
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
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
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
    def __sub__(self, other): 
        return self + (-other)
    def __rsub__(self, other):
        return other + (-self)
    def __neg__(self):
        return self * -1
    def __rmul__(self, other):
        return self * other
    def __radd__(self, other):
        return self + other
    def sum(self):
        out = Tensor(np.sum(self.data), (self,), 'sum')
        def _backward():
            self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward
        return out

    def im2col(self, kernel_size, stride= 1, padding=0):
        C, H, W = self.data.shape
        KH, KW = kernel_size
        out_h = (H + 2 * padding - KH) // stride + 1
        out_w = (W + 2 * padding - KW) // stride + 1
        img = np.pad(self.data, ((0, 0), (padding, padding), (padding, padding)), mode='constant')
        cols = np.zeros((C * KH * KW, out_h * out_w))
        for y in range(KH):
            y_max = y + stride * out_h
            for x in range(KW):
                x_max = x + stride * out_w
                cols[y*KW+x::KH*KW, :] = img[:, y:y_max:stride, x:x_max:stride].reshape(C, -1)
        out = Tensor(cols, (self,), 'im2col')
        def _backward():
            dcols = out.grad
            dimg = np.zeros_like(img)
            for y in range(KH):
                y_max = y + stride * out_h
                for x in range(KW):
                    x_max = x + stride * out_w
                    dimg[:, y:y_max:stride, x:x_max:stride] += dcols[y*KW+x::KH*KW, :].reshape(C, out_h, out_w)
            self.grad += dimg[:, padding:H+padding, padding:W+padding]
        out._backward = _backward
        return out
    
    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,), 'reshape')
        def _backward():
            self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out
    
    def transpose(self, axes):
        out = Tensor(self.data.transpose(axes), (self,), 'transpose')
        def _backward():
            self.grad += out.grad.transpose(np.argsort(axes))
        out._backward = _backward
        return out
    
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