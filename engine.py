import numpy as np


class Tensor:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), '@')

        def _backward():
            out_grad = np.atleast_2d(out.grad)
            self_data = np.atleast_2d(self.data)
            other_data = np.atleast_2d(other.data)
            self_grad = out_grad @ other_data.T
            other_grad = self_data.T @ out_grad
            self.grad += self_grad.reshape(self.data.shape)
            other.grad += other_grad.reshape(other.data.shape)

        out._backward = _backward
        return out

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            def _reduce_grad(grad, target_shape):
                # Sum away extra leading dims
                while grad.ndim > len(target_shape):
                    grad = grad.sum(axis=0)
                # Sum over broadcast dims (size==1 in target)
                for i, dim in enumerate(target_shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                return grad

            self.grad += _reduce_grad(out.grad, self.data.shape)
            other.grad += _reduce_grad(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            def _reduce(grad, target_shape):
                """Sum broadcast dimensions so grad matches target_shape."""
                while grad.ndim > len(target_shape):
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(target_shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                return grad.reshape(target_shape)

            self.grad += _reduce(other.data * out.grad, self.data.shape)
            other.grad += _reduce(self.data * out.grad, other.data.shape)

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
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        # Numerically stable sigmoid
        s = np.where(self.data >= 0,
                     1 / (1 + np.exp(-self.data)),
                     np.exp(self.data) / (1 + np.exp(self.data)))
        out = Tensor(s, (self,), 'sigmoid')

        def _backward():
            self.grad += s * (1 - s) * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Tensor(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, (self, other), '/')

        def _backward():
            def _reduce(grad, target_shape):
                while grad.ndim > len(target_shape):
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(target_shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                return grad.reshape(target_shape)

            self.grad  += _reduce((1 / other.data) * out.grad, self.data.shape)
            other.grad += _reduce((-self.data / (other.data ** 2)) * out.grad, other.data.shape)

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

    def sum(self, axis=None, keepdims=False):
        res = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(res, (self,), 'sum')

        def _backward():
            grad_to_broadcast = out.grad
            if axis is not None and not keepdims:
                shape = list(self.data.shape)
                axes = (axis,) if isinstance(axis, int) else axis
                for a in axes:
                    shape[a] = 1
                grad_to_broadcast = out.grad.reshape(shape)
            self.grad += np.ones_like(self.data) * grad_to_broadcast

        out._backward = _backward
        return out

    def exp(self):
        # Clamp to avoid overflow
        e = np.exp(np.clip(self.data, -500, 500))
        out = Tensor(e, (self,), 'exp')

        def _backward():
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(np.clip(self.data, 1e-8, None)), (self,), 'log')

        def _backward():
            self.grad += (1 / np.clip(self.data, 1e-8, None)) * out.grad

        out._backward = _backward
        return out

    def max(self, axis=None, keepdims=False):
        out = Tensor(np.max(self.data, axis=axis, keepdims=keepdims), (self,), 'max')

        def _backward():
            if axis is None:
                out_data_expanded = out.data
                out_grad_expanded = out.grad
            else:
                if not keepdims:
                    out_data_expanded = np.expand_dims(out.data, axis=axis)
                    out_grad_expanded = np.expand_dims(out.grad, axis=axis)
                else:
                    out_data_expanded = out.data
                    out_grad_expanded = out.grad

            mask = (self.data == out_data_expanded).astype(np.float64)
            # Distribute grad equally if there are ties
            mask /= np.maximum(mask.sum(axis=axis, keepdims=True), 1)
            self.grad += mask * out_grad_expanded

        out._backward = _backward
        return out

    def im2col(self, kernel_size, stride=1, padding=0):
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
                cols[y * KW + x::KH * KW, :] = img[:, y:y_max:stride, x:x_max:stride].reshape(C, -1)
        out = Tensor(cols, (self,), 'im2col')

        def _backward():
            dcols = out.grad
            dimg = np.zeros_like(img)
            for y in range(KH):
                y_max = y + stride * out_h
                for x in range(KW):
                    x_max = x + stride * out_w
                    dimg[:, y:y_max:stride, x:x_max:stride] += dcols[y * KW + x::KH * KW, :].reshape(C, out_h, out_w)
            self.grad += dimg[:, padding:H + padding, padding:W + padding]

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

    def flatten(self):
        out = Tensor(self.data.flatten(), (self,), 'flatten')

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), (self,), 'mean')

        def _backward():
            grad = out.grad
            if axis is None:
                self.grad += np.ones_like(self.data) * grad / self.data.size
            else:
                if not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                self.grad += np.ones_like(self.data) * grad / self.data.shape[axis]

        out._backward = _backward
        return out

    def concat(self, other, axis=0):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.concatenate((self.data, other.data), axis=axis), (self, other), 'concat')

        def _backward():
            grads = np.split(out.grad, [self.data.shape[axis]], axis=axis)
            self.grad += grads[0]
            other.grad += grads[1]

        out._backward = _backward
        return out

    def slice(self, start, end, axis=0):
        slices = [slice(None)] * self.data.ndim
        slices[axis] = slice(start, end)
        out = Tensor(self.data[tuple(slices)], (self,), 'slice')

        def _backward():
            grad = np.zeros_like(self.data)
            grad[tuple(slices)] = out.grad
            self.grad += grad

        out._backward = _backward
        return out

    def __getitem__(self, key):
        out = Tensor(self.data[key], (self,), 'getitem')

        def _backward():
            grad = np.zeros_like(self.data)
            np.add.at(grad, key, out.grad)  # handles repeated indices correctly
            self.grad += grad

        out._backward = _backward
        return out

    @staticmethod
    def stack(tensors, axis=0):
        data = np.stack([t.data for t in tensors], axis=axis)
        out = Tensor(data, tuple(tensors), 'stack')

        def _backward():
            grads = np.split(out.grad, len(tensors), axis=axis)
            for i, t in enumerate(tensors):
                t.grad += np.squeeze(grads[i], axis=axis)

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        # Initialise scalar or array gradient
        self.grad = np.ones_like(self.data, dtype=np.float64)
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op='{self._op}')"