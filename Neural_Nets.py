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

    def forward(self, x):
        col = x.im2col((self.kh, self.kw), self.stride, self.padding)
        
        res = self.w.matmul(col) + self.b
        
        _, h, w = x.data.shape
        out_h = (h + 2 * self.padding - self.kh) // self.stride + 1
        out_w = (w + 2 * self.padding - self.kw) // self.stride + 1
        
        return res.reshape(self.out_c, out_h, out_w)

    def parameters(self):
        return [self.w, self.b]
    

class Flatten(Module):
    def forward(self, x):
        return x.flatten()
    def parameters(self):
        return []

class LayerNorm(Module):
    def __init__(self, num_features, eps=1e-5, label=''):
        self.gamma = Tensor(np.ones(num_features), label=f'{label}_gamma')
        self.beta = Tensor(np.zeros(num_features), label=f'{label}_beta')
        self.eps = eps
    def forward(self , x):
        mean = x.mean(axis = 0, keepdims = True)
        var = ((x - mean) ** 2).mean(axis = 0, keepdims = True)
        x_normalized = (x - mean) / ((var + self.eps)**0.5)
        out = self.gamma * x_normalized + self.beta
        return out
    def parameters(self):
        return [self.gamma, self.beta]
    

class Dropout(Module):
    def __init__(self, p = 0.2):
        self.p = p
        self.training = True
    def forward(self, x):
        if not self.training:
            return x
        random_mask = (np.random.rand(*x.data.shape) > self.p).astype(np.float32)
        out = Tensor(x.data * random_mask, (x,), 'dropout') 
        def _backward():
            x.grad += out.grad * random_mask
        out._backward = _backward
        return out * (1.0 / (1.0 - self.p))
    def parameters(self):
        return []

class Adam_Optimiser:
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


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, label = ''):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_f = Tensor(np.random.randn(input_size + hidden_size, hidden_size) * np.sqrt( 1/ (input_size + hidden_size)), label=f'{label}_W_f')
        self.b_f = Tensor(np.ones(hidden_size), label=f'{label}_b_f')  
        self.W_i = Tensor(np.random.randn(input_size + hidden_size, hidden_size) * np.sqrt( 1/ (input_size + hidden_size)), label=f'{label}_W_i')
        self.b_i = Tensor(np.zeros(hidden_size), label=f'{label}_b_i')
        self.W_c = Tensor(np.random.randn(input_size + hidden_size, hidden_size) * np.sqrt( 1/ (input_size + hidden_size)), label=f'{label}_W_c')
        self.b_c = Tensor(np.zeros(hidden_size), label=f'{label}_b_c')
        self.W_o = Tensor(np.random.randn(input_size + hidden_size, hidden_size) * np.sqrt( 1/ (input_size + hidden_size)), label=f'{label}_W_o')
        self.b_o = Tensor(np.zeros(hidden_size), label=f'{label}_b_o')

    def forward(self, x, h_prev, c_prev):
        combined = x.concat(h_prev)
        f_t = (combined.matmul(self.W_f) + self.b_f).sigmoid()
        i_t = (combined.matmul(self.W_i) + self.b_i).sigmoid()
        g_t = (combined.matmul(self.W_c) + self.b_c).tanh()
        o_t = (combined.matmul(self.W_o) + self.b_o).sigmoid()
        c_next = f_t * c_prev + i_t * g_t
        h_next = o_t * c_next.tanh()
        return h_next, c_next
    
    def parameters(self):
        return [self.W_f, self.b_f, self.W_i, self.b_i, self.W_c, self.b_c, self.W_o, self.b_o]
    
class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers, label = ''):
        self.cells = [LSTMCell(input_size if i == 0 else hidden_size, hidden_size, label=f'lstm_cell_{i}') for i in range(num_layers)]
    def forward(self, x, h_prev = None, c_prev = None):

        if h_prev is None:
            h_prev = [Tensor(np.zeros((self.cells[0].hidden_size,))) for _ in self.cells]
        if c_prev is None:
            c_prev = [Tensor(np.zeros((self.cells[0].hidden_size,))) for _ in self.cells]

        all_hidden_states = []
        for t in range(x.data.shape[0]):
            x_t = x[t] 
            for i, cell in enumerate(self.cells):
                h_next, c_next = cell(x_t, h_prev[i], c_prev[i])
                h_prev[i], c_prev[i] = h_next, c_next
                x_t = h_next
            all_hidden_states.append(h_prev[-1])
        return all_hidden_states , (h_prev, c_prev)
    
    def parameters(self):
        params = []
        for cell in self.cells:
            params.extend(cell.parameters())
        return params