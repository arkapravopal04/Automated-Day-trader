import numpy as np
from engine import Tensor
from module import Module
 
  
class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)
 
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
    def __init__(self, in_features, out_features, w_scale=1.0):
        self.W = Tensor(
            np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features) * w_scale,
            label='W'
        )
        self.b = Tensor(np.zeros(out_features), label='b')
 
    def forward(self, x):
        return x.matmul(self.W) + self.b
 
    def parameters(self):
        return [self.W, self.b]
 
 

class SGD:
    def __init__(self, parameters, lr=0.01, momentum=0.9):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.velocities = {id(p): np.zeros_like(p.data) for p in parameters}

    def _clip_grad_norm(self, max_norm=0.5):
        total_norm = np.sqrt(sum(
            np.sum(p.grad ** 2) for p in self.parameters
        ))
        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-8)
            for p in self.parameters:
                p.grad *= scale

    def step(self):
        self._clip_grad_norm(max_norm=0.5)
        for param in self.parameters:
            v = self.momentum * self.velocities[id(param)] - self.lr * param.grad
            self.velocities[id(param)] = v
            param.data += v

    def zero_grad(self):
        for param in self.parameters:
            param.grad = np.zeros_like(param.data)


class Conv2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, label='', window_size=48):
        self.in_c    = in_channels
        self.out_c   = out_channels
        self.kh, self.kw = (kernel_size if isinstance(kernel_size, tuple)
                            else (kernel_size, kernel_size))
        self.stride  = stride
        self.padding = padding
 
        fan_in      = in_channels * self.kh * self.kw * (window_size // 2)
        std         = np.sqrt(2.0 / fan_in)
        fan_in_cols = in_channels * self.kh * self.kw
        self.w = Tensor(np.random.randn(out_channels, fan_in_cols) * std,
                        label=f'{label}_w')
        self.b = Tensor(np.zeros((out_channels, 1)), label=f'{label}_b')
 
    def forward(self, x):
        col   = x.im2col((self.kh, self.kw), self.stride, self.padding)
        res   = self.w.matmul(col) + self.b
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
        self.gamma       = Tensor(np.ones(num_features),  label=f'{label}_gamma')
        self.beta        = Tensor(np.zeros(num_features), label=f'{label}_beta')
        self.eps         = eps
        self.num_features = num_features
 
    def forward(self, x):
        mean  = x.mean(axis=-1, keepdims=True) if x.data.ndim > 1 else x.mean()
        diff  = x - mean
        var   = ((diff ** 2).mean(axis=-1, keepdims=True)
                 if x.data.ndim > 1 else (diff ** 2).mean())
        x_norm = diff / ((var + self.eps) ** 0.5)
        return x_norm * self.gamma + self.beta
 
    def parameters(self):
        return [self.gamma, self.beta]
 
 
class Dropout(Module):
    def __init__(self, p=0.2):
        self.p        = p
        self.training = True
 
    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        mask  = (np.random.rand(*x.data.shape) > self.p).astype(np.float64)
        scale = 1.0 / (1.0 - self.p)
        out   = Tensor(x.data * mask * scale, (x,), 'dropout')
 
        def _backward():
            x.grad += out.grad * mask * scale
 
        out._backward = _backward
        return out
 
    def parameters(self):
        return []
 
class Adam_Optimiser:
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.parameters  = parameters
        self.lr          = lr
        self.betas       = betas
        self.eps         = eps
        self.weight_decay = weight_decay
        self.m = {id(p): np.zeros_like(p.data) for p in parameters}
        self.v = {id(p): np.zeros_like(p.data) for p in parameters}
        self.t = 0

    def _clip_grad_norm(self, max_norm=0.5):
        """
        Enforces tight submodule local gradient clipping.
        Restricts the step changes of individual layers to a max norm of 0.5.
        """
        total_norm = np.sqrt(sum(
            np.sum(p.grad ** 2) for p in self.parameters if p.grad is not None
        ))
        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-8)
            for p in self.parameters:
                if p.grad is not None:
                    p.grad *= scale

    def step(self):
        self.t += 1
        # Tightened gradient norm clipping constraint down to 0.5
        self._clip_grad_norm(max_norm=0.5)
        if self.weight_decay != 0.0:
            for param in self.parameters:
                param.data *= (1.0 - self.lr * self.weight_decay)
        beta1, beta2 = self.betas
        for param in self.parameters:
            grad = param.grad
            if grad is None or np.all(grad == 0):
                continue
            self.m[id(param)] = beta1 * self.m[id(param)] + (1 - beta1) * grad
            self.v[id(param)] = beta2 * self.v[id(param)] + (1 - beta2) * (grad ** 2)
            m_hat = self.m[id(param)] / (1 - beta1 ** self.t)
            v_hat = self.v[id(param)] / (1 - beta2 ** self.t)
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for param in self.parameters:
            param.grad = np.zeros_like(param.data)

 
class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, label=''):
        self.input_size  = input_size
        self.hidden_size = hidden_size
        scale = np.sqrt(1.0 / (input_size + hidden_size))
        self.W_f = Tensor(np.random.randn(input_size + hidden_size, hidden_size) * scale, label=f'{label}_W_f')
        self.b_f = Tensor(np.ones(hidden_size),  label=f'{label}_b_f')  
        self.W_i = Tensor(np.random.randn(input_size + hidden_size, hidden_size) * scale, label=f'{label}_W_i')
        self.b_i = Tensor(np.zeros(hidden_size), label=f'{label}_b_i')
        self.W_c = Tensor(np.random.randn(input_size + hidden_size, hidden_size) * scale, label=f'{label}_W_c')
        self.b_c = Tensor(np.zeros(hidden_size), label=f'{label}_b_c')
        self.W_o = Tensor(np.random.randn(input_size + hidden_size, hidden_size) * scale, label=f'{label}_W_o')
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
        return [self.W_f, self.b_f, self.W_i, self.b_i,
                self.W_c, self.b_c, self.W_o, self.b_o]
 
 
class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers, label=''):
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.cells = [
            LSTMCell(input_size if i == 0 else hidden_size, hidden_size,
                     label=f'{label}_cell_{i}')
            for i in range(num_layers)
        ]
 
    def forward(self, x, h_prev=None, c_prev=None, tbptt_chunk=16):
        T = x.data.shape[0]
        if h_prev is None:
            h_prev = [Tensor(np.zeros(self.cells[0].hidden_size)) for _ in self.cells]
        if c_prev is None:
            c_prev = [Tensor(np.zeros(self.cells[0].hidden_size)) for _ in self.cells]
 
        all_hidden_states = []
        for t in range(T):
            if t > 0 and t % tbptt_chunk == 0:
                h_prev = [Tensor(h.data.copy()) for h in h_prev]
                c_prev = [Tensor(c.data.copy()) for c in c_prev]
 
            x_t = x[t]
            for i, cell in enumerate(self.cells):
                h_next, c_next = cell(x_t, h_prev[i], c_prev[i])
                h_prev[i] = h_next
                c_prev[i] = c_next
                x_t = h_next
            all_hidden_states.append(h_prev[-1])
 
        return all_hidden_states, (h_prev, c_prev)
 
    def parameters(self):
        params = []
        for cell in self.cells:
            params.extend(cell.parameters())
        return params
 
 
class Attention(Module):
    def __init__(self, hidden_size):
        scale        = np.sqrt(1.0 / hidden_size)
        self.W_q     = Tensor(np.random.randn(hidden_size, hidden_size) * scale, label='W_q')
        self.W_k     = Tensor(np.random.randn(hidden_size, hidden_size) * scale, label='W_k')
        self.W_v     = Tensor(np.random.randn(hidden_size, hidden_size) * scale, label='W_v')
        self.hidden_size = hidden_size
 
    def forward(self, hidden_states):
        if isinstance(hidden_states, list):
            h_stack = Tensor.stack([h.reshape(-1) for h in hidden_states], axis=0)
        else:
            h_stack = hidden_states
 
        Q      = h_stack.matmul(self.W_q)
        K      = h_stack.matmul(self.W_k)
        V      = h_stack.matmul(self.W_v)
        d_k    = Q.data.shape[-1]
        K_T    = K.transpose((1, 0))
        scores = Q.matmul(K_T) * (d_k ** -0.5)
 
        weights = scores.softmax(axis=-1)
        return weights.matmul(V)
 
    def parameters(self):
        return [self.W_q, self.W_k, self.W_v]
 
 
class MultiHeadAttention(Module):
    def __init__(self, hidden_size, num_heads=4):
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.hidden_size = hidden_size
        self.num_heads   = num_heads
        self.head_dim    = hidden_size // num_heads
 
        scale    = np.sqrt(1.0 / hidden_size)
        self.W_q = [Tensor(np.random.randn(hidden_size, self.head_dim) * scale) for _ in range(num_heads)]
        self.W_k = [Tensor(np.random.randn(hidden_size, self.head_dim) * scale) for _ in range(num_heads)]
        self.W_v = [Tensor(np.random.randn(hidden_size, self.head_dim) * scale) for _ in range(num_heads)]
        self.W_o = Tensor(np.random.randn(hidden_size, hidden_size) * np.sqrt(1.0 / (hidden_size * self.num_heads)))
 
    def forward(self, hidden_states):
        if isinstance(hidden_states, list):
            h_stack = Tensor.stack([h.reshape(-1) for h in hidden_states], axis=0)
        else:
            h_stack = hidden_states
 
        head_outputs = []
        for i in range(self.num_heads):
            Q = h_stack.matmul(self.W_q[i])
            K = h_stack.matmul(self.W_k[i])
            V = h_stack.matmul(self.W_v[i])
            d_k = Q.data.shape[-1]
            K_T = K.transpose((1, 0))
            scores = Q.matmul(K_T) * (d_k ** -0.5)
 
            weights = scores.softmax(axis=-1)
            head_outputs.append(weights.matmul(V))
 
        combined = head_outputs[0]
        for h_out in head_outputs[1:]:
            combined = combined.concat(h_out, axis=1)
 
        return combined.matmul(self.W_o)
 
    def parameters(self):
        params = []
        for i in range(self.num_heads):
            params += [self.W_q[i], self.W_k[i], self.W_v[i]]
        params.append(self.W_o)
        return params
 
 
class FusionLayers(Module):
    def __init__(self, lstm_hidden_size, cnn_out_channels, nlp_hidden_size,
                 hidden_size, risk_size=8):
        self.lstm_proj   = Linear(lstm_hidden_size, hidden_size)
        self.cnn_proj    = Linear(cnn_out_channels, hidden_size)
        self.nlp_proj    = Linear(nlp_hidden_size,  hidden_size)
        self.regime_proj = Linear(3, hidden_size)
        self.risk_proj   = Linear(risk_size, hidden_size)
        
        self.out_proj    = Linear(hidden_size, hidden_size, w_scale=0.01)
 
        self.lstm_norm   = LayerNorm(hidden_size)
        self.cnn_norm    = LayerNorm(hidden_size)
        self.nlp_norm    = LayerNorm(hidden_size)
        self.regime_norm = LayerNorm(hidden_size)
        self.risk_norm   = LayerNorm(hidden_size)
 
        self.attention   = MultiHeadAttention(hidden_size, num_heads=4)
 
    def forward(self, lstm_out, cnn_out, nlp_out, regime_out, risk_out=None):
        lstm_hidden   = self.lstm_norm(self.lstm_proj(lstm_out))
        cnn_hidden    = self.cnn_norm(self.cnn_proj(cnn_out))
        nlp_hidden    = self.nlp_norm(self.nlp_proj(nlp_out))
 
        regime_probs  = regime_out.softmax(axis=-1)
        regime_hidden = self.regime_norm(self.regime_proj(regime_probs))
 
        signals = [lstm_hidden, cnn_hidden, nlp_hidden, regime_hidden]
 
        if risk_out is not None:
            risk_hidden = self.risk_norm(self.risk_proj(risk_out))
            signals.append(risk_hidden)
 
        fused      = self.attention(signals)
        fused_mean = fused.mean(axis=0, keepdims=True)
        return self.out_proj(fused_mean)
 
    def parameters(self):
        params = []
        for sub in [self.lstm_proj,   self.cnn_proj,    self.nlp_proj,
                    self.regime_proj, self.risk_proj,    self.out_proj,
                    self.lstm_norm,   self.cnn_norm,     self.nlp_norm,
                    self.regime_norm, self.risk_norm,    self.attention]:
            params.extend(sub.parameters())
        return params
 
 
class RegimeDetector(Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        self.lstm      = LSTM(input_size, hidden_size, num_layers, label='regime_lstm')
        self.attention = Attention(hidden_size)
        
        self.linear    = Linear(hidden_size, 3, w_scale=0.01)
 
    def forward(self, x):
        hidden_states, _ = self.lstm(x)
        attn_out  = self.attention(hidden_states)
        last_step = attn_out[-1].reshape(1, -1)  
        logits    = self.linear(last_step)
        return logits.reshape(3)
 
    def parameters(self):
        params = []
        for sub in [self.lstm, self.attention, self.linear]:
            params.extend(sub.parameters())
        return params
