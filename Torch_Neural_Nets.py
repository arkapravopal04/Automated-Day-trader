"""
Neural network components — identical architecture to the custom engine version
but implemented with torch.nn for GPU acceleration and reliable autograd.
"""

import torch
import torch.nn as nn
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_tensor(x, dtype=torch.float32):
    """Convert numpy array or tensor to a float tensor on DEVICE."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=DEVICE)
    return torch.tensor(x, dtype=dtype, device=DEVICE)


class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.bias)


class LayerNorm(nn.LayerNorm):
    def __init__(self, num_features, eps=1e-5, label=''):
        super().__init__(num_features, eps=eps)


class Dropout(nn.Dropout):
    def __init__(self, p=0.2):
        super().__init__(p=p)


class Flatten(nn.Module):
    def forward(self, x):
        return x.flatten()


class Sequential(nn.Sequential):
    pass


class Conv2D(nn.Module):
    """
    Input shape: (C, H, W) — single sample, no batch dim.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, label=''):
        super().__init__()
        kh, kw = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.conv = nn.Conv2d(in_channels, out_channels, (kh, kw),
                              stride=stride, padding=padding)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        out = self.conv(x.unsqueeze(0))  
        return out.squeeze(0)           
    
class LSTM(nn.Module):
    """
    Same interface as before: forward(x) → (all_hidden_states, (h, c))
    x: (T, input_size) tensor
    Returns list of T hidden state tensors each shape (hidden_size,)
    """
    def __init__(self, input_size, hidden_size, num_layers, label=''):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False)
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                n = param.shape[0]
                param.data[n // 4: n // 2].fill_(1.0)

    def forward(self, x, h_prev=None, c_prev=None):
        x_seq = x.unsqueeze(1)                        
        out, (h_n, c_n) = self.lstm(x_seq, None)      
        all_hidden = [out[t, 0] for t in range(out.shape[0])]
        return all_hidden, (h_n, c_n)


class Attention(nn.Module):
    """
    Self-attention over a list of hidden states.
    Input:  list of T tensors each (hidden_size,)  OR  tensor (T, hidden_size)
    Output: tensor (T, hidden_size)
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.scale = hidden_size ** -0.5
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)

    def forward(self, hidden_states):
        if isinstance(hidden_states, list):
            h_stack = torch.stack(hidden_states, dim=0)  
        else:
            h_stack = hidden_states

        Q = self.W_q(h_stack)                     
        K= self.W_k(h_stack)                     
        V= self.W_v(h_stack)                     
        scores= torch.mm(Q, K.t()) * self.scale       
        weights = torch.softmax(scores, dim=-1)         
        return torch.mm(weights, V)             

class FusionLayers(nn.Module):
    """
    Fuses LSTM, CNN, NLP and Regime signals.
    All inputs expected as (1, dim) tensors.
    """
    def __init__(self, lstm_hidden_size, cnn_out_channels, nlp_hidden_size, hidden_size):
        super().__init__()
        self.lstm_proj= Linear(lstm_hidden_size, hidden_size)
        self.cnn_proj = Linear(cnn_out_channels, hidden_size)
        self.nlp_proj= Linear(nlp_hidden_size,  hidden_size)
        self.regime_proj = Linear(3, hidden_size)
        self.out_proj= Linear(hidden_size, hidden_size)

        self.lstm_norm= LayerNorm(hidden_size)
        self.cnn_norm = LayerNorm(hidden_size)
        self.nlp_norm= LayerNorm(hidden_size)
        self.regime_norm = LayerNorm(hidden_size)

        self.attention = Attention(hidden_size)

    def forward(self, lstm_out, cnn_out, nlp_out, regime_out):
        # All inputs: (1, dim)
        lstm_hidden= self.lstm_norm(self.lstm_proj(lstm_out))
        cnn_hidden= self.cnn_norm(self.cnn_proj(cnn_out))
        nlp_hidden= self.nlp_norm(self.nlp_proj(nlp_out))
        regime_probs = torch.softmax(regime_out, dim=-1)
        regime_hidden = self.regime_norm(self.regime_proj(regime_probs))

        signals= torch.cat([lstm_hidden, cnn_hidden,
                                nlp_hidden, regime_hidden], dim=0)
        fused = self.attention(signals)     
        fused_mean = fused.mean(dim=0, keepdim=True)  
        return self.out_proj(fused_mean)                 

class RegimeDetector(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.lstm= LSTM(input_size, hidden_size, num_layers, label='regime_lstm')
        self.attention = Attention(hidden_size)
        self.linear = Linear(hidden_size, 3)

    def forward(self, x):
        hidden_states, _ = self.lstm(x)
        attn_out= self.attention(hidden_states)   
        last_step = attn_out[-1].unsqueeze(0)       
        return self.linear(last_step)               


class Adam_Optimiser(torch.optim.Adam):
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(parameters, lr=lr, betas=betas, eps=eps)


class SGD(torch.optim.SGD):
    def __init__(self, parameters, lr=0.01, momentum=0.9):
        super().__init__(parameters, lr=lr, momentum=momentum)