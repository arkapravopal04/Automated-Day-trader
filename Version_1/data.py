import yfinance as yf
import numpy as np
import pandas as pd


def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    data = data.dropna()
    return data

def transform_data(data):
    data = data.pct_change()
    data = data.dropna()
    return data

def build_windows(data, window_size, raw_data=None):
    X, y, prices = [], [], []
    for i in range(len(data) - window_size):
        X.append(data.iloc[i:i+window_size].values)
        y.append(1 if data.iloc[i+window_size]['Close'] > 0 else 0)
        if raw_data is not None:
            prices.append(raw_data.iloc[i+window_size]['Close'])
        else:
            prices.append(0.0)
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int32), np.array(prices, dtype=np.float64)

class DataLoader:
    def __init__(self, X, y, batch_size=32, shuffle = True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(X)
        self.num_batches = int(np.ceil(self.num_samples / batch_size))
    
    def __iter__(self):
        indices = np.random.permutation(self.num_samples) if self.shuffle else np.arange(self.num_samples)
        for start in range(0, self.num_samples, self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            yield self.X[batch_idx], self.y[batch_idx]
        
    
    def __len__(self):
        return self.num_batches

def generate_regime_labels(X, threshold=0.001):
    returns = X[:, :, 3].mean(axis=1)
    labels = np.where(returns > threshold, 2,
                      np.where(returns < -threshold, 0, 1))
    return labels.astype(np.int32)