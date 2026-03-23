import yfinance as yf
import numpy as np
import pandas as pd


def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
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
    return np.array(X), np.array(y), np.array(prices)

class DataLoader:
    def __init__(self, X, y, batch_size=32):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.num_samples = len(X)
        self.num_batches = int(np.ceil(self.num_samples / batch_size))
    
    def __iter__(self):
        self.indices = np.random.permutation(self.num_samples)
        self.current_batch = 0
        return self
    
    def __len__(self):
        return self.num_batches
    def __next__(self):
        if self.current_batch < self.num_batches:
            start = self.current_batch * self.batch_size
            end = min(start + self.batch_size, self.num_samples)
            batch_indices = self.indices[start:end]
            batch_X = self.X[batch_indices]
            batch_y = self.y[batch_indices]
            self.current_batch += 1
            return batch_X, batch_y
        else:
            raise StopIteration
        
        

def generate_regime_labels(X, threshold=0.001):
    returns = X[:, :, 3].mean(axis=1) 
    conditions = [
        returns > threshold,       
        returns < -threshold,      
    ]
    choices = [2, 0]
    labels = np.select(conditions, choices, default=1)  
    return labels