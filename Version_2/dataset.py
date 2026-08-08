"""
PyTorch Multi-Ticker Rollout Dataset
Combines the preprocessed features of all tickers into a strictly time-aligned,
multi-asset PyTorch tensor. It provides windowed slices of the market state 
[n_envs, window_size, features] suitable for feeding directly into sequence models 
or vectorized Reinforcement Learning environments. Handles GPU/CPU placement.
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())
from paths import PROCESSED_DIR, TRAIN_FRAC, VAL_FRAC, is_kaggle

class MultiTickerRolloutDataset(Dataset):
    """
    A PyTorch Dataset that manages synchronized multi-asset time series data.
    """
    
    def __init__(self, window_size: int, split: str = 'train', device: str = None):
        """
        Initializes the dataset and loads the necessary splits into memory.
        
        Args:
            window_size (int): The sequence length (timesteps) required for one environment observation.
            split (str): One of 'train', 'val', or 'test'. Filters data based on
                        TRAIN_FRAC/VAL_FRAC fractions of available history (see paths.py),
                        applied chronologically so test is always the most recent data.
            device (str, optional): Target hardware for the tensor ('cpu' or 'cuda'). 
                                    If None, automatically detects CUDA availability.
        """
        self.window_size = window_size
        self.split = split
        
        # Autodetect hardware if device is not explicitly passed
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"🎮 Target Device for PyTorch Tensors: {self.device} ({'Kaggle' if is_kaggle() else 'Local'})")

        meta_path = os.path.join(PROCESSED_DIR, "metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata not found at {meta_path}. Run preprocess.py first.")

        with open(meta_path, "r") as f:
            self.metadata = json.load(f)
            
        self.feature_names = self.metadata["features"]
        self.tickers = sorted(list(self.metadata["norm_constants"].keys()))
        self.n_envs = len(self.tickers)
        
        print("=" * 60)
        print(f"BUILDING ROLLOUT DATASET [{split.upper()} SPLIT]")
        print("=" * 60)
        
        # Synchronize disparate ticker data into a single multidimensional tensor
        self.data_tensor, self.aligned_dates = self._load_and_align_data()
        
        # Enforce temporal splits to guarantee no test-data leakage during training.
        # Uses the same TRAIN_FRAC / VAL_FRAC (row-position based) that
        # preprocess.py used to compute normalization stats, so this scales
        # automatically with however much history was fetched (3yr, 6yr, ...)
        # instead of relying on a fixed calendar date that could drift out of
        # sync with the data you actually have.
        T = len(self.aligned_dates)
        train_end_idx = int(T * TRAIN_FRAC)
        val_end_idx = int(T * (TRAIN_FRAC + VAL_FRAC))

        if split == 'train':
            idx_slice = slice(0, train_end_idx)
        elif split == 'val':
            idx_slice = slice(train_end_idx, val_end_idx)
        elif split == 'test':
            idx_slice = slice(val_end_idx, T)
        else:
            raise ValueError(f"Invalid split name: {split}. Must be train, val, or test.")

        train_end_date = self.aligned_dates[train_end_idx - 1] if train_end_idx > 0 else None
        val_end_date = self.aligned_dates[val_end_idx - 1] if val_end_idx > 0 else None
        print(f"Split boundaries -> train ends: {train_end_date} | val ends: {val_end_date} "
              f"(train_frac={TRAIN_FRAC}, val_frac={VAL_FRAC})")

        # Push the finalized split tensor directly to the target device
        self.data_tensor = self.data_tensor[idx_slice].to(self.device)
        self.aligned_dates = self.aligned_dates[idx_slice]
        
        self._run_sanity_checks()

    def _load_and_align_data(self) -> Tuple[torch.Tensor, pd.DatetimeIndex]:
        """
        Loads all individual ticker feature files, performs an outer join on their
        DateTime indexes to perfectly align the timestamps, and handles missing data.
        
        Returns:
            Tuple[torch.Tensor, pd.DatetimeIndex]: 
                - 3D PyTorch Tensor of shape (Total_Timesteps, N_Tickers, N_Features)
                - The unified DateTime Index.
        """
        dfs = []
        for ticker in self.tickers:
            path = os.path.join(PROCESSED_DIR, f"{ticker}_features.parquet")
            
            print(f"📊 [DATASET] Importing preprocessed 5-min candles for {ticker}...")
            df = pd.read_parquet(path)
            print(f"  └─ Loaded {len(df)} processed 5-min candles across {len(df.columns)} features.")
            
            # Use MultiIndex to preserve the ticker name post-join
            df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
            dfs.append(df)
            
        print("\n⏳ Aligning timestamps across all 14 tickers (Outer Join)...")
        master_df = pd.concat(dfs, axis=1)
        
        # Handle misaligned market schedules (e.g. trading halts for a specific asset)
        # Forward fill the previous known value. For unfillable NaNs at the start, use 0.
        master_df = master_df.ffill().fillna(0)
        
        T = len(master_df)
        F = len(self.feature_names)
        
        tensor_np = np.zeros((T, self.n_envs, F), dtype=np.float32)
        
        # Map the dataframe into the strict [Timestep, Ticker, Feature] array
        for i, ticker in enumerate(self.tickers):
            tensor_np[:, i, :] = master_df[ticker][self.feature_names].values
            
        print(f"Time alignment complete. Total synchronized timesteps: {T}\n")
        return torch.tensor(tensor_np, dtype=torch.float32), master_df.index

    def _run_sanity_checks(self):
        """
        Validates the integrity of the final tensor. Ensures no NaNs leaked through, 
        checks for abnormally large chronological gaps, and verifies dimensions.
        """
        print(f"🔍 Running sanity checks on [{self.split.upper()}] split...")
        
        # 1. Verification of missing values
        assert not torch.isnan(self.data_tensor).any(), "Sanity Check Failed: NaNs detected in tensor!"
        
        # 2. Chronological continuity check
        time_deltas = self.aligned_dates.to_series().diff().dropna()
        max_gap = time_deltas.max() if len(time_deltas) > 0 else pd.Timedelta(seconds=0)
        # 5 days allows for 3-day holiday weekends without breaking
        assert max_gap <= pd.Timedelta(days=5), f"Sanity Check Failed: Data gap of {max_gap} exceeds tolerance!"
        
        # 3. Shape validation
        assert self.data_tensor.shape[1] == self.n_envs, f"Expected {self.n_envs} tickers, got {self.data_tensor.shape[1]}"
        assert self.data_tensor.shape[2] == len(self.feature_names), "Feature dimension mismatch!"
        
        print(f"Sanity checks passed. Final Tensor Shape: {self.data_tensor.shape} on {self.data_tensor.device}\n")

    def __len__(self) -> int:
        """
        Returns the number of valid rolling windows available in this split.
        """
        return max(0, len(self.data_tensor) - self.window_size + 1)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Extracts a sequence window of data.
        
        Args:
            idx (int): The starting index of the window.
            
        Returns:
            torch.Tensor: A tensor sliced and transposed to shape:
                          [n_envs, window_size, features]
        """
        # Slices [window, n_envs, features] -> transposes to [n_envs, window, features]
        window = self.data_tensor[idx : idx + self.window_size]
        return window.transpose(0, 1)

if __name__ == "__main__":
    # Test instantiation (auto-detects CPU/GPU)
    dataset = MultiTickerRolloutDataset(window_size=60, split='train')
    
    if len(dataset) > 0:
        sample_batch = dataset[0]
        print(f"Sample Batch Shape [n_envs, window_size, features]: {sample_batch.shape}")
        print(f"Sample Batch Device: {sample_batch.device}")