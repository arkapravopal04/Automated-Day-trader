"""
Diagnostic script for the Algorithmic Trading Data Pipeline.
Run this script locally to verify that your environment, API keys, 
data folders, and PyTorch datasets are correctly configured.
"""

import os
import sys
import traceback

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

sys.path.append(BASE_DIR)
from paths import RAW_DIR, PROCESSED_DIR, is_kaggle

# Terminal colors for better readability
class Colors:
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_status(step: str, success: bool, message: str = ""):
    if success:
        print(f"{Colors.OKGREEN} [PASS]{Colors.ENDC} {step}")
        if message:
            print(f"   └─ {message}")
    else:
        print(f"{Colors.FAIL}[FAIL]{Colors.ENDC} {step}")
        if message:
            print(f"   └─ {message}")

def run_diagnostics():
    print(f"\n{Colors.BOLD}=== RUNNING PIPELINE DIAGNOSTICS ==={Colors.ENDC}\n")
    print(f"{Colors.BOLD}Environment:{Colors.ENDC} {'Kaggle' if is_kaggle() else 'Local'}")
    print(f"{Colors.BOLD}Raw dir:{Colors.ENDC} {RAW_DIR}")
    print(f"{Colors.BOLD}Processed dir:{Colors.ENDC} {PROCESSED_DIR}\n")

    print(f"{Colors.BOLD}[1/5] Checking Dependencies...{Colors.ENDC}")
    try:
        import torch
        import pandas as pd
        import pyarrow
        import alpaca
        print_status("Core Libraries Imported", True, f"PyTorch: {torch.__version__}, Pandas: {pd.__version__}")
    except ImportError as e:
        print_status("Core Libraries Imported", False, str(e))
        print("   └─ Suggestion: Run `pip install torch pandas pyarrow alpaca-py python-dotenv`")
        return

    print(f"\n{Colors.BOLD}[2/5] Checking Alpaca API Credentials...{Colors.ENDC}")
    try:
        from fetch_alpaca import get_alpaca_credentials
        api_key, secret_key = get_alpaca_credentials()
        if api_key and secret_key:
            # Mask the keys for security
            masked_api = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
            print_status("Credentials Found", True, f"API Key: {masked_api}")
        else:
            print_status("Credentials Found", False, "Could not find Alpaca keys in .env, env vars, or Kaggle.")
    except Exception as e:
        print_status("Credentials Found", False, f"Error importing fetch_alpaca: {e}")

    print(f"\n{Colors.BOLD}[3/5] Checking Raw Data (Parquet)...{Colors.ENDC}")
    parquet_dir = RAW_DIR
    if os.path.exists(parquet_dir):
        files = [f for f in os.listdir(parquet_dir) if f.endswith('.parquet')]
        if files:
            print_status("Raw Parquet Files", True, f"Found {len(files)} ticker files in {parquet_dir}")
        else:
            print_status("Raw Parquet Files", False, f"Directory exists but no .parquet files found in {parquet_dir}.")
    else:
        print_status("Raw Parquet Files", False, f"Directory not found: {parquet_dir}. Run fetch_alpaca.py first.")

    print(f"\n{Colors.BOLD}[4/5] Checking Preprocessed Features...{Colors.ENDC}")
    processed_dir = PROCESSED_DIR
    if os.path.exists(processed_dir):
        meta_path = os.path.join(processed_dir, "metadata.json")
        if os.path.exists(meta_path):
            try:
                import json
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                features = meta.get("features", [])
                print_status("Metadata JSON", True, f"Found metadata with {len(features)} features.")
            except Exception as e:
                print_status("Metadata JSON", False, f"Error reading metadata.json: {e}")
        else:
            print_status("Metadata JSON", False, "metadata.json not found. Run preprocess.py.")
            
        processed_files = [f for f in os.listdir(processed_dir) if f.endswith('.parquet')]
        if processed_files:
            print_status("Processed Parquet Files", True, f"Found {len(processed_files)} processed files.")
        else:
            print_status("Processed Parquet Files", False, "No processed .parquet files found.")
    else:
        print_status("Preprocessed Features", False, f"Directory not found: {processed_dir}. Run preprocess.py first.")

    print(f"\n{Colors.BOLD}[5/5] Checking PyTorch Dataset Integration...{Colors.ENDC}")
    try:
        from dataset import MultiTickerRolloutDataset
        
        # Check if metadata exists before trying to instantiate to avoid redundant tracebacks
        if not os.path.exists(os.path.join(processed_dir, "metadata.json")):
            print_status("Dataset Instantiation", False, "Cannot test dataset without metadata.json.")
        else:
            dataset = MultiTickerRolloutDataset(window_size=60, split='train')
            print_status("Dataset Instantiation", True, f"Successfully loaded {dataset.split} split. Total windows: {len(dataset)}")
            
            if len(dataset) > 0:
                sample = dataset[0]
                import torch
                if isinstance(sample, torch.Tensor):
                    print_status("Tensor Generation", True, f"Output shape [n_envs, window_size, features]: {sample.shape}")
                    print_status("Device Placement", True, f"Tensor is on device: {sample.device}")
                else:
                    print_status("Tensor Generation", False, "Dataset did not return a PyTorch Tensor.")
            else:
                print_status("Tensor Generation", False, "Dataset is empty. Cannot test tensor shapes.")
                
    except Exception as e:
        print_status("Dataset Integration", False, "Failed to load dataset.")
        print(f"\n{Colors.WARNING}Traceback for Dataset Error:{Colors.ENDC}")
        traceback.print_exc()
        
    print(f"\n{Colors.BOLD}=== DIAGNOSTICS COMPLETE ==={Colors.ENDC}\n")

if __name__ == "__main__":
    run_diagnostics()