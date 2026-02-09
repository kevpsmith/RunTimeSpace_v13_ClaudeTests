import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import sys
# Add the parent directory of bulk_runs (this_studio) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from torch.utils.data import DataLoader 
from pytorch_lightning import Trainer
from data_acquisition.RL_Data_Acquisition_v12 import PolygonDataFetcher #Use the file that fixes j index error on val and predict
from data_acquisition.RL_Data_Prep_v12 import PolygonDataPrep
import pickle
from pytorch_lightning.callbacks import ModelCheckpoint
from datetime import datetime, timedelta
import pandas as pd
import time

def validate_datasets(data, growth_rates, prediction_data, tickers):
    """
    Validate all datasets for NaN, Inf, and zero values
    
    Args:
        data: Training data array
        growth_rates: Growth rates array
        prediction_data: Prediction data array
        tickers: List of tickers
    """
    def check_array(arr, name):
        """Helper function to check a single array"""
        if isinstance(arr, np.ndarray):
            nan_count = np.isnan(arr).sum()
            inf_count = np.isinf(arr).sum()
            zero_count = (arr == 0).sum()
            total_elements = arr.size
            
            print(f"\n{name} Validation:")
            print(f"Shape: {arr.shape}")
            print(f"NaN count: {nan_count} ({(nan_count/total_elements)*100:.4f}% of elements)")
            print(f"Inf count: {inf_count} ({(inf_count/total_elements)*100:.4f}% of elements)")
            print(f"Zero count: {zero_count} ({(zero_count/total_elements)*100:.4f}% of elements)")
            print(f"Min value: {np.min(arr[~np.isnan(arr) & ~np.isinf(arr)])}")
            print(f"Max value: {np.max(arr[~np.isnan(arr) & ~np.isinf(arr)])}")
            print(f"Mean value: {np.mean(arr[~np.isnan(arr) & ~np.isinf(arr)])}")
            
            # Check for extreme values
            abs_vals = np.abs(arr[~np.isnan(arr) & ~np.isinf(arr)])
            extreme_count = (abs_vals > 1e6).sum()
            if extreme_count > 0:
                print(f"Warning: Found {extreme_count} values with absolute value > 1e6")
            
            # Feature-wise analysis for training data
            if len(arr.shape) >= 3:  # If we have feature dimension
                print("\nFeature-wise statistics:")
                for i in range(arr.shape[-1]):
                    feature = arr[..., i]
                    nan_in_feature = np.isnan(feature).sum()
                    inf_in_feature = np.isinf(feature).sum()
                    if nan_in_feature > 0 or inf_in_feature > 0:
                        print(f"Feature {i}: NaN count = {nan_in_feature}, Inf count = {inf_in_feature}")
                    
                    # Calculate statistics excluding NaN and Inf
                    valid_data = feature[~np.isnan(feature) & ~np.isinf(feature)]
                    if len(valid_data) > 0:
                        print(f"Feature {i} - Min: {valid_data.min():.4f}, Max: {valid_data.max():.4f}, "
                              f"Mean: {valid_data.mean():.4f}, Std: {valid_data.std():.4f}")
            
            return nan_count > 0 or inf_count > 0

    # Validate each dataset
    print("\nValidating datasets...")
    has_issues = False
    
    if check_array(data, "Training Data"):
        has_issues = True
        print("WARNING: Issues found in training data")
    
    if check_array(growth_rates, "Growth Rates"):
        has_issues = True
        print("WARNING: Issues found in growth rates")
    
    if check_array(prediction_data, "Prediction Data"):
        has_issues = True
        print("WARNING: Issues found in prediction data")
    
    # Validate relationships between datasets
    print("\nValidating dataset relationships:")
    print(f"Number of tickers: {len(tickers)}")
    print(f"Training data episodes: {len(data)}")
    print(f"Growth rates episodes: {len(growth_rates)}")
    print(f"Prediction data shape: {prediction_data.shape}")
    
    if has_issues:
        print("\nWARNING: Dataset validation found issues that may affect model training!")
    else:
        print("\nAll datasets passed basic validation checks.")

dates = [
         '2025-01-07', '2025-01-08', '2025-01-09', '2025-01-10',
         '2025-01-13', '2025-01-14', '2025-01-15', '2025-01-16', '2025-01-17'
         ]

key = "cvV9m9XNz41uD7SMCLqftmzWKwDCI_9x"
for date in dates:
    ticker_dir = os.path.join('data', 'daily_data')
    analysis_date_end = date
    end_date = datetime.strptime(analysis_date_end, '%Y-%m-%d')
    start_date = end_date - timedelta(days=750)
    ticker_file = f'{analysis_date_end}_daily_ticker_list'
    train_dataset = f'{analysis_date_end}_daily_train_dataset'
    val_dataset = f'{analysis_date_end}_daily_val_dataset'
    predict_dataset = f'{analysis_date_end}_daily_predict_dataset'
    train_file = f'{analysis_date_end}_train_data'
    growth_file = f'{analysis_date_end}_growth_data'
    prediction_file = f'{analysis_date_end}_prediction_data_jFIX'

    ticker_path = os.path.join(ticker_dir, ticker_file)
    train_path = os.path.join(ticker_dir, train_dataset)
    val_path = os.path.join(ticker_dir, val_dataset)
    predict_path = os.path.join(ticker_dir, predict_dataset)
    growth_path = os.path.join(ticker_dir, growth_file)

    DataFetcher = PolygonDataFetcher(key, analysis_date_end)
    DataFetcher.get_all_tickers(ticker_path)
    DataFetcher.assemble_market_data(ticker_path)
    DataFetcher.assemble_dataset(ticker_path, train_path, val_path, predict_path, start_date, end_date)

    DataPrep = PolygonDataPrep(ticker_path, train_path, val_path, predict_path)
    DataPrep.training_prep(train_path, growth_path)
    DataPrep.prediction_prep(predict_path)

