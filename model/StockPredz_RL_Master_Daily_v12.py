import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
import torch
from torch.utils.data import DataLoader 
from pytorch_lightning import Trainer
from GPU_StockPredz_RL_Environment_v12 import SequenceSelectionEnv 
from GPU_StockPredz_RL_Model_v12 import TransformerPolicyNetwork
from GPU_StockPredz_RL_Dataset_v12 import SequenceSelectionDataset
from RL_Data_Acquisition_v12 import PolygonDataFetcher #Use the file that fixes j index error on val and predict
from RL_Data_Prep_v12 import PolygonDataPrep
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
         '2025-01-06', '2025-01-07', '2025-01-08', '2025-01-09', '2025-01-10',
         '2025-01-13', '2025-01-14', '2025-01-15', '2025-01-16', '2025-01-17'
         ]

key = "cvV9m9XNz41uD7SMCLqftmzWKwDCI_9x"
for date in dates:
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

    DataFetcher = PolygonDataFetcher(key, analysis_date_end)
    DataFetcher.get_all_tickers(ticker_file)
    DataFetcher.assemble_market_data(ticker_file)
    DataFetcher.assemble_dataset(ticker_file, train_dataset, val_dataset, predict_dataset, start_date)

    DataPrep = PolygonDataPrep(ticker_file, train_dataset, val_dataset, predict_dataset)
    DataPrep.training_prep(train_file, growth_file)
    DataPrep.prediction_prep(prediction_file)

    # with open(ticker_file, 'rb') as l:
    #     tickers = pickle.load(l)
    # with open(train_file,'rb') as f: 
    #     data = np.array(pickle.load(f))
    # with open(growth_file, 'rb') as f:
    #     growth_rates = np.array(pickle.load(f))
    # with open(prediction_file, 'rb') as f:
    #     prediction_data = np.array(pickle.load(f))

    # #validate_datasets(data, growth_rates, prediction_data, tickers)

    # episodes = int(len(data))-1

    # train_env = SequenceSelectionEnv(data[:-1], growth_rates[:-1])
    # val_env = SequenceSelectionEnv(data[-1:], growth_rates[-1:])

    # num_sequences = len(tickers)
    # sequence_length = data.shape[1]
    # d_model = 256
    # nhead = 8
    # learning_rate = 1e-4
    # layers = 8
    # size = 100 #how many top stocks are you trying to select

    # model = TransformerPolicyNetwork(train_env, val_env, d_model, nhead, lr = learning_rate, num_layers = layers, size=size)

    # train_dataset = SequenceSelectionDataset(train_env, model, num_episodes=episodes, randomize_series=True)
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)  # keep shuffle=False since we're randomizing series internally

    # val_dataset = SequenceSelectionDataset(val_env, model, num_episodes=1, randomize_series=False)
    # val_loader = DataLoader(val_dataset, batch_size=1)

    # os.makedirs("my_model_checkpoints", exist_ok=True)
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath="my_model_checkpoints",  # Directory to save checkpoints
    #     filename="best-checkpoint",      # Filename pattern
    #     save_top_k=1,                    # Only save the best model
    #     monitor="val_loss",              # Metric to monitor (e.g., validation loss)
    #     mode="min"                       # Minimize the monitored metric (e.g., for loss)
    # )

    # trainer = Trainer(
    #     max_epochs=10,
    #     accelerator = "gpu",
    #     devices = 1,
    #     log_every_n_steps=1,
    #     callbacks=[checkpoint_callback]
    # )

    # trainer.fit(model, train_loader, val_loader)

    # model.eval()
    # predict_tensor = torch.tensor(prediction_data, dtype=torch.float32).to(model.device)
    # stock_tensor = predict_tensor[:, :, :12]  
    # regime_tensor = predict_tensor[:, :, 12:]  

    # predictions = []
    
    # print("Stock Tensor Shape:", predict_tensor.shape)  # Should be (batch, num_stocks, 12)
    # print("Regime Tensor Shape:", regime_tensor.shape)  # Should be (batch, num_stocks, 3)
    # with torch.no_grad():
    #     select_probs, decline_probs, double_digit_probs = model(stock_tensor, regime_tensor)

    # select_probs = select_probs.squeeze()
    # decline_probs = decline_probs.squeeze()
    # double_digit_probs = double_digit_probs.squeeze()

    # select_action = (select_probs > 0.5).int().numpy()
    # decline_action = (decline_probs > 0.5).int().numpy()

    # for i, ticker in enumerate(tickers):
    #     predictions.append({
    #         'Ticker': ticker,
    #         'Select Probability': select_probs[i].item(),
    #         'Decline Probability': decline_probs[i].item(),
    #         'Double-Digit Probability': double_digit_probs[i].item(),
    #         'Select Action': select_action[i],
    #         'Decline Action': decline_action[i]            
    #     })

    # predictions_df = pd.DataFrame(predictions)

    # os.makedirs("predictions_output_random", exist_ok=True)
    # predictions_df = pd.DataFrame(predictions)
    # predictions_df.to_excel(f'predictions_output_random/{analysis_date_end}_v12_NewData_RegimeAtt_DoubleDigit_random_random.xlsx', index=False)