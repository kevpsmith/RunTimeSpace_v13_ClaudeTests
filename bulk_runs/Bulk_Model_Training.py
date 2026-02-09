import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import sys
# Add the parent directory of bulk_runs (this_studio) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from torch.utils.data import DataLoader 
from pytorch_lightning import Trainer
from model.GPU_StockPredz_RL_Environment_v12 import SequenceSelectionEnv 
from model.GPU_StockPredz_RL_Model_v12 import TransformerPolicyNetwork
from model.GPU_StockPredz_RL_Dataset_v12 import SequenceSelectionDataset
import pickle
from pytorch_lightning.callbacks import ModelCheckpoint
from datetime import datetime, timedelta
import pandas as pd
import time

dates = [
         '2025-01-06', '2025-01-07', '2025-01-08', '2025-01-09', '2025-01-10',
         '2025-01-13', '2025-01-14', 
         '2025-01-15', '2025-01-16', '2025-01-17'
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

    with open(ticker_path, 'rb') as l:
        tickers = pickle.load(l)
    with open(train_path,'rb') as f: 
        data = np.array(pickle.load(f))
    with open(growth_path, 'rb') as f:
        growth_rates = np.array(pickle.load(f))
    with open(predict_path, 'rb') as f:
        prediction_data = np.array(pickle.load(f))

    #validate_datasets(data, growth_rates, prediction_data, tickers)

    episodes = int(len(data))-1

    train_env = SequenceSelectionEnv(data[:-1], growth_rates[:-1])
    val_env = SequenceSelectionEnv(data[-1:], growth_rates[-1:])

    num_sequences = len(tickers)
    sequence_length = data.shape[1]
    d_model = 256
    nhead = 8
    learning_rate = 1e-4
    layers = 8
    size = 100 #how many top stocks are you trying to select

    model = TransformerPolicyNetwork(train_env, val_env, d_model, nhead, lr = learning_rate, num_layers = layers, size=size)

    train_dataset = SequenceSelectionDataset(train_env, model, num_episodes=episodes, randomize_series=True)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)  # keep shuffle=False since we're randomizing series internally

    val_dataset = SequenceSelectionDataset(val_env, model, num_episodes=1, randomize_series=False)
    val_loader = DataLoader(val_dataset, batch_size=1)

    os.makedirs("my_model_checkpoints", exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath="my_model_checkpoints",  # Directory to save checkpoints
        filename="best-checkpoint",      # Filename pattern
        save_top_k=1,                    # Only save the best model
        monitor="val_loss",              # Metric to monitor (e.g., validation loss)
        mode="min"                       # Minimize the monitored metric (e.g., for loss)
    )

    trainer = Trainer(
        max_epochs=10,
        accelerator = "gpu",
        devices = 1,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, train_loader, val_loader)

    model.eval()
    predict_tensor = torch.tensor(prediction_data, dtype=torch.float32).to(model.device)
    stock_tensor = predict_tensor[:, :, :12]  
    regime_tensor = predict_tensor[:, :, 12:]  

    predictions = []
    
    print("Stock Tensor Shape:", predict_tensor.shape)  # Should be (batch, num_stocks, 12)
    print("Regime Tensor Shape:", regime_tensor.shape)  # Should be (batch, num_stocks, 3)
    with torch.no_grad():
        select_probs, decline_probs, double_digit_probs = model(stock_tensor, regime_tensor)

    select_probs = select_probs.squeeze()
    decline_probs = decline_probs.squeeze()
    double_digit_probs = double_digit_probs.squeeze()

    select_action = (select_probs > 0.5).int().numpy()
    decline_action = (decline_probs > 0.5).int().numpy()

    for i, ticker in enumerate(tickers):
        predictions.append({
            'Ticker': ticker,
            'Select Probability': select_probs[i].item(),
            'Decline Probability': decline_probs[i].item(),
            'Double-Digit Probability': double_digit_probs[i].item(),
            'Select Action': select_action[i],
            'Decline Action': decline_action[i]            
        })

    predictions_df = pd.DataFrame(predictions)

    os.makedirs("predictions_output_random", exist_ok=True)
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_excel(f'predictions_output_random/{analysis_date_end}_v12_NewData_RegimeAtt_DoubleDigit_random_random.xlsx', index=False)