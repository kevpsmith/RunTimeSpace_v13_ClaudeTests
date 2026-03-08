"""
Backtest Runner: Full pipeline for Jan-Jul 2025
Runs data acquisition + training + prediction for each trading day.

Usage:
    python backtest/Backtest_Runner.py                    # Run full Jan-Jul 2025
    python backtest/Backtest_Runner.py --start 2025-03-01 # Start from a specific date
    python backtest/Backtest_Runner.py --skip-acquisition  # Skip data acquisition (use existing data)
"""
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from model.GPU_StockPredz_RL_Environment_v12 import SequenceSelectionEnv
from model.GPU_StockPredz_RL_Model_v12 import TransformerPolicyNetwork
from model.GPU_StockPredz_RL_Dataset_v12 import SequenceSelectionDataset
from data_acquisition.RL_Data_Acquisition_v12 import PolygonDataFetcher
from data_acquisition.RL_Data_Prep_v12 import PolygonDataPrep
import pickle
from datetime import datetime, timedelta
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import time
import json
import traceback


def generate_trading_days(start_date, end_date):
    """Generate US trading days (weekdays minus federal holidays)."""
    us_holidays = USFederalHolidayCalendar()
    us_business_day = CustomBusinessDay(calendar=us_holidays)
    trading_days = pd.date_range(start=start_date, end=end_date, freq=us_business_day)
    return [d.strftime('%Y-%m-%d') for d in trading_days]


def run_data_acquisition(date, key, ticker_dir):
    """Run data acquisition and prep for a single date."""
    analysis_date_end = date
    end_date = datetime.strptime(analysis_date_end, '%Y-%m-%d')
    start_date = end_date - timedelta(days=750)

    ticker_file = f'{analysis_date_end}_daily_ticker_list'
    train_dataset = f'{analysis_date_end}_daily_train_dataset'
    val_dataset = f'{analysis_date_end}_daily_val_dataset'
    predict_dataset = f'{analysis_date_end}_daily_predict_dataset'
    growth_file = f'{analysis_date_end}_growth_data'

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


def run_training_and_prediction(date, ticker_dir, output_dir):
    """Run model training and generate predictions for a single date."""
    analysis_date_end = date

    ticker_file = f'{analysis_date_end}_daily_ticker_list'
    train_dataset = f'{analysis_date_end}_daily_train_dataset'
    train_file = f'{analysis_date_end}_train_data'
    growth_file = f'{analysis_date_end}_growth_data'
    predict_dataset = f'{analysis_date_end}_daily_predict_dataset'

    ticker_path = os.path.join(ticker_dir, ticker_file)
    train_path = os.path.join(ticker_dir, train_dataset)
    predict_path = os.path.join(ticker_dir, predict_dataset)
    growth_path = os.path.join(ticker_dir, growth_file)

    with open(ticker_path, 'rb') as l:
        tickers = pickle.load(l)
    with open(train_path, 'rb') as f:
        data = np.array(pickle.load(f))
    with open(growth_path, 'rb') as f:
        growth_rates = np.array(pickle.load(f))
    with open(predict_path, 'rb') as f:
        prediction_data = np.array(pickle.load(f))

    episodes = int(len(data)) - 1

    train_env = SequenceSelectionEnv(data[:-1], growth_rates[:-1])
    val_env = SequenceSelectionEnv(data[-1:], growth_rates[-1:])

    d_model = 256
    nhead = 8
    learning_rate = 1e-4
    layers = 8
    size = 100

    model = TransformerPolicyNetwork(train_env, val_env, d_model, nhead, lr=learning_rate, num_layers=layers, size=size)

    train_dataset = SequenceSelectionDataset(train_env, model, num_episodes=episodes, randomize_series=True)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    val_dataset = SequenceSelectionDataset(val_env, model, num_episodes=1, randomize_series=False)
    val_loader = DataLoader(val_dataset, batch_size=1)

    checkpoint_dir = os.path.join("backtest_checkpoints", analysis_date_end)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-checkpoint",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )

    trainer = Trainer(
        max_epochs=10,
        accelerator="gpu",
        devices=1,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, train_loader, val_loader)

    # Generate predictions
    model.eval()
    predict_tensor = torch.tensor(prediction_data, dtype=torch.float32).to(model.device)
    stock_tensor = predict_tensor[:, :, :12]
    regime_tensor = predict_tensor[:, :, 12:]

    with torch.no_grad():
        select_probs, decline_probs, double_digit_probs = model(stock_tensor, regime_tensor)

    select_probs = select_probs.squeeze()
    decline_probs = decline_probs.squeeze()
    double_digit_probs = double_digit_probs.squeeze()

    select_action = (select_probs > 0.5).int().numpy()
    decline_action = (decline_probs > 0.5).int().numpy()

    predictions = []
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
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{analysis_date_end}_v12_NewData_RegimeAtt_DoubleDigit_random_random.xlsx')
    predictions_df.to_excel(output_file, index=False)
    print(f"[OK] Predictions saved: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description='Run backtest for Jan-Jul 2025')
    parser.add_argument('--start', type=str, default='2025-01-02', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2025-07-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--skip-acquisition', action='store_true', help='Skip data acquisition, use existing data')
    parser.add_argument('--skip-training', action='store_true', help='Skip training, use existing predictions')
    parser.add_argument('--key', type=str, default='cvV9m9XNz41uD7SMCLqftmzWKwDCI_9x', help='Polygon API key')
    parser.add_argument('--output-dir', type=str, default='backtest_predictions', help='Output directory for predictions')
    args = parser.parse_args()

    trading_days = generate_trading_days(args.start, args.end)
    print(f"Total trading days to backtest: {len(trading_days)}")
    print(f"Date range: {trading_days[0]} to {trading_days[-1]}")

    ticker_dir = os.path.join('data', 'daily_data')
    os.makedirs(ticker_dir, exist_ok=True)

    # Track progress
    progress_file = os.path.join(args.output_dir, 'backtest_progress.json')
    completed_dates = set()
    failed_dates = {}
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
            completed_dates = set(progress.get('completed', []))
            failed_dates = progress.get('failed', {})
        print(f"Resuming: {len(completed_dates)} dates already completed, {len(failed_dates)} previously failed")

    for idx, date in enumerate(trading_days):
        if date in completed_dates:
            print(f"[{idx+1}/{len(trading_days)}] Skipping {date} (already completed)")
            continue

        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(trading_days)}] Processing {date}")
        print(f"{'='*60}")

        try:
            # Step 1: Data Acquisition
            if not args.skip_acquisition:
                print(f"[{date}] Starting data acquisition...")
                start_time = time.time()
                run_data_acquisition(date, args.key, ticker_dir)
                acq_time = time.time() - start_time
                print(f"[{date}] Data acquisition complete ({acq_time:.1f}s)")

            # Step 2: Training + Prediction
            if not args.skip_training:
                print(f"[{date}] Starting training and prediction...")
                start_time = time.time()
                run_training_and_prediction(date, ticker_dir, args.output_dir)
                train_time = time.time() - start_time
                print(f"[{date}] Training and prediction complete ({train_time:.1f}s)")

            completed_dates.add(date)

        except Exception as e:
            print(f"[ERROR] Failed on {date}: {str(e)}")
            traceback.print_exc()
            failed_dates[date] = str(e)

        # Save progress after each date
        os.makedirs(args.output_dir, exist_ok=True)
        with open(progress_file, 'w') as f:
            json.dump({
                'completed': sorted(list(completed_dates)),
                'failed': failed_dates,
                'total_dates': len(trading_days),
                'last_updated': datetime.now().isoformat()
            }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Backtest complete: {len(completed_dates)}/{len(trading_days)} days succeeded")
    if failed_dates:
        print(f"Failed dates ({len(failed_dates)}):")
        for d, err in sorted(failed_dates.items()):
            print(f"  {d}: {err}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
