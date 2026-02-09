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
import polygon
from pandas.tseries.offsets import BDay

dates = [
         '2025-01-06', '2025-01-07', '2025-01-08', '2025-01-09', '2025-01-10',
         '2025-01-13', '2025-01-14', 
         '2025-01-15', '2025-01-16', '2025-01-17',
         '2025-02-19', '2025-02-25', '2025-02-26', '2025-02-27', '2025-02-28',
         '2025-03-03', '2025-03-04', '2025-03-06',
         '2025-03-10', '2025-03-11', '2025-03-12', '2025-03-13', '2025-03-14',
         
         ]

pred_dir = os.path.join('predictions_output_random')
results = []
key = "cvV9m9XNz41uD7SMCLqftmzWKwDCI_9x"
stocks_client = polygon.StocksClient(key, connect_timeout = 240, read_timeout = 240)
for date in dates:
    date_pandas = pd.to_datetime(date)
    start_date = (date_pandas + BDay(1)).to_pydatetime()
    end_date = (date_pandas + BDay(6)).to_pydatetime()
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    file_name = f'{date}_v12_NewData_RegimeAtt_DoubleDigit_random_random.xlsx'
    filepath = os.path.join(pred_dir, file_name)
    df = pd.read_excel(filepath)
    df = df.sort_values(by='Select Probability', ascending=False)
    mask = ((df['Select Probability'] > 0.33) & (df['Decline Probability'] < 0.40))
    df_filtered = df[mask]
    # Iterate through each row of df_filtered
    for idx, row in df_filtered.iterrows():
        ticker = row['Ticker']

        # Fetch the "weekly" bar for that range
        ohlc_data = stocks_client.get_aggregate_bars(
                    ticker, 
                    start_date_str, 
                    end_date_str, 
                    timespan='day', 
                    full_range=True, 
                    run_parallel=False
                )

        if ohlc_data is None:
            print(f"No data returned for {ticker} from {start_date_str} to {end_date_str}")
            # Optionally set them to NaN or some placeholder
            df_filtered.at[idx, 'Open'] = None
            df_filtered.at[idx, 'High'] = None
            df_filtered.at[idx, 'Low'] = None
            df_filtered.at[idx, 'Close'] = None
            continue
        
        if ohlc_data:
            agg_open = ohlc_data[0]['o']           # open of the first day
            high_values = [item['h'] for item in ohlc_data if 'h' in item]
            agg_high = max(high_values) 
            low_values = [item['l'] for item in ohlc_data if 'l' in item]
            agg_low = min(low_values)
            agg_close = ohlc_data[-1]['c']
            # We got a dict with {open, high, low, close}
            df_filtered.at[idx, 'Open'] = agg_open
            df_filtered.at[idx, 'High'] = agg_high
            df_filtered.at[idx, 'Low'] = agg_low
            df_filtered.at[idx, 'Close'] = agg_close

            oh_return = (agg_high - agg_open) / agg_open
            oc_return = (agg_close - agg_open) / agg_open

            if oh_return >=.045 and oh_return > oc_return:
                df_filtered.at[idx, 'return'] = oh_return
            else:
                df_filtered.at[idx, 'return'] = oc_return

    if df_filtered.empty:
        print(f"Skipping {date}, no valid rows after filtering.")
        continue  # Skip this iteration and move to the next date
        
    average_return = df_filtered['return'].mean()
    results.append({'Date': date, 'Average_return': average_return})
    df.update(df_filtered[["Open", "High", "Low", "Close", "return"]])
    df.to_excel(filepath, index=False)

df_results = pd.DataFrame(results)
average_average = df_results['Average_return'].mean()
save_final = f'{dates[0]}_to_{dates[-1]}_average return_{average_average:.3f}.xlsx' 
save_final_file = os.path.join(pred_dir, save_final)
df_results.to_excel(save_final_file, index=False)
