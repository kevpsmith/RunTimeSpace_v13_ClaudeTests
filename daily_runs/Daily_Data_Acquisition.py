import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from torch.utils.data import DataLoader 
from pytorch_lightning import Trainer
from data_acquisition.RL_Data_Acquisition_v12 import PolygonDataFetcher #Use the file that fixes j index error on val and predict
from data_acquisition.RL_Data_Prep_v12 import PolygonDataPrep
from pytorch_lightning.callbacks import ModelCheckpoint
from datetime import datetime, timedelta
import pandas as pd
import time

class DailyDataAcquisition:
    def __init__(self, base_dir, date):
        self.base_dir = base_dir #should be "Kevin Patrick Smith / Vision-model / RuntimeSpace_v12_Production"
        self.analysis_date_end = date
        self.key = "cvV9m9XNz41uD7SMCLqftmzWKwDCI_9x"

    def daily_acquire(self):
        key = self.key
        ticker_dir = os.path.join(self.base_dir, 'data', 'daily_data')
        analysis_date_end = self.analysis_date_end
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

if __name__ == "__main__":
    today = datetime.today().strftime("%Y-%m-%d")
    base_dir = "/teamspace/studios/this_studio"
    acquisition = DailyDataAcquisition(base_dir, today)
    acquisition.daily_acquire()

