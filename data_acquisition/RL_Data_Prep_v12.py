#This is updated code to get greater data from polygon.io
    #new data includes the following
        #Closing price by stock by day
        #Quarterly earnings by stock by day
        #Market cap by stock by day
        #Shares outstanding by stock by day
        #volume by stock by day
        #trades by stock by day
        #SIC Code
#This is my import stack
import polygon
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import math
import pickle 
import pandas as pd
import torch
import numpy as np

class PolygonDataPrep:
    def __init__(self, ticker_list, train_dataset, val_dataset, predict_dataset):
        with open(ticker_list, 'rb') as t:
            self.tickers = pickle.load(t)
        with open(train_dataset, 'rb') as t:
            self.tupple_data = pickle.load(t)
        with open(val_dataset, 'rb') as t:
            self.val_data = pickle.load(t)
        with open(predict_dataset, 'rb') as t:
            self.predict_dataset = pickle.load(t)

    def training_prep(self, train_file, growth_file):      
        full_length = len(self.tupple_data)
        num_sequences = len(self.tickers)
        num_episodes = int(full_length/num_sequences)
        sequence_length = len(self.tupple_data[0][0])
        data_list = []
        growth_rates_list = []

        for i in range(num_episodes):
            episode_data = []
            episode_growth_rates = []
            for row in self.tupple_data[i * num_sequences : (i+1) * num_sequences]:
                state, targets = row
                episode_data.append(state)
                episode_growth_rates.append(targets[0])
            data_list.append(np.array(episode_data))
            growth_rates_list.append(np.array(episode_growth_rates))
        ###################################################################
        ###################################################################
        full_length = len(self.val_data)
        sequence_length = len(self.val_data[0][0])
        num_episodes = int(full_length/num_sequences)

        for i in range(num_episodes):
            episode_data = []
            episode_growth_rates = []
            for row in self.val_data[i * num_sequences : (i+1) * num_sequences]:
                state, targets = row
                episode_data.append(state)
                episode_growth_rates.append(targets[0])
            data_list.append(np.array(episode_data))
            growth_rates_list.append(np.array(episode_growth_rates))

        with open(train_file, 'wb') as f:
            pickle.dump(data_list, f)
        with open(growth_file, 'wb') as f:
            pickle.dump(growth_rates_list, f) 

    def prediction_prep(self, prediction_file):
        full_length = len(self.predict_dataset)
        num_sequences = len(self.tickers)
        sequence_length_predict = len(self.predict_dataset[0])
        num_episodes_predict = int(full_length/num_sequences)
        data_list_predict = []

        for i in range(num_episodes_predict):
            episode_data = []
            episode_growth_rates = []
            for row in self.predict_dataset[i * num_sequences : (i+1) * num_sequences]:
                state = row
                episode_data.append(state)
            data_list_predict.append(np.array(episode_data))

        with open(prediction_file, 'wb') as f:
            pickle.dump(data_list_predict, f)   