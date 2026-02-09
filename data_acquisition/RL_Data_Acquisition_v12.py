import polygon
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import math
import pickle 
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import Dict, Any, List, Optional
from data_acquisition.Helper_feature_calcs import EnhancedFeatures
import os

class PolygonDataFetcher:
    def __init__(self, key, analysis_date_end, max_workers=10, rate_limit_pause=0.12):
        """
        Initialize the data fetcher with improved configuration
        
        Args:
            key: Polygon.io API key
            analysis_date_end: End date for analysis
            max_workers: Maximum number of concurrent threads
            rate_limit_pause: Pause between API calls to respect rate limits
        """
        self.key = key
        self.stocks_client = polygon.StocksClient(key, connect_timeout = 240, read_timeout = 240)
        self.reference_client = polygon.ReferenceClient(key, connect_timeout = 240, read_timeout = 240)
        # Date Configuration
        self.analysis_date_end = pd.to_datetime(analysis_date_end, format='%Y-%m-%d')
        self.analysis_date_start = self.analysis_date_end - relativedelta(years=5)
        self.data_set_start = self.analysis_date_end - relativedelta(years=4)
        self.check_date = pd.to_datetime(self.analysis_date_end + relativedelta(days=7), format='%Y-%m-%d')
        self.buy_date = pd.to_datetime(self.analysis_date_end + relativedelta(days=0), format='%Y-%m-%d')
        self.tickers = []
        self.max_workers = max_workers
        self.rate_limit_pause = rate_limit_pause
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)


    def clean_dataframe(self, df: pd.DataFrame, fill_method='ffill') -> pd.DataFrame:
            """
            Clean a DataFrame with consistent handling of missing values
            
            Args:
                df: Input DataFrame
                fill_method: Method to fill missing values ('ffill', 'bfill', or 'zero')
                
            Returns:
                Cleaned DataFrame
            """
            # Replace various null values with np.nan for consistency
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.replace(['NaN', 'nan', 'None', '', 'NULL', 'N/A'], np.nan)
            
            # Replace zeros with np.nan before forward filling
            df = df.replace(0, np.nan)
            
            # Forward fill only
            df = df.ffill()

            # Fill any remaining NaNs with 1
            df = df.fillna(1)

            # Check remaining NaN values
            final_nan_count = df.isna().sum().sum()
            self.logger.info(f"Remaining NaN count after cleaning: {final_nan_count}")
            
            if final_nan_count > 0:
                # If there are still NaNs, log where they are
                nan_locations = df.isna().sum()
                nan_locations = nan_locations[nan_locations > 0]
                self.logger.warning(f"Remaining NaNs by column:\n{nan_locations}")

                # Additional cleaning for remaining NaNs
                for column in nan_locations.index:
                    # First try backward fill for any leading NaNs
                    if df[column].iloc[0] is np.nan:
                        df[column] = df[column].bfill()
                    
                    # If still have NaNs, fill with column mean/median
                    if df[column].isna().any():
                        df[column] = df[column].fillna(1)
                        self.logger.info(f"Filled remaining NaNs in {column} with median value: 1")
                
                # Final verification
                final_check = df.isna().sum().sum()
                self.logger.info(f"Final NaN count after additional cleaning: {final_check}")
                if final_check > 0:
                    self.logger.error("WARNING: Still have NaN values after all cleaning attempts!")
                else:
                    self.logger.info("All NaN values successfully cleaned")
                        
            return df

    def assemble_market_data(self, tickers):
        # Load tickers
        with open(tickers, 'rb') as t:
            TICKERS = pickle.load(t)
        TICKERS = sorted(list(set(TICKERS)))

        # Get dates from first ticker
        dates = []
        placeholder = self.stocks_client.get_aggregate_bars(
            TICKERS[1], 
            self.analysis_date_start, 
            self.analysis_date_end, 
            timespan='day',
            full_range=True, 
            run_parallel=False
        )
        for k in placeholder:
            date = datetime.fromtimestamp(k['t']/1000)
            if date.weekday() < 5:
                dates.append(date.strftime('%Y-%m-%d'))
        dates = sorted(dates)

        # Initialize DataFrames
        dataframes = {
            'closing': pd.DataFrame(index=TICKERS, columns=dates),
            'high': pd.DataFrame(index=TICKERS, columns=dates),    # Added
            'low': pd.DataFrame(index=TICKERS, columns=dates),     # Added
            'volume': pd.DataFrame(index=TICKERS, columns=dates),
            'trades': pd.DataFrame(index=TICKERS, columns=dates),
            'rsi_daily': pd.DataFrame(index=TICKERS, columns=dates),
            'rsi_weekly': pd.DataFrame(index=TICKERS, columns=dates),
            'macd_daily': pd.DataFrame(index=TICKERS, columns=dates),
            'macd_weekly': pd.DataFrame(index=TICKERS, columns=dates), 
            'QuarterlyEarnings': pd.DataFrame(index=TICKERS, columns=dates),
            'DaysSince': pd.DataFrame(index=TICKERS, columns=dates),
        }

        # Process price data in parallel
        def fetch_price_data(ticker):
            try:
                bars = self.stocks_client.get_aggregate_bars(
                    ticker,
                    self.analysis_date_start.strftime('%Y-%m-%d'),
                    self.analysis_date_end.strftime('%Y-%m-%d'),
                    timespan='day',
                    full_range=True,
                    run_parallel=False
                )
                time.sleep(self.rate_limit_pause)
                return ticker, bars
            except Exception as e:
                self.logger.error(f"Error fetching price data for {ticker}: {str(e)}")
                return ticker, None

        # Fetch price data
        print("Fetching price data...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for ticker, bars in executor.map(fetch_price_data, TICKERS):
                if bars:
                    for bar in bars:
                        date = datetime.fromtimestamp(bar['t']/1000).strftime('%Y-%m-%d')
                        if date in dates:
                            dataframes['closing'].loc[ticker, date] = bar.get('c', np.nan)
                            dataframes['high'].loc[ticker, date] = bar.get('h', np.nan)     # Added
                            dataframes['low'].loc[ticker, date] = bar.get('l', np.nan)      # Added
                            dataframes['volume'].loc[ticker, date] = bar.get('v', np.nan)
                            dataframes['trades'].loc[ticker, date] = bar.get('n', np.nan)

        # Process RSI data in parallel
        def fetch_rsi_data(ticker):
            try:
                rsi_d = self.stocks_client.get_rsi(ticker, timestamp_gte=self.analysis_date_start, timespan='day', window_size=14, adjusted=True, limit=5000)
                time.sleep(self.rate_limit_pause)
                rsi_w = self.stocks_client.get_rsi(ticker, timestamp_gte=self.analysis_date_start, timespan='week', window_size=14, adjusted=True, limit=5000)
                time.sleep(self.rate_limit_pause)
                return ticker, (rsi_d, rsi_w)
            except Exception as e:
                self.logger.error(f"Error fetching RSI data for {ticker}: {str(e)}")
                return ticker, None

        # Fetch RSI data
        print("Fetching RSI data...")
        dates_array = np.array([pd.to_datetime(d) for d in dates])
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for ticker, rsi_data in executor.map(fetch_rsi_data, TICKERS):
                if rsi_data:
                    rsi_d, rsi_w = rsi_data
                    if 'results' in rsi_d and 'values' in rsi_d['results']:
                        for entry in rsi_d['results']['values']:
                            date = datetime.fromtimestamp(entry['timestamp']/1000).strftime('%Y-%m-%d')
                            if date in dates:
                                dataframes['rsi_daily'].loc[ticker, date] = entry['value']
                    # Modified weekly RSI handling
                    if 'results' in rsi_w and 'values' in rsi_w['results']:
                        for entry in rsi_w['results']['values']:
                            sunday_date = pd.to_datetime(entry['timestamp'], unit='ms')
                            # Find index of next date after Sunday
                            idx = np.searchsorted(dates_array, sunday_date)
                            if idx < len(dates):  # Make sure we found a valid date
                                next_date = dates[idx]
                                dataframes['rsi_weekly'].loc[ticker, next_date] = entry['value']

                    # Forward fill weekly values
                    dataframes['rsi_weekly'].loc[ticker] = dataframes['rsi_weekly'].loc[ticker].ffill()

        # Process MACD data in parallel
        def fetch_macd_data(ticker):
            try:
                macd_d = self.stocks_client.get_macd(ticker, timestamp_gte=self.analysis_date_start, timespan='day', 
                                                    long_window_size=26, short_window_size=12, signal_window_size=9, 
                                                    adjusted=True, limit=5000)
                time.sleep(self.rate_limit_pause)
                macd_w = self.stocks_client.get_macd(ticker, timestamp_gte=self.analysis_date_start, timespan='week',
                                                    long_window_size=26, short_window_size=12, signal_window_size=9,
                                                    adjusted=True, limit=5000)
                time.sleep(self.rate_limit_pause)
                return ticker, (macd_d, macd_w)
            except Exception as e:
                self.logger.error(f"Error fetching MACD data for {ticker}: {str(e)}")
                return ticker, None

        # Fetch MACD data
        print("Fetching MACD data...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for ticker, macd_data in executor.map(fetch_macd_data, TICKERS):
                if macd_data:
                    macd_d, macd_w = macd_data
                    if 'results' in macd_d and 'values' in macd_d['results']:
                        for entry in macd_d['results']['values']:
                            date = datetime.fromtimestamp(entry['timestamp']/1000).strftime('%Y-%m-%d')
                            if date in dates:
                                dataframes['macd_daily'].loc[ticker, date] = entry['histogram']
                    # Modified weekly MACD handling
                    if 'results' in macd_w and 'values' in macd_w['results']:
                        for entry in macd_w['results']['values']:
                            sunday_date = pd.to_datetime(entry['timestamp'], unit='ms')
                            idx = np.searchsorted(dates_array, sunday_date)
                            if idx < len(dates):
                                next_date = dates[idx]
                                dataframes['macd_weekly'].loc[ticker, next_date] = entry['histogram']

                    # Forward fill weekly values
                    dataframes['macd_weekly'].loc[ticker] = dataframes['macd_weekly'].loc[ticker].ffill()

        # Process quarterly earnings data in parallel
        def fetch_earnings_data(ticker):
            try:
                earnings = self.reference_client.get_stock_financials_vx(ticker, filing_date_gte=self.analysis_date_start)
                time.sleep(self.rate_limit_pause)
                if 'results' in earnings:
                    # Create two series with just filing dates to start
                    series = pd.Series(index=dates, dtype=float)
                    days_series = pd.Series(index=dates, dtype=float)
                    
                    # Sort by filing date
                    sorted_earnings = sorted(earnings['results'], key=lambda x: pd.to_datetime(x['filing_date']))
                    
                    # Convert filing dates to numpy array once
                    filing_dates_array = np.array([pd.to_datetime(e['filing_date']) for e in sorted_earnings])
                    
                    # Set values at filing dates and backward fill
                    for earning in sorted_earnings:
                        filing_date = pd.to_datetime(earning['filing_date']).strftime('%Y-%m-%d')
                        if filing_date in dates:
                            value = earning.get('financials', {}).get('income_statement', {}).get('net_income_loss', {}).get('value', np.nan)
                            series[filing_date] = value
                    
                    # Backward fill earnings values
                    series = series.bfill()
                    
                    # Vectorized days since calculation
                    date_objects = pd.to_datetime(dates)
                    for i, curr_date in enumerate(date_objects):
                        # Find index of most recent filing date
                        idx = np.searchsorted(filing_dates_array, curr_date)
                        if idx > 0:  # If we found a filing date before current date
                            most_recent = filing_dates_array[idx-1]
                            days_series[dates[i]] = (curr_date - most_recent).days
                        else:
                            days_series[dates[i]] = np.nan
                    
                    return ticker, (series, days_series)
                return ticker, None
            except Exception as e:
                self.logger.error(f"Error fetching earnings data for {ticker}: {str(e)}")
                return ticker, None

        # Fetch earnings data
        print("Fetching earnings data...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for ticker, earnings_data in executor.map(fetch_earnings_data, TICKERS):
                if earnings_data:
                    series, days_series = earnings_data
                    dataframes['QuarterlyEarnings'].loc[ticker] = series
                    dataframes['DaysSince'].loc[ticker] = days_series

        # Clean all DataFrames
        for key in dataframes:
            dataframes[key] = self.clean_dataframe(dataframes[key])

        date_string=self.analysis_date_end.strftime("%Y-%m-%d")

        # Save DataFrames
        save_mappings = {
            'closing': f'{date_string}_01ClosingPrices.pkl',
            'high': f'{date_string}_01HighPrices.pkl',         # Added
            'low': f'{date_string}_01LowPrices.pkl',           # Added
            'volume': f'{date_string}_02VolumeTraded.pkl',
            'trades': f'{date_string}_03NumberTrades.pkl',
            'QuarterlyEarnings': f'{date_string}_04QuarterlyEarnings.pkl',
            'DaysSince': f'{date_string}_08DaysSince.pkl',
            'rsi_daily': f'{date_string}_09RSIDays.pkl',
            'rsi_weekly': f'{date_string}_10RSIWeeks.pkl',
            'macd_daily': f'{date_string}_12MACDDays.pkl',
            'macd_weekly': f'{date_string}_13MACDWeeks.pkl'
        }

        placeholder_dir = os.path.join('data', 'placeholders')
        os.makedirs(placeholder_dir, exist_ok=True)

        for key, filename in save_mappings.items():
            out_path=os.path.join(placeholder_dir, filename)
            with open(out_path, 'wb') as f:
                pickle.dump(dataframes[key], f)

    def get_all_tickers(self, filename):
        save_path = filename
        responses = []
        range_limit = 1000
        response = self.reference_client.get_tickers(symbol_type='CS', market='stocks', exchange='XNAS', active=True, limit=range_limit, date=self.analysis_date_end) #you should consider adding parameters to limit the total input
        for i in range(range_limit):
            responses.append(response['results'][i]['ticker'])
        while "next_url" in response.keys():
            response = self.reference_client.get_next_page(response)
            for i in range(len(response['results'])):
                responses.append(response['results'][i]['ticker'])

        response = self.reference_client.get_tickers(symbol_type='CS', market='stocks', exchange='XNYS', active=True, limit=range_limit, date=self.analysis_date_end) #you should consider adding parameters to limit the total input
        for i in range(range_limit):
            responses.append(response['results'][i]['ticker'])
        while "next_url" in response.keys():
            response = self.reference_client.get_next_page(response)
            for i in range(len(response['results'])):
                responses.append(response['results'][i]['ticker'])

        date_string=self.analysis_date_end.strftime("%Y-%m-%d")
        filename = f'{date_string}_AllTickersEnd'
        placeholder_dir = os.path.join('data', 'placeholders')
        os.makedirs(placeholder_dir, exist_ok=True)
        out_path=os.path.join(placeholder_dir, filename)

        with open(out_path, 'wb') as f:
            pickle.dump(responses, f)
        print(f"all pages recevied. total pages: {len(responses)}")

        responses2 = []
        response = self.reference_client.get_tickers(symbol_type='CS', market='stocks', exchange='XNAS', active=True, limit=range_limit, date=self.analysis_date_start) #you should consider adding parameters to limit the total input
        for i in range(range_limit):
            responses2.append(response['results'][i]['ticker'])
        while "next_url" in response.keys():
            response = self.reference_client.get_next_page(response)
            for i in range(len(response['results'])):
                responses2.append(response['results'][i]['ticker'])

        response = self.reference_client.get_tickers(symbol_type='CS', market='stocks', exchange='XNYS', active=True, limit=range_limit, date=self.analysis_date_start) #you should consider adding parameters to limit the total input
        for i in range(range_limit):
            responses2.append(response['results'][i]['ticker'])
        while "next_url" in response.keys():
            response = self.reference_client.get_next_page(response)
            for i in range(len(response['results'])):
                responses2.append(response['results'][i]['ticker'])

        date_string=self.analysis_date_end.strftime("%Y-%m-%d")
        filename = f'{date_string}_AllTickersStart'
        placeholder_dir = os.path.join('data', 'placeholders')
        os.makedirs(placeholder_dir, exist_ok=True)
        out_path=os.path.join(placeholder_dir, filename)

        with open(out_path, 'wb') as f:
            pickle.dump(responses2, f)
        print(f"all pages recevied. total pages: {len(responses2)}")
    
        TICKERS = [i for i in responses if i in responses2]
        TICKERS = sorted(list(set(TICKERS)))
        with open(save_path, 'wb') as l:
            pickle.dump(TICKERS, l)

    def forward_fill(self, array):
        if len(array) == 0:
            return array
        valid_mask = ~np.isnan(array)
        indices = np.where(valid_mask, np.arange(len(array)), 0)
        np.maximum.accumulate(indices, out=indices)
        return array[indices]
    
    def precompute_constants(self, j):
        return {
            'yesterday': j - 1,
            'two_days': j - 2,
            'three_days': j - 3,
            'four_days': j-4,
            'last_week': j - 5,
            'last_month': j - 21,
            'last_quarter': j - 63,
            'last_year': j - 252,
            'next_week': slice(j + 1, j + 6)}

    def compute_features(self, closing_prices, high_prices, low_prices, indices, today_col,
                    QuarterlyEarnings, VolumeTraded, NumberTrades, DaysSince,
                    RelativeStrengthIndexD, RelativeStrengthIndexW, MovingAverageCDD,
                    MovingAverageCDW, regime_indicators=None):
        """Enhanced compute_features with optional regime indicators."""
        # Calculate existing features
        if not hasattr(self, 'feature_calculator'):
            self.feature_calculator = EnhancedFeatures()
            
        data = {
            'close': pd.to_numeric(closing_prices.iloc[:, today_col], errors='coerce'),
            'high': pd.to_numeric(high_prices.iloc[:, today_col], errors='coerce'),
            'low': pd.to_numeric(low_prices.iloc[:, today_col], errors='coerce'),
            'volume': pd.to_numeric(VolumeTraded.iloc[:, today_col], errors='coerce'),
            'rsi': pd.to_numeric(RelativeStrengthIndexD.iloc[:, today_col], errors='coerce'),
            'macd_histogram': pd.to_numeric(MovingAverageCDD.iloc[:, today_col], errors='coerce')
        }
        
        features_df = pd.DataFrame(data).fillna(0)

        # Calculate base features (existing calculations)
        price_momentum = self.feature_calculator.calculate_price_momentum(features_df)
        atr = self.feature_calculator.calculate_atr(features_df['high'], features_df['low'], features_df['close'])
        vwap = self.feature_calculator.calculate_vwap(features_df)
        ma_metrics = self.feature_calculator.calculate_moving_averages(features_df['close'])
        vol_metrics = self.feature_calculator.calculate_volume_metrics(features_df['volume'], features_df['close'])
        tech_indicators = self.feature_calculator.calculate_technical_indicators(features_df)
        
        # Calculate next week stats
        next_week_data = closing_prices.iloc[:, indices['next_week']].astype(float)
        next_week_avg = next_week_data.iloc[:, -2:].mean(axis=1)
        next_week_max = next_week_data.max(axis=1)
        
        current_prices = pd.to_numeric(features_df['close'], errors='coerce')
        current_volume = pd.to_numeric(features_df['volume'], errors='coerce')
        
        forward_week_pct_change = np.where(current_prices != 0,
            (next_week_avg - current_prices) / current_prices, 0)
        
        # Apply volume penalty
        forward_week_pct_change = np.where(
            current_volume < 850_000,
            -abs(forward_week_pct_change),
            forward_week_pct_change
        )

        # Combine all features
        features = {
            'momentum_1d': price_momentum['momentum_1d'].astype(np.float32),
            'momentum_5d': price_momentum['momentum_5d'].astype(np.float32),
            'momentum_20d': price_momentum['momentum_20d'].astype(np.float32),
            'atr': atr.astype(np.float32),
            'vwap': vwap.astype(np.float32),
            'relative_volume': vol_metrics['relative_volume_20d'].astype(np.float32),
            'mfi': vol_metrics['mfi'].astype(np.float32),
            'volume_trend': vol_metrics['volume_trend_5d'].astype(np.float32),
            'rsi_divergence': tech_indicators['rsi_divergence'].astype(np.float32),
            'macd_slope': tech_indicators['macd_slope'].astype(np.float32),
            'macd_crossover': tech_indicators['macd_crossover'].astype(np.float32),
            'bb_position': tech_indicators['bb_position'].astype(np.float32),
            'forward_week_pct_change': forward_week_pct_change.astype(np.float32),
            'next_week_max': next_week_max.astype(np.float32)
        }

        # Add regime features if provided
        if regime_indicators is not None:
            features.update({
                'vix_proxy': regime_indicators['vix_proxy'].iloc[:, today_col].astype(np.float32),
                'market_breadth': regime_indicators['market_breadth'].iloc[:, today_col].astype(np.float32),
                'volatility_regime': regime_indicators['volatility_regime'].iloc[:, today_col].astype(np.float32)
            })

        # Clean features
        for key in features:
            features[key] = np.nan_to_num(features[key], nan=0.0)

        return features

    def calculate_market_regime_indicators(self, closing_prices_df, volume_traded_df):
        """Calculate market regime indicators using the loaded dataframes."""
        import numpy as np
        
        # Initialize regime indicators with same shape as input dataframes
        vix_proxy = pd.DataFrame(index=closing_prices_df.index, columns=closing_prices_df.columns)
        market_breadth = pd.DataFrame(index=closing_prices_df.index, columns=closing_prices_df.columns)
        volatility_regime = pd.DataFrame(index=closing_prices_df.index, columns=closing_prices_df.columns)
        
        # Calculate rolling windows for the entire market
        for date_idx, date in enumerate(closing_prices_df.columns):
            # VIX proxy using rolling standard deviation of returns
            if date_idx >= 5:  # Need at least 5 days for 5-day returns
                returns = closing_prices_df.iloc[:, max(0, date_idx-21):date_idx].pct_change(periods=5).iloc[:, -1]
                rolling_std = returns.rolling(window=21).std() * np.sqrt(252)  # Annualized
                vix_proxy[date] = rolling_std
            
            # Market breadth (% of stocks above 50-day MA)
            if date_idx >= 50:  # Need at least 50 days for 50-day MA
                ma_50 = closing_prices_df.iloc[:, max(0, date_idx-50):date_idx].mean(axis=1)
                market_breadth[date] = (closing_prices_df[date] > ma_50).astype(float)
            
            # Volatility regime based on recent volume
            if date_idx >= 21:  # Need at least 21 days for volatility calculation
                vol_data = volume_traded_df.iloc[:, max(0, date_idx-21):date_idx]
                vol_std = vol_data.std(axis=1)
                vol_threshold_high = vol_std.quantile(0.75)
                vol_threshold_low = vol_std.quantile(0.25)
                
                volatility_regime[date] = np.where(
                    vol_std > vol_threshold_high, 2,  # High volatility
                    np.where(vol_std < vol_threshold_low, 0, 1)  # Low or Medium volatility
                )
        
        # Clean and return the regime indicators
        return {
            'vix_proxy': self.clean_dataframe(vix_proxy),
            'market_breadth': self.clean_dataframe(market_breadth),
            'volatility_regime': self.clean_dataframe(volatility_regime)
        }


    def assemble_dataset(self, ticker_list, train_dataset, val_dataset, predict_dataset, data_set_start, data_set_end):
        # Load DataFrames
        date_string=self.analysis_date_end.strftime("%Y-%m-%d")
        save_mappings = {
            'closing': f'{date_string}_01ClosingPrices.pkl',
            'high': f'{date_string}_01HighPrices.pkl',         # Added
            'low': f'{date_string}_01LowPrices.pkl',           # Added
            'volume': f'{date_string}_02VolumeTraded.pkl',
            'trades': f'{date_string}_03NumberTrades.pkl',
            'QuarterlyEarnings': f'{date_string}_04QuarterlyEarnings.pkl',
            'DaysSince': f'{date_string}_08DaysSince.pkl',
            'rsi_daily': f'{date_string}_09RSIDays.pkl',
            'rsi_weekly': f'{date_string}_10RSIWeeks.pkl',
            'macd_daily': f'{date_string}_12MACDDays.pkl',
            'macd_weekly': f'{date_string}_13MACDWeeks.pkl'}

        placeholder_dir = os.path.join('data', 'placeholders')

        # Load the datasets using the correct mappings
        datasets = {}
        for key, file_name in save_mappings.items():
            file_path = os.path.join(placeholder_dir, file_name)
            if os.path.exists(file_path):
                datasets[key] = pd.read_pickle(file_path)
            else:
                print(f"Warning: File {file_path} not found!")

        # Load tickers
        with open(ticker_list, 'rb') as t:
            tickers = sorted(list(set(pickle.load(t))))

        # Print shapes and check indexes
        print("Shape check:")
        for key, df in datasets.items():
            print(f"{key} shape: {df.shape}")
            print(f"{key} missing tickers: {set(tickers) - set(df.index)}")
            print(f"{key} extra tickers: {set(df.index) - set(tickers)}")

        closing_prices = datasets['closing']
        high_prices = datasets['high']
        low_prices = datasets['low']
        volume_traded = datasets['volume']
        number_trades = datasets['trades']
        quarterly_earnings = datasets['QuarterlyEarnings']
        days_since = datasets['DaysSince']
        rsi_days = datasets['rsi_daily']
        rsi_weeks = datasets['rsi_weekly']
        macd_days = datasets['macd_daily']
        macd_weeks = datasets['macd_weekly']

        regime_indicators = self.calculate_market_regime_indicators(
            closing_prices,  # already loaded from '01ClosingPrices.pkl'
            volume_traded   # already loaded from '02VolumeTraded.pkl'
        )

        # Prepare data containers
        train_features, train_targets = [], []
        predict_features, predict_targets = [], []
        val_features, val_targets = [], []

        # Prepare dataset
        dataset = []

        if isinstance(data_set_start, str):
            data_set_start = pd.to_datetime(data_set_start)

        # Iterate over time steps
        for j in range(0, closing_prices.shape[1] - 6):
            current_date = pd.to_datetime(closing_prices.columns[j])
            if current_date >= data_set_start:
                indices = self.precompute_constants(j)

                features = self.compute_features(
                    closing_prices, high_prices, low_prices,
                    indices, j,
                    quarterly_earnings,
                    volume_traded,
                    number_trades,
                    days_since,
                    rsi_days,
                    rsi_weeks,
                    macd_days,
                    macd_weeks,
                    regime_indicators=regime_indicators  # Add this parameter
                )

                for i in range(len(features['momentum_1d'])):  # Assuming all features have the same length
                    features_array = [
                        features['momentum_1d'][i][0],
                        features['momentum_5d'][i][0],
                        features['momentum_20d'][i][0],
                        features['atr'][i],
                        features['vwap'][i],
                        features['relative_volume'][i],
                        features['mfi'][i],
                        features['volume_trend'][i],
                        features['rsi_divergence'][i],
                        features['macd_slope'][i],
                        features['macd_crossover'][i],
                        features['bb_position'][i],
                        features['vix_proxy'][i], features['market_breadth'][i], features['volatility_regime'][i]
                    ]
                    combined_targets = [
                        features['forward_week_pct_change'][i],
                        features['next_week_max'][i]
                    ]
                    # Append to dataset
                    dataset.append((features_array, combined_targets))

        # Check the overall length of the dataset
        print("Overall dataset length:", len(dataset))  # Total number of tuples

        # Check the first entry
        features, targets = dataset[0]
        print("First dataset entry features shape:", len(features))  # Should be 12
        print("First dataset entry targets shape:", len(targets))    # Should be 2        

        # Save dataset as a single file
        with open(train_dataset, 'wb') as f:
            pickle.dump(dataset, f)

        print(f"Dataset assembly complete with {len(dataset)} rows.")

        # Assemble prediction dataset
        predict_indices = self.precompute_constants(closing_prices.shape[1] - 1)
        predict_features = self.compute_features(
            closing_prices, high_prices, low_prices,
            predict_indices, closing_prices.shape[1] - 1,
            quarterly_earnings,
            volume_traded,
            number_trades,
            days_since,
            rsi_days,
            rsi_weeks,
            macd_days,
            macd_weeks,
            regime_indicators=regime_indicators  # Add this
        )

        for key in predict_features:
            print(f"{key} type:", type(predict_features[key]), "shape:", predict_features[key].shape if hasattr(predict_features[key], 'shape') else 'no shape')

        predict_tuples = [
            np.array([
                predict_features['momentum_1d'][i][0],
                predict_features['momentum_5d'][i][0],
                predict_features['momentum_20d'][i][0],
                predict_features['atr'][i],
                predict_features['vwap'][i],
                predict_features['relative_volume'][i],
                predict_features['mfi'][i],
                predict_features['volume_trend'][i],
                predict_features['rsi_divergence'][i],
                predict_features['macd_slope'][i],
                predict_features['macd_crossover'][i],
                predict_features['bb_position'][i],
                predict_features['vix_proxy'][i],
                predict_features['market_breadth'][i],
                predict_features['volatility_regime'][i]
            ], dtype=np.float32)
            for i in range(len(tickers))
        ]

        print(predict_tuples[0])  # Should output: [12 items] (numpy array)
        print(predict_tuples[0].shape)  # Should output: (12,)

        with open(predict_dataset, 'wb') as f:
            pickle.dump(predict_tuples, f)
        print("Prediction dataset assembly complete.")

        # Assemble validation dataset
        val_indices = self.precompute_constants(closing_prices.shape[1] - 6)
        val_features = self.compute_features(
            closing_prices, high_prices, low_prices,
            val_indices, closing_prices.shape[1] - 6,
            quarterly_earnings,
            volume_traded,
            number_trades,
            days_since,
            rsi_days,
            rsi_weeks,
            macd_days,
            macd_weeks,
            regime_indicators=regime_indicators  # Add this
        )

        val_tuples = [
            (np.array([
                val_features['momentum_1d'][i][0],
                val_features['momentum_5d'][i][0],
                val_features['momentum_20d'][i][0],
                val_features['atr'][i],
                val_features['vwap'][i],
                val_features['relative_volume'][i],
                val_features['mfi'][i],
                val_features['volume_trend'][i],
                val_features['rsi_divergence'][i],
                val_features['macd_slope'][i],
                val_features['macd_crossover'][i],
                val_features['bb_position'][i],
                val_features['vix_proxy'][i],
                val_features['market_breadth'][i],
                val_features['volatility_regime'][i]
            ], dtype=np.float32),
            np.array([
                val_features['forward_week_pct_change'][i],
                val_features['next_week_max'][i]
            ], dtype=np.float32))
            for i in range(len(tickers))
        ]

        with open(val_dataset, 'wb') as f:
            pickle.dump(val_tuples, f)
        print("Validation dataset assembly complete.")

    # Helper function to save batches
    def _save_batch(self, dataset_path, features, targets, batch_index):
        """Helper function to save a batch to disk."""
        batch_data = {'features': np.array(features), 'targets': np.array(targets)}
        with open(f"{dataset_path}_batch_{batch_index}.pkl", 'wb') as f:
            pickle.dump(batch_data, f)
        print(f"Saved batch {batch_index} with shape {batch_data['features'].shape}")
    
    def get_fwd_price(self, ticker_list):
        # Load the list of tickers
        with open(ticker_list, 'rb') as t:
            TICKERS = pickle.load(t)
        
        # Remove duplicates and sort
        TICKERS = sorted(list(set(TICKERS)))
        
        # Initialize a list to store closing prices
        closing_prices = []
        
        # Iterate over tickers and fetch data
        for i in TICKERS:
            try:
                placeholder = self.stocks_client.get_aggregate_bars(
                    i, 
                    self.check_date, 
                    self.check_date, 
                    timespan='day', 
                    full_range=True, 
                    run_parallel=False
                )
                # Append the closing price ('c') if it exists
                if 'c' in placeholder[0]:
                    closing_prices.append(placeholder[0]['c'])
                else:
                    print(f"Closing price not found for {i}")
                    closing_prices.append(None)
            except Exception as e:
                print(f"Failed to fetch data for {i}: {e}")
                closing_prices.append(None)
        
        # Return the list of closing prices
        return closing_prices

    def get_purchase_price(self, ticker_list):
        # Load the list of tickers
        with open(ticker_list, 'rb') as t:
            TICKERS = pickle.load(t)
        
        # Remove duplicates and sort
        TICKERS = sorted(list(set(TICKERS)))
        
        # Initialize a list to store closing prices
        purchase_prices = []
        
        # Iterate over tickers and fetch data
        for i in TICKERS:
            try:
                placeholder = self.stocks_client.get_aggregate_bars(
                    i, 
                    self.buy_date, 
                    self.buy_date, 
                    timespan='day',
                    full_range=True, 
                    run_parallel=False
                )
                # Append the closing price ('c') if it exists
                if 'o' in placeholder[0]:
                    purchase_prices.append(placeholder[0]['c'])
                else:
                    print(f"Opening price not found for {i}")
                    purchase_prices.append(None)
            except Exception as e:
                print(f"Failed to fetch data for {i}: {e}")
                purchase_prices.append(None)
        
        # Return the list of closing prices
        return purchase_prices