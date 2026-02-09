import numpy as np
import pandas as pd

class EnhancedFeatures:
    def calculate_price_momentum(self, df, periods=[1, 5, 20]):
        """Calculate price momentum over multiple periods"""
        momentum = {}
        for period in periods:
            momentum[f'momentum_{period}d'] = df.pct_change(period)
        return momentum
    
    def calculate_atr(self, high, low, close, period=14):
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.DataFrame({'TR1': tr1, 'TR2': tr2, 'TR3': tr3}).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def calculate_vwap(self, df):
        """Calculate VWAP using high, low, close prices and volume"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return vwap
    
    def calculate_moving_averages(self, close, periods=[20, 50, 200]):
        """Calculate price relative to multiple moving averages"""
        ma_metrics = {}
        for period in periods:
            ma = close.rolling(window=period).mean()
            ma_metrics[f'price_to_ma_{period}'] = close / ma - 1
        return ma_metrics
    
    def calculate_volume_metrics(self, volume, close, periods=[20]):
        """Calculate volume-based metrics"""
        vol_metrics = {}
        
        # Relative volume
        for period in periods:
            avg_volume = volume.rolling(window=period).mean()
            vol_metrics[f'relative_volume_{period}d'] = volume / avg_volume
        
        # Money Flow Index
        typical_price = close  # Simplified version
        raw_money_flow = typical_price * volume
        mfi_period = 14
        
        pos_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
        neg_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
        
        money_ratio = pos_flow.rolling(window=mfi_period).sum() / neg_flow.rolling(window=mfi_period).sum()
        vol_metrics['mfi'] = 100 - (100 / (1 + money_ratio))
        
        # Volume trend
        vol_metrics['volume_trend_5d'] = volume.pct_change(5)
        vol_metrics['volume_trend_20d'] = volume.pct_change(20)
        
        return vol_metrics
    
    def calculate_technical_indicators(self, df, rsi_period=14, bb_period=20):
        """Calculate technical indicators"""
        tech_indicators = {}
        
        # RSI Divergence
        rsi = df['rsi']
        price = df['close']
        
        # Calculate slopes for price and RSI over 5-day period
        price_slope = price.diff(5) / 5
        rsi_slope = rsi.diff(5) / 5
        
        tech_indicators['rsi_divergence'] = np.where(
            (price_slope > 0) & (rsi_slope < 0), -1,  # Bearish divergence
            np.where((price_slope < 0) & (rsi_slope > 0), 1, 0)  # Bullish divergence
        )
        
        # MACD Histogram Analysis
        macd_hist = df['macd_histogram']
        tech_indicators['macd_slope'] = macd_hist.diff()
        tech_indicators['macd_crossover'] = np.where(
            (macd_hist > 0) & (macd_hist.shift() < 0), 1,
            np.where((macd_hist < 0) & (macd_hist.shift() > 0), -1, 0)
        )
        
        # Bollinger Bands
        std = df['close'].rolling(window=bb_period).std()
        middle_band = df['close'].rolling(window=bb_period).mean()
        upper_band = middle_band + (std * 2)
        lower_band = middle_band - (std * 2)
        
        tech_indicators['bb_position'] = (df['close'] - middle_band) / (upper_band - lower_band)
        
        return tech_indicators