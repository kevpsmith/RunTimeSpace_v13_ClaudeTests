import os
import json
import pandas as pd
from Broker_API import BrokerAPI
from datetime import datetime
import pytz

class DailyTrading:
    def __init__(self, date, api_key, base_dir, config_path):
        self.date = date
        self.api = BrokerAPI(config_path)
        self.base_dir = base_dir
        self.logs_dir = os.path.join(self.base_dir, 'logs')
        self.predictions_dir = os.path.join(self.base_dir, 'predictions_output_random')
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.predictions_dir, exist_ok=True)
        self.predictions_file = os.path.join(self.predictions_dir, f'{self.date}_v12_NewData_RegimeAtt_DoubleDigit_random_random.xlsx')

    def get_settled_cash(self):
        settled_cash = self.api.get_settled_cash()
        print(f"[INFO] Settled cash available: ${settled_cash:.2f}")
        return settled_cash

    def load_predictions(self):
        if not os.path.exists(self.predictions_file):
            raise FileNotFoundError(f"Predictions file not found: {self.predictions_file}")
        
        predictions = pd.read_excel(self.predictions_file)
        print(f"[INFO] Loaded predictions from {self.predictions_file}")
        return predictions
    
    def process_predictions(self, predictions, settled_cash, stock_price_dict):
        do_not_buy = ["CGNX"]
        trades = []
        remaining_cash = settled_cash
        selected_stocks = predictions[
            (predictions['Select Probability']>0.33) &
            (predictions['Decline Probability']<0.4)].sort_values(by='Select Probability', ascending=False)
        
        for _, row in selected_stocks.iterrows():
            ticker = row['Ticker']
            if ticker in do_not_buy:
                print(f"[INFO] Skipping {ticker} - It's on the restricted list.")
                continue
            
            price = stock_price_dict.get(ticker)
            #prev_close = prev_close_prices.get(ticker)
            if not price:
                continue
            
            max_cash_per_stock = remaining_cash / len(selected_stocks)
            quantity = max(1, int(max_cash_per_stock//price))
            total_spend = quantity * price

            if total_spend > remaining_cash:
                continue
                
            trades.append({
                'Ticker': ticker,
                'Quantity': quantity,
                'Buy Price': price * 1.005,
                'Max Price': price * 1.045,
                'Trigger Price': price *1.045,
                'Trailing Percent': 0.5
            })

            remaining_cash -= total_spend

            if remaining_cash <= 0:
                break
        print(f"[INFO] Generated{len(trades)} trades. Remaining cash ${remaining_cash:.2f}")
        return trades
    
    def execute_trades(self, trades):
        results = []
        for trade in trades:
            ticker = trade['Ticker']
            quantity = int(trade['Quantity'])
            buy_price = round(trade['Buy Price'],2)
            cancel_price = round(trade['Max Price'],2)  # 4.5% above previous close
            trigger_price = round(trade['Max Price'],2)  # Trigger for trailing stop
            trailing_percent = 0.5  # 0.5% trailing stop
            try:
                # Place the 1st triggers sequential order
                result = self.api.place_1st_triggers_sequential_order(
                    ticker=ticker,
                    quantity=quantity,
                    limit_price=buy_price,
                    cancel_price=cancel_price,
                    trigger_price=trigger_price,
                    trailing_percent=trailing_percent
                )
                results.append({'Trade': trade, 'Status': 'SUCCESS', 'Details': result})
            except Exception as e:
                results.append({'Trade': trade, 'Status': 'FAILED', 'Details': str(e)})
            
        print(f'[INFO] Executed {len(results)} trades.')
        return results

    def log_trades(self, results):
        log_file = os.path.join(self.logs_dir, f"{self.date}_trade_log.csv")
        df = pd.DataFrame(results)
        df.to_csv(log_file, index=False)
        print(f"[INFO] Trade log saved to {log_file}")

    def run(self):
        status = self.api.check_if_open()
        if not status:  # Check if the market was open earlier today
            print("[INFO] Market was not open earlier today. Exiting...")
            #return  # Exit early if market was not open
        predictions = self.load_predictions()
        settled_cash = self.get_settled_cash()
        stock_prices = self.api.get_polygon_stock_prices(predictions['Ticker'].tolist())
        #prev_closes = self.api.get_previous_closes(predictions['Ticker'].tolist())
        trades = self.process_predictions(predictions, settled_cash, stock_prices)
        results = self.execute_trades(trades)
        self.log_trades(results)

if __name__ == "__main__":
    ny_tz = pytz.timezone("America/New_York")
    today = datetime.now(ny_tz).strftime("%Y-%m-%d")
    base_dir = "/teamspace/studios/this_studio"
    # Get the directory of the current script (Trading.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.json")
    # Load the config file
    with open(config_path, "r") as file:
        config = json.load(file)
    apikey = config.get("client_id")  # Adjust key if needed
    trading = DailyTrading(today, apikey, base_dir, config_path)
    trading.run()


