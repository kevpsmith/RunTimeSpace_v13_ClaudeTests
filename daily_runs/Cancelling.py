import os
import json
import pandas as pd
from Broker_API import BrokerAPI
from datetime import datetime, timedelta, timedelta
import pytz

class DailyCancelling:
    def __init__(self, date, api_key, base_dir, config_path):
        self.date = date
        self.api = BrokerAPI(config_path)
        self.base_dir = base_dir
        self.logs_dir = os.path.join(self.base_dir, 'logs')
        self.predictions_dir = os.path.join(self.base_dir, 'predictions_output_random')
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.predictions_dir, exist_ok=True)
        self.predictions_file = os.path.join(self.predictions_dir, f'{self.date}_v12_NewData_RegimeAtt_DoubleDigit_random_random.xlsx')

    def cancel_and_replace_old_orders(self):
        """
        Cancels old purchase orders (older than 2 days) with no replacement.
        Cancels old sell orders (older than 5 days) and replaces them with market sell orders.
        """
        analysis_date = datetime.strptime(self.date, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
        five_days_ago = analysis_date - timedelta(days=5)
        two_days_ago = analysis_date - timedelta(days=2)

        open_orders = self.api.get_open_orders()

        for order in open_orders:
            order_id = order["orderId"]
            entered_time = datetime.strptime(order["enteredTime"], "%Y-%m-%dT%H:%M:%S%z")
            instruction = order["orderLegCollection"][0]["instruction"]  # BUY or SELL
            position_effect = order["orderLegCollection"][0]["positionEffect"]  # OPENING or CLOSING
            status = order["status"]

            if status in ["CANCELED", "REJECTED"]:
                continue

            # Rule 1: Cancel purchase orders older than 2 days (NO replacement)
            if instruction == "BUY" and position_effect == "OPENING":
                if entered_time < two_days_ago:
                    print(f"[INFO] Cancelling old BUY order {order_id} (entered: {entered_time})")
                    self.api.cancel_order(order_id)

            # Rule 2: Cancel and replace sell orders older than 5 days (Market Sell)
            elif instruction == "SELL" and position_effect == "CLOSING":
                if status != "AWAITING_PARENT_ORDER" and entered_time < five_days_ago:
                    quantity = order["quantity"]
                    symbol = order["orderLegCollection"][0]["instrument"]["symbol"]
                    
                    print(f"[INFO] Cancelling old SELL order {order_id} (entered: {entered_time})")
                    self.api.cancel_order(order_id)
                    
                    print(f"[INFO] Placing Market SELL order for {symbol}, {quantity} shares")
                    self.api.place_market_sell_order(symbol, quantity)

    def log_cancellations(self):
        log_file = os.path.join(self.logs_dir, f"{self.date}_cancellation_log.csv")
        df = pd.DataFrame(self.api.get_open_orders())
        df.to_csv(log_file, index=False)
        print(f"[INFO] Cancellation log saved to {log_file}")

    def run(self):
        if not self.api.check_if_open():
            print("[INFO] Market was not open earlier today. Exiting...")
            #return
        self.cancel_and_replace_old_orders()
        self.log_cancellations()

if __name__ == "__main__":
    ny_tz = pytz.timezone("America/New_York")
    today = datetime.now(ny_tz).strftime("%Y-%m-%d")
    base_dir = "/teamspace/studios/this_studio"
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    with open(config_path, "r") as file:
        config = json.load(file)
    apikey = config.get("client_id")
    cancelling = DailyCancelling(today, apikey, base_dir, config_path)
    cancelling.run()