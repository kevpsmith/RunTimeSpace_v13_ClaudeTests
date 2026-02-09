import sys
import os
# Add the correct path for 'daily_runs/' so Python can find the module
sys.path.append(os.path.abspath("/teamspace/studios/this_studio/daily_runs"))
import lightning as L
import time
import datetime
import pytz
from Daily_Data_Acquisition import DailyDataAcquisition
from Daily_Model_Training import DailyModelTraining
from Trading import DailyTrading

class DailyFlow(L.LightningFlow):
    def __init__(self, base_dir, date, api_key):
        super().__init__()
        self.base_dir = base_dir
        self.date = date
        self.api_key = api_key

        # Define Work Modules
        self.data_acquisition = DailyDataAcquisition(base_dir, date)
        self.model_training = DailyModelTraining(base_dir, date, device="gpu")
        self.trading = DailyTrading(date, api_key, base_dir)

        # Track execution state
        self.state = {
            "data_acquisition_done": False,
            "model_training_done": False,
            "trading_done": False
        }

    def wait_until_415_pm(self):
        """
        Waits until 4:15 PM EST before starting the pipeline.
        If already past 4:15 PM EST, it proceeds immediately.
        """
        est = pytz.timezone("US/Eastern")
        now = datetime.datetime.now(est)
        target_time = now.replace(hour=16, minute=15, second=0, microsecond=0)

        if now < target_time:
            wait_seconds = (target_time - now).total_seconds()
            print(f"[INFO] Waiting {wait_seconds / 60:.2f} minutes until 4:15 PM EST...")
            time.sleep(wait_seconds)

        print("[INFO] It's 4:15 PM EST or later. Starting workflow...")

    def run(self):
        """
        Execute the full pipeline in sequence, ensuring it starts after 4:15 PM EST.
        """
        # Ensure we only start after 4:15 PM EST
        self.wait_until_415_pm()

        # Step 1: Run Data Acquisition
        if not self.state["data_acquisition_done"]:
            print("[INFO] Running Data Acquisition...")
            self.data_acquisition.daily_acquire()
            self.state["data_acquisition_done"] = True

        # Step 2: Run Model Training
        elif not self.state["model_training_done"]:
            print("[INFO] Running Model Training on GPU...")
            self.model_training.daily_train()
            self.state["model_training_done"] = True

        # Step 3: Run Trading Execution
        elif not self.state["trading_done"]:
            print("[INFO] Running Trading Execution...")
            self.trading.run()
            self.state["trading_done"] = True

        else:
            print("[INFO] All tasks completed for the day.")
