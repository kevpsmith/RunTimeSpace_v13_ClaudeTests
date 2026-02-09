import sys
import os

# Add the correct path for 'daily_runs/' so Python can find the module
sys.path.append(os.path.abspath("/teamspace/studios/this_studio/daily_runs"))
from Daily_Data_Acquisition import DailyDataAcquisition

test_data = DailyDataAcquisition(base_dir="/teamspace/studios/this_studio", date="2025-02-19")
test_data.daily_acquire()
