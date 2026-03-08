"""
Smoke Test: Run the full backtest pipeline for January 2025 only.

This is a quick validation run to make sure everything works end-to-end
before committing to the full Jan-Jul 2025 backtest.

Runs every trading day in January 2025 (~21 days):
  1. Data acquisition from Polygon
  2. Model training (10 epochs)
  3. Prediction generation
  4. Evaluation against actuals + benchmarks (SPY, QQQ, IWM, DIA)

Usage:
    python backtest/Backtest_Smoke_Test.py
    python backtest/Backtest_Smoke_Test.py --skip-acquisition   # reuse existing data
    python backtest/Backtest_Smoke_Test.py --eval-only           # skip pipeline, just evaluate existing predictions
"""
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from datetime import datetime
import time

from Backtest_Runner import generate_trading_days, run_data_acquisition, run_training_and_prediction
from Backtest_Evaluator import main as run_evaluation


START_DATE = '2025-01-02'
END_DATE = '2025-01-31'
OUTPUT_DIR = 'backtest_smoke_test'


def main():
    parser = argparse.ArgumentParser(description='Smoke test: January 2025 backtest')
    parser.add_argument('--skip-acquisition', action='store_true', help='Skip data acquisition, use existing data')
    parser.add_argument('--eval-only', action='store_true', help='Skip pipeline entirely, just evaluate existing predictions')
    parser.add_argument('--key', type=str, default='cvV9m9XNz41uD7SMCLqftmzWKwDCI_9x', help='Polygon API key')
    args = parser.parse_args()

    trading_days = generate_trading_days(START_DATE, END_DATE)
    print(f"Smoke Test: January 2025")
    print(f"Trading days: {len(trading_days)}")
    print(f"Dates: {trading_days[0]} to {trading_days[-1]}")
    print()

    ticker_dir = os.path.join('data', 'daily_data')
    os.makedirs(ticker_dir, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not args.eval_only:
        total_start = time.time()
        completed = 0
        failed = 0

        for idx, date in enumerate(trading_days):
            print(f"\n{'='*60}")
            print(f"[{idx+1}/{len(trading_days)}] Processing {date}")
            print(f"{'='*60}")

            try:
                # Step 1: Data Acquisition
                if not args.skip_acquisition:
                    print(f"  Data acquisition...")
                    t0 = time.time()
                    run_data_acquisition(date, args.key, ticker_dir)
                    print(f"  Done ({time.time()-t0:.1f}s)")

                # Step 2: Training + Prediction
                print(f"  Training + prediction...")
                t0 = time.time()
                run_training_and_prediction(date, ticker_dir, OUTPUT_DIR)
                print(f"  Done ({time.time()-t0:.1f}s)")

                completed += 1

            except Exception as e:
                print(f"  [FAILED] {e}")
                failed += 1

        elapsed = time.time() - total_start
        print(f"\n{'='*60}")
        print(f"Pipeline complete: {completed} succeeded, {failed} failed")
        print(f"Total time: {elapsed/60:.1f} minutes")
        print(f"{'='*60}")

    # Step 3: Evaluate results
    print(f"\nRunning evaluation...")
    sys.argv = [
        'Backtest_Evaluator.py',
        '--pred-dir', OUTPUT_DIR,
        '--key', args.key,
        '--start', START_DATE,
        '--end', END_DATE,
    ]
    run_evaluation()


if __name__ == "__main__":
    main()
