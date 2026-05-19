"""
Backtest Evaluator: Evaluate model predictions vs actual returns and benchmark ETFs.

Compares model stock picks against SPY, QQQ, IWM, and DIA over the same periods.

Usage:
    python backtest/Backtest_Evaluator.py
    python backtest/Backtest_Evaluator.py --pred-dir backtest_predictions
    python backtest/Backtest_Evaluator.py --pred-dir predictions_output_random  # evaluate existing predictions
"""
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import pandas as pd
from polygon import RESTClient
from pandas.tseries.offsets import BDay
from datetime import datetime
import time
import json


BENCHMARKS = ['SPY', 'QQQ', 'IWM', 'DIA']

# Same filter rules used in Trading.py and Bulk_Result_Check.py
SELECT_PROB_THRESHOLD = 0.33
DECLINE_PROB_THRESHOLD = 0.40

# Exit rule: if the open-to-high return >= 4.5% and beats open-to-close, take the high exit
HIGH_EXIT_THRESHOLD = 0.045


def get_prediction_dates(pred_dir):
    """Scan the prediction directory for available date files."""
    dates = []
    for fname in os.listdir(pred_dir):
        if fname.endswith('_v12_NewData_RegimeAtt_DoubleDigit_random_random.xlsx'):
            date_str = fname[:10]  # Extract YYYY-MM-DD
            try:
                datetime.strptime(date_str, '%Y-%m-%d')
                dates.append(date_str)
            except ValueError:
                continue
    return sorted(dates)


def fetch_ohlc_for_tickers(client, tickers, start_date_str, end_date_str):
    """Fetch aggregated OHLC data for a list of tickers over a date range."""
    results = {}
    for ticker in tickers:
        try:
            ohlc_data = list(client.list_aggs(
                ticker,
                1,
                'day',
                start_date_str,
                end_date_str,
            ))
            if ohlc_data:
                agg_open = ohlc_data[0].open
                agg_high = max(item.high for item in ohlc_data if item.high is not None)
                agg_low = min(item.low for item in ohlc_data if item.low is not None)
                agg_close = ohlc_data[-1].close
                results[ticker] = {
                    'Open': agg_open,
                    'High': agg_high,
                    'Low': agg_low,
                    'Close': agg_close
                }
            time.sleep(0.12)  # rate limit
        except Exception as e:
            print(f"  Warning: no data for {ticker}: {e}")
    return results


def calculate_return(ohlc):
    """Calculate return using the same logic as Bulk_Result_Check.py (4.5% high-exit rule)."""
    if ohlc['Open'] == 0:
        return 0.0
    oh_return = (ohlc['High'] - ohlc['Open']) / ohlc['Open']
    oc_return = (ohlc['Close'] - ohlc['Open']) / ohlc['Open']
    if oh_return >= HIGH_EXIT_THRESHOLD and oh_return > oc_return:
        return oh_return
    return oc_return


def evaluate_single_date(client, pred_dir, date):
    """Evaluate model predictions for a single date."""
    file_name = f'{date}_v12_NewData_RegimeAtt_DoubleDigit_random_random.xlsx'
    filepath = os.path.join(pred_dir, file_name)

    if not os.path.exists(filepath):
        return None

    df = pd.read_excel(filepath)
    df = df.sort_values(by='Select Probability', ascending=False)

    # Apply the same filter as Trading.py
    mask = (df['Select Probability'] > SELECT_PROB_THRESHOLD) & (df['Decline Probability'] < DECLINE_PROB_THRESHOLD)
    df_filtered = df[mask].copy()

    if df_filtered.empty:
        print(f"  No stocks passed filter for {date}")
        return None

    # Forward period: next business day to +5 business days (1 trading week)
    date_pandas = pd.to_datetime(date)
    start_date = (date_pandas + BDay(1)).to_pydatetime()
    end_date = (date_pandas + BDay(6)).to_pydatetime()
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    # Fetch actual forward returns for selected stocks
    selected_tickers = df_filtered['Ticker'].tolist()
    print(f"  {date}: {len(selected_tickers)} stocks selected, fetching forward returns...")
    ohlc_data = fetch_ohlc_for_tickers(client, selected_tickers, start_str, end_str)

    stock_returns = []
    for ticker in selected_tickers:
        if ticker in ohlc_data:
            ret = calculate_return(ohlc_data[ticker])
            stock_returns.append(ret)

    if not stock_returns:
        print(f"  No return data available for {date}")
        return None

    model_avg_return = np.mean(stock_returns)

    # Fetch benchmark returns for the same period
    benchmark_returns = {}
    benchmark_ohlc = fetch_ohlc_for_tickers(client, BENCHMARKS, start_str, end_str)
    for bench in BENCHMARKS:
        if bench in benchmark_ohlc:
            benchmark_returns[bench] = calculate_return(benchmark_ohlc[bench])
        else:
            benchmark_returns[bench] = None

    result = {
        'Date': date,
        'Model_Return': model_avg_return,
        'Num_Stocks_Selected': len(selected_tickers),
        'Num_With_Data': len(stock_returns),
    }
    for bench in BENCHMARKS:
        result[f'{bench}_Return'] = benchmark_returns.get(bench)

    return result


def generate_summary(results_df, output_dir):
    """Generate summary statistics and save report."""
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS SUMMARY")
    print("=" * 80)

    # Overall model performance
    model_returns = results_df['Model_Return'].dropna()
    print(f"\nModel Performance ({len(model_returns)} trading days):")
    print(f"  Average Weekly Return:  {model_returns.mean()*100:.2f}%")
    print(f"  Median Weekly Return:   {model_returns.median()*100:.2f}%")
    print(f"  Std Dev:                {model_returns.std()*100:.2f}%")
    print(f"  Win Rate:               {(model_returns > 0).mean()*100:.1f}%")
    print(f"  Best Week:              {model_returns.max()*100:.2f}%")
    print(f"  Worst Week:             {model_returns.min()*100:.2f}%")

    # Cumulative return (compounded)
    cumulative_model = (1 + model_returns).prod() - 1
    print(f"  Cumulative Return:      {cumulative_model*100:.2f}%")

    # Sharpe-like ratio (annualized, assuming ~52 weeks/year)
    if model_returns.std() > 0:
        sharpe = (model_returns.mean() / model_returns.std()) * np.sqrt(52)
        print(f"  Annualized Sharpe:      {sharpe:.2f}")

    # Benchmark performance
    print(f"\nBenchmark Comparison (same {len(model_returns)} periods):")
    print(f"  {'Benchmark':<10} {'Avg Wkly':>10} {'Cumulative':>12} {'Win Rate':>10} {'vs Model':>12}")
    print(f"  {'-'*54}")

    for bench in BENCHMARKS:
        col = f'{bench}_Return'
        if col in results_df.columns:
            bench_returns = results_df[col].dropna()
            if len(bench_returns) > 0:
                avg = bench_returns.mean()
                cumulative = (1 + bench_returns).prod() - 1
                win_rate = (bench_returns > 0).mean()
                excess = model_returns.mean() - avg
                print(f"  {bench:<10} {avg*100:>9.2f}% {cumulative*100:>11.2f}% {win_rate*100:>9.1f}% {excess*100:>+11.2f}%")

    # Model outperformance frequency
    print(f"\n  How often the Model beat each benchmark:")
    for bench in BENCHMARKS:
        col = f'{bench}_Return'
        if col in results_df.columns:
            valid = results_df.dropna(subset=['Model_Return', col])
            if len(valid) > 0:
                beat_pct = (valid['Model_Return'] > valid[col]).mean()
                print(f"    vs {bench}: {beat_pct*100:.1f}% of weeks ({int(beat_pct*len(valid))}/{len(valid)})")

    # Monthly breakdown
    results_df['Month'] = pd.to_datetime(results_df['Date']).dt.to_period('M')
    print(f"\nMonthly Breakdown:")
    print(f"  {'Month':<10} {'Model':>10} {'SPY':>10} {'QQQ':>10} {'IWM':>10} {'DIA':>10} {'#Picks':>8}")
    print(f"  {'-'*68}")

    for month, group in results_df.groupby('Month'):
        model_monthly = group['Model_Return'].mean()
        row = f"  {str(month):<10} {model_monthly*100:>9.2f}%"
        for bench in BENCHMARKS:
            col = f'{bench}_Return'
            if col in group.columns:
                val = group[col].mean()
                row += f" {val*100:>9.2f}%" if pd.notna(val) else f" {'N/A':>9}"
            else:
                row += f" {'N/A':>9}"
        avg_picks = group['Num_Stocks_Selected'].mean()
        row += f" {avg_picks:>7.0f}"
        print(row)

    # Save detailed results
    output_file = os.path.join(output_dir, 'backtest_results_detailed.xlsx')
    results_df.to_excel(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")

    # Save summary
    summary = {
        'date_range': f"{results_df['Date'].min()} to {results_df['Date'].max()}",
        'num_trading_days': len(model_returns),
        'model': {
            'avg_weekly_return': float(model_returns.mean()),
            'cumulative_return': float(cumulative_model),
            'win_rate': float((model_returns > 0).mean()),
            'std_dev': float(model_returns.std()),
        },
        'benchmarks': {}
    }
    for bench in BENCHMARKS:
        col = f'{bench}_Return'
        if col in results_df.columns:
            br = results_df[col].dropna()
            if len(br) > 0:
                summary['benchmarks'][bench] = {
                    'avg_weekly_return': float(br.mean()),
                    'cumulative_return': float((1 + br).prod() - 1),
                    'win_rate': float((br > 0).mean()),
                }

    summary_file = os.path.join(output_dir, 'backtest_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Evaluate backtest predictions vs benchmarks')
    parser.add_argument('--pred-dir', type=str, default='backtest_predictions',
                        help='Directory containing prediction Excel files')
    parser.add_argument('--key', type=str, default='cvV9m9XNz41uD7SMCLqftmzWKwDCI_9x',
                        help='Polygon API key')
    parser.add_argument('--start', type=str, default=None,
                        help='Only evaluate dates from this date forward')
    parser.add_argument('--end', type=str, default=None,
                        help='Only evaluate dates up to this date')
    args = parser.parse_args()

    client = RESTClient(api_key=args.key)

    # Find all available prediction dates
    dates = get_prediction_dates(args.pred_dir)
    if not dates:
        print(f"No prediction files found in {args.pred_dir}")
        return

    if args.start:
        dates = [d for d in dates if d >= args.start]
    if args.end:
        dates = [d for d in dates if d <= args.end]

    print(f"Evaluating {len(dates)} prediction dates from {dates[0]} to {dates[-1]}")

    # Check for existing partial results
    results_file = os.path.join(args.pred_dir, 'backtest_results_detailed.xlsx')
    existing_results = {}
    if os.path.exists(results_file):
        existing_df = pd.read_excel(results_file)
        for _, row in existing_df.iterrows():
            existing_results[row['Date']] = row.to_dict()
        print(f"Found {len(existing_results)} previously evaluated dates")

    results = []
    for idx, date in enumerate(dates):
        print(f"[{idx+1}/{len(dates)}] {date}", end="")

        # Use cached result if available
        if date in existing_results:
            results.append(existing_results[date])
            print(" (cached)")
            continue

        print()
        result = evaluate_single_date(client, args.pred_dir, date)
        if result:
            results.append(result)

        # Save intermediate results every 10 dates
        if (idx + 1) % 10 == 0 and results:
            intermediate_df = pd.DataFrame(results)
            intermediate_df.to_excel(results_file, index=False)
            print(f"  [Saved intermediate results: {len(results)} dates]")

    if not results:
        print("No results to evaluate.")
        return

    results_df = pd.DataFrame(results)
    generate_summary(results_df, args.pred_dir)


if __name__ == "__main__":
    main()
