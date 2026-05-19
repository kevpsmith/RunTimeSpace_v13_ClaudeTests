"""
IC Analysis: Compute Spearman rank IC for each feature against forward 1-week returns.

Reads .pkl files saved by PolygonDataFetcher (in data/placeholders/), computes
forward returns from closing prices, then measures cross-sectional Spearman rank
correlation (Information Coefficient) for every available feature.

Includes two sanity checks:
  - random_noise: synthetic random feature, should produce IC ~ 0 and |t-stat| < 2
  - perfect_signal: forward return itself, should produce IC ~ 1.0

Usage:
    python ic_analysis.py
    python ic_analysis.py --data-dir data/placeholders --date-prefix 2025-01-06
    python ic_analysis.py --horizon 10  # 2-week forward returns
"""
import os
import glob
import argparse
import pickle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PKL_FEATURE_MAP = {
    '01ClosingPrices': 'close_level',
    '01HighPrices': 'high_level',
    '01LowPrices': 'low_level',
    '02VolumeTraded': 'volume',
    '03NumberTrades': 'num_trades',
    '04QuarterlyEarnings': 'quarterly_earnings',
    '08DaysSince': 'days_since_filing',
    '09RSIDays': 'rsi_daily',
    '10RSIWeeks': 'rsi_weekly',
    '12MACDDays': 'macd_daily',
    '13MACDWeeks': 'macd_weekly',
}


def find_date_prefix(data_dir):
    pattern = os.path.join(data_dir, '*_01ClosingPrices.pkl')
    files = glob.glob(pattern)
    if not files:
        return None
    prefixes = [os.path.basename(f).replace('_01ClosingPrices.pkl', '') for f in files]
    return sorted(prefixes)[-1]


def load_features(data_dir, date_prefix):
    features = {}
    for pkl_suffix, feature_name in PKL_FEATURE_MAP.items():
        filepath = os.path.join(data_dir, f'{date_prefix}_{pkl_suffix}.pkl')
        if os.path.exists(filepath):
            df = pd.read_pickle(filepath)
            df = df.apply(pd.to_numeric, errors='coerce')
            features[feature_name] = df
            print(f"  Loaded {feature_name}: {df.shape}")
        else:
            print(f"  Skipped {feature_name}: {filepath} not found")
    return features


def compute_forward_returns(closing_prices, horizon=5):
    """
    Forward return at date t = (close[t+horizon] - close[t]) / close[t].

    No lookahead bias: the feature value at date t is known at time t.
    The return uses close[t] (the entry price, known at t) and
    close[t+horizon] (the exit price, only known at t+horizon).
    Dates without a valid t+horizon are left as NaN.
    """
    dates = closing_prices.columns.tolist()
    forward_returns = pd.DataFrame(
        np.nan, index=closing_prices.index, columns=dates
    )
    for i in range(len(dates) - horizon):
        current = closing_prices.iloc[:, i].astype(float)
        future = closing_prices.iloc[:, i + horizon].astype(float)
        forward_returns.iloc[:, i] = np.where(
            current != 0, (future - current) / current, np.nan
        )
    return forward_returns


def test_no_lookahead():
    """
    Verify forward return computation has no lookahead bias.
    Uses synthetic data with known values. Fails loudly on any violation.
    """
    print("\n--- Lookahead Bias Test ---")
    np.random.seed(42)

    tickers = [f'STOCK_{i}' for i in range(10)]
    dates = [f'2025-01-{d:02d}' for d in range(1, 21)]
    prices = np.random.uniform(50, 200, size=(10, 20))
    close_df = pd.DataFrame(prices, index=tickers, columns=dates)

    fwd = compute_forward_returns(close_df, horizon=5)

    # Test 1: values match manual calculation
    for t in range(15):
        for s in range(10):
            expected = (prices[s, t + 5] - prices[s, t]) / prices[s, t]
            actual = fwd.iloc[s, t]
            assert abs(actual - expected) < 1e-10, (
                f"FAIL: mismatch at t={t}, stock={s}: {actual} != {expected}"
            )
    print("  Test 1 PASSED: forward returns match manual calculation")

    # Test 2: last `horizon` dates must be NaN
    for t in range(15, 20):
        assert fwd.iloc[:, t].isna().all(), (
            f"FAIL: date index {t} should be NaN (no future data)"
        )
    print("  Test 2 PASSED: trailing dates are NaN")

    # Test 3: changing close[t-1] does NOT change forward_return[t]
    modified = close_df.copy()
    modified.iloc[:, 4] = 9999.0
    fwd_mod = compute_forward_returns(modified, horizon=5)
    pd.testing.assert_series_equal(
        fwd.iloc[:, 5], fwd_mod.iloc[:, 5], check_names=False
    )
    print("  Test 3 PASSED: past price change does not affect forward return")

    # Test 4: changing close[t+5] DOES change forward_return[t]
    modified2 = close_df.copy()
    modified2.iloc[:, 10] = 9999.0
    fwd_mod2 = compute_forward_returns(modified2, horizon=5)
    assert not fwd.iloc[:, 5].equals(fwd_mod2.iloc[:, 5]), (
        "FAIL: changing close[t+5] did not affect forward_return[t]"
    )
    print("  Test 4 PASSED: future price change correctly affects forward return")

    print("  ALL LOOKAHEAD TESTS PASSED")


def compute_cross_sectional_ic(feature_df, forward_returns_df, min_stocks=30):
    """Spearman rank IC per date between feature and forward returns."""
    common_dates = sorted(set(feature_df.columns) & set(forward_returns_df.columns))
    ics = []
    for date in common_dates:
        feat = feature_df[date].astype(float)
        ret = forward_returns_df[date].astype(float)
        valid = feat.notna() & ret.notna() & np.isfinite(feat) & np.isfinite(ret)
        if valid.sum() < min_stocks:
            continue
        corr, _ = spearmanr(feat[valid].values, ret[valid].values)
        if not np.isnan(corr):
            ics.append(corr)
    return ics


def compute_derived_features(features):
    """Simple derived features from raw .pkl data."""
    derived = {}
    if 'close_level' in features:
        close = features['close_level']
        derived['momentum_5d'] = close.pct_change(periods=5, axis=1)
        derived['momentum_20d'] = close.pct_change(periods=20, axis=1)
        daily_ret = close.pct_change(axis=1)
        derived['volatility_5d'] = daily_ret.T.rolling(window=5).std().T
    if 'volume' in features:
        vol = features['volume']
        vol_ma20 = vol.T.rolling(window=20).mean().T
        derived['relative_volume_20d'] = vol / vol_ma20
    return derived


def summarize_ic(ic_dict):
    rows = []
    for name, ics in ic_dict.items():
        if not ics:
            continue
        arr = np.array(ics)
        n = len(arr)
        mean = arr.mean()
        std = arr.std(ddof=1) if n > 1 else 0.0
        t = mean / (std / np.sqrt(n)) if std > 0 else 0.0
        ir = mean / std if std > 0 else 0.0
        pct_pos = (arr > 0).mean() * 100
        rows.append({
            'feature': name,
            'mean_IC': round(mean, 4),
            'std_IC': round(std, 4),
            't_stat': round(t, 2),
            'IC_IR': round(ir, 4),
            'pct_positive': round(pct_pos, 1),
            'num_dates': n,
        })
    return pd.DataFrame(rows).sort_values('t_stat', ascending=False, key=abs)


def run_sanity_checks(ic_dict):
    print("\n--- Sanity Checks ---")

    if 'random_noise' in ic_dict and ic_dict['random_noise']:
        arr = np.array(ic_dict['random_noise'])
        mean = arr.mean()
        std = arr.std(ddof=1) if len(arr) > 1 else 0.0
        t = mean / (std / np.sqrt(len(arr))) if std > 0 else 0.0
        assert abs(t) < 2.0, (
            f"FAIL: random_noise |t-stat| = {abs(t):.2f} >= 2.0 "
            f"(mean_IC={mean:.4f}, n={len(arr)})"
        )
        print(f"  PASSED: random_noise  mean_IC={mean:.4f}  t-stat={t:.2f}")

    if 'perfect_signal' in ic_dict and ic_dict['perfect_signal']:
        arr = np.array(ic_dict['perfect_signal'])
        mean = arr.mean()
        assert mean >= 0.95, (
            f"FAIL: perfect_signal mean_IC = {mean:.4f} (expected >= 0.95)"
        )
        print(f"  PASSED: perfect_signal  mean_IC={mean:.4f}")


def main():
    parser = argparse.ArgumentParser(description='IC Analysis: feature predictiveness')
    parser.add_argument('--data-dir', type=str,
                        default=os.path.join('data', 'placeholders'),
                        help='Directory containing .pkl files')
    parser.add_argument('--date-prefix', type=str, default=None,
                        help='Date prefix for .pkl files (auto-detected if omitted)')
    parser.add_argument('--output', type=str, default='ic_results.csv',
                        help='Output CSV path')
    parser.add_argument('--horizon', type=int, default=5,
                        help='Forward return horizon in trading days (default: 5)')
    args = parser.parse_args()

    test_no_lookahead()

    if args.date_prefix is None:
        args.date_prefix = find_date_prefix(args.data_dir)
    if args.date_prefix is None:
        print(f"\nNo .pkl files found in {args.data_dir}")
        print("Run PolygonDataFetcher first to generate data, or specify --date-prefix")
        return

    print(f"\nLoading features from {args.data_dir} (prefix: {args.date_prefix})...")
    features = load_features(args.data_dir, args.date_prefix)

    if 'close_level' not in features:
        print("ERROR: closing prices not found — cannot compute forward returns")
        return

    print(f"\nComputing {args.horizon}-day forward returns...")
    forward_returns = compute_forward_returns(features['close_level'], horizon=args.horizon)
    valid_count = forward_returns.dropna(axis=1, how='all').shape[1]
    print(f"  Dates with valid forward returns: {valid_count}")

    print("\nComputing derived features...")
    derived = compute_derived_features(features)

    all_features = {}
    for name, df in features.items():
        if name not in ('close_level', 'high_level', 'low_level'):
            all_features[name] = df
    all_features.update(derived)

    np.random.seed(12345)
    all_features['random_noise'] = pd.DataFrame(
        np.random.randn(*features['close_level'].shape),
        index=features['close_level'].index,
        columns=features['close_level'].columns,
    )
    all_features['perfect_signal'] = forward_returns

    print(f"\nComputing Spearman rank IC for {len(all_features)} features...")
    ic_dict = {}
    for name, df in all_features.items():
        ics = compute_cross_sectional_ic(df, forward_returns)
        ic_dict[name] = ics
        if ics:
            print(f"  {name}: {len(ics)} dates, mean IC = {np.mean(ics):.4f}")
        else:
            print(f"  {name}: no valid dates")

    run_sanity_checks(ic_dict)

    summary = summarize_ic(ic_dict)
    print("\n" + "=" * 90)
    print("IC ANALYSIS RESULTS")
    print("=" * 90)
    print(summary.to_string(index=False))
    print("=" * 90)

    summary.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
