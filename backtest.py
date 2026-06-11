"""
WHEAT ACCURACY BACKTEST
========================
Run this once in GitHub Actions (or locally) to find which combinations
of conditions actually preceded a 2.5% target hit before a 1.5% stop hit.

Usage:
  python3 backtest.py

Output:
  - Prints each condition's individual accuracy
  - Prints all combinations that achieve 70%+ accuracy
  - Saves results to backtest_results.json
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import combinations
import json
import warnings
warnings.filterwarnings('ignore')

STOP_PCT   = 0.015
TARGET_PCT = 0.025
LOOKBACK_DAYS = 730  # 2 years

# ── fetch data ────────────────────────────────────────────────────────────────

def fetch_data():
    print("Fetching ZW=F daily + hourly data...")
    end   = datetime.now()
    start = end - timedelta(days=LOOKBACK_DAYS)

    daily  = yf.Ticker("ZW=F").history(start=start, end=end, interval='1d', auto_adjust=False)
    hourly = yf.Ticker("ZW=F").history(start=start, end=end, interval='1h', auto_adjust=False)
    corn   = yf.Ticker("ZC=F").history(start=start, end=end, interval='1d', auto_adjust=False)
    soy    = yf.Ticker("ZS=F").history(start=start, end=end, interval='1d', auto_adjust=False)

    daily.index  = daily.index.tz_localize(None)  if daily.index.tz  else daily.index
    hourly.index = hourly.index.tz_localize(None) if hourly.index.tz else hourly.index
    corn.index   = corn.index.tz_localize(None)   if corn.index.tz   else corn.index
    soy.index    = soy.index.tz_localize(None)    if soy.index.tz    else soy.index

    print(f"Daily candles:  {len(daily)}")
    print(f"Hourly candles: {len(hourly)}")
    return daily, hourly, corn, soy

# ── build indicators ──────────────────────────────────────────────────────────

def build_features(daily, corn, soy):
    df = daily.copy()

    # Returns
    df['ret_1d']  = df['Close'].pct_change(1)
    df['ret_3d']  = df['Close'].pct_change(3)
    df['ret_5d']  = df['Close'].pct_change(5)

    # Trend
    df['sma20'] = df['Close'].rolling(20).mean()
    df['sma50'] = df['Close'].rolling(50).mean()
    df['above_sma20'] = (df['Close'] > df['sma20']).astype(int)
    df['above_sma50'] = (df['Close'] > df['sma50']).astype(int)
    df['trend_aligned'] = ((df['sma20'] > df['sma50'])).astype(int)  # 1=uptrend

    # RSI
    delta = df['Close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))
    df['rsi_oversold']  = (df['rsi'] < 35).astype(int)
    df['rsi_overbought']= (df['rsi'] > 65).astype(int)
    df['rsi_neutral']   = ((df['rsi'] >= 40) & (df['rsi'] <= 60)).astype(int)

    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['macd']        = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_bullish']= (df['macd'] > df['macd_signal']).astype(int)

    # ATR — volatility gate
    hl   = df['High'] - df['Low']
    hc   = (df['High'] - df['Close'].shift()).abs()
    lc   = (df['Low']  - df['Close'].shift()).abs()
    df['atr']     = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df['atr_pct'] = df['atr'] / df['Close']
    # ATR must be > 1.0% for target to be reachable
    df['vol_ok']  = (df['atr_pct'] > 0.010).astype(int)
    # ATR must be < 3.5% — too volatile = unpredictable
    df['vol_not_extreme'] = (df['atr_pct'] < 0.035).astype(int)
    df['vol_good'] = (df['vol_ok'] & df['vol_not_extreme']).astype(int)

    # Volume
    df['vol_avg'] = df['Volume'].rolling(20).mean()
    df['vol_ratio'] = df['Volume'] / df['vol_avg']
    df['vol_high']  = (df['vol_ratio'] > 1.2).astype(int)
    df['vol_low']   = (df['vol_ratio'] < 0.8).astype(int)

    # Bollinger Bands
    bb_mid   = df['Close'].rolling(20).mean()
    bb_std   = df['Close'].rolling(20).std()
    df['bb_upper'] = bb_mid + 2 * bb_std
    df['bb_lower'] = bb_mid - 2 * bb_std
    df['near_bb_lower'] = (df['Close'] < bb_mid - 1.5 * bb_std).astype(int)
    df['near_bb_upper'] = (df['Close'] > bb_mid + 1.5 * bb_std).astype(int)
    df['inside_bb']     = (~(df['near_bb_lower'].astype(bool) | df['near_bb_upper'].astype(bool))).astype(int)

    # Wheat/corn ratio
    corn_aligned = corn['Close'].reindex(df.index, method='ffill')
    soy_aligned  = soy['Close'].reindex(df.index,  method='ffill')
    wc_ratio = df['Close'] / corn_aligned
    ws_ratio = df['Close'] / soy_aligned
    df['wc_zscore'] = (wc_ratio - wc_ratio.rolling(60).mean()) / wc_ratio.rolling(60).std()
    df['ws_zscore'] = (ws_ratio - ws_ratio.rolling(60).mean()) / ws_ratio.rolling(60).std()
    df['wc_bullish'] = (df['wc_zscore'] > 0.5).astype(int)
    df['wc_bearish'] = (df['wc_zscore'] < -0.5).astype(int)

    # Price position in 52-week range
    df['high52'] = df['Close'].rolling(252).max()
    df['low52']  = df['Close'].rolling(252).min()
    df['range_pct'] = (df['Close'] - df['low52']) / (df['high52'] - df['low52'])
    df['in_lower_half'] = (df['range_pct'] < 0.4).astype(int)
    df['in_upper_half'] = (df['range_pct'] > 0.6).astype(int)

    # Momentum consistency — same direction 2 days in a row
    df['momentum_up']   = ((df['ret_1d'] > 0) & (df['ret_3d'] > 0)).astype(int)
    df['momentum_down'] = ((df['ret_1d'] < 0) & (df['ret_3d'] < 0)).astype(int)

    # Seasonal month
    df['month'] = df.index.month
    # Historically bullish months for wheat: Mar, Apr, May, Nov, Dec
    df['bullish_month'] = df['month'].isin([3, 4, 5, 11, 12]).astype(int)
    # Historically bearish months: Jun, Jul, Aug
    df['bearish_month'] = df['month'].isin([6, 7, 8]).astype(int)

    return df.dropna()

# ── validate outcome using next-day hourly bars ───────────────────────────────

def compute_outcomes(daily, hourly):
    """
    For each daily close, simulate entering next morning.
    Use hourly bars to check if +2.5% target or -1.5% stop was hit first.
    Returns a Series: True = WIN (target hit first), False = LOSS (stop hit first), NaN = still open
    """
    outcomes_up   = {}
    outcomes_down = {}

    daily_dates = daily.index.normalize().unique()

    for i in range(len(daily_dates) - 2):
        entry_date  = daily_dates[i]
        exit_window = daily_dates[i + 1] if i + 1 < len(daily_dates) else None

        if exit_window is None:
            continue

        entry_price = float(daily.loc[daily.index.normalize() == entry_date, 'Close'].iloc[-1])
        stop_up     = entry_price * (1 - STOP_PCT)
        target_up   = entry_price * (1 + TARGET_PCT)
        stop_down   = entry_price * (1 + STOP_PCT)
        target_down = entry_price * (1 - TARGET_PCT)

        # Get next day's hourly bars
        next_day_mask = (hourly.index.normalize() == exit_window)
        next_bars = hourly[next_day_mask]

        if next_bars.empty:
            continue

        # Walk bars — UP trade
        win_up = loss_up = win_down = loss_down = None
        for _, bar in next_bars.iterrows():
            if win_up is None and loss_up is None:
                if bar['Low'] <= stop_up:
                    loss_up = True
                if bar['High'] >= target_up:
                    win_up = True
                if win_up and loss_up:
                    # both in same bar — conservative: count as loss
                    win_up = None
                    break

            if win_down is None and loss_down is None:
                if bar['High'] >= stop_down:
                    loss_down = True
                if bar['Low'] <= target_down:
                    win_down = True
                if win_down and loss_down:
                    win_down = None
                    break

        # If neither hit — use close vs entry
        if win_up is None and loss_up is None:
            final_close = float(next_bars['Close'].iloc[-1])
            win_up = final_close > entry_price

        if win_down is None and loss_down is None:
            final_close = float(next_bars['Close'].iloc[-1])
            win_down = final_close < entry_price

        outcomes_up[entry_date]   = bool(win_up)   if win_up   else False
        outcomes_down[entry_date] = bool(win_down)  if win_down else False

    return pd.Series(outcomes_up), pd.Series(outcomes_down)

# ── backtest each condition ───────────────────────────────────────────────────

def backtest_conditions(df, outcomes_up, outcomes_down):
    """
    For each condition, calculate:
    - UP accuracy when condition is True
    - DOWN accuracy when condition is True
    - Sample size
    """

    # Align outcomes with features
    df_aligned = df[df.index.isin(outcomes_up.index)].copy()
    up_outcomes   = outcomes_up.reindex(df_aligned.index)
    down_outcomes = outcomes_down.reindex(df_aligned.index)

    conditions = [
        'above_sma20', 'above_sma50', 'trend_aligned',
        'rsi_oversold', 'rsi_overbought', 'rsi_neutral',
        'macd_bullish',
        'vol_good', 'vol_high', 'vol_low',
        'near_bb_lower', 'near_bb_upper', 'inside_bb',
        'wc_bullish', 'wc_bearish',
        'in_lower_half', 'in_upper_half',
        'momentum_up', 'momentum_down',
        'bullish_month', 'bearish_month',
    ]

    results = []

    for cond in conditions:
        if cond not in df_aligned.columns:
            continue

        mask = df_aligned[cond] == 1

        if mask.sum() < 10:
            continue

        up_acc   = up_outcomes[mask].mean()
        down_acc = down_outcomes[mask].mean()
        n        = mask.sum()

        results.append({
            'condition': cond,
            'n': int(n),
            'up_accuracy':   round(float(up_acc),   3),
            'down_accuracy': round(float(down_acc),  3),
            'best_direction': 'UP' if up_acc > down_acc else 'DOWN',
            'best_accuracy':  round(max(up_acc, down_acc), 3),
        })

    results.sort(key=lambda x: x['best_accuracy'], reverse=True)
    return results

# ── find best combinations ────────────────────────────────────────────────────

def find_best_combinations(df, outcomes_up, outcomes_down, top_conditions, target_accuracy=0.75):
    """
    Test combinations of 3-5 conditions to find those exceeding target_accuracy.
    Only tests the top 10 individual conditions to keep compute manageable.
    """
    df_aligned    = df[df.index.isin(outcomes_up.index)].copy()
    up_outcomes   = outcomes_up.reindex(df_aligned.index)
    down_outcomes = outcomes_down.reindex(df_aligned.index)

    top_conds = [r['condition'] for r in top_conditions[:12]]
    combo_results = []

    print(f"\nTesting combinations of top {len(top_conds)} conditions...")

    for r in [3, 4, 5]:
        for combo in combinations(top_conds, r):
            combo = list(combo)
            mask  = pd.Series(True, index=df_aligned.index)
            for c in combo:
                if c in df_aligned.columns:
                    mask = mask & (df_aligned[c] == 1)

            n = mask.sum()
            if n < 8:
                continue

            up_acc   = float(up_outcomes[mask].mean())
            down_acc = float(down_outcomes[mask].mean())

            if max(up_acc, down_acc) >= target_accuracy:
                combo_results.append({
                    'conditions':    combo,
                    'n':             int(n),
                    'up_accuracy':   round(up_acc,   3),
                    'down_accuracy': round(down_acc,  3),
                    'best_direction': 'UP' if up_acc > down_acc else 'DOWN',
                    'best_accuracy':  round(max(up_acc, down_acc), 3),
                })

    combo_results.sort(key=lambda x: (x['best_accuracy'], x['n']), reverse=True)
    return combo_results

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("WHEAT ACCURACY BACKTEST")
    print(f"Stop: {STOP_PCT:.1%} | Target: {TARGET_PCT:.1%} | Lookback: {LOOKBACK_DAYS}d")
    print("=" * 60)

    # 1. Fetch data
    daily, hourly, corn, soy = fetch_data()

    # 2. Build features
    print("\nBuilding features...")
    df = build_features(daily, corn, soy)
    print(f"Feature rows: {len(df)}")

    # 3. Compute outcomes
    print("\nComputing outcomes (stop/target hits)...")
    outcomes_up, outcomes_down = compute_outcomes(daily, hourly)
    print(f"Outcomes computed: {len(outcomes_up)}")

    baseline_up   = float(outcomes_up.mean())
    baseline_down = float(outcomes_down.mean())
    print(f"\nBaseline (no filter): UP={baseline_up:.1%} DOWN={baseline_down:.1%}")
    print(f"Average daily range: {((daily['High']-daily['Low'])/daily['Close']*100).mean():.2f}%")

    # 4. Test individual conditions
    print("\n" + "=" * 60)
    print("INDIVIDUAL CONDITION ACCURACY")
    print("=" * 60)
    condition_results = backtest_conditions(df, outcomes_up, outcomes_down)

    for r in condition_results[:15]:
        print(f"  {r['condition']:<25} n={r['n']:>4} | "
              f"UP={r['up_accuracy']:.1%} DOWN={r['down_accuracy']:.1%} | "
              f"Best: {r['best_direction']} @ {r['best_accuracy']:.1%}")

    # 5. Find best combinations
    print("\n" + "=" * 60)
    print("COMBINATIONS ACHIEVING 75%+ ACCURACY")
    print("=" * 60)
    combo_results = find_best_combinations(
        df, outcomes_up, outcomes_down,
        condition_results,
        target_accuracy=0.75
    )

    if combo_results:
        for r in combo_results[:20]:
            print(f"  {r['best_direction']} @ {r['best_accuracy']:.1%} (n={r['n']}) | "
                  f"Conditions: {' + '.join(r['conditions'])}")
    else:
        print("  No combinations hit 75% — printing best found:")
        combo_results_all = find_best_combinations(
            df, outcomes_up, outcomes_down,
            condition_results,
            target_accuracy=0.60
        )
        for r in combo_results_all[:10]:
            print(f"  {r['best_direction']} @ {r['best_accuracy']:.1%} (n={r['n']}) | "
                  f"Conditions: {' + '.join(r['conditions'])}")

    # 6. Save results
    output = {
        'run_date':          datetime.now().isoformat(),
        'baseline_up':       baseline_up,
        'baseline_down':     baseline_down,
        'total_days':        len(outcomes_up),
        'individual_conditions': condition_results,
        'best_combinations': combo_results[:20] if combo_results else [],
    }

    with open('backtest_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Results saved to backtest_results.json")
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Baseline accuracy:  UP={baseline_up:.1%} | DOWN={baseline_down:.1%}")
    if combo_results:
        best = combo_results[0]
        print(f"Best combo found:   {best['best_direction']} @ {best['best_accuracy']:.1%} "
              f"(n={best['n']} trades)")
        print(f"Conditions: {' + '.join(best['conditions'])}")
    print("\nNext step: share backtest_results.json and we build the gate from real numbers.")


if __name__ == "__main__":
    main()
