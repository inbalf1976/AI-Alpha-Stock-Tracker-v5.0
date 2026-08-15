"""
WHEAT ACCURACY BACKTEST v2 - WITH TRAIN/HOLDOUT VALIDATION
=============================================================
Run this once in GitHub Actions (or locally) to find which combinations
of conditions actually preceded a 2.5% target hit before a 1.5% stop hit.

CHANGELOG (this version):
  Added a chronological train/holdout split. The original version
  searched hundreds of condition combinations against the FULL dataset
  and reported the single best result — a textbook overfitting setup,
  since testing hundreds of combos against the same data you're
  measuring against will always turn up something that looks like
  100% by chance, even with no real edge.

  Now: combos are DISCOVERED only on the training period (older ~70%
  of the lookback window), then separately RE-TESTED on the holdout
  period (most recent ~30%, which also happens to include this year's
  drought/price action). If a combo's accuracy collapses on the
  holdout, that's your proof it was overfit, not a real edge.

Usage:
  python3 backtest.py

Output:
  - Prints each condition's individual accuracy (on train data)
  - Prints best combinations found on TRAIN, then their REAL accuracy
    on the HOLDOUT they never saw
  - Saves results to backtest_results.json, with both numbers labeled
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import combinations
import json
import warnings
warnings.filterwarnings('ignore')

STOP_PCT      = 0.015
TARGET_PCT    = 0.025
LOOKBACK_DAYS = 730  # 2 years
TRAIN_FRACTION = 0.70  # older 70% = train, most recent 30% = holdout
STABILITY_N_PERIODS = 4  # for the fold-by-fold stability check, see report_stability_by_period()

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

    df['ret_1d']  = df['Close'].pct_change(1)
    df['ret_3d']  = df['Close'].pct_change(3)
    df['ret_5d']  = df['Close'].pct_change(5)

    df['sma20'] = df['Close'].rolling(20).mean()
    df['sma50'] = df['Close'].rolling(50).mean()
    df['above_sma20'] = (df['Close'] > df['sma20']).astype(int)
    df['above_sma50'] = (df['Close'] > df['sma50']).astype(int)
    df['trend_aligned'] = ((df['sma20'] > df['sma50'])).astype(int)

    delta = df['Close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))
    df['rsi_oversold']  = (df['rsi'] < 35).astype(int)
    df['rsi_overbought']= (df['rsi'] > 65).astype(int)
    df['rsi_neutral']   = ((df['rsi'] >= 40) & (df['rsi'] <= 60)).astype(int)

    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['macd']        = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_bullish']= (df['macd'] > df['macd_signal']).astype(int)

    hl   = df['High'] - df['Low']
    hc   = (df['High'] - df['Close'].shift()).abs()
    lc   = (df['Low']  - df['Close'].shift()).abs()
    df['atr']     = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df['atr_pct'] = df['atr'] / df['Close']
    df['vol_ok']  = (df['atr_pct'] > 0.010).astype(int)
    df['vol_not_extreme'] = (df['atr_pct'] < 0.035).astype(int)
    df['vol_good'] = (df['vol_ok'] & df['vol_not_extreme']).astype(int)

    df['vol_avg'] = df['Volume'].rolling(20).mean()
    df['vol_ratio'] = df['Volume'] / df['vol_avg']
    df['vol_high']  = (df['vol_ratio'] > 1.2).astype(int)
    df['vol_low']   = (df['vol_ratio'] < 0.8).astype(int)

    bb_mid   = df['Close'].rolling(20).mean()
    bb_std   = df['Close'].rolling(20).std()
    df['bb_upper'] = bb_mid + 2 * bb_std
    df['bb_lower'] = bb_mid - 2 * bb_std
    df['near_bb_lower'] = (df['Close'] < bb_mid - 1.5 * bb_std).astype(int)
    df['near_bb_upper'] = (df['Close'] > bb_mid + 1.5 * bb_std).astype(int)
    df['inside_bb']     = (~(df['near_bb_lower'].astype(bool) | df['near_bb_upper'].astype(bool))).astype(int)

    corn_aligned = corn['Close'].reindex(df.index, method='ffill')
    soy_aligned  = soy['Close'].reindex(df.index,  method='ffill')
    wc_ratio = df['Close'] / corn_aligned
    ws_ratio = df['Close'] / soy_aligned
    df['wc_zscore'] = (wc_ratio - wc_ratio.rolling(60).mean()) / wc_ratio.rolling(60).std()
    df['ws_zscore'] = (ws_ratio - ws_ratio.rolling(60).mean()) / ws_ratio.rolling(60).std()
    df['wc_bullish'] = (df['wc_zscore'] > 0.5).astype(int)
    df['wc_bearish'] = (df['wc_zscore'] < -0.5).astype(int)

    df['high52'] = df['Close'].rolling(252).max()
    df['low52']  = df['Close'].rolling(252).min()
    df['range_pct'] = (df['Close'] - df['low52']) / (df['high52'] - df['low52'])
    df['in_lower_half'] = (df['range_pct'] < 0.4).astype(int)
    df['in_upper_half'] = (df['range_pct'] > 0.6).astype(int)

    df['momentum_up']   = ((df['ret_1d'] > 0) & (df['ret_3d'] > 0)).astype(int)
    df['momentum_down'] = ((df['ret_1d'] < 0) & (df['ret_3d'] < 0)).astype(int)

    df['month'] = df.index.month
    df['bullish_month'] = df['month'].isin([3, 4, 5, 11, 12]).astype(int)
    df['bearish_month'] = df['month'].isin([6, 7, 8]).astype(int)

    return df.dropna()

# ── validate outcome using next-day hourly bars ───────────────────────────────

def compute_outcomes(daily, hourly):
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

        next_day_mask = (hourly.index.normalize() == exit_window)
        next_bars = hourly[next_day_mask]

        if next_bars.empty:
            continue

        win_up = loss_up = win_down = loss_down = None
        for _, bar in next_bars.iterrows():
            if win_up is None and loss_up is None:
                if bar['Low'] <= stop_up:
                    loss_up = True
                if bar['High'] >= target_up:
                    win_up = True
                if win_up and loss_up:
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

        if win_up is None and loss_up is None:
            final_close = float(next_bars['Close'].iloc[-1])
            win_up = final_close > entry_price

        if win_down is None and loss_down is None:
            final_close = float(next_bars['Close'].iloc[-1])
            win_down = final_close < entry_price

        outcomes_up[entry_date]   = bool(win_up)   if win_up   else False
        outcomes_down[entry_date] = bool(win_down)  if win_down else False

    return pd.Series(outcomes_up), pd.Series(outcomes_down)

# ── train/holdout split ───────────────────────────────────────────────────────

def split_train_holdout(df, outcomes_up, outcomes_down, train_fraction=TRAIN_FRACTION):
    """
    Chronological split — NOT random. Train = older data, holdout =
    most recent data (which naturally includes this year's drought
    and price action). This is the correct way to validate a trading
    signal: you can only ever trade forward in time, never backward,
    so the test must respect that same direction.
    """
    dates = df.index.sort_values()
    split_idx = int(len(dates) * train_fraction)
    split_date = dates[split_idx]

    train_df = df[df.index < split_date]
    holdout_df = df[df.index >= split_date]

    train_up   = outcomes_up[outcomes_up.index < split_date]
    train_down = outcomes_down[outcomes_down.index < split_date]
    holdout_up   = outcomes_up[outcomes_up.index >= split_date]
    holdout_down = outcomes_down[outcomes_down.index >= split_date]

    print(f"\nTrain/Holdout split at {split_date.date()}:")
    print(f"  Train:   {len(train_df)} rows ({train_df.index.min().date()} to {train_df.index.max().date()})")
    print(f"  Holdout: {len(holdout_df)} rows ({holdout_df.index.min().date()} to {holdout_df.index.max().date()})")

    return train_df, holdout_df, train_up, train_down, holdout_up, holdout_down

# ── backtest each condition ───────────────────────────────────────────────────

def backtest_conditions(df, outcomes_up, outcomes_down):
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

# ── find best combinations (TRAIN ONLY) ───────────────────────────────────────

def find_best_combinations(df, outcomes_up, outcomes_down, top_conditions, target_accuracy=0.75):
    df_aligned    = df[df.index.isin(outcomes_up.index)].copy()
    up_outcomes   = outcomes_up.reindex(df_aligned.index)
    down_outcomes = outcomes_down.reindex(df_aligned.index)

    top_conds = [r['condition'] for r in top_conditions[:12]]
    combo_results = []

    print(f"\nTesting combinations of top {len(top_conds)} conditions (TRAIN data only)...")

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

# ── re-test INDIVIDUAL conditions on holdout data (was missing before) ────────

def evaluate_condition_on_holdout(holdout_df, holdout_up, holdout_down, condition):
    """
    Same validation logic as combos, but for a single condition.
    This was the missing piece: individual conditions were reported
    from TRAIN data only, with no check for whether they hold up on
    data they never saw. A condition that looks solid on train but
    collapses on holdout is just as untrustworthy as an overfit combo
    — it was just discovered with a smaller search, not zero search.
    """
    df_aligned = holdout_df[holdout_df.index.isin(holdout_up.index)].copy()
    up_outcomes   = holdout_up.reindex(df_aligned.index)
    down_outcomes = holdout_down.reindex(df_aligned.index)

    if condition not in df_aligned.columns:
        return {'n': 0, 'up_accuracy': None, 'down_accuracy': None, 'note': 'Condition column missing'}

    mask = df_aligned[condition] == 1
    n = int(mask.sum())
    if n == 0:
        return {'n': 0, 'up_accuracy': None, 'down_accuracy': None, 'note': 'Condition never occurred in holdout period'}

    up_acc   = float(up_outcomes[mask].mean())
    down_acc = float(down_outcomes[mask].mean())

    return {
        'n': n,
        'up_accuracy':   round(up_acc, 3),
        'down_accuracy': round(down_acc, 3),
        'best_direction': 'UP' if up_acc > down_acc else 'DOWN',
        'best_accuracy':  round(max(up_acc, down_acc), 3),
    }

# ── re-test a combo on holdout data it never saw ──────────────────────────────

def evaluate_combo_on_holdout(holdout_df, holdout_up, holdout_down, conditions):
    """
    THE KEY VALIDATION STEP. Takes a combo that was found to look good
    on TRAIN data, and checks its real accuracy on HOLDOUT data it was
    never fitted to. If accuracy collapses here, the combo was overfit
    noise, not a real edge — regardless of how good it looked on train.
    """
    df_aligned = holdout_df[holdout_df.index.isin(holdout_up.index)].copy()
    up_outcomes   = holdout_up.reindex(df_aligned.index)
    down_outcomes = holdout_down.reindex(df_aligned.index)

    mask = pd.Series(True, index=df_aligned.index)
    for c in conditions:
        if c in df_aligned.columns:
            mask = mask & (df_aligned[c] == 1)

    n = int(mask.sum())
    if n == 0:
        return {'n': 0, 'up_accuracy': None, 'down_accuracy': None, 'note': 'Condition never occurred in holdout period'}

    up_acc   = float(up_outcomes[mask].mean())
    down_acc = float(down_outcomes[mask].mean())

    return {
        'n': n,
        'up_accuracy':   round(up_acc, 3),
        'down_accuracy': round(down_acc, 3),
        'best_direction': 'UP' if up_acc > down_acc else 'DOWN',
        'best_accuracy':  round(max(up_acc, down_acc), 3),
    }

# ── stability check: does accuracy hold across different time periods? ────────
# ADDED 2026-08-15 — purely additive reporting, does not change which
# conditions get validated or feed generate_validated_conditions.py.
# The existing train/holdout split gives ONE holdout accuracy number
# per condition — this can't distinguish "consistently good" from
# "got lucky/unlucky in this particular stretch." This function
# splits the FULL lookback window into several chronological periods
# and reports each already-validated condition's accuracy in each
# period separately, using the SAME direction chosen on train (never
# re-picks the best direction per period — that would let a condition
# "flip" to whichever direction looks good in each slice, which would
# silently reintroduce the exact overfitting this file's whole
# train/holdout design exists to catch).

def report_stability_by_period(df, outcomes_up, outcomes_down, conditions_with_direction, n_periods=STABILITY_N_PERIODS):
    """
    conditions_with_direction: list of {'condition': str, 'best_direction': 'UP'/'DOWN'}
    — normally the conditions that already survived the train/holdout
    check (see main()). Reporting only; returns a list of dicts
    suitable for saving to backtest_results.json alongside (not
    replacing) the existing train/holdout fields.
    """
    df_aligned    = df[df.index.isin(outcomes_up.index)].copy()
    up_outcomes   = outcomes_up.reindex(df_aligned.index)
    down_outcomes = outcomes_down.reindex(df_aligned.index)

    dates = df_aligned.index.sort_values()
    period_chunks = np.array_split(dates, n_periods)

    print(f"\n{'Condition [dir]':<26} | " + " | ".join(f"Period {i+1}" for i in range(n_periods)))
    print("-" * (28 + 16 * n_periods))

    results = []
    for cw in conditions_with_direction:
        cond, direction = cw['condition'], cw['best_direction']
        if cond not in df_aligned.columns:
            continue
        outcomes = up_outcomes if direction == 'UP' else down_outcomes

        period_records = []
        display_cells = []
        for chunk in period_chunks:
            in_period = df_aligned.index.isin(chunk)
            mask = (df_aligned[cond] == 1) & in_period
            n = int(mask.sum())
            acc = round(float(outcomes[mask].mean()), 3) if n > 0 else None

            period_records.append({
                'start': str(pd.Timestamp(chunk[0]).date()) if len(chunk) else None,
                'end':   str(pd.Timestamp(chunk[-1]).date()) if len(chunk) else None,
                'n': n,
                'accuracy': acc,
            })
            display_cells.append(f"{acc:.0%}(n={n})" if acc is not None else "  n/a  ")

        valid_accs = [p['accuracy'] for p in period_records if p['accuracy'] is not None]
        spread = round(max(valid_accs) - min(valid_accs), 3) if len(valid_accs) >= 2 else None
        flag = "  ⚠️ unstable across periods" if (spread is not None and spread > 0.30) else ""

        print(f"{cond + ' [' + direction + ']':<26} | " + " | ".join(f"{c:<10}" for c in display_cells) + flag)

        results.append({
            'condition': cond,
            'direction': direction,
            'periods': period_records,
            'spread': spread,
        })

    return results

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("WHEAT ACCURACY BACKTEST v2 - TRAIN/HOLDOUT VALIDATION")
    print(f"Stop: {STOP_PCT:.1%} | Target: {TARGET_PCT:.1%} | Lookback: {LOOKBACK_DAYS}d")
    print(f"Train fraction: {TRAIN_FRACTION:.0%} (older) | Holdout: {1-TRAIN_FRACTION:.0%} (most recent)")
    print("=" * 60)

    daily, hourly, corn, soy = fetch_data()

    print("\nBuilding features...")
    df = build_features(daily, corn, soy)
    print(f"Feature rows: {len(df)}")

    print("\nComputing outcomes (stop/target hits)...")
    outcomes_up, outcomes_down = compute_outcomes(daily, hourly)
    print(f"Outcomes computed: {len(outcomes_up)}")

    baseline_up   = float(outcomes_up.mean())
    baseline_down = float(outcomes_down.mean())
    print(f"\nBaseline (full data, no filter): UP={baseline_up:.1%} DOWN={baseline_down:.1%}")

    # ── SPLIT ──
    train_df, holdout_df, train_up, train_down, holdout_up, holdout_down = \
        split_train_holdout(df, outcomes_up, outcomes_down)

    # ── Individual conditions — TRAIN ONLY ──
    print("\n" + "=" * 60)
    print("INDIVIDUAL CONDITION ACCURACY (TRAIN DATA)")
    print("=" * 60)
    condition_results = backtest_conditions(train_df, train_up, train_down)

    for r in condition_results[:15]:
        print(f"  {r['condition']:<25} n={r['n']:>4} | "
              f"UP={r['up_accuracy']:.1%} DOWN={r['down_accuracy']:.1%} | "
              f"Best: {r['best_direction']} @ {r['best_accuracy']:.1%}")

    # ── Individual conditions — HOLDOUT VALIDATION (the missing piece) ──
    print("\n" + "=" * 60)
    print("INDIVIDUAL CONDITIONS — DO THEY SURVIVE HOLDOUT?")
    print("=" * 60)
    print(f"{'TRAIN result':<35} | {'HOLDOUT result (real test)'}")
    print("-" * 75)

    validated_conditions = []
    for r in condition_results[:15]:
        holdout_eval = evaluate_condition_on_holdout(holdout_df, holdout_up, holdout_down, r['condition'])

        train_str = f"{r['condition']} {r['best_direction']} @ {r['best_accuracy']:.1%} (n={r['n']})"
        if holdout_eval['n'] == 0:
            holdout_str = "NEVER OCCURRED / missing in holdout"
        else:
            holdout_str = f"{holdout_eval['best_direction']} @ {holdout_eval['best_accuracy']:.1%} (n={holdout_eval['n']})"

        flag = ""
        if holdout_eval['n'] > 0 and holdout_eval['best_accuracy'] is not None:
            drop = r['best_accuracy'] - holdout_eval['best_accuracy']
            if drop > 0.25:
                flag = "  ⚠️ LIKELY OVERFIT (big drop)"
            elif drop > 0.10:
                flag = "  ⚠️ accuracy dropped meaningfully"
            elif drop < -0.05:
                flag = "  ✓ held up (even improved on holdout)"
            else:
                flag = "  ✓ held up reasonably well"

        print(f"{train_str:<35} | {holdout_str}{flag}")

        # 'held_up' feeds the stability-by-period check below — only
        # conditions that already passed the existing holdout test are
        # worth checking for period-to-period consistency. Matches the
        # same threshold as the "✓" flags above (drop <= 0.10).
        held_up = (
            holdout_eval['n'] > 0
            and holdout_eval['best_accuracy'] is not None
            and (r['best_accuracy'] - holdout_eval['best_accuracy']) <= 0.10
        )

        validated_conditions.append({
            'condition': r['condition'],
            'train': r,
            'holdout': holdout_eval,
            'held_up': held_up,
        })

    # ── Stability check: does accuracy hold across different periods? ──
    # ADDED 2026-08-15, reporting only — see report_stability_by_period()
    # docstring. Only checks conditions that already survived the
    # existing train/holdout test above.
    print("\n" + "=" * 60)
    print("STABILITY CHECK — ACCURACY ACROSS DIFFERENT TIME PERIODS")
    print("=" * 60)
    print(f"(Splits the full {LOOKBACK_DAYS}-day lookback into {STABILITY_N_PERIODS} periods.")
    print(" Reporting only — does not change which conditions are validated.")
    print(" A condition swinging wildly between periods is less trustworthy")
    print(" than one with similar accuracy in every period, even if both")
    print(" have the same overall holdout number.)")

    held_up_conditions = [
        {'condition': v['condition'], 'best_direction': v['train']['best_direction']}
        for v in validated_conditions if v['held_up']
    ]
    if held_up_conditions:
        stability_results = report_stability_by_period(df, outcomes_up, outcomes_down, held_up_conditions)
    else:
        stability_results = []
        print("  No conditions held up on holdout this run — nothing to check for stability.")

    # ── Best combinations — TRAIN ONLY ──
    print("\n" + "=" * 60)
    print("COMBINATIONS ACHIEVING 75%+ ACCURACY (TRAIN DATA)")
    print("=" * 60)
    combo_results = find_best_combinations(
        train_df, train_up, train_down,
        condition_results,
        target_accuracy=0.75
    )

    if not combo_results:
        print("  No combinations hit 75% on train — lowering threshold to 60%")
        combo_results = find_best_combinations(
            train_df, train_up, train_down,
            condition_results,
            target_accuracy=0.60
        )

    # ── THE KEY STEP: re-test top train combos on HOLDOUT ──
    print("\n" + "=" * 60)
    print("HOLDOUT VALIDATION — DOES IT SURVIVE UNSEEN DATA?")
    print("=" * 60)
    print(f"{'TRAIN result':<35} | {'HOLDOUT result (real test)'}")
    print("-" * 75)

    validated_results = []
    for r in combo_results[:15]:
        holdout_eval = evaluate_combo_on_holdout(holdout_df, holdout_up, holdout_down, r['conditions'])

        train_str = f"{r['best_direction']} @ {r['best_accuracy']:.1%} (n={r['n']})"
        if holdout_eval['n'] == 0:
            holdout_str = "NEVER OCCURRED in holdout"
        else:
            holdout_str = f"{holdout_eval['best_direction']} @ {holdout_eval['best_accuracy']:.1%} (n={holdout_eval['n']})"

        # Flag large drops as likely overfit
        flag = ""
        if holdout_eval['n'] > 0 and holdout_eval['best_accuracy'] is not None:
            drop = r['best_accuracy'] - holdout_eval['best_accuracy']
            if drop > 0.25:
                flag = "  ⚠️ LIKELY OVERFIT (big drop)"
            elif drop > 0.10:
                flag = "  ⚠️ accuracy dropped meaningfully"

        print(f"{train_str:<35} | {holdout_str}{flag}")
        print(f"  Conditions: {' + '.join(r['conditions'])}")

        validated_results.append({
            'conditions': r['conditions'],
            'train': r,
            'holdout': holdout_eval,
        })

    # ── Save everything, clearly labeled ──
    output = {
        'run_date':          datetime.now().isoformat(),
        'baseline_up_full':  baseline_up,
        'baseline_down_full': baseline_down,
        'train_period':      f"{train_df.index.min().date()} to {train_df.index.max().date()}",
        'holdout_period':    f"{holdout_df.index.min().date()} to {holdout_df.index.max().date()}",
        'individual_conditions_train': condition_results,
        'individual_conditions_train_and_holdout': validated_conditions,
        'combinations_train_and_holdout': validated_results,
        'stability_by_period': stability_results,
    }

    with open('backtest_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n✅ Results saved to backtest_results.json (now includes train AND holdout numbers)")
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Baseline (full data): UP={baseline_up:.1%} | DOWN={baseline_down:.1%}")
    print("\nOnly trust a combo for ConvictionGate if its HOLDOUT accuracy is")
    print("close to its TRAIN accuracy. A big gap between the two columns")
    print("above means that combo was fitted to noise, not a real pattern —")
    print("do not wire it into wheat_monitor_pro.py's ConvictionGate.")


if __name__ == "__main__":
    main()
