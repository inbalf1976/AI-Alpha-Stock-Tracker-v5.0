name: V5 Spike Direction Backtest

on:
  workflow_dispatch:
    inputs:
      period:
        description: "yfinance intraday backtest period"
        required: false
        default: "60d"
      horizon_bars:
        description: "15m bars evaluated after each confirmed spike"
        required: false
        default: "32"
  schedule:
    - cron: "30 14 * * 1"

permissions:
  contents: read

jobs:
  backtest:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install yfinance pandas numpy requests scikit-learn

      - name: Run V5 directional spike backtest
        env:
          BACKTEST_PERIOD: ${{ inputs.period || '60d' }}
          SPIKE_HORIZON_BARS: ${{ inputs.horizon_bars || '32' }}
          PYTHONUNBUFFERED: "1"
        run: |
          python backtest_v5_spike.py

      - name: Upload backtest report
        uses: actions/upload-artifact@v4
        with:
          name: v5-spike-backtest
          path: |
            backtest_v5_spike_report.json
            backtest_v5_spike_trades.csv
          if-no-files-found: error
          retention-days: 30
