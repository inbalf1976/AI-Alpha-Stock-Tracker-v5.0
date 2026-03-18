name: Wheat Monitor PRO - Ultimate Edition
on:
  schedule:
    - cron: '0 1 * * 1-5'  # 01:00 UTC = 03:00 Israel time, weekdays only (Mon-Fri)
  workflow_dispatch:
  push:
    branches: [ main, master ]
    paths-ignore:
      - 'wheat_monitor_state.json'
jobs:
  monitor:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    permissions:
      contents: write
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Setup Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.10'
    
    - name: Install packages
      run: |
        pip install yfinance pandas numpy scikit-learn requests
        pip install tensorflow-cpu keras
        pip install xgboost beautifulsoup4 lxml
    
    - name: Run Professional Monitor
      env:
        TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
        TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
      run: python wheat_monitor_pro.py
    
    - name: Commit state to repository
      run: |
        git config --local user.email "github-actions[bot]@users.noreply.github.com"
        git config --local user.name "github-actions[bot]"
        git add -f wheat_monitor_state.json || true
        git diff --quiet && git diff --staged --quiet || git commit -m "Update monitor state [skip ci]"
        git push || echo "Nothing to push"
