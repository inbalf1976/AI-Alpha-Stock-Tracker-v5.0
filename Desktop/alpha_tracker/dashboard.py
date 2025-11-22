import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import requests
import plotly.graph_objs as go
import time
import threading
import os
import warnings
import joblib
import json
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
import traceback
from enum import Enum
from typing import Tuple, List
import socket  # <-- STEP 1: Added exactly at line 20
# ... [all previous code unchanged up to get_error_statistics()] ...

def get_error_statistics():
    """Get error statistics with robust file handling"""
    try:
        if not ERROR_LOG_PATH.exists():
            return {"total": 0, "by_severity": {}, "recent": []}
       
        with open(ERROR_LOG_PATH, 'r') as f:
            errors = json.load(f)
       
        by_severity = {}
        for error in errors:
            sev = error.get('severity', 'UNKNOWN')
            by_severity[sev] = by_severity.get(sev, 0) + 1
       
        return {"total": len(errors), "by_severity": by_severity, "recent": errors[-10:]}
    except (json.JSONDecodeError, FileNotFoundError, Exception) as e:
        logger.warning(f"Error reading statistics: {e}")
        return {"total": 0, "by_severity": {}, "recent": []}

# ================================
# ACCESS CONTROL SYSTEM
# ================================

def is_local_user():
    """
    Determine if the user is accessing locally (PC) or remotely (online).
    Returns True for local access, False for online/remote access.
    """
    try:
        # Check if running on Streamlit Cloud or similar hosting
        if os.getenv('STREAMLIT_SHARING_MODE') or os.getenv('STREAMLIT_SERVER_HEADLESS'):
            return False
        
        # Check if we're in a cloud environment
        cloud_indicators = [
            'HEROKU', 'AWS', 'AZURE', 'GOOGLE_CLOUD', 
            'STREAMLIT_CLOUD', 'RENDER', 'RAILWAY'
        ]
        if any(os.getenv(indicator) for indicator in cloud_indicators):
            return False
        
        # If we can get the hostname and it's localhost, we're local
        hostname = socket.gethostname()
        if hostname in ['localhost', '127.0.0.1', 'DESKTOP', 'LAPTOP'] or hostname.startswith('DESKTOP-') or hostname.startswith('LAPTOP-'):
            return True
        
        # Default to local for development (when no cloud indicators present)
        return True
        
    except Exception as e:
        logger.warning(f"Could not determine user location: {e}")
        # Default to restricting access if we can't determine
        return False

def require_local_access(function_name):
    """
    Decorator/check function to restrict access to local users only.
    Returns True if access allowed, False if blocked.
    """
    if not is_local_user():
        st.error(f"Access Restricted: '{function_name}' is only available for local PC users.")
        st.info("This function is disabled for online users for security and system stability reasons.")
        logger.warning(f"Blocked online user from accessing: {function_name}")
        return False
    return True

# ... [rest of code continues] ...

# Later in sidebar — all 7 buttons now protected:

    if st.button("Force Retrain", use_container_width=True):
        if require_local_access("Force Retrain"):  # <-- Protected
            with st.spinner("Retraining..."):
                try:
                    result = train_self_learning_model(ticker, days=1, force_retrain=True)
                    if result[0] is not None:
                        st.success("Retrained!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Retraining failed - check error logs")
                except Exception as e:
                    log_error(ErrorSeverity.ERROR, "force_retrain", e, ticker=ticker, user_message="Retrain failed")

    if st.button("Bootstrap All Models", use_container_width=True):
        if require_local_access("Bootstrap All Models"):  # <-- Protected
            with st.spinner("Training all models... This will take 5-10 minutes"):
                # ... bootstrap logic ...

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start", use_container_width=True, key="daemon_start"):
            if require_local_access("Start Learning Daemon"):  # <-- Protected
                try:
                    save_daemon_config(True)
                    threading.Thread(target=continuous_learning_daemon, daemon=True).start()
                    st.success("Started!")
                    time.sleep(0.5)
                    st.rerun()

    with col2:
        if st.button("Stop", use_container_width=True, key="daemon_stop"):
            if require_local_access("Stop Learning Daemon"):  # <-- Protected
                try:
                    save_daemon_config(False)
                    st.success("Stopped!")
                    time.sleep(0.5)
                    st.rerun()

    if st.button("Test Telegram", use_container_width=True):
        if require_local_access("Test Telegram"):  # <-- Protected
            try:
                success = send_telegram_alert("TEST ALERT\nAI - Alpha Tracker v4.1")
                st.success("Sent!") if success else st.error("Check keys")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Alerts", use_container_width=True):
            if require_local_access("Start Alerts"):  # <-- Protected
                try:
                    save_monitoring_config(True)
                    threading.Thread(target=monitor_6percent_pre_move, daemon=True).start()
                    st.success("Started!")
                    time.sleep(0.5)
                    st.rerun()

    with col2:
        if st.button("Stop Alerts", use_container_width=True):
            if require_local_access("Stop Alerts"):  # <-- Protected
                try:
                    save_monitoring_config(False)
                    st.success("Stopped!")
                    time.sleep(0.5)
                    st.rerun()
