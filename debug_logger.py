"""
Advanced Debug Logger for Wheat Trading System
Catches bugs, logs everything, shows EXACT error locations

Features:
- Logs to file with timestamps and line numbers
- Catches exceptions with full traceback
- Sends critical errors to Telegram
- Tracks system health metrics
- Error summary JSON for analysis
"""

import logging
import traceback
import sys
from datetime import datetime
from pathlib import Path
import json
import os

class DebugLogger:
    """
    Comprehensive debug logger that:
    1. Logs everything to timestamped files
    2. Catches exceptions with full traceback
    3. Shows exact line numbers where errors occur
    4. Sends critical errors to Telegram
    5. Tracks system health and error trends
    """
    
    def __init__(self, log_dir="logs", telegram_alerts=True):
        """
        Initialize debug logger
        
        Args:
            log_dir: Directory to store log files
            telegram_alerts: Send critical errors to Telegram
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.telegram_alerts = telegram_alerts
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"wheat_monitor_{timestamp}.log"
        
        # Create error summary file
        self.error_file = self.log_dir / "errors.json"
        
        # Create health metrics file
        self.health_file = self.log_dir / "system_health.json"
        
        # Setup Python logging
        self._setup_logging()
        
        # Counters
        self.error_count = 0
        self.warning_count = 0
        self.start_time = datetime.now()
        
        # Track API calls
        self.api_calls = {
            'weather': 0,
            'wasde': 0,
            'yfinance': 0,
            'telegram': 0
        }
        
        self.info("🚀 Debug Logger initialized")
        self.info(f"📁 Log file: {self.log_file}")
    
    def _setup_logging(self):
        """Setup Python's logging module with custom formatting"""
        # Create logger
        self.logger = logging.getLogger('WheatMonitor')
        self.logger.setLevel(logging.DEBUG)
        
        # Remove any existing handlers
        self.logger.handlers = []
        
        # File handler - EVERYTHING goes here (DEBUG and above)
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler - Only INFO and above
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Detailed formatter with line numbers
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(funcName)s() | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Simpler console formatter
        console_formatter = logging.Formatter(
            '%(levelname)-8s | %(message)s'
        )
        
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def debug(self, message):
        """
        Log debug message (verbose detail, only in file)
        Use for: Variable values, loop iterations, detailed state
        """
        self.logger.debug(message)
    
    def info(self, message):
        """
        Log info message (normal operation)
        Use for: Progress updates, successful operations
        """
        self.logger.info(message)
    
    def warning(self, message):
        """
        Log warning (something unusual but not critical)
        Use for: Degraded operation, unusual values, retry attempts
        """
        self.logger.warning(f"⚠️  {message}")
        self.warning_count += 1
    
    def error(self, message, exception=None, send_telegram=False):
        """
        Log error with full details
        
        Args:
            message: Error description
            exception: Exception object (if available)
            send_telegram: Send error alert to Telegram
        
        Example:
            try:
                risky_operation()
            except Exception as e:
                logger.error("Operation failed", exception=e, send_telegram=True)
        """
        self.error_count += 1
        
        # Get caller information
        frame = sys._getframe(1)
        filename = Path(frame.f_code.co_filename).name
        line_number = frame.f_lineno
        function_name = frame.f_code.co_name
        
        error_detail = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'file': filename,
            'line': line_number,
            'function': function_name,
            'exception': None,
            'exception_type': None,
            'traceback': None
        }
        
        # Log to console/file
        self.logger.error(f"❌ {message}")
        self.logger.error(f"   📍 Location: {filename}:{line_number} in {function_name}()")
        
        # Add exception details if provided
        if exception:
            error_detail['exception'] = str(exception)
            error_detail['exception_type'] = type(exception).__name__
            error_detail['traceback'] = traceback.format_exc()
            
            self.logger.error(f"   🐛 Exception: {type(exception).__name__}: {exception}")
            
            # Log full traceback to file only (not console)
            file_only_logger = logging.getLogger('WheatMonitor.FileOnly')
            file_only_logger.setLevel(logging.DEBUG)
            if not file_only_logger.handlers:
                file_handler = logging.FileHandler(self.log_file)
                file_handler.setFormatter(logging.Formatter('%(message)s'))
                file_only_logger.addHandler(file_handler)
            
            file_only_logger.error("   📋 Full traceback:")
            file_only_logger.error(traceback.format_exc())
        
        # Save to error summary file
        self._save_error_summary(error_detail)
        
        # Send to Telegram if critical
        if send_telegram and self.telegram_alerts:
            self._send_telegram_error(error_detail)
    
    def critical(self, message, exception=None):
        """
        Log critical error (system-breaking)
        Automatically sends Telegram alert
        
        Use for: System crashes, data unavailable, complete failures
        
        Example:
            if df is None:
                logger.critical("No wheat data - cannot run predictions!")
        """
        self.logger.critical(f"🚨 CRITICAL: {message}")
        self.error(message, exception, send_telegram=True)
    
    def exception(self, message):
        """
        Log exception with automatic traceback capture
        Use this inside except blocks
        
        Example:
            try:
                risky_operation()
            except:
                logger.exception("Operation failed")
        """
        exc_type, exc_value, exc_traceback = sys.exc_info()
        
        if exc_type is None:
            self.warning(f"exception() called but no exception active: {message}")
            return
        
        self.error_count += 1
        
        # Get detailed traceback
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        tb_string = ''.join(tb_lines)
        
        # Extract error location
        tb_list = traceback.extract_tb(exc_traceback)
        if tb_list:
            last_call = tb_list[-1]
            filename = Path(last_call.filename).name
            line_number = last_call.lineno
            function_name = last_call.name
        else:
            filename = "unknown"
            line_number = 0
            function_name = "unknown"
        
        error_detail = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'file': filename,
            'line': line_number,
            'function': function_name,
            'exception': str(exc_value),
            'exception_type': exc_type.__name__ if exc_type else 'Unknown',
            'traceback': tb_string
        }
        
        # Log to console
        self.logger.error(f"❌ EXCEPTION: {message}")
        self.logger.error(f"   🐛 Type: {exc_type.__name__}")
        self.logger.error(f"   📍 Location: {filename}:{line_number} in {function_name}()")
        self.logger.error(f"   💬 Details: {exc_value}")
        
        # Log full traceback to file only
        file_only_logger = logging.getLogger('WheatMonitor.FileOnly')
        if not file_only_logger.handlers:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(logging.Formatter('%(message)s'))
            file_only_logger.addHandler(file_handler)
        
        file_only_logger.error("   📋 Full traceback:")
        file_only_logger.error(tb_string)
        
        # Save to error file
        self._save_error_summary(error_detail)
        
        return error_detail
    
    def track_api_call(self, api_name, success=True):
        """
        Track API calls for monitoring
        
        Args:
            api_name: 'weather', 'wasde', 'yfinance', 'telegram'
            success: Whether call succeeded
        """
        if api_name in self.api_calls:
            self.api_calls[api_name] += 1
        
        if success:
            self.debug(f"✓ API call: {api_name}")
        else:
            self.warning(f"✗ API call failed: {api_name}")
    
    def section(self, title):
        """
        Log a section header (for readability)
        
        Example:
            logger.section("FETCHING DATA")
        """
        separator = "=" * 60
        self.info(f"\n{separator}")
        self.info(f"  {title}")
        self.info(f"{separator}")
    
    def _save_error_summary(self, error_detail):
        """Save error to summary JSON file"""
        try:
            # Load existing errors
            if self.error_file.exists():
                with open(self.error_file, 'r') as f:
                    errors = json.load(f)
            else:
                errors = []
            
            # Add new error
            errors.append(error_detail)
            
            # Keep only last 100 errors
            if len(errors) > 100:
                errors = errors[-100:]
            
            # Save
            with open(self.error_file, 'w') as f:
                json.dump(errors, f, indent=2)
        
        except Exception as e:
            # Don't use logger here to avoid infinite loop
            print(f"Failed to save error summary: {e}")
    
    def _send_telegram_error(self, error_detail):
        """Send critical error alert to Telegram"""
        if not self.telegram_bot_token or not self.telegram_chat_id:
            self.debug("Telegram not configured, skipping error alert")
            return
        
        try:
            import requests
            
            # Format error message
            message = f"""
🚨 *CRITICAL ERROR* 🚨

{error_detail['message']}

📍 *Location:*
File: `{error_detail['file']}`
Line: {error_detail['line']}
Function: `{error_detail['function']}()`

⏰ *Time:* {error_detail['timestamp']}

🐛 *Exception:*
`{error_detail['exception_type']}: {error_detail['exception']}`

📋 Check full logs: `{self.log_file.name}`
"""
            
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            data = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, data=data, timeout=10)
            
            if response.status_code == 200:
                self.debug("✓ Telegram error alert sent")
            else:
                self.debug(f"✗ Telegram error alert failed: {response.status_code}")
        
        except Exception as e:
            # Don't use logger here to avoid infinite loop
            print(f"Failed to send Telegram error: {e}")
    
    def save_health_metrics(self):
        """
        Save system health metrics to file
        Call this at end of each run
        """
        try:
            runtime = (datetime.now() - self.start_time).total_seconds()
            
            health = {
                'timestamp': datetime.now().isoformat(),
                'runtime_seconds': runtime,
                'errors': self.error_count,
                'warnings': self.warning_count,
                'api_calls': self.api_calls,
                'log_file': str(self.log_file),
                'status': 'FAILED' if self.error_count > 0 else 'SUCCESS'
            }
            
            # Load history
            if self.health_file.exists():
                with open(self.health_file, 'r') as f:
                    history = json.load(f)
            else:
                history = []
            
            # Add current run
            history.append(health)
            
            # Keep last 50 runs
            if len(history) > 50:
                history = history[-50:]
            
            # Save
            with open(self.health_file, 'w') as f:
                json.dump(history, f, indent=2)
            
            self.info(f"💚 Health metrics saved: {self.error_count} errors, {self.warning_count} warnings")
        
        except Exception as e:
            print(f"Failed to save health metrics: {e}")
    
    def get_summary(self):
        """
        Get summary of current run
        
        Returns:
            dict with run statistics
        """
        runtime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'runtime_seconds': runtime,
            'errors': self.error_count,
            'warnings': self.warning_count,
            'api_calls': self.api_calls.copy(),
            'log_file': str(self.log_file)
        }
    
    def close(self):
        """
        Close logger and save final metrics
        Call this at end of script
        """
        self.save_health_metrics()
        
        runtime = (datetime.now() - self.start_time).total_seconds()
        
        self.info(f"\n{'='*60}")
        self.info(f"📊 RUN SUMMARY:")
        self.info(f"   Runtime: {runtime:.1f}s")
        self.info(f"   Errors: {self.error_count}")
        self.info(f"   Warnings: {self.warning_count}")
        self.info(f"   Status: {'❌ FAILED' if self.error_count > 0 else '✅ SUCCESS'}")
        self.info(f"{'='*60}\n")
        
        # Flush and close handlers
        for handler in self.logger.handlers:
            handler.flush()
            handler.close()


# Convenience function to create logger
def create_logger(log_dir="logs", telegram_alerts=True):
    """
    Create and return a debug logger instance
    
    Args:
        log_dir: Directory for log files
        telegram_alerts: Enable Telegram alerts for critical errors
    
    Returns:
        DebugLogger instance
    
    Example:
        logger = create_logger()
        logger.info("System starting...")
    """
    return DebugLogger(log_dir=log_dir, telegram_alerts=telegram_alerts)


# Example usage
if __name__ == "__main__":
    # Create logger
    logger = create_logger()
    
    # Normal logging
    logger.info("Starting wheat monitor...")
    logger.debug("Debug info: price = 620.25")
    
    # Warning
    logger.warning("Unusual volume detected")
    
    # Error without exception
    logger.error("Failed to fetch weather data")
    
    # Error with exception
    try:
        result = 10 / 0
    except Exception as e:
        logger.error("Division error", exception=e)
    
    # Exception with automatic traceback
    try:
        data = {'key': 'value'}
        print(data['nonexistent'])
    except:
        logger.exception("Dictionary key error")
    
    # Critical error (auto-sends Telegram)
    logger.critical("System cannot continue!")
    
    # API tracking
    logger.track_api_call('weather', success=True)
    logger.track_api_call('wasde', success=False)
    
    # Section headers
    logger.section("MAKING PREDICTION")
    logger.info("Ensemble models trained")
    
    # Get summary
    summary = logger.get_summary()
    print(f"\nRun summary: {summary}")
    
    # Close logger
    logger.close()
