"""
system_monitor.py - System monitoring and resource management
"""

import psutil
import gc
from typing import Dict, List, Tuple
from dataclasses import dataclass
import time

@dataclass
class SystemMetrics:
    memory_mb: float
    cpu_percent: float
    disk_percent: float
    thread_count: int
    timestamp: float

class SystemMonitor:
    def __init__(self, warning_threshold_mb=1000, critical_threshold_mb=2000):
        self.process = psutil.Process()
        self.warning_threshold = warning_threshold_mb
        self.critical_threshold = critical_threshold_mb
        self.metrics_history: List[SystemMetrics] = []
        
    def collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        cpu_percent = self.process.cpu_percent(interval=0.1)
        disk_percent = psutil.disk_usage('/').percent
        
        metrics = SystemMetrics(
            memory_mb=memory_mb,
            cpu_percent=cpu_percent,
            disk_percent=disk_percent,
            thread_count=len(self.process.threads()),
            timestamp=time.time()
        )
        
        self.metrics_history.append(metrics)
        # Keep last 100 metrics
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
            
        return metrics
    
    def check_memory(self) -> Tuple[str, str]:
        """Check memory status"""
        metrics = self.collect_metrics()
        
        if metrics.memory_mb > self.critical_threshold:
            return "critical", f"Memory usage critical: {metrics.memory_mb:.0f}MB"
        elif metrics.memory_mb > self.warning_threshold:
            return "warning", f"Memory usage high: {metrics.memory_mb:.0f}MB"
        else:
            return "ok", f"Memory usage normal: {metrics.memory_mb:.0f}MB"
    
    def optimize_memory(self):
        """Optimize memory usage"""
        gc.collect()
        # Clear TensorFlow/Keras sessions if available
        try:
            from tensorflow.keras import backend as K
            K.clear_session()
        except:
            pass
