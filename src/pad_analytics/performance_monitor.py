"""
Performance monitoring and metrics for PAD Analytics.

This module provides performance tracking, timing, memory usage monitoring,
and metrics collection for various operations in the PAD Analytics pipeline.
"""

import time
import psutil
import threading
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import functools


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    operation: str
    start_time: float
    end_time: float
    duration: float
    memory_start_mb: float
    memory_end_mb: float
    memory_peak_mb: float
    cpu_percent: float
    batch_size: Optional[int] = None
    items_processed: Optional[int] = None
    errors: int = 0
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def throughput(self) -> Optional[float]:
        """Calculate throughput (items per second)."""
        if self.items_processed and self.duration > 0:
            return self.items_processed / self.duration
        return None
    
    @property
    def memory_usage_mb(self) -> float:
        """Calculate memory usage during operation."""
        return self.memory_end_mb - self.memory_start_mb
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        result = asdict(self)
        result['throughput'] = self.throughput
        result['memory_usage_mb'] = self.memory_usage_mb
        return result


class PerformanceMonitor:
    """
    Performance monitoring and metrics collection system.
    
    Tracks timing, memory usage, CPU usage, and throughput for
    various operations in the PAD Analytics pipeline.
    """
    
    def __init__(self, enable_detailed_monitoring: bool = True):
        """
        Initialize the performance monitor.
        
        Args:
            enable_detailed_monitoring: Whether to track detailed metrics
        """
        self.enable_detailed_monitoring = enable_detailed_monitoring
        self.metrics_history: List[PerformanceMetrics] = []
        self._lock = threading.Lock()
        self._active_operations: Dict[str, Dict[str, Any]] = {}
        
        # System monitoring
        self.process = psutil.Process()
        
    def start_operation(self, operation: str, batch_size: Optional[int] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Start monitoring a new operation.
        
        Args:
            operation: Name/description of the operation
            batch_size: Size of the batch being processed
            metadata: Additional metadata for the operation
            
        Returns:
            Operation ID for tracking
        """
        if not self.enable_detailed_monitoring:
            return operation
        
        operation_id = f"{operation}_{time.time():.6f}"
        
        with self._lock:
            self._active_operations[operation_id] = {
                'operation': operation,
                'start_time': time.time(),
                'memory_start_mb': self._get_memory_usage_mb(),
                'cpu_start': self._get_cpu_percent(),
                'batch_size': batch_size,
                'metadata': metadata or {},
                'peak_memory_mb': self._get_memory_usage_mb()
            }
        
        return operation_id
    
    def end_operation(self, operation_id: str, items_processed: Optional[int] = None,
                     errors: int = 0) -> Optional[PerformanceMetrics]:
        """
        End monitoring an operation and record metrics.
        
        Args:
            operation_id: The operation ID returned by start_operation
            items_processed: Number of items processed during operation
            errors: Number of errors that occurred
            
        Returns:
            PerformanceMetrics object if monitoring was enabled
        """
        if not self.enable_detailed_monitoring or operation_id not in self._active_operations:
            return None
        
        end_time = time.time()
        memory_end_mb = self._get_memory_usage_mb()
        
        with self._lock:
            op_data = self._active_operations.pop(operation_id)
            
            metrics = PerformanceMetrics(
                operation=op_data['operation'],
                start_time=op_data['start_time'],
                end_time=end_time,
                duration=end_time - op_data['start_time'],
                memory_start_mb=op_data['memory_start_mb'],
                memory_end_mb=memory_end_mb,
                memory_peak_mb=op_data['peak_memory_mb'],
                cpu_percent=self._get_cpu_percent(),
                batch_size=op_data['batch_size'],
                items_processed=items_processed,
                errors=errors,
                metadata=op_data['metadata']
            )
            
            self.metrics_history.append(metrics)
            
        return metrics
    
    def update_peak_memory(self, operation_id: str) -> None:
        """
        Update peak memory usage for an active operation.
        
        Args:
            operation_id: The operation ID to update
        """
        if not self.enable_detailed_monitoring or operation_id not in self._active_operations:
            return
        
        current_memory = self._get_memory_usage_mb()
        
        with self._lock:
            if operation_id in self._active_operations:
                self._active_operations[operation_id]['peak_memory_mb'] = max(
                    self._active_operations[operation_id]['peak_memory_mb'],
                    current_memory
                )
    
    @contextmanager
    def monitor_operation(self, operation: str, batch_size: Optional[int] = None,
                         items_processed: Optional[int] = None,
                         metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for monitoring an operation.
        
        Args:
            operation: Name/description of the operation
            batch_size: Size of the batch being processed
            items_processed: Number of items processed
            metadata: Additional metadata
            
        Yields:
            PerformanceMetrics object (or None if monitoring disabled)
        """
        operation_id = self.start_operation(operation, batch_size, metadata)
        errors = 0
        
        try:
            yield operation_id
        except Exception as e:
            errors = 1
            raise
        finally:
            metrics = self.end_operation(operation_id, items_processed, errors)
            if metrics:
                # Update items_processed if it was provided in the context manager
                if items_processed is not None:
                    metrics.items_processed = items_processed
    
    def get_metrics_summary(self, operation_filter: Optional[str] = None,
                           last_n: Optional[int] = None) -> Dict[str, Any]:
        """
        Get a summary of performance metrics.
        
        Args:
            operation_filter: Filter metrics by operation name
            last_n: Only include the last N metrics
            
        Returns:
            Dictionary containing metrics summary
        """
        with self._lock:
            metrics = self.metrics_history.copy()
        
        # Apply filters
        if operation_filter:
            metrics = [m for m in metrics if operation_filter in m.operation]
        
        if last_n:
            metrics = metrics[-last_n:]
        
        if not metrics:
            return {}
        
        # Calculate summary statistics
        durations = [m.duration for m in metrics]
        throughputs = [m.throughput for m in metrics if m.throughput is not None]
        memory_usages = [m.memory_usage_mb for m in metrics]
        error_counts = [m.errors for m in metrics]
        
        summary = {
            'total_operations': len(metrics),
            'total_errors': sum(error_counts),
            'error_rate': sum(error_counts) / len(metrics) if metrics else 0,
            'duration_stats': {
                'min': min(durations),
                'max': max(durations),
                'avg': sum(durations) / len(durations),
                'total': sum(durations)
            },
            'memory_stats': {
                'min_mb': min(memory_usages),
                'max_mb': max(memory_usages),
                'avg_mb': sum(memory_usages) / len(memory_usages)
            }
        }
        
        if throughputs:
            summary['throughput_stats'] = {
                'min_items_per_sec': min(throughputs),
                'max_items_per_sec': max(throughputs),
                'avg_items_per_sec': sum(throughputs) / len(throughputs)
            }
        
        # Operation breakdown
        operation_counts = {}
        for metric in metrics:
            op_name = metric.operation
            if op_name not in operation_counts:
                operation_counts[op_name] = 0
            operation_counts[op_name] += 1
        
        summary['operation_breakdown'] = operation_counts
        
        return summary
    
    def export_metrics(self, file_path: Union[str, Path], 
                      operation_filter: Optional[str] = None) -> None:
        """
        Export metrics to a JSON file.
        
        Args:
            file_path: Path to save the metrics file
            operation_filter: Filter metrics by operation name
        """
        with self._lock:
            metrics = self.metrics_history.copy()
        
        # Apply filter
        if operation_filter:
            metrics = [m for m in metrics if operation_filter in m.operation]
        
        # Convert to dictionaries
        metrics_data = [m.to_dict() for m in metrics]
        
        # Add summary
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'summary': self.get_metrics_summary(operation_filter),
            'metrics': metrics_data
        }
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
    
    def clear_metrics(self) -> None:
        """Clear all stored metrics."""
        with self._lock:
            self.metrics_history.clear()
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            return self.process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def _get_cpu_percent(self) -> float:
        """Get current CPU usage percentage."""
        try:
            return self.process.cpu_percent()
        except Exception:
            return 0.0


def performance_monitor(operation: str, batch_size: Optional[int] = None,
                       monitor_instance: Optional[PerformanceMonitor] = None):
    """
    Decorator for monitoring function performance.
    
    Args:
        operation: Name/description of the operation
        batch_size: Size of the batch being processed
        monitor_instance: PerformanceMonitor instance to use
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            monitor = monitor_instance or getattr(args[0], 'performance_monitor', None)
            
            if not monitor:
                # No monitor available, just call the function
                return func(*args, **kwargs)
            
            # Determine batch size from arguments if not specified
            actual_batch_size = batch_size
            if actual_batch_size is None:
                # Try to infer batch size from common argument patterns
                for arg in args:
                    if isinstance(arg, (list, tuple)):
                        actual_batch_size = len(arg)
                        break
                
                for key, value in kwargs.items():
                    if 'batch' in key.lower() and isinstance(value, (list, tuple)):
                        actual_batch_size = len(value)
                        break
            
            with monitor.monitor_operation(operation, actual_batch_size) as operation_id:
                result = func(*args, **kwargs)
                
                # Try to determine items processed from result
                items_processed = None
                if isinstance(result, (list, tuple)):
                    items_processed = len(result)
                elif hasattr(result, '__len__'):
                    try:
                        items_processed = len(result)
                    except Exception:
                        pass
                
                # Update metrics with items processed
                if items_processed is not None:
                    monitor.end_operation(operation_id, items_processed)
                
                return result
        
        return wrapper
    return decorator


# Global performance monitor instance
_global_monitor = PerformanceMonitor()


def get_global_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return _global_monitor


def set_global_monitoring(enabled: bool) -> None:
    """Enable or disable global performance monitoring."""
    global _global_monitor
    _global_monitor.enable_detailed_monitoring = enabled


def get_system_info() -> Dict[str, Any]:
    """
    Get current system information.
    
    Returns:
        Dictionary containing system metrics
    """
    try:
        memory = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()
        
        return {
            'cpu_count': cpu_count,
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_total_gb': memory.total / 1024 / 1024 / 1024,
            'memory_available_gb': memory.available / 1024 / 1024 / 1024,
            'memory_percent': memory.percent,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {'error': str(e), 'timestamp': datetime.now().isoformat()}