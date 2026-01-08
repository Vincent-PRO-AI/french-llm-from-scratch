"""
Reusable Tracing Module for French LLM Training Scripts
Provides context managers and decorators for tracing training operations
"""

from contextlib import contextmanager
from typing import Generator, Any, Callable, Optional
from functools import wraps
import time
from tracing_setup import get_tracer


class TracingContext:
    """Helper class for managing traced operations."""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.tracer, self.meter = get_tracer(__name__)
        self.start_time = None
        self.span = None
    
    @contextmanager
    def trace_operation(self, span_name: str, attributes: Optional[dict] = None) -> Generator:
        """
        Context manager for tracing a named operation.
        
        Usage:
            with TracingContext("training").trace_operation("forward_pass", {"batch_size": 32}):
                # operation code
        """
        with self.tracer.start_as_current_span(span_name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            
            start_time = time.time()
            try:
                yield span
                span.set_attribute("status", "success")
            except Exception as e:
                span.set_attribute("status", "error")
                span.set_attribute("error.type", type(e).__name__)
                span.set_attribute("error.message", str(e))
                raise
            finally:
                duration = time.time() - start_time
                span.set_attribute("duration_seconds", duration)
    
    @contextmanager
    def trace_checkpoint(self, checkpoint_step: int, loss: float) -> Generator:
        """Trace checkpoint save operation."""
        with self.trace_operation(
            "checkpoint_save",
            {
                "step": checkpoint_step,
                "loss": loss,
            }
        ):
            yield
    
    @contextmanager
    def trace_evaluation(self, eval_step: int, metrics: dict) -> Generator:
        """Trace evaluation operation."""
        with self.trace_operation(
            "evaluation",
            {
                "step": eval_step,
                "validation_loss": metrics.get("val_loss", 0.0),
                "training_loss": metrics.get("train_loss", 0.0),
            }
        ):
            yield


def trace_training_step(func: Callable) -> Callable:
    """Decorator to trace a training step function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        ctx = TracingContext("training")
        step = kwargs.get("step", args[0] if args else 0)
        
        with ctx.trace_operation(
            "training_step",
            {"step": step}
        ) as span:
            result = func(*args, **kwargs)
            if isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        span.set_attribute(f"result.{key}", value)
            return result
    
    return wrapper


def trace_data_loading(func: Callable) -> Callable:
    """Decorator to trace data loading operations."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        ctx = TracingContext("data")
        
        with ctx.trace_operation("data_loading") as span:
            start = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start
            
            if hasattr(result, '__len__'):
                span.set_attribute("num_samples", len(result))
            span.set_attribute("load_time_seconds", duration)
            
            return result
    
    return wrapper


def trace_model_export(func: Callable) -> Callable:
    """Decorator to trace model export operations."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        ctx = TracingContext("export")
        
        with ctx.trace_operation("model_export") as span:
            start = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start
            
            span.set_attribute("export_format", kwargs.get("format", "unknown"))
            span.set_attribute("export_time_seconds", duration)
            
            return result
    
    return wrapper
