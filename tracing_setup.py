"""
OpenTelemetry Tracing Setup for French LLM Training
Initializes OTLP exporter to send traces to AI Toolkit
"""

# pyright: reportMissingImports=false

try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.resources import Resource
except Exception:
    trace = metrics = None  # type: ignore[assignment]
    TracerProvider = BatchSpanProcessor = OTLPSpanExporter = None  # type: ignore[assignment]
    MeterProvider = PeriodicExportingMetricReader = OTLPMetricExporter = None  # type: ignore[assignment]
    Resource = None  # type: ignore[assignment]
import os
from typing import Optional

# OTLP endpoint (AI Toolkit default)
OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")


def setup_tracing(
    service_name: str,
    service_version: str = "1.0.0",
    environment: str = "development",
    otlp_endpoint: Optional[str] = None
) -> tuple:
    """
    Initialize OpenTelemetry tracing and metrics exporters.
    
    Args:
        service_name: Name of the service (e.g., "french-llm-training")
        service_version: Version of the service
        environment: Environment (development, production, etc.)
        otlp_endpoint: Custom OTLP endpoint (defaults to localhost:4318)
    
    Returns:
        tuple: (tracer, meter) for use in application code
    """
    
    if otlp_endpoint is None:
        otlp_endpoint = OTLP_ENDPOINT
    
    # Create resource with service metadata
    if Resource is None:
        return None, None
    resource = Resource.create({
        "service.name": service_name,
        "service.version": service_version,
        "deployment.environment": environment,
    })
    
    # Setup Trace Exporter and Provider
    if OTLPSpanExporter and TracerProvider and BatchSpanProcessor and trace is not None:
        otlp_trace_exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint
        )
        trace_provider = TracerProvider(resource=resource)
        trace_provider.add_span_processor(BatchSpanProcessor(otlp_trace_exporter))
        trace.set_tracer_provider(trace_provider)
    
    # Setup Metrics Exporter and Provider
    meter = None
    if OTLPMetricExporter and MeterProvider and PeriodicExportingMetricReader and metrics is not None:
        otlp_metric_exporter = OTLPMetricExporter(
            endpoint=otlp_endpoint
        )
        metric_reader = PeriodicExportingMetricReader(otlp_metric_exporter)
        metrics_provider = MeterProvider(
            resource=resource,
            metric_readers=[metric_reader]
        )
        metrics.set_meter_provider(metrics_provider)
        meter = metrics.get_meter(__name__)
    
    # Get tracer and meter instances
    tracer = trace.get_tracer(__name__) if trace is not None else None
    
    print(f"✅ Tracing initialized for {service_name}")
    print(f"   OTLP Endpoint: {otlp_endpoint}")
    print(f"   Service: {service_name} v{service_version}")
    
    return tracer, meter


# Convenience function for getting existing tracer/meter
def get_tracer(name: str = __name__) -> tuple:
    """Get current tracer and meter instances."""
    tracer = trace.get_tracer(name) if trace is not None else None
    meter = metrics.get_meter(name) if metrics is not None else None
    return tracer, meter
