"""tracing.py — OpenTelemetry for the gateway (#13): one span per request, one child span per backend attempt,
W3C `traceparent` propagated to the backend so engine-side spans (if any) join the same trace.

Configuration (environment, read on first use; tests can call configure(mode) directly):
  OTEL_SERVICE_NAME            default "s2s-gateway"
  S2S_TRACING=console|memory|otlp|off   default "off" (no overhead unless asked)
  OTEL_EXPORTER_OTLP_ENDPOINT  used when S2S_TRACING=otlp (needs opentelemetry-exporter-otlp installed)
`memory` keeps finished spans in `MEMORY_EXPORTER` for tests.
We hold our own TracerProvider (instead of the process-global one, which can only be set once) so the mode can be
changed in tests; context propagation is global and works either way.
"""
from __future__ import annotations

import os

from opentelemetry import propagate, trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

MEMORY_EXPORTER = InMemorySpanExporter()
_provider: TracerProvider | None = None
_mode: str | None = None


def configure(mode: str | None = None) -> None:
    global _provider, _mode
    mode = mode or os.environ.get("S2S_TRACING", "off")
    if mode == _mode:
        return
    _mode = mode
    if mode == "off":
        _provider = None
        return
    _provider = TracerProvider(resource=Resource.create({"service.name": os.environ.get("OTEL_SERVICE_NAME", "s2s-gateway")}))
    if mode == "console":
        _provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    elif mode == "memory":
        _provider.add_span_processor(SimpleSpanProcessor(MEMORY_EXPORTER))
    elif mode == "otlp":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # optional dependency
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        _provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))


def tracer():
    if _mode is None:
        configure()
    return _provider.get_tracer("s2s.gateway") if _provider else trace.NoOpTracer()


def inject(headers: dict) -> dict:
    """Add W3C trace-context headers for the current span to an outgoing request."""
    propagate.inject(headers)
    return headers
