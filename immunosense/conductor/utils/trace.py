"""Trace ID management for following one bucket evaluation across the system.

A single bucket evaluation touches the registry, up to five adapters, the
quality layer, and (Sprint 6+) fusion and decision. A shared trace_id stitches
all the resulting Layer A events and log lines together so a developer can ask
"show me everything that happened in patient001's Tuesday-T2 evaluation" and
get a coherent thread.

This module is deliberately tiny. It wraps the same UUID scheme used by
BaseAgent and events.types so trace ids look identical wherever they're minted.
"""

from __future__ import annotations

from immunosense.events.types import new_trace_id


class TraceContext:
    """A small holder for the trace id of one logical unit of work.

    Usage:
        ctx = TraceContext.for_bucket("patient001_2026-05-27_T2")
        ... pass ctx.trace_id into adapter.run(...) and event creation ...
    """

    def __init__(self, trace_id: str):
        self.trace_id = trace_id

    @classmethod
    def new(cls, prefix: str = "conductor") -> "TraceContext":
        return cls(new_trace_id(prefix))

    @classmethod
    def for_bucket(cls, bucket_id: str) -> "TraceContext":
        """Mint a trace id anchored to a bucket id for readability.

        The bucket id is embedded so the trace is human-greppable, with a
        short random suffix to keep it unique across re-evaluations of the
        same bucket (e.g. a flare-button re-eval after the scheduled one).
        """
        suffix = new_trace_id("").lstrip("-")
        return cls(f"eval-{bucket_id}-{suffix[:8]}")

    def __repr__(self) -> str:
        return f"TraceContext({self.trace_id!r})"
