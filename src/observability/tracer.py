# src/observability/tracer.py
"""
Distributed tracing for debugging and optimization.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Any
from contextlib import contextmanager
import uuid
import time
import json
from pathlib import Path
from loguru import logger


@dataclass
class SpanEvent:
    """Event within a span."""
    name: str
    timestamp: float
    attributes: dict = field(default_factory=dict)


@dataclass
class Span:
    """A span representing a unit of work."""
    span_id: str
    name: str
    parent_id: Optional[str]
    trace_id: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "ok"
    attributes: dict = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)
    
    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0


@dataclass
class Trace:
    """Complete trace of a request."""
    trace_id: str
    spans: list[Span]
    query: str
    timestamp: datetime
    total_duration_ms: float
    outcome: str  # success, fallback, error
    attributes: dict = field(default_factory=dict)
    

class RAGTracer:
    """
    Trace RAG pipeline execution for debugging and analysis.
    
    Tracks:
    - Retrieval timing and results
    - Reranking decisions
    - Generation latency
    - Confidence scores at each stage
    """
    
    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.traces_file = self.storage_path / "traces.jsonl"
        
        self._current_trace_id: Optional[str] = None
        self._current_spans: list[Span] = []
        self._span_stack: list[str] = []  # Stack of span IDs
        
    @contextmanager
    def trace(self, query: str):
        """Start a new trace for a query."""
        self._current_trace_id = str(uuid.uuid4())
        self._current_spans = []
        self._span_stack = []
        start_time = time.time()
        
        try:
            yield self
        finally:
            total_duration = (time.time() - start_time) * 1000
            
            # Determine outcome
            outcome = "success"
            for span in self._current_spans:
                if span.status == "fallback":
                    outcome = "fallback"
                    break
                elif span.status == "error":
                    outcome = "error"
                    break
                    
            trace = Trace(
                trace_id=self._current_trace_id,
                spans=self._current_spans,
                query=query,
                timestamp=datetime.now(timezone.utc),
                total_duration_ms=total_duration,
                outcome=outcome
            )
            
            self._persist_trace(trace)
            self._current_trace_id = None
            
    @contextmanager
    def span(self, name: str, **attributes):
        """Create a span within the current trace."""
        if not self._current_trace_id:
            yield None
            return
            
        span_id = str(uuid.uuid4())
        parent_id = self._span_stack[-1] if self._span_stack else None
        
        span = Span(
            span_id=span_id,
            name=name,
            parent_id=parent_id,
            trace_id=self._current_trace_id,
            start_time=time.time(),
            attributes=attributes
        )
        
        self._span_stack.append(span_id)
        
        try:
            yield span
        except Exception as e:
            span.status = "error"
            span.attributes['error'] = str(e)
            raise
        finally:
            span.end_time = time.time()
            self._current_spans.append(span)
            self._span_stack.pop()
            
    def add_event(self, name: str, **attributes):
        """Add event to current span."""
        if not self._span_stack:
            return
            
        current_span_id = self._span_stack[-1]
        for span in self._current_spans:
            if span.span_id == current_span_id:
                span.events.append(SpanEvent(
                    name=name,
                    timestamp=time.time(),
                    attributes=attributes
                ))
                break
                
    def set_attribute(self, key: str, value: Any):
        """Set attribute on current span."""
        if not self._span_stack:
            return
            
        current_span_id = self._span_stack[-1]
        for span in self._current_spans:
            if span.span_id == current_span_id:
                span.attributes[key] = value
                break
                
    def _persist_trace(self, trace: Trace):
        """Save trace to disk."""
        data = {
            'trace_id': trace.trace_id,
            'query': trace.query,
            'timestamp': trace.timestamp.isoformat(),
            'total_duration_ms': trace.total_duration_ms,
            'outcome': trace.outcome,
            'attributes': trace.attributes,
            'spans': [
                {
                    'span_id': s.span_id,
                    'name': s.name,
                    'parent_id': s.parent_id,
                    'duration_ms': s.duration_ms,
                    'status': s.status,
                    'attributes': s.attributes,
                    'events': [
                        {'name': e.name, 'attributes': e.attributes}
                        for e in s.events
                    ]
                }
                for s in trace.spans
            ]
        }
        
        with open(self.traces_file, 'a') as f:
            f.write(json.dumps(data) + '\n')
            
    def get_recent_traces(self, limit: int = 100) -> list[dict]:
        """Get recent traces for analysis."""
        traces = []
        if self.traces_file.exists():
            with open(self.traces_file, 'r') as f:
                for line in f:
                    traces.append(json.loads(line))
                    
        return traces[-limit:]
    
    def get_slow_queries(
        self, 
        threshold_ms: float = 5000, 
        limit: int = 10
    ) -> list[dict]:
        """Get queries exceeding latency threshold."""
        traces = self.get_recent_traces(1000)
        slow = []
        for t in traces:
            # Use .get() to safely check for the key, and default to 0 if missing
            duration = t.get('total_duration_ms', 0)
            if duration > threshold_ms:
                slow.append(t)
        return sorted(slow, key=lambda t: t['total_duration_ms'], reverse=True)[:limit]
