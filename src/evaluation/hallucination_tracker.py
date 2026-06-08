# src/evaluation/hallucination_tracker.py
"""
Track and analyze potential hallucinations over time.
Acts as a read-only parser on top of the master trace logs.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from collections import defaultdict
import json
from pathlib import Path


@dataclass
class HallucinationEvent:
    """Record of a potential hallucination incident extracted from traces."""
    event_id: str
    timestamp: datetime
    query: str
    response_snippet: str
    detection_method: str  # fallback_triggered, low_confidence
    confidence_score: float
    sources_available: int
    was_fallback_triggered: bool
    user_feedback: Optional[str] = None


@dataclass
class HallucinationMetrics:
    """Aggregated hallucination metrics."""
    total_queries: int
    fallback_count: int
    fallback_rate: float
    low_confidence_count: int
    citation_issues_count: int
    avg_confidence: float
    by_detection_method: dict[str, int]


class HallucinationTracker:
    """
    Parses the master RAG traces to compute hallucination metrics.
    
    This provides:
    1. Historical analysis for system improvement
    2. Alerting when rates exceed thresholds
    """
    
    def __init__(
        self,
        storage_path: Path,
        alert_threshold: float = 0.1  # Alert if >10% are potential hallucinations
    ):
        self.storage_path = Path(storage_path)
        # Point directly to the master trace file handled by RAGTracer
        self.events_file = self.storage_path.parent / "traces" / "traces.jsonl"
        self.alert_threshold = alert_threshold
        
        self._events: list[HallucinationEvent] = []
        self._total_queries = 0
        self._load_events()
        
    def _load_events(self):
        """Read the master traces file and filter for hallucination events."""
        if not self.events_file.exists():
            return
            
        with open(self.events_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    self._total_queries += 1  # Every trace counts as a query
                    
                    is_fallback = False
                    confidence = 0.0
                    fallback_reason = "Fallback Triggered"
                    
                    # 1. Dig into the spans to find the real data
                    spans = data.get('spans', [])
                    for span in spans:
                        # Find the confidence score
                        if span.get('name') == 'confidence_scoring':
                            confidence = float(span.get('attributes', {}).get('top_confidence', 0.0))
                            
                        # Check if a fallback occurred
                        if span.get('name') == 'fallback':
                            is_fallback = True
                            fallback_reason = span.get('attributes', {}).get('reason', 'Fallback Triggered')
                    
                    # 2. Flag as an event if it was a fallback OR confidence was dangerously low
                    if is_fallback or confidence < 0.5:
                        detection_method = "fallback_triggered" if is_fallback else "low_confidence"
                        
                        event = HallucinationEvent(
                            event_id=data.get('trace_id', f"hal_{self._total_queries}"),
                            timestamp=datetime.fromisoformat(data['timestamp']),
                            query=data.get('query', ''),
                            response_snippet=fallback_reason,
                            detection_method=detection_method,
                            confidence_score=confidence,
                            sources_available=0, # Optional: pull from hybrid_retrieval span if needed
                            was_fallback_triggered=is_fallback
                        )
                        self._events.append(event)
                except Exception:
                    # Silently skip any malformed lines
                    continue
            
    def get_metrics(
        self, 
        since: Optional[datetime] = None
    ) -> HallucinationMetrics:
        """Compute hallucination metrics based on parsed traces."""
        events = self._events
        if since:
            events = [e for e in events if e.timestamp >= since]
            
        if not events and self._total_queries == 0:
            return HallucinationMetrics(
                total_queries=0,
                fallback_count=0,
                fallback_rate=0.0,
                low_confidence_count=0,
                citation_issues_count=0,
                avg_confidence=0.0,
                by_detection_method={}
            )
            
        fallback_count = sum(1 for e in events if e.was_fallback_triggered)
        
        by_method = defaultdict(int)
        confidences = []
        
        for event in events:
            by_method[event.detection_method] += 1
            confidences.append(event.confidence_score)
            
        return HallucinationMetrics(
            total_queries=self._total_queries,
            fallback_count=fallback_count,
            fallback_rate=fallback_count / max(self._total_queries, 1),
            low_confidence_count=by_method.get('low_confidence', 0),
            citation_issues_count=by_method.get('citation_missing', 0) + 
                                   by_method.get('invalid_citations', 0),
            avg_confidence=sum(confidences) / len(confidences) if confidences else 0.0,
            by_detection_method=dict(by_method)
        )
    
    def check_alerts(self) -> list[str]:
        """Check if any alert thresholds are exceeded."""
        alerts = []
        metrics = self.get_metrics()
        
        if metrics.fallback_rate > self.alert_threshold:
            alerts.append(
                f"HIGH: Fallback rate ({metrics.fallback_rate:.1%}) "
                f"exceeds threshold ({self.alert_threshold:.1%})"
            )
            
        if metrics.total_queries > 0 and metrics.avg_confidence < 0.5 and metrics.fallback_count > 0:
            alerts.append(
                f"MEDIUM: Average confidence ({metrics.avg_confidence:.2f}) across flagged events is critically low."
            )
            
        return alerts