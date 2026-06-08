# src/observability/dashboard.py
"""
Streamlit Dashboard for Monitoring RAG Observability.
Supports both programmatic framework imports and CLI execution.

To run manually:
streamlit run src/observability/dashboard.py
"""

import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd


current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.evaluation.hallucination_tracker import HallucinationTracker
from src.observability.tracer import RAGTracer

tracer_PATH = project_root / "data" / "traces"
tracker_PATH = project_root / "data" / "hallucination_tracking"
#print(f"🔍 Tracer Storage Path: {tracer_PATH.resolve()}")
#print(f"🔍 Tracker Storage Path: {tracker_PATH.resolve()}")
class Dashboard:
    """Framework wrapper class to satisfy project structural imports."""
    def render(self) -> str:
        """Returns initialization status for the observability module."""
        return "Streamlit Observability Dashboard Engine Ready. Run via CLI to view UI."

@st.cache_data(ttl=2)
def load_trackers():
    """Cache trackers to minimize file I/O overhead on UI re-renders."""
    tracker = HallucinationTracker(storage_path=tracker_PATH)
    tracer = RAGTracer(storage_path=tracer_PATH)
    return tracker, tracer

def run_streamlit_ui():
    """Main execution block for the Streamlit UI presentation layer."""
    st.set_page_config(page_title="RAG Observability Dashboard", layout="wide")
    st.title("📊 RAG Observability & Health Dashboard")
    
    try:
        hallucination_tracker, rag_tracer = load_trackers()
    except Exception as e:
        st.error(f"Failed to load storage files from {tracker_PATH}. Error: {e}")
        return

   
    st.header("System Health Overview")
    metrics = hallucination_tracker.get_metrics()
    #print(f"📈 Current Metrics: {metrics}")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Queries", metrics.total_queries)
    with col2:
        delta_color = "normal" if metrics.fallback_rate <= 0.1 else "inverse"
        st.metric("Fallback Rate", f"{metrics.fallback_rate:.1%}", delta_color=delta_color)
    with col3:
        st.metric("Avg Confidence", f"{metrics.avg_confidence:.2f}")
    with col4:
        st.metric("Citation Issues", metrics.citation_issues_count)

    st.markdown("---")

    # --- ALERTS SECTION ---
    alerts = hallucination_tracker.check_alerts()
    if alerts:
        st.subheader("⚠️ Active System Alerts")
        for alert in alerts:
            if "HIGH" in alert:
                st.error(alert)
            else:
                st.warning(alert)
    else:
        st.success("✅ No active alerts. System is healthy.")

    st.markdown("---")

    # --- MAIN TABS ---
    tab1, tab2 = st.tabs(["Hallucination Log", "Performance & Traces"])

    with tab1:
        st.subheader("Recent Potential Hallucinations")
        events = hallucination_tracker._events 
        
        if events:
            df_events = pd.DataFrame([
                {
                    "Timestamp": e.timestamp.strftime("%Y-%m-%d %H:%M:%S") if hasattr(e.timestamp, 'strftime') else str(e.timestamp),
                    "Query": e.query,
                    "Detection Method": e.detection_method,
                    "Confidence": round(e.confidence_score, 2),
                    "Fallback Triggered": bool(e.was_fallback_triggered),
                    "Snippet": e.response_snippet
                }
                for e in reversed(events[-50:])
            ])
            st.dataframe(df_events, use_container_width=True)
        else:
            st.info("No hallucination events recorded yet.")

    with tab2:
        st.subheader("Latency & Slow Queries")
        
        # 1. Place the slider at the very top
        threshold = st.slider(
            "Slow Query Threshold (ms)", 
            min_value=1000, max_value=10000, value=5000, step=500
        )
        
        slow_queries = rag_tracer.get_slow_queries(threshold_ms=threshold, limit=10)
        
        if slow_queries:
            st.warning(f"Found {len(slow_queries)} queries exceeding {threshold}ms")
            
            # 2. Build the dynamic data directly below the slider
            slow_data = []
            for t in slow_queries:
                # Check for actual outcome
                final_outcome = t.get('outcome', 'success')
                for span in t.get('spans', []):
                    if span.get('name') == 'fallback':
                        final_outcome = 'fallback'
                        break

                # Base row
                row = {
                    "Timestamp": t.get('timestamp', '')[:19],
                    "Query": t.get('query', ''),
                    "Total (ms)": round(t.get('total_duration_ms', 0), 0),
                    "Outcome": final_outcome
                }
                
                # Dynamic span columns
                for span in t.get('spans', []):
                    span_name = span.get('name', 'unknown')
                    span_duration = round(span.get('duration_ms', 0), 0)
                    row[f"{span_name} (ms)"] = span_duration
                    
                slow_data.append(row)
            
            # Create DataFrame
            df_slow = pd.DataFrame(slow_data).fillna(0)
            
            # Order columns: Base columns first, then sorted span columns
            cols = df_slow.columns.tolist()
            base_cols = ["Timestamp", "Query", "Total (ms)", "Outcome"]
            span_cols = sorted([c for c in cols if c not in base_cols])
            
            df_slow = df_slow[base_cols + span_cols]
            
            # 3. Render the full-width table underneath
            st.dataframe(
                df_slow, 
                use_container_width=True, 
                hide_index=True
            )
        else:
            st.success(f"No queries exceeding {threshold}ms")


if __name__ == "__main__":
    run_streamlit_ui()
