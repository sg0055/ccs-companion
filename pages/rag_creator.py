# pages/rag_creator.py
"""
RAG Creator — Streamlit page for bulk document ingestion.
Allows selecting a folder, choosing ingestion mode, tracking progress live,
and reviewing a full completion report.
"""

import sys
import json
import time
import threading
from pathlib import Path
import streamlit as st

# Ensure project root is on path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.main import HallucinationProofRAG
from src.ingestion.batch_processor import BatchProcessor, BatchReport, FileResult

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Creator",
    page_icon="🗂️",
    layout="wide",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
    border: 1px solid rgba(139, 92, 246, 0.25);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    margin-bottom: 0.5rem;
}
.metric-card .metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #a78bfa;
    line-height: 1;
}
.metric-card .metric-label {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-top: 0.35rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── File log rows ── */
.log-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.45rem 0.7rem;
    border-radius: 8px;
    font-size: 0.85rem;
    margin-bottom: 0.3rem;
    border-left: 3px solid transparent;
}
.log-row.success  { background: rgba(34,197,94,0.08);  border-color: #22c55e; }
.log-row.duplicate{ background: rgba(99,102,241,0.08); border-color: #6366f1; }
.log-row.error    { background: rgba(239,68,68,0.08);  border-color: #ef4444; }
.log-row.skipped  { background: rgba(148,163,184,0.06);border-color: #94a3b8; }

/* ── Stage pill ── */
.stage-pill {
    display: inline-block;
    background: rgba(139,92,246,0.15);
    color: #a78bfa;
    border-radius: 99px;
    padding: 0.15rem 0.65rem;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}

/* ── Section header ── */
.section-header {
    font-size: 1.05rem;
    font-weight: 600;
    color: #e2e8f0;
    margin: 1.2rem 0 0.6rem 0;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session state helpers
# ─────────────────────────────────────────────
def _init_state():
    defaults = {
        "rag_running": False,
        "rag_cancelled": False,
        "rag_report": None,
        "rag_log": [],           # list[FileResult]
        "rag_current_file": "",
        "rag_progress": 0.0,
        "rag_total": 0,
        "rag_done_count": 0,
        "rag_live_chunks": 0,
        "rag_live_dupes": 0,
        "rag_live_errors": 0,
        "rag_stage": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────
# Load RAG engine (cached)
# ─────────────────────────────────────────────
def _load_rag():
    try:
        from state import get_rag_engine
        return get_rag_engine()
    except Exception as exc:
        return exc   # Return the error; handle below


# ─────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────
def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


def _status_icon(status: str) -> str:
    return {"success": "✅", "duplicate": "🔁", "error": "❌", "skipped": "⏭️"}.get(status, "•")


def _validate_folder(folder_str: str, convert_pdf: bool, recursive: bool):
    """Return (files_found, error_string)."""
    if not folder_str.strip():
        return 0, "Please enter a folder path."
    folder = Path(folder_str.strip())
    if not folder.exists():
        return 0, f"Folder does not exist: `{folder}`"
    if not folder.is_dir():
        return 0, f"Path is not a folder: `{folder}`"

    from src.ingestion.batch_processor import PDF_EXTENSIONS, MD_EXTENSIONS
    pattern = "**/*" if recursive else "*"
    extensions = MD_EXTENSIONS | (PDF_EXTENSIONS if convert_pdf else set())
    files = [f for f in folder.glob(pattern) if f.is_file() and f.suffix.lower() in extensions]
    if not files:
        exts = ", ".join(sorted(extensions))
        return 0, f"No supported files ({exts}) found in folder."
    return len(files), None


# ─────────────────────────────────────────────
# Background ingestion thread
# ─────────────────────────────────────────────
def _run_ingestion(rag, folder_path, convert_pdf, recursive):
    """Runs in a background thread; updates session_state via callbacks."""

    processor = BatchProcessor(rag)
    # Store processor so the cancel button can reach it
    st.session_state["_batch_processor"] = processor

    def _callback(stage, file_index, total_files, current_file, result):
        st.session_state["rag_stage"] = stage
        st.session_state["rag_current_file"] = current_file
        st.session_state["rag_total"] = total_files
        st.session_state["rag_progress"] = file_index / max(total_files, 1)

        if result is not None:
            st.session_state["rag_log"].append(result)
            st.session_state["rag_done_count"] = file_index
            if result.status == "success":
                st.session_state["rag_live_chunks"] += result.chunks_created
                st.session_state["rag_live_dupes"] += (
                    result.exact_duplicates + result.near_duplicates
                )
            elif result.status == "error":
                st.session_state["rag_live_errors"] += 1

    try:
        report = processor.process_folder(
            folder_path=Path(folder_path),
            convert_pdf=convert_pdf,
            recursive=recursive,
            progress_callback=_callback,
        )
    except Exception as exc:
        # Catch any unexpected top-level error
        report = BatchReport(
            total_files=st.session_state.get("rag_total", 0),
            failed=1,
        )
        report.errors.append({"filename": folder_path, "error": str(exc)})

    st.session_state["rag_report"] = report
    st.session_state["rag_running"] = False
    st.session_state["rag_progress"] = 1.0


# ─────────────────────────────────────────────
# Page Header
# ─────────────────────────────────────────────
st.markdown("# 🗂️ RAG Creator")
st.caption("Bulk-ingest documents into the knowledge base • Powered by docling + FAISS")
st.divider()

rag_or_error = _load_rag()
if isinstance(rag_or_error, Exception):
    st.error(
        f"❌ **Failed to load RAG engine** — {type(rag_or_error).__name__}: {rag_or_error}\n\n"
        "Make sure all required packages are installed (`pip install -r requirements.txt`) "
        "and the `data/` directory is accessible."
    )
    st.stop()

rag = rag_or_error


# ─────────────────────────────────────────────
# Section A — Configuration
# ─────────────────────────────────────────────
if not st.session_state["rag_running"] and st.session_state["rag_report"] is None:

    st.markdown('<div class="section-header">📁 Folder Selection</div>', unsafe_allow_html=True)

    folder_path = st.text_input(
        "Folder path",
        placeholder=r"e.g. C:\Users\ADIO\Documents\handbooks",
        help="Paste the full path to the folder containing your PDFs or Markdown files.",
        key="folder_path_input",
    )

    recursive = st.checkbox(
        "📂 Include subfolders",
        value=False,
        help="Scan all nested subfolders recursively.",
    )
    
    convert_pdf = True  # Forced to True for markdown chunker compatibility

    # Validate folder on the fly
    if folder_path.strip():
        file_count, err = _validate_folder(folder_path, convert_pdf, recursive)
        if err:
            st.error(f"⚠️ {err}")
            st.stop()
        else:
            folder = Path(folder_path.strip())
            total_size_mb = sum(
                f.stat().st_size for f in folder.rglob("*") if f.is_file()
            ) / 1_048_576
            st.success(
                f"✅ Found **{file_count} files** · "
                f"Total size ~**{total_size_mb:.1f} MB**"
            )

    st.markdown('<div class="section-header">⚙️ Advanced Settings</div>', unsafe_allow_html=True)
    with st.expander("Chunking overrides (optional)", expanded=False):
        from src.config import settings as cfg
        chunk_size = st.slider(
            "Chunk size (characters)",
            min_value=128, max_value=2048,
            value=cfg.chunking.chunk_size, step=64,
        )
        chunk_overlap = st.slider(
            "Chunk overlap (characters)",
            min_value=0, max_value=512,
            value=cfg.chunking.chunk_overlap, step=16,
        )
        if chunk_size != cfg.chunking.chunk_size or chunk_overlap != cfg.chunking.chunk_overlap:
            cfg.chunking.chunk_size = chunk_size
            cfg.chunking.chunk_overlap = chunk_overlap
            rag.chunker.chunk_size = chunk_size
            st.info(f"Chunking overridden: size={chunk_size}, overlap={chunk_overlap}")

    st.markdown("")
    btn_disabled = not folder_path.strip() or bool(_validate_folder(folder_path, convert_pdf, recursive)[1])
    if st.button("🚀 Start Ingestion", type="primary", disabled=btn_disabled, use_container_width=True):
        # Reset live stats
        for k in ["rag_log", "rag_live_chunks", "rag_live_dupes", "rag_live_errors",
                  "rag_done_count", "rag_progress", "rag_total", "rag_report", "rag_cancelled"]:
            st.session_state[k] = [] if k == "rag_log" else (0 if isinstance(st.session_state[k], (int, float)) else None)
        st.session_state["rag_running"] = True
        st.session_state["rag_cancelled"] = False

        thread = threading.Thread(
            target=_run_ingestion,
            args=(rag, folder_path.strip(), convert_pdf, recursive),
            daemon=True,
        )
        thread.start()
        st.rerun()


# ─────────────────────────────────────────────
# Section B — Live Progress
# ─────────────────────────────────────────────
if st.session_state["rag_running"]:

    st.markdown('<div class="section-header">⚡ Live Progress</div>', unsafe_allow_html=True)

    # Top-level stats
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{st.session_state["rag_done_count"]}</div>'
            f'<div class="metric-label">Files Done</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{st.session_state["rag_live_chunks"]:,}</div>'
            f'<div class="metric-label">Chunks Created</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{st.session_state["rag_live_dupes"]:,}</div>'
            f'<div class="metric-label">Duplicates Removed</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{st.session_state["rag_live_errors"]}</div>'
            f'<div class="metric-label">Errors</div></div>',
            unsafe_allow_html=True,
        )

    # Progress bar
    progress = st.session_state["rag_progress"]
    total = st.session_state["rag_total"]
    done = st.session_state["rag_done_count"]
    st.progress(progress, text=f"Processing **{done}/{total}** files…")

    # Current file
    if st.session_state["rag_current_file"]:
        st.markdown(
            f'<span class="stage-pill">⚙️ {st.session_state["rag_stage"].upper()}</span> '
            f'&nbsp;`{st.session_state["rag_current_file"]}`',
            unsafe_allow_html=True,
        )

    # Cancel button
    if st.button("⏹ Cancel Ingestion", type="secondary"):
        proc = st.session_state.get("_batch_processor")
        if proc:
            proc.cancel()
        st.session_state["rag_cancelled"] = True
        st.warning("⏳ Cancellation requested — finishing current file…")

    # Scrollable live log
    st.markdown('<div class="section-header">📋 File Log</div>', unsafe_allow_html=True)
    log_container = st.container(height=300)
    with log_container:
        for r in reversed(st.session_state["rag_log"][-100:]):
            icon = _status_icon(r.status)
            detail = ""
            if r.status == "success":
                detail = f"chunks: {r.chunks_created} · dupes: {r.exact_duplicates + r.near_duplicates} · v{r.version} · {_fmt_time(r.duration_seconds)}"
            elif r.status == "duplicate":
                detail = "already in knowledge base"
            elif r.status == "error":
                detail = f"{r.error_type}: {r.error_message[:80]}…" if r.error_message and len(r.error_message) > 80 else r.error_message
            st.markdown(
                f'<div class="log-row {r.status}">'
                f'{icon} <strong>{r.filename}</strong>'
                f'<span style="margin-left:auto;color:#94a3b8;font-size:0.8rem">{detail}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Auto-refresh while running
    time.sleep(1.5)
    st.rerun()


# ─────────────────────────────────────────────
# Section C — Completion Report
# ─────────────────────────────────────────────
if st.session_state["rag_report"] is not None and not st.session_state["rag_running"]:

    report: BatchReport = st.session_state["rag_report"]

    if report.was_cancelled:
        st.warning("⏹ Ingestion was cancelled by user.")
    else:
        st.success("✅ Ingestion complete!")

    st.markdown('<div class="section-header">📊 Summary Report</div>', unsafe_allow_html=True)

    # Summary metric cards
    cols = st.columns(6)
    metrics = [
        ("Total Files",        report.total_files,             "📁"),
        ("Successful",         report.successful,              "✅"),
        ("Chunks Created",     report.total_chunks_created,    "🧩"),
        ("Exact Dupes Removed",report.total_exact_duplicates,  "🔍"),
        ("Near Dupes Removed", report.total_near_duplicates,   "≈"),
        ("Errors",             report.failed,                  "❌"),
    ]
    for col, (label, value, icon) in zip(cols, metrics):
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{icon} {value:,}</div>'
                f'<div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True,
            )

    c1, c2 = st.columns(2)
    with c1:
        st.info(f"⏱️ **Total time:** {_fmt_time(report.total_time_seconds)}")
    with c2:
        avg = report.total_time_seconds / max(report.successful, 1)
        st.info(f"📄 **Avg per file:** {_fmt_time(avg)}")

    # Per-file results table
    st.markdown('<div class="section-header">📋 Per-File Breakdown</div>', unsafe_allow_html=True)
    if report.per_file_results:
        import pandas as pd
        df = pd.DataFrame([
            {
                "File": r.filename,
                "Status": r.status.upper(),
                "Chunks": r.chunks_created,
                "Exact Dupes": r.exact_duplicates,
                "Near Dupes": r.near_duplicates,
                "Version": r.version if r.version else "—",
                "Time": _fmt_time(r.duration_seconds),
                "Error": r.error_message or "",
            }
            for r in report.per_file_results
        ])
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Status": st.column_config.TextColumn(width="small"),
                "Chunks": st.column_config.NumberColumn(width="small"),
                "Exact Dupes": st.column_config.NumberColumn(width="small"),
                "Near Dupes": st.column_config.NumberColumn(width="small"),
            }
        )

    # Error details
    if report.errors:
        st.markdown('<div class="section-header">🔴 Error Details</div>', unsafe_allow_html=True)
        with st.expander(f"View {len(report.errors)} error(s)", expanded=True):
            for err in report.errors:
                st.markdown(
                    f'<div class="log-row error">'
                    f'❌ <strong>{err.get("filename","unknown")}</strong>'
                    f'<span style="margin-left:1rem;color:#fca5a5">'
                    f'{err.get("error_type","Error")}: {err.get("error","")}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # Download report
    st.markdown("")
    report_json = json.dumps(
        {
            "summary": {
                "total_files": report.total_files,
                "successful": report.successful,
                "duplicates_skipped": report.duplicates_skipped,
                "failed": report.failed,
                "total_chunks_created": report.total_chunks_created,
                "total_exact_duplicates": report.total_exact_duplicates,
                "total_near_duplicates": report.total_near_duplicates,
                "total_time_seconds": round(report.total_time_seconds, 2),
                "was_cancelled": report.was_cancelled,
            },
            "per_file": [
                {
                    "filename": r.filename,
                    "status": r.status,
                    "chunks_created": r.chunks_created,
                    "exact_duplicates": r.exact_duplicates,
                    "near_duplicates": r.near_duplicates,
                    "version": r.version,
                    "duration_seconds": round(r.duration_seconds, 2),
                    "error": r.error_message,
                }
                for r in report.per_file_results
            ],
            "errors": report.errors,
        },
        indent=2,
    )
    col_dl, col_new = st.columns([1, 1])
    with col_dl:
        st.download_button(
            label="📥 Download Report (JSON)",
            data=report_json,
            file_name="rag_ingestion_report.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_new:
        if st.button("🔄 Start New Ingestion", use_container_width=True, type="primary"):
            for k in ["rag_report", "rag_log", "rag_running", "rag_live_chunks",
                      "rag_live_dupes", "rag_live_errors", "rag_done_count",
                      "rag_progress", "rag_total", "rag_current_file"]:
                st.session_state[k] = [] if k == "rag_log" else (False if k in ("rag_running",) else None if k == "rag_report" else 0)
            st.rerun()
