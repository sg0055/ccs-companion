# src/ingestion/batch_processor.py
"""
Batch ingestion engine for processing folders of PDFs and Markdown files.
Supports progress callbacks, per-file error isolation, and docling PDF conversion.
"""

import time
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
from loguru import logger


@dataclass
class FileResult:
    """Result for a single file ingestion."""
    filename: str
    file_path: str
    status: str                   # 'success' | 'duplicate' | 'skipped' | 'error'
    chunks_created: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    version: int = 0
    duration_seconds: float = 0.0
    error_message: Optional[str] = None
    error_type: Optional[str] = None


@dataclass
class BatchReport:
    """Full report after batch ingestion completes."""
    total_files: int = 0
    successful: int = 0
    duplicates_skipped: int = 0
    failed: int = 0
    total_chunks_created: int = 0
    total_exact_duplicates: int = 0
    total_near_duplicates: int = 0
    total_time_seconds: float = 0.0
    per_file_results: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    was_cancelled: bool = False


# Supported file extensions
PDF_EXTENSIONS = {".pdf"}
MD_EXTENSIONS = {".md", ".markdown", ".txt"}


class BatchProcessor:
    """
    Processes a folder of documents through the RAG ingestion pipeline.

    Supports two ingestion modes:
      - convert_pdf=True:  Uses docling to convert PDFs → Markdown first (better
                           accuracy for tables and complex layouts).
      - convert_pdf=False: Ingests PDFs or Markdown files directly (faster,
                           suitable for simple text-heavy documents).

    Error isolation: a single file failure never stops the batch.
    """

    def __init__(self, rag_engine):
        """
        Args:
            rag_engine: Initialized HallucinationProofRAG instance.
        """
        self.rag = rag_engine
        self._cancelled = False

    def cancel(self):
        """Signal the batch to stop after the current file finishes."""
        self._cancelled = True
        logger.info("BatchProcessor: cancellation requested")

    def scan_folder(
        self,
        folder_path: Path,
        recursive: bool = False,
        convert_pdf: bool = True,
    ) -> list[Path]:
        """
        Return a sorted list of files to process from the given folder.

        Args:
            folder_path: Root folder to scan.
            recursive:   If True, also scans subfolders.
            convert_pdf: If True, include PDFs. If False, only include MD files.

        Raises:
            FileNotFoundError: If folder_path does not exist.
            NotADirectoryError: If folder_path is not a directory.
            ValueError: If the folder contains no supported files.
        """
        folder = Path(folder_path)

        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
        if not folder.is_dir():
            raise NotADirectoryError(f"Path is not a folder: {folder}")

        pattern = "**/*" if recursive else "*"
        extensions = MD_EXTENSIONS | (PDF_EXTENSIONS if convert_pdf else set())

        files = sorted(
            f for f in folder.glob(pattern)
            if f.is_file() and f.suffix.lower() in extensions
        )

        if not files:
            exts = ", ".join(sorted(extensions))
            raise ValueError(
                f"No supported files ({exts}) found in: {folder}"
                + (" (including subfolders)" if recursive else "")
            )

        return files

    def process_folder(
        self,
        folder_path: Path,
        convert_pdf: bool = True,
        recursive: bool = False,
        progress_callback: Optional[Callable] = None,
    ) -> BatchReport:
        """
        Ingest all supported files in a folder through the RAG pipeline.

        Args:
            folder_path:       Path to the folder containing documents.
            convert_pdf:       If True, PDFs are converted to MD via docling first.
            recursive:         If True, scan subfolders recursively.
            progress_callback: Optional callable(stage, file_index, total_files,
                               current_file, result) called after each file.

        Returns:
            BatchReport with full statistics and per-file results.
        """
        self._cancelled = False
        report = BatchReport()
        start_time = time.time()

        # --- Scan phase ---
        try:
            files = self.scan_folder(folder_path, recursive=recursive, convert_pdf=convert_pdf)
        except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
            logger.error("Folder scan failed: {}", exc)
            report.failed = 1
            report.errors.append({"filename": str(folder_path), "error": str(exc)})
            report.total_time_seconds = time.time() - start_time
            return report

        report.total_files = len(files)
        logger.info("BatchProcessor: found {} files to process", len(files))

        # --- Ingestion phase ---
        for idx, file_path in enumerate(files):
            if self._cancelled:
                logger.info("BatchProcessor: cancelled after {} files", idx)
                report.was_cancelled = True
                break

            if progress_callback:
                progress_callback(
                    stage="processing",
                    file_index=idx,
                    total_files=len(files),
                    current_file=file_path.name,
                    result=None,
                )

            result = self._process_single_file(file_path, convert_pdf=convert_pdf)
            report.per_file_results.append(result)

            # Accumulate stats
            if result.status == "success":
                report.successful += 1
                report.total_chunks_created += result.chunks_created
                report.total_exact_duplicates += result.exact_duplicates
                report.total_near_duplicates += result.near_duplicates
            elif result.status == "duplicate":
                report.duplicates_skipped += 1
            elif result.status == "error":
                report.failed += 1
                report.errors.append({
                    "filename": result.filename,
                    "error_type": result.error_type,
                    "error": result.error_message,
                })

            logger.info(
                "File {}/{} | {} | {} | chunks={}",
                idx + 1, len(files), result.filename, result.status, result.chunks_created,
            )

            if progress_callback:
                progress_callback(
                    stage="done",
                    file_index=idx + 1,
                    total_files=len(files),
                    current_file=file_path.name,
                    result=result,
                )

        report.total_time_seconds = time.time() - start_time
        logger.info(
            "Batch complete: {}/{} OK, {} skipped, {} failed in {:.1f}s",
            report.successful, report.total_files,
            report.duplicates_skipped, report.failed,
            report.total_time_seconds,
        )
        return report

    def _process_single_file(self, file_path: Path, convert_pdf: bool) -> FileResult:
        """
        Ingest a single file with full error handling.

        Handles:
          - Docling conversion errors (PDF → MD)
          - Parser errors (bad encoding, empty file)
          - Chunking/embedding errors
          - Unexpected exceptions
        """
        start = time.time()
        result = FileResult(
            filename=file_path.name,
            file_path=str(file_path),
            status="error",
        )

        try:
            target_path = file_path

            # Convert PDF → Markdown if needed
            if file_path.suffix.lower() == ".pdf" and convert_pdf:
                logger.debug("Converting PDF to MD: {}", file_path.name)
                target_path = self._convert_pdf_to_md(file_path)

            # Run the existing ingest_document pipeline
            ingest_result = self.rag.ingest_document(target_path)

            if ingest_result.get("status") == "duplicate":
                result.status = "duplicate"
            else:
                result.status = "success"
                result.chunks_created = ingest_result.get("chunks_created", 0)
                result.exact_duplicates = ingest_result.get("duplicates_removed", 0)
                result.version = ingest_result.get("version", 1)

        except FileNotFoundError as exc:
            result.error_type = "FileNotFoundError"
            result.error_message = f"File not found: {exc}"
            logger.error("FileNotFoundError for {}: {}", file_path.name, exc)

        except PermissionError as exc:
            result.error_type = "PermissionError"
            result.error_message = f"Cannot read file (permission denied): {exc}"
            logger.error("PermissionError for {}: {}", file_path.name, exc)

        except UnicodeDecodeError as exc:
            result.error_type = "UnicodeDecodeError"
            result.error_message = f"File has unreadable encoding: {exc}"
            logger.error("UnicodeDecodeError for {}: {}", file_path.name, exc)

        except ImportError as exc:
            result.error_type = "ImportError"
            result.error_message = (
                f"Required library not installed: {exc}. "
                "Run: pip install docling"
            )
            logger.error("ImportError for {}: {}", file_path.name, exc)

        except RuntimeError as exc:
            result.error_type = "RuntimeError"
            result.error_message = str(exc)
            logger.error("RuntimeError for {}: {}", file_path.name, exc)

        except Exception as exc:
            result.error_type = type(exc).__name__
            result.error_message = str(exc)
            logger.exception(
                "Unexpected error processing {}: {} | {}",
                file_path.name, type(exc).__name__, exc,
            )

        result.duration_seconds = time.time() - start
        return result

    def _convert_pdf_to_md(self, pdf_path: Path) -> Path:
        """
        Convert a PDF to Markdown using docling and return the path to the temp MD file.

        Raises:
            ImportError: If docling is not installed.
            RuntimeError: If docling conversion fails.
        """
        try:
            from docling.document_converter import DocumentConverter
        except ImportError:
            raise ImportError(
                "docling is not installed. Run: pip install docling==2.98.0"
            )

        try:
            converter = DocumentConverter()
            result = converter.convert(str(pdf_path))
            md_content = result.document.export_to_markdown()

            if not md_content or not md_content.strip():
                raise RuntimeError(
                    f"Docling returned empty content for: {pdf_path.name}"
                )

            # Write to a temp file with the same stem name so the parser picks
            # up the correct document title
            tmp_dir = Path(tempfile.mkdtemp())
            md_path = tmp_dir / (pdf_path.stem + ".md")
            md_path.write_text(md_content, encoding="utf-8")

            logger.debug(
                "Converted {} → MD ({} chars)", pdf_path.name, len(md_content)
            )
            return md_path

        except Exception as exc:
            if isinstance(exc, (ImportError, RuntimeError)):
                raise
            raise RuntimeError(
                f"Docling failed to convert '{pdf_path.name}': {exc}"
            ) from exc
