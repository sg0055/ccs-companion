# src/ingestion/parser.py
"""
Lightweight Markdown document parsing and metadata extraction.
"""

import hashlib
import os
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DocumentMetadata:
    """Rich metadata for source tracking and trust scoring."""
    doc_id: str
    filename: str
    file_hash: str  
    title: Optional[str] = None
    author: Optional[str] = None
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    ingestion_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source_type: str = "markdown"
    version: int = 1


@dataclass
class ParsedDocument:
    """Complete parsed document ready for the MarkdownChunker."""
    metadata: DocumentMetadata
    raw_text: str  


class DocumentParser:
    """
    Reads prepared Markdown files and generates rich tracking metadata
    before passing the text to the chunking layer.
    """
    
    def parse(self, file_path: Path) -> ParsedDocument:
        """Parse MD file and extract content with metadata."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
            
        # Compute file hash for deduplication
        file_hash = self._compute_hash(file_path)
        
        
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
            
        # Extract basic OS-level metadata
        stat = file_path.stat()
        creation_date = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)
        mod_date = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        
        metadata = DocumentMetadata(
            doc_id=file_hash[:16],  
            filename=file_path.name,
            file_hash=file_hash,
            title=file_path.stem.replace("_", " ").title(), 
            creation_date=creation_date,
            modification_date=mod_date,
        )
        
        print(f"📄 Loaded {file_path.name}: {len(raw_text)} chars")
        
        return ParsedDocument(
            metadata=metadata,
            raw_text=raw_text
        )
    
    def _compute_hash(self, file_path: Path) -> str:
        """SHA-256 hash for deduplication."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()