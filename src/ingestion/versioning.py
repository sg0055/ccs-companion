# src/ingestion/versioning.py
"""
Document versioning for tracking updates and maintaining history.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import json
from pathlib import Path


@dataclass
class DocumentVersion:
    """A specific version of a document."""
    version_id: str
    doc_id: str
    version_number: int
    file_hash: str
    chunk_ids: list[str]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True
    change_summary: Optional[str] = None


class VersionManager:
    """
    Track document versions and handle updates.
    
    When a document is re-uploaded:
    1. Compare hashes to detect actual changes
    2. If changed, create new version
    3. Mark old chunks as inactive (but retain for history)
    4. Update active version pointer
    """
    
    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.versions_file = self.storage_path / "versions.json"
        self._versions: dict[str, list[DocumentVersion]] = {}
        self._load_versions()
        
    def _load_versions(self):
        """Load version history from disk."""
        if self.versions_file.exists():
            with open(self.versions_file, 'r') as f:
                data = json.load(f)
                for doc_id, versions in data.items():
                    self._versions[doc_id] = [
                        DocumentVersion(**v) for v in versions
                    ]
                    
    def _save_versions(self):
        """Persist version history to disk."""
        data = {}
       
        for doc_id, versions in self._versions.items():
           
            data[doc_id] = [
                {
                    'version_id': v.version_id,
                    'doc_id': v.doc_id,
                    'version_number': v.version_number,
                    'file_hash': v.file_hash,
                    'chunk_ids': v.chunk_ids,
                    'created_at': v.created_at if isinstance(v.created_at, str) else v.created_at.isoformat(),
                    'is_active': v.is_active,
                    'change_summary': v.change_summary
                }
                for v in versions
            ]
        with open(self.versions_file, 'w') as f:
            json.dump(data, f, indent=2)
            
    def create_version(
        self, 
        doc_id: str, 
        file_hash: str, 
        chunk_ids: list[str],
        change_summary: Optional[str] = None
    ) -> DocumentVersion:
        """Create a new version for a document."""
        existing = self._versions.get(doc_id, [])
        
        
        if existing:
            latest = max(existing, key=lambda v: v.version_number)
            if latest.file_hash == file_hash:
                
                return latest
                
           
            latest.is_active = False
            version_number = latest.version_number + 1
        else:
            version_number = 1
            
        
        version = DocumentVersion(
            version_id=f"{doc_id}_v{version_number}",
            doc_id=doc_id,
            version_number=version_number,
            file_hash=file_hash,
            chunk_ids=chunk_ids,
            change_summary=change_summary
        )
        
        if doc_id not in self._versions:
            self._versions[doc_id] = []
        self._versions[doc_id].append(version)
        
        self._save_versions()
        return version
    
    def get_active_version(self, doc_id: str) -> Optional[DocumentVersion]:
        """Get the currently active version of a document."""
        versions = self._versions.get(doc_id, [])
        active = [v for v in versions if v.is_active]
        return active[0] if active else None
    
    def get_version_history(self, doc_id: str) -> list[DocumentVersion]:
        """Get all versions of a document."""
        return sorted(
            self._versions.get(doc_id, []),
            key=lambda v: v.version_number,
            reverse=True
        )
    
    def get_active_chunk_ids(self) -> set[str]:
        """Get all chunk IDs from active document versions."""
        chunk_ids = set()
        for versions in self._versions.values():
            for version in versions:
                if version.is_active:
                    chunk_ids.update(version.chunk_ids)
        return chunk_ids
