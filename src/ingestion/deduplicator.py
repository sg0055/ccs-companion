# src/ingestion/deduplicator.py
"""
Multi-level deduplication to prevent index bloat and retrieval noise.
"""

import hashlib
from typing import Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class DeduplicationResult:
    """Result of deduplication process."""
    original_count: int
    deduplicated_count: int
    exact_duplicates: int
    near_duplicates: int
    duplicate_map: dict[str, str]  


class ChunkDeduplicator:
    """
    Three-level deduplication:
    1. Exact hash match (identical content)
    2. Near-duplicate detection (high similarity)
    3. Cross-document deduplication
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.95,
        use_minhash: bool = True
    ):
        self.similarity_threshold = similarity_threshold
        self.use_minhash = use_minhash
        self.hash_index: dict[str, str] = {}  
        
    def deduplicate(
        self, 
        chunks: list,  
        cross_document: bool = True
    ) -> tuple[list, DeduplicationResult]:
        """
        Remove duplicate chunks, keeping the first occurrence.
        
        Returns:
            Tuple of (deduplicated_chunks, deduplication_stats)
        """
        exact_duplicates = 0
        near_duplicates = 0
        duplicate_map = {}
        unique_chunks = []
        
        
        content_hashes = {}
        for chunk in chunks:
            content_hash = self._hash_content(chunk.content)
            
            if content_hash in content_hashes:
                
                exact_duplicates += 1
                duplicate_map[chunk.chunk_id] = content_hashes[content_hash]
            else:
                content_hashes[content_hash] = chunk.chunk_id
                unique_chunks.append(chunk)
                
        
        if len(unique_chunks) > 1:
            unique_chunks, near_dups, near_map = self._find_near_duplicates(
                unique_chunks
            )
            near_duplicates = near_dups
            duplicate_map.update(near_map)
            
        return unique_chunks, DeduplicationResult(
            original_count=len(chunks),
            deduplicated_count=len(unique_chunks),
            exact_duplicates=exact_duplicates,
            near_duplicates=near_duplicates,
            duplicate_map=duplicate_map
        )
    
    def _hash_content(self, content: str) -> str:
        """Create normalized hash of content."""
        
        normalized = content.lower().strip()
        normalized = ' '.join(normalized.split())  
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _find_near_duplicates(
        self, 
        chunks: list
    ) -> tuple[list, int, dict]:
        """Find and remove near-duplicate chunks using TF-IDF."""
        if len(chunks) < 2:
            return chunks, 0, {}
            
        
        texts = [c.content for c in chunks]
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
        except ValueError:
           
            return chunks, 0, {}
            
        
        similarities = cosine_similarity(tfidf_matrix)
        
        
        duplicates = set()
        duplicate_map = {}
        
        for i in range(len(chunks)):
            if i in duplicates:
                continue
            for j in range(i + 1, len(chunks)):
                if j in duplicates:
                    continue
                if similarities[i, j] >= self.similarity_threshold:
                    
                    duplicates.add(j)
                    duplicate_map[chunks[j].chunk_id] = chunks[i].chunk_id
                    
        
        unique = [c for idx, c in enumerate(chunks) if idx not in duplicates]
        
        return unique, len(duplicates), duplicate_map


class DocumentDeduplicator:
    """Document-level deduplication based on file hash."""
    
    def __init__(self):
        self.document_hashes: dict[str, str] = {}  
        
    def is_duplicate(self, file_hash: str, doc_id: str) -> Optional[str]:
        """
        Check if document is a duplicate.
        Returns existing doc_id if duplicate, None otherwise.
        """
        if file_hash in self.document_hashes:
            return self.document_hashes[file_hash]
        
        self.document_hashes[file_hash] = doc_id
        return None
