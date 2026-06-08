# src/retrieval/bm25.py
"""
BM25 sparse retrieval using the rank_bm25 library.
Complementing semantic search with exact keyword matching.
"""

import pickle
from pathlib import Path
from typing import Optional
import numpy as np
from dataclasses import dataclass
from rank_bm25 import BM25Okapi

from src.ingestion.normalizer import TextNormalizer


@dataclass
class BM25Result:
    """Single BM25 retrieval result."""
    chunk_id: str
    score: float
    matched_terms: list[str]


class BM25Index:
    """BM25 retrieval implemented with rank_bm25."""

    def __init__(
        self,
        storage_path: Optional[Path] = None
    ):
        self.storage_path = Path(storage_path) if storage_path else None
        self.normalizer = TextNormalizer()

        self.chunk_ids: list[str] = []
        self.doc_lengths: list[int] = []
        self.avg_doc_length: float = 0
        self.vocabulary: set[str] = set()
        self.tokenized_corpus: list[list[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self._built = False

    def build_index(self, chunks: list) -> None:
        """Build BM25 index from chunks."""
        self.chunk_ids = []
        self.doc_lengths = []
        self.tokenized_corpus = []
        self.vocabulary = set()

        for chunk in chunks:
            normalized = self.normalizer.normalize_for_bm25(chunk.content)
            tokens = normalized.split()

            self.chunk_ids.append(chunk.chunk_id)
            self.doc_lengths.append(len(tokens))
            self.tokenized_corpus.append(tokens)
            self.vocabulary.update(tokens)

        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        self.bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None
        self._built = self.bm25 is not None

        if self.storage_path:
            self._save_index()

    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> list[BM25Result]:
        """Search the index and return top-k results."""
        if not self._built or self.bm25 is None:
            raise RuntimeError("Index not built. Call build_index first.")

        normalized_query = self.normalizer.normalize_for_bm25(query)
        query_terms = normalized_query.split()

        if not query_terms:
            return []

        scores = self.bm25.get_scores(query_terms)
        ranked_indices = np.argsort(scores)[::-1]

        results: list[BM25Result] = []
        query_terms_set = set(query_terms)
        for idx in ranked_indices:
            score = float(scores[idx])
            if score <= 0:
                continue

            matched_terms = sorted(query_terms_set.intersection(self.tokenized_corpus[idx]))
            results.append(BM25Result(
                chunk_id=self.chunk_ids[idx],
                score=score,
                matched_terms=matched_terms
            ))

            if len(results) >= top_k:
                break

        return results

    def _save_index(self):
        """Persist index to disk."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        index_file = self.storage_path / "bm25_index.pkl"

        index_data = {
            'chunk_ids': self.chunk_ids,
            'doc_lengths': self.doc_lengths,
            'avg_doc_length': self.avg_doc_length,
            'vocabulary': list(self.vocabulary),
            'tokenized_corpus': self.tokenized_corpus,
        }

        with open(index_file, 'wb') as f:
            pickle.dump(index_data, f)

    def load_index(self) -> bool:
        """Load index from disk."""
        if not self.storage_path:
            return False

        index_file = self.storage_path / "bm25_index.pkl"
        if not index_file.exists():
            return False

        with open(index_file, 'rb') as f:
            data = pickle.load(f)

        self.chunk_ids = data.get('chunk_ids', [])
        self.doc_lengths = data.get('doc_lengths', [])
        self.avg_doc_length = data.get('avg_doc_length', 0)
        self.vocabulary = set(data.get('vocabulary', []))
        self.tokenized_corpus = data.get('tokenized_corpus', [])

        if not self.tokenized_corpus and 'term_freqs' in data:
            self.tokenized_corpus = []
            for tf in data['term_freqs']:
                tokens: list[str] = []
                for term, count in tf.items():
                    tokens.extend([term] * count)
                self.tokenized_corpus.append(tokens)

        self.bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None
        self._built = self.bm25 is not None
        return self._built
    
    @property
    def is_built(self) -> bool:
        """Check if index has been built."""
        return self._built
    
    def add_documents(self, chunks: list) -> None:
        """Add new documents to existing index incrementally."""
        if not self._built:
            raise RuntimeError("Index not built. Call build_index first.")
        
        for chunk in chunks:
            normalized = self.normalizer.normalize_for_bm25(chunk.content)
            tokens = normalized.split()
            
            self.chunk_ids.append(chunk.chunk_id)
            self.doc_lengths.append(len(tokens))
            self.tokenized_corpus.append(tokens)
            self.vocabulary.update(tokens)
        
        # Rebuild BM25 with all documents
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        if self.storage_path:
            self._save_index()
