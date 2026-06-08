# src/reranking/source_confidence.py
"""
Source confidence scoring for trustworthiness assessment.
Combines multiple signals: freshness, extraction quality, retrieval consistency.
"""

from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional
import numpy as np
from loguru import logger

from src.config import settings
from src.retrieval.hybrid import RetrievalResult


@dataclass
class ConfidenceScores:
    """Breakdown of confidence components."""
    freshness_score: float      # How recent is the source?
    trust_score: float          # Extraction quality, source type
    consistency_score: float    # Do multiple chunks agree?
    retrieval_score: float      # How well did it match the query?
    overall_confidence: float   # Weighted combination
    
    
@dataclass
class ScoredResult:
    """Retrieval result with confidence scores."""
    result: RetrievalResult
    confidence: ConfidenceScores
    final_score: float


class SourceConfidenceScorer:
    """
    Score source trustworthiness based on multiple factors.
    
    This is crucial for hallucination prevention - we want to:
    1. Prefer recent documents over outdated ones
    2. Trust native text extraction over OCR
    3. Require consistency across multiple retrieved chunks
    4. Factor in retrieval score (relevance to query)
    """
    
    def __init__(
        self,
        freshness_decay_days: int = None,
        min_trust_score: float = None,
        consistency_threshold: float = None
    ):
        self.freshness_decay_days = (
            freshness_decay_days or settings.confidence.freshness_decay_days
        )
        self.min_trust_score = (
            min_trust_score or settings.confidence.min_source_trust
        )
        self.consistency_threshold = (
            consistency_threshold or settings.confidence.consistency_threshold
        )
        
        # Weights for combining scores
        self.weights = {
            'freshness': 0,
            'trust': 0,
            'consistency': 0.5,
            'retrieval': 0.5
        }
        
    def score_results(
        self,
        results: list[RetrievalResult],
        query: str
    ) -> list[ScoredResult]:
        """
        Score all results and compute consistency.
        
        Args:
            results: Reranked retrieval results
            query: Original query (for consistency analysis)
            
        Returns:
            Results with confidence scores
        """
        if not results:
            return []
            
        # Calculate individual scores
        scored = []
        for result in results:
            freshness = self._score_freshness(result)
            trust = self._score_trust(result)
            retrieval = self._normalize_retrieval_score(result)
            
            scored.append({
                'result': result,
                'freshness': freshness,
                'trust': trust,
                'retrieval': retrieval
            })
            
        # Calculate consistency (requires multiple results)
        consistency_scores = self._score_consistency(scored)
        
        # Combine into final scores
        final_results = []
        for item, consistency in zip(scored, consistency_scores):
            confidence = ConfidenceScores(
                freshness_score=item['freshness'],
                trust_score=item['trust'],
                consistency_score=consistency,
                retrieval_score=item['retrieval'],
                overall_confidence=self._compute_overall(
                    item['freshness'],
                    item['trust'],
                    consistency,
                    item['retrieval']
                )
            )
            
            final_results.append(ScoredResult(
                result=item['result'],
                confidence=confidence,
                final_score=confidence.overall_confidence * item['retrieval']
            ))
            
        # Sort by final score
        final_results.sort(key=lambda x: x.final_score, reverse=True)
        
        return final_results
    
    def _score_freshness(self, result: RetrievalResult) -> float:
        """Score based on document age."""
        metadata = result.chunk.metadata
        
        # Try creation date, then ingestion timestamp
        date_str = metadata.get('creation_date') or metadata.get('ingestion_timestamp')
        
        if not date_str:
            return 0.7  # Unknown date gets moderate score
            
        try:
            doc_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            return 0.7
            
        # Calculate age in days
        now = datetime.now(timezone.utc)
        if doc_date.tzinfo is None:
            doc_date = doc_date.replace(tzinfo=timezone.utc)
        age_days = (now - doc_date).days
        
        # Exponential decay
        decay_rate = 1 / self.freshness_decay_days
        freshness = np.exp(-decay_rate * age_days)
        
        return max(0.1, min(1.0, freshness))
    
    def _score_trust(self, result: RetrievalResult) -> float:
        """Score based on source reliability."""
        metadata = result.chunk.metadata
        
        base_score = 0.8  # Default trust
        
        # Penalize OCR'd documents (less reliable)
        if metadata.get('is_scanned', False):
            base_score *= 0.85
            
        # Factor in extraction confidence
        extraction_conf = metadata.get('extraction_confidence', 1.0)
        base_score *= extraction_conf
        
        # Boost for known reliable metadata
        if metadata.get('author'):
            base_score *= 1.05
        if metadata.get('title'):
            base_score *= 1.02
            
        return min(1.0, base_score)
    
    def _score_consistency(
        self, 
        scored_items: list[dict]
    ) -> list[float]:
        """
        Score consistency across retrieved chunks.
        
        If multiple chunks from the same document agree, boost confidence.
        If information contradicts across documents, lower confidence.
        """
        if len(scored_items) < 2:
            return [0.7] * len(scored_items)  # No comparison possible
            
        # Group by document
        doc_chunks = {}
        for idx, item in enumerate(scored_items):
            doc_id = item['result'].chunk.doc_id
            if doc_id not in doc_chunks:
                doc_chunks[doc_id] = []
            doc_chunks[doc_id].append(idx)
            
        # Calculate consistency scores
        consistency_scores = []
        
        for idx, item in enumerate(scored_items):
            doc_id = item['result'].chunk.doc_id
            same_doc_chunks = doc_chunks[doc_id]
            
            if len(same_doc_chunks) > 1:
                # Multiple chunks from same doc - high consistency
                consistency = 0.9
            elif len(doc_chunks) == 1:
                # All from same doc
                consistency = 0.85
            else:
                # Information from multiple documents
                # This is actually good for verification
                consistency = 0.8
                
            consistency_scores.append(consistency)
            
        return consistency_scores
    
    def _normalize_retrieval_score(self, result: RetrievalResult) -> float:
        """Normalize retrieval score to 0-1 range."""
        # Cross-encoder scores are typically 0-1
        # But we ensure it's bounded
        score = result.combined_score
        return max(0.0, min(1.0, score))
    
    def _compute_overall(
        self,
        freshness: float,
        trust: float,
        consistency: float,
        retrieval: float
    ) -> float:
        """Compute weighted overall confidence."""
        overall = (
            self.weights['freshness'] * freshness +
            self.weights['trust'] * trust +
            self.weights['consistency'] * consistency +
            self.weights['retrieval'] * retrieval
        )
        return overall
    
    def passes_threshold(self, scored_result: ScoredResult) -> bool:
        """Check if result meets minimum confidence thresholds."""
        conf = scored_result.confidence
        
        return (
            conf.overall_confidence >= settings.confidence.min_retrieval_score and
            conf.trust_score >= self.min_trust_score
        )
