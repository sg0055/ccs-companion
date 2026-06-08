# src/reranking/cross_encoder.py
"""
Cross-encoder reranking for precise relevance scoring.
More accurate than bi-encoders but slower (only on top candidates).
"""

import numpy as np
from typing import Optional
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from loguru import logger

from src.config import settings
from src.retrieval.hybrid import RetrievalResult


class CrossEncoderReranker:
    """
    Cross-encoder model for precise query-passage relevance scoring.
    
    Unlike bi-encoders (which encode query and passage separately),
    cross-encoders process the pair together, enabling full attention
    between query and passage tokens. This gives much better accuracy
    but is too slow for initial retrieval.
    
    Use case: Re-rank top-50 candidates to get final top-5.
    """
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.reranker.model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading cross-encoder {self.model_name} on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        ).to(self.device)
        self.model.eval()
        
    @torch.no_grad()
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = None
    ) -> list[RetrievalResult]:
        """
        Rerank retrieval results using cross-encoder scores.
        
        Args:
            query: Original query
            results: Initial retrieval results
            top_k: Number of results to return
            
        Returns:
            Reranked results with updated scores
        """
        top_k = top_k or settings.reranker.top_k
        
        if not results:
            return []
            
        # Prepare query-passage pairs
        pairs = [(query, r.chunk.content) for r in results]
        
        # Batch scoring
        scores = self._score_pairs(pairs)
        
       
        reranked = []
        for result, ce_score in zip(results, scores):
           
            reranked.append(RetrievalResult(
                chunk_id=result.chunk_id,
                chunk=result.chunk,
                semantic_score=result.semantic_score,
                bm25_score=result.bm25_score,
                combined_score=ce_score,  # Replace with CE score
                matched_terms=result.matched_terms,
                rank_semantic=result.rank_semantic,
                rank_bm25=result.rank_bm25
            ))
            
        # Sort by cross-encoder score
        reranked.sort(key=lambda x: x.combined_score, reverse=True)
        
        logger.debug(f"Reranked {len(results)} -> top {top_k}")
        
        return reranked[:top_k]
    
    def _score_pairs(
        self, 
        pairs: list[tuple[str, str]],
        batch_size: int = None
    ) -> list[float]:
        """Score query-passage pairs in batches."""
        batch_size = batch_size or settings.reranker.batch_size
        all_scores = []
        
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                [p[0] for p in batch],
                [p[1] for p in batch],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Score
            outputs = self.model(**inputs)
            scores = outputs.logits.squeeze(-1)
            
            # Convert to probabilities if multi-class, else use raw scores
            if scores.dim() > 1:
                scores = torch.softmax(scores, dim=-1)[:, 1]
            else:
                scores = torch.sigmoid(scores)
                
            all_scores.extend(scores.cpu().tolist())
            
        return all_scores
    
    def score_single(self, query: str, passage: str) -> float:
        """Score a single query-passage pair."""
        return self._score_pairs([(query, passage)])[0]
