# src/retrieval/hybrid.py
"""
Hybrid retrieval combining BM25 and semantic search.
Reciprocal Rank Fusion for score combination.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from loguru import logger

from src.config import settings
from src.retrieval.embedder import EmbeddingManager
from src.retrieval.bm25 import BM25Index, BM25Result
from src.retrieval.ann import ANNIndex
from src.ingestion.chunker import Chunk


@dataclass
class RetrievalResult:
    """A single retrieval result with scores from both methods."""
    chunk_id: str
    chunk: Chunk
    semantic_score: float
    bm25_score: float
    combined_score: float
    matched_terms: list[str]  # BM25 matched terms
    rank_semantic: int
    rank_bm25: int
    

class HybridRetriever:
    """
    Combine BM25 keyword search with semantic vector search.
    
    Uses Reciprocal Rank Fusion (RRF) to combine rankings:
    RRF(d) = Σ 1/(k + rank(d))
    
    where k is a constant (typically 60) and rank(d) is the 
    position of document d in each result list.
    """
    
    def __init__(
        self,
        embedder: EmbeddingManager,
        ann_index: ANNIndex,
        bm25_index: BM25Index,
        chunk_store: dict[str, Chunk],  # chunk_id -> Chunk
        bm25_weight: float = None,
        semantic_weight: float = None,
        rrf_k: int = 60
    ):
        self.embedder = embedder
        self.ann_index = ann_index
        self.bm25_index = bm25_index
        self.chunk_store = chunk_store
        
        self.bm25_weight = bm25_weight or settings.retrieval.bm25_weight
        self.semantic_weight = semantic_weight or settings.retrieval.semantic_weight
        self.rrf_k = rrf_k
        
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        use_bm25: bool = True,
        use_semantic: bool = True
    ) -> list[RetrievalResult]:
        """
        Perform hybrid retrieval combining both methods.
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_bm25: Whether to include BM25 results
            use_semantic: Whether to include semantic results
            
        Returns:
            List of RetrievalResult sorted by combined score
        """
        top_k = top_k or settings.retrieval.final_top_k
        initial_k = settings.retrieval.initial_candidates
        
        semantic_results = {}
        bm25_results = {}
        
        # Semantic search
        if use_semantic:
            query_embedding = self.embedder.embed_query(query)
            ann_results = self.ann_index.search(query_embedding, top_k=initial_k)
            
            for rank, (chunk_id, score) in enumerate(ann_results, 1):
                semantic_results[chunk_id] = {
                    'score': score,
                    'rank': rank
                }
                
        # BM25 search
        if use_bm25:
            bm25_hits = self.bm25_index.search(query, top_k=initial_k)
            
            for rank, hit in enumerate(bm25_hits, 1):
                bm25_results[hit.chunk_id] = {
                    'score': hit.score,
                    'rank': rank,
                    'matched_terms': hit.matched_terms
                }
                
        # Combine results using RRF
        all_chunk_ids = set(semantic_results.keys()) | set(bm25_results.keys())
        
        combined_results = []
        for chunk_id in all_chunk_ids:
            # Get chunk from store
            chunk = self.chunk_store.get(chunk_id)
            if chunk is None:
                logger.warning(f"Chunk {chunk_id} not found in store")
                continue
                
            # Calculate RRF score
            rrf_score = 0.0
            sem_score = 0.0
            #sem_rank = len(all_chunk_ids) + 1  # Default rank if not found
            bm25_score = 0.0
            #bm25_rank = len(all_chunk_ids) + 1
            matched_terms = []
            # A stable ceiling representing "just outside our search limit"
            fallback_rank = initial_k + 1
            sem_rank = fallback_rank
            bm25_rank = fallback_rank
            
            if chunk_id in semantic_results:
                sem_data = semantic_results[chunk_id]
                sem_score = sem_data['score']
                sem_rank = sem_data['rank']
                rrf_score += self.semantic_weight / (self.rrf_k + sem_rank)
                
            if chunk_id in bm25_results:
                bm25_data = bm25_results[chunk_id]
                bm25_score = bm25_data['score']
                bm25_rank = bm25_data['rank']
                matched_terms = bm25_data['matched_terms']
                rrf_score += self.bm25_weight / (self.rrf_k + bm25_rank)
                
            combined_results.append(RetrievalResult(
                chunk_id=chunk_id,
                chunk=chunk,
                semantic_score=sem_score,
                bm25_score=bm25_score,
                combined_score=rrf_score,
                matched_terms=matched_terms,
                rank_semantic=sem_rank,
                rank_bm25=bm25_rank
            ))
            
        # Sort by combined score
        combined_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        logger.debug(
            f"Hybrid retrieval: {len(semantic_results)} semantic, "
            f"{len(bm25_results)} BM25, {len(combined_results)} combined"
        )
        
        return combined_results[:top_k]
    
    def retrieve_semantic_only(
        self, 
        query: str, 
        top_k: int = None
    ) -> list[RetrievalResult]:
        """Semantic-only retrieval (useful for certain query types)."""
        return self.retrieve(query, top_k, use_bm25=False, use_semantic=True)
    
    def retrieve_keyword_only(
        self, 
        query: str, 
        top_k: int = None
    ) -> list[RetrievalResult]:
        """BM25-only retrieval (useful for exact term matching)."""
        return self.retrieve(query, top_k, use_bm25=True, use_semantic=False)

"""Why hybrid retrieval matters:

BM25 catches exact matches: When users search for specific terms, acronyms, or names
Semantic catches meaning: When users describe concepts in their own words
RRF is robust: Doesn't require score normalization across different methods"""