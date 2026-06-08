# src/retrieval/embedder.py
"""
Embedding generation with batching, caching, and normalization.
"""

import numpy as np
from typing import Union
import torch
from sentence_transformers import SentenceTransformer
from loguru import logger

from src.config import settings


class EmbeddingManager:
    """
    Generate and manage embeddings with:
    - Efficient batching
    - GPU acceleration when available
    - L2 normalization for cosine similarity
    - Instruction-aware embeddings (for asymmetric retrieval)
    """
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.embedding.model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading embedding model {self.model_name} on {self.device}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        
        self.dimension = self.model.get_embedding_dimension()
        
        # Instruction prefixes for asymmetric retrieval
        # BGE models perform better with these
        self.query_prefix = "Represent this sentence for searching relevant passages: "
        self.passage_prefix = ""  # No prefix for passages
        
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a search query with instruction prefix."""
        prefixed_query = self.query_prefix + query
        embedding = self.model.encode(
            prefixed_query,
            normalize_embeddings=settings.embedding.normalize,
            convert_to_numpy=True
        )
        return embedding
    
    def embed_passages(
        self, 
        passages: list[str],
        batch_size: int = None,
        show_progress: bool = False
    ) -> np.ndarray:
        """Embed multiple passages efficiently."""
        batch_size = batch_size or settings.embedding.batch_size
        
        # Add passage prefix if specified
        if self.passage_prefix:
            passages = [self.passage_prefix + p for p in passages]
            
        embeddings = self.model.encode(
            passages,
            batch_size=batch_size,
            normalize_embeddings=settings.embedding.normalize,
            convert_to_numpy=True,
            show_progress_bar=show_progress
        )
        
        return embeddings
    
    def embed_single(self, text: str, is_query: bool = False) -> np.ndarray:
        """Embed a single text."""
        if is_query:
            return self.embed_query(text)
        return self.embed_passages([text])[0]
    
    def compute_similarity(
        self, 
        query_embedding: np.ndarray, 
        passage_embeddings: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarities between query and passages."""
        # If embeddings are normalized, dot product = cosine similarity
        if settings.embedding.normalize:
            return np.dot(passage_embeddings, query_embedding)
        else:
            # Manual cosine similarity
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            passage_norms = passage_embeddings / np.linalg.norm(
                passage_embeddings, axis=1, keepdims=True
            )
            return np.dot(passage_norms, query_norm)
