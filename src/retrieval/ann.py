# src/retrieval/ann.py
"""
Approximate Nearest Neighbor search using FAISS.
Provides fast semantic retrieval at scale.
"""

import numpy as np
from pathlib import Path
from typing import Optional
import faiss
from loguru import logger

from src.config import settings


class ANNIndex:
    """
    FAISS-based ANN index for fast similarity search.
    
    Index types by use case:
    - Small (<10K): Flat index (exact, no approximation)
    - Medium (10K-1M): IVF index (inverted file)
    - Large (>1M): HNSW or IVF-PQ
    """
    
    def __init__(
        self, 
        dimension: int = None,
        index_type: str = "auto",
        storage_path: Optional[Path] = None
    ):
        self.dimension = dimension or settings.embedding.dimension
        self.index_type = index_type
        self.storage_path = Path(storage_path) if storage_path else None
        
        self.index: Optional[faiss.Index] = None
        self.chunk_ids: list[str] = []
        self._trained = False
    
    @property
    def is_built(self) -> bool:
        """Check if index has been built."""
        return self._trained
        
    def build_index(
        self, 
        embeddings: np.ndarray,
        chunk_ids: list[str],
        train_size: int = 10000
    ) -> None:
        """
        Build the ANN index from embeddings.
        
        Args:
            embeddings: (N, D) array of embeddings
            chunk_ids: Corresponding chunk identifiers
            train_size: Number of vectors to use for training (IVF indexes)
        """
        n_vectors = len(embeddings)
        self.chunk_ids = chunk_ids
        
        # Choose index type based on corpus size
        if self.index_type == "auto":
            if n_vectors < 10000:
                self.index_type = "flat"
            elif n_vectors < 100000:
                self.index_type = "ivf"
            else:
                self.index_type = "hnsw"
                
        logger.info(
            f"Building {self.index_type} index for {n_vectors} vectors "
            f"(dim={self.dimension})"
        )
        
        # Create index
        if self.index_type == "flat":
            # Exact search - best for small corpora
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for normalized vectors
            
        elif self.index_type == "ivf":
            # IVF for medium corpora
            n_lists = int(np.sqrt(n_vectors))  # Rule of thumb
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer, self.dimension, n_lists, faiss.METRIC_INNER_PRODUCT
            )
            # Train on subset
            train_vectors = embeddings[:min(train_size, n_vectors)]
            self.index.train(train_vectors.astype(np.float32))
            self.index.nprobe = min(10, n_lists)  # Search this many lists
            
        elif self.index_type == "hnsw":
            # HNSW for large corpora
            self.index = faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 64
            
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
            
        # Add vectors
        self.index.add(embeddings.astype(np.float32))
        self._trained = True
        
        if self.storage_path:
            self._save_index()
            
        logger.info(f"Index built: {self.index.ntotal} vectors indexed")
        
    def search(
        self, 
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> list[tuple[str, float]]:
        """
        Search for nearest neighbors.
        
        Returns:
            List of (chunk_id, score) tuples
        """
        if not self._trained:
            raise RuntimeError("Index not built. Call build_index first.")
            
        # Ensure correct shape
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        # Search
        scores, indices = self.index.search(
            query_embedding.astype(np.float32), 
            top_k
        )
        
        # Map back to chunk IDs
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:  # FAISS returns -1 for not-found
                continue
            results.append((self.chunk_ids[idx], float(score)))
            
        return results
    
    def batch_search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 10
    ) -> list[list[tuple[str, float]]]:
        """Search for multiple queries at once."""
        if not self._trained:
            raise RuntimeError("Index not built. Call build_index first.")
            
        scores, indices = self.index.search(
            query_embeddings.astype(np.float32),
            top_k
        )
        
        all_results = []
        for query_indices, query_scores in zip(indices, scores):
            results = []
            for idx, score in zip(query_indices, query_scores):
                if idx < 0:
                    continue
                results.append((self.chunk_ids[idx], float(score)))
            all_results.append(results)
            
        return all_results
    
    def _save_index(self):
        """Save index to disk."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(
            self.index, 
            str(self.storage_path / "faiss_index.bin")
        )
        
        # Save chunk ID mapping
        import json
        with open(self.storage_path / "chunk_ids.json", 'w') as f:
            json.dump(self.chunk_ids, f)
            
    def load_index(self) -> bool:
        """Load index from disk."""
        if not self.storage_path:
            return False
            
        index_path = self.storage_path / "faiss_index.bin"
        ids_path = self.storage_path / "chunk_ids.json"
        
        if not index_path.exists() or not ids_path.exists():
            return False
            
        self.index = faiss.read_index(str(index_path))
        
        import json
        with open(ids_path, 'r') as f:
            self.chunk_ids = json.load(f)
            
        self._trained = True
        return True
    
    def add_vectors(
        self, 
        embeddings: np.ndarray, 
        chunk_ids: list[str]
    ) -> None:
        """Add new vectors to existing index."""
        if not self._trained:
            raise RuntimeError("Index not built. Call build_index first.")
            
        self.index.add(embeddings.astype(np.float32))
        self.chunk_ids.extend(chunk_ids)
        
        if self.storage_path:
            self._save_index()
