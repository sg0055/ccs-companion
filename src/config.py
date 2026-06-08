# src/config.py
"""
Central configuration for the RAG system.
All thresholds, model names, and paths are defined here.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from pathlib import Path
import os


class EmbeddingConfig(BaseSettings):
    """Configuration for embedding models."""
    model_name: str = "BAAI/bge-large-en-v1.5"
    dimension: int = 1024
    batch_size: int = 64
    normalize: bool = True
    
    
class RerankerConfig(BaseSettings):
    """Configuration for cross-encoder reranking."""
    #model_name: str = "BAAI/bge-reranker-v2-m3"
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2" # A smaller model for faster reranking during testing
    top_k: int = 5 
    batch_size: int = 32
    

class RetrievalConfig(BaseSettings):
    """Configuration for hybrid retrieval."""
    bm25_weight: float = 0.3
    semantic_weight: float = 0.7
    initial_candidates: int = 10  # Candidates before reranking
    final_top_k: int = 5
    

class ConfidenceConfig(BaseSettings):
    """Thresholds for hallucination prevention."""
    min_retrieval_score: float = 0.65  
    min_source_trust: float = 0.5
    freshness_decay_days: int = 365 
    consistency_threshold: float = 0.7  
    citation_required: bool = True
    # If true, force fallback for all queries (useful for testing)
    force_fallback: bool = False
    debug_mode: bool = False # If true, include detailed debug info in responses
    

class ChunkingConfig(BaseSettings):
    """Configuration for document chunking."""
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 100
    

class CacheConfig(BaseSettings):
    """Caching configuration."""
    enabled: bool = True
    query_cache_ttl: int = 3600  # 1 hour
    embedding_cache_ttl: int = 86400  # 24 hours
    max_cache_size: int = 10000
    redis_url: Optional[str] = "redis://localhost:6379"
    

class LLMConfig(BaseSettings):
    """LLM configuration for generation."""
    model_name: str = "gemma3:1b"  # Or local model like gemma3-1b / ollama:gemma3-1b
    temperature: float = 0.1  # Low temperature for factual responses
    max_tokens: int = 1000
    api_key: Optional[str] = Field(default="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", alias="OPENAI_API_KEY")
    

class StorageConfig(BaseSettings):
    """Storage paths and database configuration."""
    data_dir: Path = Path("./data")
    vector_db_path: Path = Path("./data/vector_store")
    bm25_index_path: Path = Path("./data/bm25_index")
    document_store_path: Path = Path("./data/documents")
    

class Settings(BaseSettings):
    """Master configuration combining all sub-configs."""
    embedding: EmbeddingConfig = EmbeddingConfig()
    reranker: RerankerConfig = RerankerConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    confidence: ConfidenceConfig = ConfidenceConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    cache: CacheConfig = CacheConfig()
    llm: LLMConfig = LLMConfig()
    storage: StorageConfig = StorageConfig()

    
    # Observability
    enable_tracing: bool = True
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"


# Singleton instance
settings = Settings()



"""Key decisions explained:

Pydantic Settings: Type-safe configuration with environment variable support
Low temperature (0.1): Reduces creative outputs, keeps responses factual
High confidence thresholds: Better to say "I don't know" than hallucinate
Separate configs per module: Clean separation, easy testing"""