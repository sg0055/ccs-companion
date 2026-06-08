# src/main.py
"""
Main RAG pipeline integrating all components.
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from loguru import logger

from src.config import settings
from src.ingestion.parser import DocumentParser, ParsedDocument
from src.ingestion.normalizer import TextNormalizer
from src.ingestion.chunker import SemanticChunker, Chunk
from src.ingestion.deduplicator import ChunkDeduplicator, DocumentDeduplicator
from src.ingestion.versioning import VersionManager
from src.retrieval.embedder import EmbeddingManager
from src.retrieval.bm25 import BM25Index
from src.retrieval.ann import ANNIndex
from src.retrieval.hybrid import HybridRetriever
from src.reranking.cross_encoder import CrossEncoderReranker
from src.reranking.source_confidence import SourceConfidenceScorer
from src.generation.context_builder import ContextBuilder, PreparedContext
from src.generation.constrained_generator import ConstrainedGenerator
from src.generation.citation_validator import CitationValidator
from src.generation.fallback_handler import FallbackHandler, FallbackReason
from src.caching.cache_manager import CacheManager
from src.observability.tracer import RAGTracer
from src.evaluation.hallucination_tracker import HallucinationTracker


@dataclass
class QueryResponse:
    """Complete response to a user query."""
    answer: str
    citations: list[dict]
    confidence_level: str
    is_fallback: bool
    fallback_reason: Optional[str]
    trace_id: Optional[str]
    raw_confidence: float = 0.0
    

class HallucinationProofRAG:
    """
    Complete RAG pipeline with hallucination prevention.
    
    Pipeline:
    1. Ingest documents → chunk → deduplicate → embed → index
    2. Query → hybrid retrieval → rerank → confidence score
    3. Build context → constrained generation → citation validation
    4. Fallback handling → observability → return response
    """
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = Path(data_dir or settings.storage.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initializing Hallucination-Proof RAG pipeline...")
        
        # Ingestion components
        self.parser = DocumentParser()
        self.normalizer = TextNormalizer()
        self.chunker = SemanticChunker(
            chunk_size=settings.chunking.chunk_size,
            chunk_overlap=settings.chunking.chunk_overlap
        )
        self.chunk_deduplicator = ChunkDeduplicator()
        self.doc_deduplicator = DocumentDeduplicator()
        self.version_manager = VersionManager(self.data_dir / "versions")
        
        # Retrieval components
        self.embedder = EmbeddingManager()
        self.bm25_index = BM25Index(storage_path=self.data_dir / "bm25")
        self.ann_index = ANNIndex(
            dimension=self.embedder.dimension,
            storage_path=self.data_dir / "ann"
        )
        
        # Store chunks for retrieval
        self.chunk_store: dict[str, Chunk] = {}
        self.chunk_store_path = self.data_dir / "chunks"
        self.chunk_store_path.mkdir(parents=True, exist_ok=True)
        
        # Reranking
        self.reranker = CrossEncoderReranker()
        self.confidence_scorer = SourceConfidenceScorer()
        
        # Generation
        self.context_builder = ContextBuilder()
        self.generator = ConstrainedGenerator()
        self.citation_validator = CitationValidator(require_all_claims_cited=False)
        self.fallback_handler = FallbackHandler()
        
        # Infrastructure
        self.cache = CacheManager()
        self.tracer = RAGTracer(self.data_dir / "traces")
        self.hallucination_tracker = HallucinationTracker(
            self.data_dir / "hallucination_tracking"
        )
        
        
        self._load_chunk_store()
        self._load_indexes()
        
        logger.info("RAG pipeline initialized")
        
    def _load_indexes(self):
        """Load existing indexes from disk."""
        if self.bm25_index.load_index():
            logger.info("Loaded existing BM25 index")
        if self.ann_index.load_index():
            logger.info("Loaded existing ANN index")

    def _load_chunk_store(self):
        """Load persisted chunks from disk into the in-memory store."""
        if not self.chunk_store_path.exists():
            return

        loaded = 0
        for chunk_file in self.chunk_store_path.glob("*.json"):
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                chunk = Chunk(
                    chunk_id=data['chunk_id'],
                    content=data['content'],
                    doc_id=data['doc_id'],
                    page_numbers=data['page_numbers'],
                    start_char=data['start_char'],
                    end_char=data['end_char'],
                    chunk_index=data['chunk_index'],
                    total_chunks=data['total_chunks'],
                    metadata=data['metadata']
                )
                self.chunk_store[chunk.chunk_id] = chunk
                loaded += 1
            except Exception as exc:
                logger.warning(
                    "Failed to load chunk file %s: %s",
                    chunk_file.name,
                    exc
                )

        logger.info("Loaded {} persisted chunks", loaded)

    def _persist_chunk(self, chunk: Chunk):
        """Persist a single chunk to disk."""
        chunk_file = self.chunk_store_path / f"{chunk.chunk_id}.json"
        with open(chunk_file, 'w', encoding='utf-8') as f:
            json.dump({
                'chunk_id': chunk.chunk_id,
                'content': chunk.content,
                'doc_id': chunk.doc_id,
                'page_numbers': chunk.page_numbers,
                'start_char': chunk.start_char,
                'end_char': chunk.end_char,
                'chunk_index': chunk.chunk_index,
                'total_chunks': chunk.total_chunks,
                'metadata': chunk.metadata
            }, f, ensure_ascii=False, indent=2)
            
    def ingest_document(self, file_path: Path) -> dict:
        """
        Ingest a Markdown/PDF document into the system.
        
        Returns:
            Dict with ingestion statistics
        """
        file_path = Path(file_path)
        logger.info(f"Ingesting document: {file_path.name}")
        
       
        parsed = self.parser.parse(file_path)
        
        # Check for duplicate document
        existing_doc = self.doc_deduplicator.is_duplicate(
            parsed.metadata.file_hash,
            parsed.metadata.doc_id
        )
        if existing_doc:
            logger.info(f"Document already exists: {existing_doc}")
            return {
                'status': 'duplicate',
                'existing_doc_id': existing_doc
            }
            
        
        parsed.raw_text = self.normalizer.normalize(parsed.raw_text)
        
        
        chunks = self.chunker.chunk_document(parsed)
        logger.info(f"Created {len(chunks)} chunks")
        
        
        chunks, dedup_result = self.chunk_deduplicator.deduplicate(chunks)
        logger.info(
            f"Deduplication: {dedup_result.exact_duplicates} exact, "
            f"{dedup_result.near_duplicates} near duplicates removed"
        )
        
        
        for chunk in chunks:
            self.chunk_store[chunk.chunk_id] = chunk
            self._persist_chunk(chunk)
            
        
        chunk_ids = [c.chunk_id for c in chunks]
        version = self.version_manager.create_version(
            doc_id=parsed.metadata.doc_id,
            file_hash=parsed.metadata.file_hash,
            chunk_ids=chunk_ids
        )
        
        
        texts = [c.content for c in chunks]
        chunk_ids = [c.chunk_id for c in chunks]
        embeddings = self.embedder.embed_passages(texts, show_progress=True)
        
        
        self._update_indexes(chunks, embeddings, chunk_ids)
        
        
        self.cache.invalidate('retrieval')
        
        return {
            'status': 'success',
            'doc_id': parsed.metadata.doc_id,
            'version': version.version_number,
            'chunks_created': len(chunks),
            'duplicates_removed': dedup_result.exact_duplicates + dedup_result.near_duplicates
        }
    
    def _update_indexes(self, chunks: list[Chunk], embeddings, chunk_ids: list[str]):
        """Update retrieval indexes with new chunks (incremental)."""
        
        if not self.bm25_index.is_built:
            
            all_chunks = list(self.chunk_store.values())
            self.bm25_index.build_index(all_chunks)
            logger.info(f"Built BM25 index with {len(all_chunks)} chunks")
            
            
            all_chunk_ids = [c.chunk_id for c in all_chunks]
            self.ann_index.build_index(embeddings, all_chunk_ids)
            logger.info(f"Built FAISS index with {len(all_chunk_ids)} vectors")
        else:
            
            self.bm25_index.add_documents(chunks)
            logger.info(f"Added {len(chunks)} new documents to BM25 index")
            
            self.ann_index.add_vectors(embeddings, chunk_ids)
            logger.info(f"Added {len(chunk_ids)} new embeddings to FAISS index")
        
    def query(self, query: str) -> QueryResponse:
        """
        Process a user query through the full pipeline.
        """
        with self.tracer.trace(query) as trace:
           
            response = self._process_query(query)
            
           
            trace_outcome = "fallback" if response.is_fallback else "success"
            
          
            if hasattr(trace, 'outcome'):
                trace.outcome = trace_outcome
            elif isinstance(trace, dict):
                trace['outcome'] = trace_outcome
                
            
            metrics = {
                "was_fallback": response.is_fallback,
                "confidence_score": response.raw_confidence,
                "fallback_reason": response.fallback_reason,
                "sources_available": len(response.citations)
            }
            
            if hasattr(trace, 'attributes'):
                trace.attributes.update(metrics)
            elif isinstance(trace, dict) and 'attributes' in trace:
                trace['attributes'].update(metrics)
                
            return response
            
    def _process_query(self, query: str) -> QueryResponse:
        """Internal query processing with tracing."""
        
        if settings.confidence.force_fallback:
            reason = FallbackReason.NO_RELEVANT_DOCS
            fallback = self.fallback_handler.generate_fallback(
                reason, query, PreparedContext(context_text="", sources=[], total_tokens_estimate=0, sufficient_evidence=False)
            )
            return QueryResponse(
                answer=fallback.message,
                citations=[],
                confidence_level="insufficient",
                is_fallback=True,
                fallback_reason=reason.value,
                trace_id=self.tracer._current_trace_id
            )

        
        with self.tracer.span("hybrid_retrieval") as span:
            retriever = HybridRetriever(
                embedder=self.embedder,
                ann_index=self.ann_index,
                bm25_index=self.bm25_index,
                chunk_store=self.chunk_store
            )
            
            initial_results = retriever.retrieve(
                query, 
                top_k=settings.retrieval.initial_candidates
            )
            
            span.attributes['candidates'] = len(initial_results)
            
        
        with self.tracer.span("reranking") as span:
            reranked = self.reranker.rerank(
                query, 
                initial_results,
                top_k=settings.reranker.top_k
            )
            span.attributes['reranked_count'] = len(reranked)
            
        
        with self.tracer.span("confidence_scoring") as span:
            scored_results = self.confidence_scorer.score_results(
                reranked, 
                query
            )
            
            if scored_results:
                span.attributes['top_confidence'] = scored_results[0].confidence.overall_confidence
                
        
        with self.tracer.span("context_building"):
            context = self.context_builder.build_context(scored_results, query)
            
        
        with self.tracer.span("generation") as span:
            response = self.generator.generate(query, context)
            span.attributes['tokens'] = response.tokens_used
            span.attributes['confidence'] = response.confidence_level
            
        
        with self.tracer.span("citation_validation"):
            validation = self.citation_validator.validate(response, context)
            
        
        needs_fallback, fallback_reason = self.fallback_handler.should_fallback(
            confidence_score=response,
            has_sources=len(context.sources) > 0 if context and context.sources else False,
            validation_result=validation
        )
        
        if needs_fallback and fallback_reason:
            with self.tracer.span("fallback", reason=fallback_reason.value):
                
                fallback = self.fallback_handler.create_fallback(
                    reason=fallback_reason,
                    context=context,
                    response=response,
                    validation_result=validation
                )
                
                
                
                return QueryResponse(
                    answer=fallback.message,
                    citations=[],
                    confidence_level="insufficient",
                    is_fallback=True,
                    fallback_reason=fallback_reason.value,
                    trace_id=self.tracer._current_trace_id if hasattr(self, 'tracer') else None,
                    raw_confidence=scored_results[0].confidence.overall_confidence if scored_results else 0.0
                )
                
        
       
        
        
        citations = [
            {
                'ref': s.ref_id,
                'document': s.doc_title,
                'section': s.section,
                'confidence': s.confidence
            }
            for s in context.sources
            if s.ref_id in response.citations_used
        ]
        
        return QueryResponse(
            answer=response.answer,
            citations=citations,
            confidence_level=response.confidence_level,
            is_fallback=False,
            fallback_reason=None,
            trace_id=self.tracer._current_trace_id,
            raw_confidence=scored_results[0].confidence.overall_confidence if scored_results else 0.0
        )
        
    def get_health_status(self) -> dict:
        """Get system health metrics."""
        hal_metrics = self.hallucination_tracker.get_metrics()
        alerts = self.hallucination_tracker.check_alerts()
        cache_stats = self.cache.get_stats()
        
        return {
            'documents_indexed': len(set(c.doc_id for c in self.chunk_store.values())),
            'chunks_indexed': len(self.chunk_store),
            'hallucination_metrics': {
                'total_queries': hal_metrics.total_queries,
                'fallback_rate': f"{hal_metrics.fallback_rate:.1%}",
                'avg_confidence': f"{hal_metrics.avg_confidence:.2f}"
            },
            'alerts': alerts,
            'cache': cache_stats
        }

if __name__ == "__main__":
    import sys
    
    rag = HallucinationProofRAG()
    
    if len(sys.argv) > 1:
        for md_path in sys.argv[1:]:
            result = rag.ingest_document(Path(md_path))
            print(f"Ingested {md_path}: {result}")
            
    print("\nRAG system ready. Enter queries (Ctrl+C to exit):")
    
    while True:
        try:
            query = input("\nQuery: ").strip()
            if not query:
                continue
                
            response = rag.query(query)
            
            print(f"\n{'='*60}")
            print(f"Confidence: {response.confidence_level}")
            if response.is_fallback:
                print(f"⚠️  Fallback triggered: {response.fallback_reason}")
            print(f"\n{response.answer}")
            
            if response.citations:
                print(f"\n📚 Sources:")
                for cite in response.citations:
                    
                    section_name = cite.get('section', 'General Rules')
                    print(f"  [{cite['ref']}] Document: {cite['document']} | Section: {section_name}")
                    
        except KeyboardInterrupt:
            print("\nExiting...")
            break