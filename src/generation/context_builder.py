# src/generation/context_builder.py
"""
Build structured context for LLM consumption.
Ensures all information is traceable to sources using Markdown structural hierarchy.
"""

from dataclasses import dataclass
from typing import Optional
import tiktoken

from src.reranking.source_confidence import ScoredResult


try:
    ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    ENC = None


@dataclass
class SourceReference:
    """A citation reference for a piece of context."""
    ref_id: str             
    doc_title: str
    filename: str
    section: str
    chunk_id: str         
    confidence: float
    

@dataclass
class PreparedContext:
    """Context package ready for LLM."""
    context_text: str                    
    sources: list[SourceReference]      
    total_tokens_estimate: int          
    sufficient_evidence: bool           


class ContextBuilder:
    """
    Build citation-ready context from scored retrieval results.
    """
    
    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
        
    def _get_token_count(self, text: str) -> int:
        """Accurately compute token payload footprint."""
        if ENC is not None:
            return len(ENC.encode(text))
        return len(text.split())
        
    def build_context(
        self, 
        scored_results: list[ScoredResult], 
        query: str
    ) -> PreparedContext:
        """
        Package chunks into a single context string with clear citation IDs.
        Filters out low-confidence chunks based on guardrails.
        """
        
        
        has_sufficient = self._assess_sufficiency(scored_results)
        
        context_parts = []
        sources = []
        current_tokens = 0
        
        for idx, scored in enumerate(scored_results, 1):
           
            result = scored.result
            source = result.chunk
            conf = scored.confidence
            #print(source.metadata)
           
            doc_title = source.metadata.get('title', 'Unknown Document')
            hierarchy = source.metadata.get('hierarchy', 'General Rules')
            
            ref_id = str(idx)
            
            
            source_header = f"[{ref_id}] Document: {doc_title} | Section: {hierarchy}"
            
            chunk_text = f"{source_header}\n{source.content}"
            chunk_tokens = self._get_token_count(chunk_text)
            
            if current_tokens + chunk_tokens > self.max_tokens:
                break
                
            context_parts.append(chunk_text)
            current_tokens += chunk_tokens
            
           
            sources.append(SourceReference(
                ref_id=ref_id,
                doc_title=doc_title,
                filename=source.metadata.get('filename', 'unknown'),
                section=hierarchy,
                chunk_id=source.chunk_id,
                confidence=conf.overall_confidence if hasattr(conf, 'overall_confidence') else 0.0
            ))
            
        final_context_str = "\n\n".join(context_parts)
        
       
        sources_meta = "\n\n=== SOURCE DIRECTORY ===\n" + self._format_sources(sources)
        final_context_str += sources_meta
            
        return PreparedContext(
            context_text=final_context_str,
            sources=sources,
            total_tokens_estimate=current_tokens,
            sufficient_evidence=has_sufficient
        )
        
    def _format_sources(self, sources: list[SourceReference]) -> str:
        """Creates the directory at the bottom of the prompt to enforce strict citation mapping."""
        lines = []
        for source in sources:
            conf_pct = int(source.confidence * 100)
            lines.append(
                f"- ID [{source.ref_id}] -> Document: {source.doc_title} "
                f"| Section: {source.section} [Relevance Match Confidence: {conf_pct}%]"
            )
            
        return '\n'.join(lines)
    
    def _assess_sufficiency(self, scored_results: list[ScoredResult]) -> bool:
        if not scored_results:
            return False
            
        def get_conf(res):
            return res.confidence.overall_confidence if hasattr(res.confidence, 'overall_confidence') else 0.0
        
        best_score = get_conf(scored_results[0])
        if best_score < 0.25:
            return False
            
        top_chunks_to_check = min(2, len(scored_results))
        top_avg = sum(get_conf(r) for r in scored_results[:top_chunks_to_check]) / top_chunks_to_check
        
        return top_avg >= 0.20