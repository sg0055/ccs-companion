# src/generation/fallback_handler.py
"""
Handle cases where confidence is too low or citation validation fails.
Better to admit uncertainty and provide debugging suggestions than to hallucinate.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
from loguru import logger

from src.generation.constrained_generator import GeneratedResponse
from src.generation.citation_validator import ValidationResult
from src.generation.context_builder import PreparedContext


class FallbackReason(Enum):
    """Reasons for triggering fallback."""
    NO_RELEVANT_DOCS = "no_relevant_documents"
    LOW_CONFIDENCE = "low_confidence_scores"
    CITATION_VALIDATION_FAILED = "citation_validation_failed"
    INSUFFICIENT_COVERAGE = "insufficient_citation_coverage"
    CONFLICTING_SOURCES = "conflicting_information"
    

@dataclass
class FallbackResponse:
    """A response indicating uncertainty with explanation."""
    message: str
    reason: FallbackReason
    suggestions: list[str]
    partial_info: Optional[str]  
    

class FallbackHandler:
    """
    Handle low-confidence and citation failure scenarios gracefully.
    
    Philosophy: It's better to say "I don't know" with precise debugging tips
    than to provide an answer that might be ungrounded or poorly cited.
    """
    
    def __init__(self, confidence_threshold: float = 0.3):
        self.confidence_threshold = confidence_threshold
    
    def should_fallback(
        self,
        confidence_score: any,
        has_sources: bool,
        validation_result: Optional[ValidationResult] = None
    ) -> tuple[bool, Optional[FallbackReason]]:
        """
        Evaluate if a fallback is required based on confidence, sources, and validation.
        Defensively handles both raw float values and full GeneratedResponse objects.
        
        Returns:
            Tuple containing (should_trigger_fallback: bool, reason: FallbackReason)
        """
      
        score = 1.0
        if isinstance(confidence_score, (int, float)):
            score = float(confidence_score)
        elif confidence_score is not None:
            
            if hasattr(confidence_score, 'confidence'):
                score = float(confidence_score.confidence)
            elif hasattr(confidence_score, 'confidence_score'):
                score = float(confidence_score.confidence_score)
            elif hasattr(confidence_score, 'score'):
                score = float(confidence_score.score)
            elif hasattr(confidence_score, 'confidence_level'):
            
                if confidence_score.confidence_level == 'insufficient':
                    return True, FallbackReason.LOW_CONFIDENCE

      
        if not has_sources:
            return True, FallbackReason.NO_RELEVANT_DOCS
            
        
        if score < self.confidence_threshold:
            return True, FallbackReason.LOW_CONFIDENCE
            
       
        if validation_result and not validation_result.is_valid:
            if validation_result.invalid_citations:
                return True, FallbackReason.CITATION_VALIDATION_FAILED
            else:
                return True, FallbackReason.INSUFFICIENT_COVERAGE
                
        
        return False, None
    
    def create_fallback(
        self,
        reason: FallbackReason,
        context: PreparedContext,
        response: Optional[GeneratedResponse] = None,
        validation_result: Optional[ValidationResult] = None
    ) -> FallbackResponse:
        """Create a structured fallback response based on the failure reason."""
        
       
        logger.warning(f"Fallback triggered! Reason: {reason.value}")
        if validation_result and validation_result.warnings:
            logger.warning(f"Validation Warnings: {validation_result.warnings}")

       
        if reason == FallbackReason.CITATION_VALIDATION_FAILED:
            message = (
                "⚠️ Fallback triggered: citation_validation_failed\n\n"
                "I generated a response, but it failed strict grounding verification. "
                "To maintain absolute factual accuracy, the answer was withheld."
            )
        elif reason == FallbackReason.INSUFFICIENT_COVERAGE:
            message = (
                "⚠️ Fallback triggered: insufficient_citation_coverage\n\n"
                "The available information doesn't sufficiently cover your question. "
                "Key aspects aren't fully addressed within the source documents."
            )
        elif reason == FallbackReason.NO_RELEVANT_DOCS:
            message = "I couldn't find any documents in the database relevant to your query."
        else:
            message = "I am not confident enough in the source data to answer this reliably."

        
        suggestions = self._generate_suggestions(reason, context, validation_result)
        partial_info = self._extract_partial_info(response, context, validation_result)

        return FallbackResponse(
            message=message,
            reason=reason,
            suggestions=suggestions,
            partial_info=partial_info
        )
    
    def _generate_suggestions(
        self,
        reason: FallbackReason,
        context: PreparedContext,
        validation_result: Optional[ValidationResult] = None
    ) -> list[str]:
        """Generate actionable troubleshooting steps for the user or system logs."""
        suggestions = []

        if reason in (FallbackReason.CITATION_VALIDATION_FAILED, FallbackReason.INSUFFICIENT_COVERAGE):
            suggestions.extend([
                "Check system logs to see which specific claims failed validation.",
                "Ensure your system prompt instructs the LLM to strictly use the '[N]' citation format.",
                "Verify if the LLM is omitting inline citations for metrics, years, or percentages."
            ])
            if validation_result and validation_result.uncited_claims:
                suggestions.append(f"Review unverified claims: '{validation_result.uncited_claims[0][:80]}...'")
                
        elif reason == FallbackReason.NO_RELEVANT_DOCS:
            suggestions.extend([
                "Try using alternative keywords or broader search terms.",
                "Ensure the document containing this information has been correctly ingested.",
                "Check if the retrieval similarity threshold or top_k cutoffs are too restrictive."
            ])
            
        elif reason == FallbackReason.LOW_CONFIDENCE:
            suggestions.extend([
                "Try asking a more specific question.",
                "Mention precise names, chapters, or terminology from the source documents.",
                "Break complex questions down into smaller, single-topic queries."
            ])
            
       
        if context and context.sources:
            doc_titles = list(set(s.doc_title for s in context.sources if s.doc_title and s.doc_title != "Unknown"))
            if doc_titles:
                suggestions.append(f"Your currently loaded context covers: {', '.join(doc_titles[:3])}")
                
        return suggestions
    
    def _extract_partial_info(
        self,
        response: Optional[GeneratedResponse],
        context: PreparedContext,
        validation_result: Optional[ValidationResult] = None
    ) -> Optional[str]:
        """Extract underlying debug context or source tracking flags."""
        parts = []
        
        if context and context.sources:
            topics = list(set(source.doc_title for source in context.sources if source.doc_title and source.doc_title != "Unknown"))
            if topics:
                parts.append(f"Related Sources Found: {', '.join(topics[:3])}")
                
        if validation_result:
            parts.append(f"Coverage Achieved: {validation_result.citation_coverage:.1%}")
            if validation_result.invalid_citations:
                parts.append(f"Invalid Refs Pointed to: {validation_result.invalid_citations}")

        return " | ".join(parts) if parts else None