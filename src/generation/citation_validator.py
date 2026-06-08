# src/generation/citation_validator.py
"""
Validate that generated responses are properly grounded in citations.
"""

import re
from dataclasses import dataclass
from typing import Optional

from src.generation.constrained_generator import GeneratedResponse
from src.generation.context_builder import PreparedContext


@dataclass
class ValidationResult:
    """Result of citation validation."""
    is_valid: bool
    uncited_claims: list[str]       # Statements without citations
    invalid_citations: list[str]    # Citations to non-existent sources
    citation_coverage: float        # % of sentences with citations
    warnings: list[str]
    

class CitationValidator:
    """
    Validate that responses properly cite their sources.
    
    Checks:
    1. Every factual claim has a citation
    2. All citations reference valid sources
    3. No "hallucinated" citations to non-existent sources
    """
    
    def __init__(
        self,
        require_all_claims_cited: bool = True,
        min_citation_coverage: float = 0.1
    ):
        self.require_all_claims_cited = require_all_claims_cited
        self.min_citation_coverage = min_citation_coverage
        
        # Patterns for factual claims (heuristic)
        self.factual_patterns = [
            r'\d+%',                           # Percentages
            r'\$[\d,]+',                       # Dollar amounts
            r'\b\d{4}\b',                      # Years
            r'\b(?:increased|decreased|grew|fell|rose)\b',  # Trends
            r'\b(?:according to|stated|reported|found)\b',   # Attribution words
        ]
        
    def validate(
        self,
        response: GeneratedResponse,
        context: PreparedContext
    ) -> ValidationResult:
        """
        Validate citations in a generated response.
        
        Args:
            response: The generated response to validate
            context: The context used for generation
            
        Returns:
            ValidationResult with detailed findings
        """
        answer = response.answer
        valid_refs = {s.ref_id for s in context.sources}
        
        # Find all citations in the response
        raw_matches = re.findall(r'\[(?:cite:\s*)?(\d+(?:,\s*\d+)*)\]', answer)
        cited_refs = set()
        for match in raw_matches:
            for num in re.findall(r'\d+', match):  
                    cited_refs.add(num.strip())
        
        
        invalid_citations = [
            ref for ref in cited_refs 
            if ref not in valid_refs
        ]
        
        
        sentences = self._split_sentences(answer)
        uncited_claims = []
        cited_sentences = 0
        
        for sentence in sentences:
            has_citation = bool(re.search(r'\[\d+\]', sentence))
            is_factual = self._is_factual_claim(sentence)
            
            if has_citation:
                cited_sentences += 1
            elif is_factual and self.require_all_claims_cited:
                uncited_claims.append(sentence)
                
       
        coverage = cited_sentences / len(sentences) if sentences else 0
        
       
        warnings = []
        if invalid_citations:
            warnings.append(
                f"Found {len(invalid_citations)} citations to non-existent sources"
            )
        if coverage < self.min_citation_coverage:
            warnings.append(
                f"Citation coverage ({coverage:.0%}) below threshold "
                f"({self.min_citation_coverage:.0%})"
            )
        if uncited_claims:
            warnings.append(
                f"Found {len(uncited_claims)} factual claims without citations"
            )
            
       
        is_valid = (
            len(invalid_citations) == 0 and
            coverage >= self.min_citation_coverage and
            (not self.require_all_claims_cited or len(uncited_claims) == 0)
        )
        
        ############################################## TURN OFF CITATION VALIDATOR  ####################################
        #is_valid = True
        
        return ValidationResult(
            is_valid=is_valid,
            uncited_claims=uncited_claims,
            invalid_citations=invalid_citations,
            citation_coverage=coverage,
            warnings=warnings
        )
    
    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Filter out very short segments
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def _is_factual_claim(self, sentence: str) -> bool:
        """Heuristically determine if a sentence is a factual claim."""
        # Skip questions, meta-statements
        if sentence.endswith('?'):
            return False
        if any(phrase in sentence.lower() for phrase in [
            'i cannot', "i don't know", 'insufficient', 
            'no information', 'sources do not'
        ]):
            return False
            
        # Check for factual patterns
        for pattern in self.factual_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                return True
                
        return False
