# src/generation/constrained_generator.py
"""
LLM generation with strict constraints to prevent hallucination.
"""

from dataclasses import dataclass
from typing import Optional, AsyncGenerator
import json
import subprocess
import shutil
import re
import os
from loguru import logger
from openai import OpenAI

from src.config import settings
from src.generation.context_builder import PreparedContext
@dataclass
class GeneratedResponse:
    """Response with citations and confidence metadata."""
    answer: str
    citations_used: list[str]    
    confidence_level: str        
    model_used: str
    tokens_used: int
    peak_score: float            
    
CONSTRAINED_SYSTEM_PROMPT = """You are a precise, professional assistant summarizing administrative documents from the Swamy Handbook. 

Answer the user's question by writing a clear, natural, and cohesive paragraph using ONLY the information provided in the Context below.

Context:
{context}

CRITICAL RULES:
1. NO INTERNAL MONOLOGUE: You must output ONLY the final answer to the user. Do NOT write "We need to answer", do NOT summarize the context you were given, and do NOT explain your reasoning process.
2. SYNTHESIZE: The context text may contain OCR errors or formatting artifacts. Do your best to interpret the intent naturally. Do not complain about garbled text to the user.
3. EXACT FORMAT: You MUST cite your sources using the exact format [ID]. Example: "The OBC Creamy Layer limit is ₹8,00,000 [1]."
4. SOURCE VERIFICATION: The [ID] refers to the Document and Section header in the Context. You must place a citation bracket at the end of EVERY factual sentence you write.
5. NO HALLUCINATIONS: You are FORBIDDEN from citing an ID that is not listed in the Context. If the context does not contain the answer, say exactly: "I cannot answer this based on the provided documents."
"""

class ConstrainedGenerator:
    """Generate responses strictly grounded in provided context."""
    
    def __init__(
        self,
        model_name: str = None,
        temperature: float = None,
        max_tokens: int = None
    ):
        self.model_name = settings.llm.model_name
        self.temperature = temperature or settings.llm.temperature
        self.max_tokens = 1000 
        
        self.use_ollama = None

        if self.use_ollama:
            if shutil.which("ollama") is None:
                raise RuntimeError("Ollama CLI not found on PATH.")
            self.client = None
        else:
            try:
                from openai import OpenAI
            except ImportError as err:
                raise RuntimeError("OpenAI SDK is not installed.") from err
            
            api_key = settings.llm.api_key
            if not api_key:
                logger.warning("OPENROUTER_API_KEY is not set in .env — API calls will fail.")

            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key
            )
        
    def generate(
        self,
        query: str,
        context: PreparedContext
    ) -> GeneratedResponse:
        """Generate a response grounded in the provided context."""
        
        
        peak_score = max([s.confidence for s in context.sources]) if context.sources else 0.0

        if not context.sufficient_evidence:
            return self._generate_insufficient_response(query, context, peak_score)
            
        user_prompt = self._build_user_prompt(query, context)
        
        if self.use_ollama:
            answer = self._call_ollama(self._build_ollama_prompt(query, context))
            tokens_used = 0
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": CONSTRAINED_SYSTEM_PROMPT.format(context=context.context_text)},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            if not response.choices:
                print(f"\n❌ UPSTREAM API ERROR: OpenRouter returned an empty payload.")
                print(f"Raw Response Dump: {response}\n")
                
                
                answer = "I cannot answer this question right now because the AI model provider timed out or returned an empty response. Please try again in a few seconds."
                tokens_used = 0
            else:
                answer = response.choices[0].message.content.strip()
                tokens_used = response.usage.total_tokens if response.usage else 0


        
       
        citations_used = self._extract_citations(answer, context)
        
        
        if "cannot answer this" in answer.lower():
            confidence_label = f"REFUSED (Best Context Match: {peak_score * 100:.0f}%)"
        else:
            confidence_label = f"SUCCESS (Peak Relevance Match: {peak_score * 100:.0f}%)"


        if settings.confidence.debug_mode:
            print("\n" + "▼"*60)
            print("🛠️ RAG GENERATION TELEMETRY:")
            print(f"📊 Peak Chunk Score : {peak_score * 100:.1f}%")
            print(f"📝 Tokens Used      : {tokens_used} (Input + Output)")
            print(f"🔗 Citations Parsed : {citations_used}")
            print(f"⚖️ Final Label      : {confidence_label}")
            print("-" * 60)
            print("📥 LLM INPUT CONTEXT PAYLOAD:")
            print(context.context_text)
            print("▲"*60 + "\n")


        return GeneratedResponse(
            answer=answer,
            citations_used=citations_used,
            confidence_level=confidence_label,
            model_used=self.model_name,
            tokens_used=tokens_used,
            peak_score=peak_score
        )
    
    def _build_user_prompt(self, query: str, context: PreparedContext) -> str:
        return f"**Question:** {query}"

    def _build_ollama_prompt(self, query: str, context: PreparedContext) -> str:
        return f"{CONSTRAINED_SYSTEM_PROMPT.format(context=context.context_text)}\n\n**Question:** {query}"

    def _call_ollama(self, prompt: str) -> str:
        command = ["ollama", "run", self.ollama_model, prompt]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                encoding="utf-8"
            )
            return result.stdout.strip() if result.stdout.strip() else "No response generated."
        except subprocess.CalledProcessError as err:
            logger.error("Ollama generation failed: %s", err.stderr.strip() if err.stderr else err)
            raise

    def _generate_insufficient_response(
        self,
        query: str,
        context: PreparedContext,
        peak_score: float
    ) -> GeneratedResponse:
        """Generate response when evidence is insufficient."""
        if not context.sources:
            answer = "I cannot answer this question because no relevant information was found in the available documents."
        else:
            source_list = ', '.join(s.doc_title for s in context.sources[:3])
            answer = f"The available documents ({source_list}) do not contain sufficient information to fully answer this question."
            
        return GeneratedResponse(
            answer=answer,
            citations_used=[],
            confidence_level=f"INSUFFICIENT (Best Context Match: {peak_score * 100:.0f}%)",
            model_used=self.model_name,
            tokens_used=0,
            peak_score=peak_score
        )
    
    def _extract_citations(self, answer: str, context: PreparedContext) -> list[str]:
        """Extract citation references from the answer (PATCHED)."""
        citations = re.findall(r'\[(\d+)\]', answer)
        valid_refs = {str(s.ref_id) for s in context.sources} 
        
        valid_citations = [c for c in citations if c in valid_refs]
        return list(set(valid_citations))
