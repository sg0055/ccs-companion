# src/evaluation/benchmarks.py
"""
Benchmark suite for measuring retrieval and generation quality.
"""

from dataclasses import dataclass
from typing import Optional
import time
from loguru import logger
from math import log2

@dataclass
class RetrievalMetrics:
    """Retrieval quality metrics."""
    precision_at_k: float
    recall_at_k: float
    mrr: float  
    ndcg: float 
    latency_ms: float
    

@dataclass
class GenerationMetrics:
    """Generation quality metrics."""
    citation_coverage: float
    answer_relevance: float  
    faithfulness: float      
    latency_ms: float


class RetrievalBenchmark:
    """
    Benchmark retrieval quality against labeled datasets.
    
    Requires:
    - Query set
    - Relevance judgments (which chunks are relevant to each query)
    """
    
    def __init__(self, hybrid_retriever):
        self.retriever = hybrid_retriever
        
    def run_benchmark(
        self,
        queries: list[str],
        relevance_labels: dict[str, list[str]],  
        k: int = 5
    ) -> RetrievalMetrics:
        """Run retrieval benchmark on labeled data."""
        precisions = []
        recalls = []
        mrrs = []
        ndcgs = []
        latencies = []
        
        for query in queries:
            relevant_ids = set(relevance_labels.get(query, []))
            if not relevant_ids:
                continue
                
            # Time the retrieval
            start = time.time()
            results = self.retriever.retrieve(query, top_k=k)
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            
            retrieved_ids = [r.chunk_id for r in results]
            
            # Precision@k
            relevant_retrieved = sum(1 for cid in retrieved_ids if cid in relevant_ids)
            precision = relevant_retrieved / k
            precisions.append(precision)
            
            # Recall@k
            recall = relevant_retrieved / len(relevant_ids)
            recalls.append(recall)
            
            # MRR
            mrr = 0.0
            for rank, cid in enumerate(retrieved_ids, 1):
                if cid in relevant_ids:
                    mrr = 1.0 / rank
                    break
            mrrs.append(mrr)
            
            # NDCG@k
            dcg = sum(
                (1 if cid in relevant_ids else 0) / log2(i + 2)  # log2(i+2)
                for i, cid in enumerate(retrieved_ids)
            )
            ideal_dcg = sum(1 / (i + 2) for i in range(min(k, len(relevant_ids))))
            ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0
            ndcgs.append(ndcg)
            
        return RetrievalMetrics(
            precision_at_k=sum(precisions) / len(precisions) if precisions else 0,
            recall_at_k=sum(recalls) / len(recalls) if recalls else 0,
            mrr=sum(mrrs) / len(mrrs) if mrrs else 0,
            ndcg=sum(ndcgs) / len(ndcgs) if ndcgs else 0,
            latency_ms=sum(latencies) / len(latencies) if latencies else 0
        )


class AdversarialTestSuite:
    """
    Adversarial tests to probe for hallucination vulnerabilities.
    
    Test categories:
    1. Questions about topics NOT in documents
    2. Questions mixing real and fake information
    3. Leading questions that suggest false premises
    4. Requests for information beyond document scope
    """
    
    def __init__(self):
        self.test_cases = []
        
    def add_out_of_scope_test(
        self, 
        query: str, 
        expected_fallback: bool = True
    ):
        """Add test for questions about topics not in documents."""
        self.test_cases.append({
            'type': 'out_of_scope',
            'query': query,
            'expected_fallback': expected_fallback
        })
        
    def add_false_premise_test(
        self, 
        query: str
    ):
        """Add test with false premise the model shouldn't accept."""
        self.test_cases.append({
            'type': 'false_premise',
            'query': query,
            'expected_fallback': True 
        })
        
    def add_mixed_facts_test(
        self,
        query: str,
        real_facts: list[str],
        fake_facts: list[str]
    ):
        """Test mixing real document facts with made-up ones."""
        self.test_cases.append({
            'type': 'mixed_facts',
            'query': query,
            'real_facts': real_facts,
            'fake_facts': fake_facts
        })
        
    def run_suite(self, rag_pipeline) -> dict:
        """Run all adversarial tests."""
        results = {
            'passed': 0,
            'failed': 0,
            'details': []
        }
        
        for test in self.test_cases:
            response = rag_pipeline.query(test['query'])
            
            passed = False
            
            if test['type'] == 'out_of_scope':
                # Should trigger fallback
                passed = response.confidence_level == 'insufficient' or \
                         'cannot' in response.answer.lower() or \
                         'don\'t have' in response.answer.lower()
                         
            elif test['type'] == 'false_premise':
               
                passed = 'cannot confirm' in response.answer.lower() or \
                         'no evidence' in response.answer.lower() or \
                         response.confidence_level == 'insufficient'
                         
            elif test['type'] == 'mixed_facts':
               
                for fake in test['fake_facts']:
                    if fake.lower() in response.answer.lower():
                        passed = False
                        break
                else:
                    passed = True
                    
            if passed:
                results['passed'] += 1
            else:
                results['failed'] += 1
                
            results['details'].append({
                'test_type': test['type'],
                'query': test['query'],
                'passed': passed,
                'response_snippet': response.answer[:200]
            })
            
        return results
