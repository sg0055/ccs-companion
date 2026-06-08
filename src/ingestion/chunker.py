# src/ingestion/chunker.py
"""
Markdown-Native Semantic Chunker.
Optimized specifically for pristine Docling Markdown output.
"""

import uuid
import datetime
import os
from dataclasses import dataclass, field

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)
from src.ingestion.parser import ParsedDocument

@dataclass
class Chunk:
    """Standardized chunk object for the RAG pipeline."""
    chunk_id: str
    content: str
    doc_id: str
    page_numbers: list[int]
    start_char: int
    end_char: int
    chunk_index: int
    total_chunks: int
    metadata: dict = field(default_factory=dict)

class SemanticChunker:
    """
    Slices Docling Markdown into RAG-optimized chunks using Markdown headers,
    preserving the document hierarchy (Chapter -> Section -> Rule) in the metadata.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
            ("#####", "Header 5")
        ]
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False
        )
        
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "|", " ", ""] 
        )

    def chunk_document(self, document: ParsedDocument) -> list[Chunk]:
        """Processes the Docling Markdown string into structured, context-aware chunks."""
        text = document.raw_text
        if not text:
            return []

        md_splits = self.markdown_splitter.split_text(text)
        final_splits = self.recursive_splitter.split_documents(md_splits)
        
        chunks = []
        total_chunks = len(final_splits)
        current_char_pos = 0
        
        # 1. Safely extract existing metadata
        raw_metadata = getattr(document.metadata, '__dict__', {})
        if not raw_metadata and isinstance(document.metadata, dict):
            raw_metadata = document.metadata
            
        safe_metadata = {}
        for key, value in raw_metadata.items():
            if isinstance(value, datetime.datetime):
                safe_metadata[key] = value.isoformat()
            else:
                safe_metadata[key] = value
                
        
        file_path = safe_metadata.get('filename') or safe_metadata.get('source') or safe_metadata.get('file_path') or "Unknown_Document.pdf"
        
      
        raw_filename = os.path.basename(file_path)
        clean_name = os.path.splitext(raw_filename)[0]
        document_title = clean_name.replace("_", " ").title()
        # -----------------------------------------
                
        for idx, split in enumerate(final_splits):
            chunk_text = split.page_content.strip()
            if not chunk_text:
                continue
                
            headers_metadata = split.metadata
            
            context_breadcrumbs = " > ".join(
                [val for key, val in headers_metadata.items() if key.startswith("Header")]
            )
            
            
            hierarchy_string = context_breadcrumbs if context_breadcrumbs else "General Rules"
            
            if context_breadcrumbs:
                chunk_text = f"Context: {context_breadcrumbs}\n\n{chunk_text}"
            
            chunk_length = len(chunk_text)
            
            chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                content=chunk_text,
                doc_id=safe_metadata.get('doc_id', "doc_1"),
                page_numbers=[1], 
                start_char=current_char_pos,
                end_char=current_char_pos + chunk_length,
                chunk_index=idx,
                total_chunks=total_chunks,
                metadata={
                    **safe_metadata,      
                    **headers_metadata,
                    "hierarchy": hierarchy_string, 
                    "title": document_title,       
                    "filename": raw_filename        
                }
            )
            chunks.append(chunk)
            current_char_pos += chunk_length
            
        return chunks
    




if __name__ == "__main__":
    
    import json
    
    print("🚀 Running Standalone Chunker Test...\n")
    
   
    class MockMetadata:
        def __init__(self):
            self.doc_id = "test_doc_123"
            self.filename = "C:/downloads/swamy_handbook_2026.pdf" 
            self.author = "Gov of India"
            self.processed_at = datetime.datetime.now()

    class MockParsedDocument:
        def __init__(self):
            self.metadata = MockMetadata()
            
            self.raw_text = """# Chapter 9
## Subsistence Allowance
### Review Timeline
A Government servant under suspension is not paid any pay but is allowed a Subsistence Allowance. 
The allowance is to be reviewed after the first 3 months.

# Chapter 10
## Reinstatement
The payment is subject to adjustment of any amount earned during the period.
"""

   
    chunker = SemanticChunker(chunk_size=500, chunk_overlap=50)
    dummy_doc = MockParsedDocument()
    
   
    test_chunks = chunker.chunk_document(dummy_doc)
    
    
    print(f"✅ Successfully generated {len(test_chunks)} chunks.\n")
    
    for i, chunk in enumerate(test_chunks):
        print(f"{"="*40}")
        print(f"📦 CHUNK {i+1}")
        print(f"{"="*40}")
        print(f"TEXT CONTENT:\n{chunk.content}\n")
        
        
        print("METADATA:")
        print(json.dumps(chunk.metadata, indent=2))
        print("\n")