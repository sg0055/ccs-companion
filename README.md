# Hallucination-Proof RAG By Shivam Gupta 

An enterprise-grade, deterministic Retrieval-Augmented Generation (RAG) pipeline meticulously designed around a zero-hallucination mandate. This platform is heavily engineered to ingest, normalize, hierarchically chunk, index, and query dense, highly technical, layout-complex documents (such as the Central Government Swamy Handbook).

The core philosophy of this project is **uncompromising faithfulness to source data**. By combining hybrid lexical/semantic retrieval, state-of-the-art visual document layout analysis, cross-encoder reranking, strict threshold validation, and proactive query routing, the architecture guarantees that if an answer cannot be explicitly verified from the ingested text, the system securely executes a graceful refusal rather than inventing unsupported facts.
https://ccs-companion-8zqti43u2hflu8woudtbum.streamlit.app/
---

## 🏗️ Architectural Overview & Evolution

The architecture separates the document processing lifecycle into distinct, highly specialized modules. Originally designed for standard plaintext parsing and rule-based regex chunking, the project has evolved to address the severe layout challenges posed by watermarks, background vectors, and complex tables inherent in scanned government literature.

### The Ingestion Breakthrough: From PyMuPDF to IBM Docling
Traditional PDF parsers (`pymupdf`, `pdfminer`) view documents through character bounding boxes or simple raw text streams. When confronted with translucent watermarks or geometric borders enclosing critical rules (such as OBC Creamy Layer income thresholds), these engines mistake overlapping graphic boundaries for standalone image components, dropping text and leaving unreadable placeholders (`==> picture intentionally omitted <==`).

This system circumvents layout hallucinations by employing **IBM Docling**. Docling uses layout-aware computer vision and deep learning structure models to evaluate pages exactly like a human reader. It filters background watermark noise, dissolves artificial graphic boundaries, and natively outputs structural, clean, standard Markdown.

### Semantic Mapping: Markdown Header Hierarchy Tree
Rather than dividing the document using fixed character lengths or brittle regex patterns, the chunking pipeline uses a layout-native strategy via LangChain's `MarkdownHeaderTextSplitter`. 

1. **Structural Segmentation:** The text is sliced into logical boundaries defined by the document's header hierarchy tree (`# Header 1` down to `##### Header 5`). This cleanly wraps chapters, sections, and individual sub-rules into atomic units.
2. **Contextual Breadcrumb Injection:** To resolve the "Lost in the Middle" phenomenon and ensure the LLM maintains absolute situational awareness, the hierarchy is compiled into a metadata breadcrumb and prepended directly to the text of every chunk (e.g., `Context: Section 2 > Reservations > Creamy Layer`). If a long data table spans multiple chunks, every isolated rows-block retains explicit knowledge of its parent rule.
3. **Recursive Safety Boundaries:** If an individual table or section remains exceptionally massive, a secondary `RecursiveCharacterTextSplitter` segments the block at structural Markdown delimiters (such as table pipe operators `|` or paragraph breaks `\n\n`), preserving row integrity.

---

## 🛠️ Deep-Dive Module Breakdown

### 1. Centralized Configuration (`src/config.py`)
Driven by Pydantic's `BaseSettings`, configuration is completely decoupled from system logic, exposing granular control thresholds, embedding configurations, caching mechanics, and fallback policies via an environment-driven layer (`.env` overrides).
* **EmbeddingConfig:** Governs dimensions, pooling strategies, and local model bindings.
* **RerankerConfig:** Sets the target deep cross-encoder model parameters.
* **RetrievalConfig:** Manages candidate pool sizes (`initial_candidates` net) and the balancing weights between semantic and lexical scores (`bm25_weight` vs `semantic_weight`).
* **ConfidenceConfig & Fallback Settings:** Adjusts the mathematical ceiling required for generation and toggles testing parameters like `force_fallback`.

### 2. Intelligent Query Routing (`src/main.py`)
To prevent wasting computational cycles on vector database lookups and cross-encoder inference for casual interactions, the ingress orchestrator includes an upstream **Intent Classifier Router**.
* **Chitchat Routing:** Inputs recognized as greetings, pleasantries, or general assistant-interaction bypass the retrieval layer completely, routing directly to a lightweight, prompt-contained LLM handler.
* **Search Routing:** Technical queries trigger the high-fidelity hybrid extraction loop.

### 3. Ingestion & Normalization Engine (`src/ingestion/`)
* **`parser.py`:** Standardizes document metadata schemas across multiple versions. Extracts deep source tracking values such as `doc_id`, `version`, and extraction metrics.
* **`normalizer.py`:** A proactive sanitization layer that scrubs residual formatting noise, strips malformed elements, replaces layout line-breaks (`<br>`) with whitespace, and standardizes spacing around table pipe boundaries to ensure Markdown syntax compliance.
* **`chunker.py` (Docling & Markdown Adapted):** Evaluates the normalized document, builds the hierarchical header tree, handles datetime object conversions to safe ISO-8601 strings, and serializes pristine, context-aware `Chunk` objects containing structural metadata footprints.

### 4. Hybrid Retrieval Architecture (`src/retrieval/`)
To maximize precision and recall, the retrieval engine executes a parallel, two-pronged strategy:
* **Lexical Layer (BM25 Indexing):** Built on rank-frequency text analysis, targeting exact keyword overlap (crucial for domain-specific terminology like "OBC", "CAT", "Superannuation").
* **Semantic Layer (Approximate Nearest Neighbors - ANN):** Maps text chunks into high-dimensional vector space using embedding models. It captures conceptual queries and contextual synonyms where the exact wording differs from the source text.
* **Hybrid Blending Engine:** Retrieves independent candidate pools (e.g., top 50 semantic, top 52 lexical), standardizes raw distances into linear probability distributions, and applies configured scalar alpha weights to compile a single, unified top-k candidate array.

### 5. Cross-Encoder Reranking & Confidence Filtering (`src/reranking/`)
Bi-encoders (used during initial vector generation) process query and chunk strings independently to optimize for retrieval speed, sacrificing fine-grained relational context. This pipeline repairs that limitation by feeding the unified hybrid candidates list into a dense **Cross-Encoder Model**.
* **Full-Attention Interaction:** The cross-encoder evaluates the query and the retrieved text simultaneously, allowing deep cross-attention layers to mathematically compute true semantic relevance.
* **Deterministic Threshold Filter:** Every chunk is assigned an absolute confidence score. If the highest-scoring chunk fails to cross the strict confidence ceiling (e.g., `< 50%` relevance match), the entire pipeline halts generation, issues a `REFUSED` state, and returns a predictable fallback message.

### 6. Generation, Citation Validation & Safety (`src/generation/`)
* **`context_builder.py`:** Packages the filtered, validated high-confidence chunks into an immutable, structured prompt injection context block.
* **`constrained_generator.py`:** Enforces precise output parameters on the Gemma LLM ecosystem. It restricts output boundaries to explicitly provided context strings and prevents speculative synthesis.
* **Citation Validator:** Compares generated alphanumeric references against chunk source tracking strings (`doc_id`, hierarchy metadata), automatically stripping any reference or claim that cannot be deterministically verified back to a specific source record.

### 7. Evaluation, Observability & Logs (`src/evaluation/`)
The system outputs exhaustive, runtime telemetry logs (`loguru`) at every processing phase, tracking performance bottlenecks and pipeline behavior:
* **Hybrid Retrieval Telemetry:** Reports exact hit distributions (Semantic vs BM25 vs Combined pool).
* **RAG Generation Telemetry:** Details peak chunk scores, token consumption profiles (Input + Output), and final execution labels (`REFUSED` vs `SUCCESS`).

---

## 📦 Installation & Environment Setup

### System Prerequisites
* **Python Version:** Python 3.10 to 3.13 (Python 3.11+ highly recommended for optimal performance).
* **Hardware Acceleration:** CUDA-compatible GPU environment recommended for local Cross-Encoder inference.
