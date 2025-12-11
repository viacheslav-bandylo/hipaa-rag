# RAG Improvements Implementation Plan

Based on the QA audit findings, this document outlines a step-by-step plan to transform the current "Naive RAG" into a production-grade HIPAA compliance system.

---

## Phase 1: Hybrid Search (P0 - Critical)

**Goal:** Combine vector similarity with BM25 keyword search to fix exact term/section retrieval.

### Step 1.1: Add BM25 Dependencies
- **File:** `backend/requirements.txt`
- **Action:** Add `rank-bm25==0.2.2` package
- **Effort:** 5 minutes

### Step 1.2: Create BM25 Index Service
- **File:** `backend/app/services/bm25_service.py` (new)
- **Actions:**
  - Create `BM25Service` class with singleton pattern
  - Implement `build_index(documents)` to tokenize and index all chunks
  - Implement `search(query, top_k)` returning document IDs with BM25 scores
  - Add tokenization (lowercase, remove punctuation, split on whitespace)
- **Effort:** 1-2 hours

### Step 1.3: Implement Reciprocal Rank Fusion (RRF)
- **File:** `backend/app/services/retrieval.py`
- **Actions:**
  - Add `hybrid_search(query, top_k, vector_weight=0.5)` method
  - Get top-N from vector search (pgvector)
  - Get top-N from BM25 search
  - Combine using RRF formula: `score = Σ 1/(k + rank)` where k=60
  - Return merged, deduplicated, re-ranked results
- **Effort:** 2-3 hours

### Step 1.4: Update Chat Router
- **File:** `backend/app/routers/chat.py`
- **Actions:**
  - Replace `retrieval_service.search()` with `retrieval_service.hybrid_search()`
  - Add query parameter to toggle hybrid vs. vector-only (for A/B testing)
- **Effort:** 30 minutes

### Step 1.5: Initialize BM25 on Startup
- **File:** `backend/app/main.py`
- **Actions:**
  - In lifespan handler, after DB init, load all documents and build BM25 index
  - Add rebuild trigger after ingestion completes
- **Effort:** 1 hour

### Step 1.6: Update Ingestion to Rebuild BM25
- **File:** `backend/app/routers/ingest.py`
- **Actions:**
  - After successful ingestion, call `bm25_service.build_index()`
- **Effort:** 30 minutes

**Phase 1 Deliverables:**
- [x] BM25 service with tokenization
- [x] RRF fusion in retrieval
- [x] Hybrid search endpoint
- [x] Auto-rebuild on ingestion

**Testing Checkpoint:**
- Query "§164.502" should now return exact section match in top-3
- Query "PHI definition" should return the actual definition chunk

---

## Phase 2: Metadata Filtering (P1 - High Priority)

**Goal:** Enable hard filtering by section number when user queries reference specific sections.

### Step 2.1: Create Citation Extractor
- **File:** `backend/app/services/query_analyzer.py` (new)
- **Actions:**
  - Create `QueryAnalyzer` class
  - Implement `extract_citations(query)` using regex patterns:
    - `§\s*(\d+\.\d+)` - matches "§ 164.502"
    - `Section\s+(\d+\.\d+)` - matches "Section 164.502"
    - `(\d{3}\.\d{3})` - matches bare "164.502"
  - Return list of extracted section numbers
- **Effort:** 1 hour

### Step 2.2: Add Filtered Search to Retrieval
- **File:** `backend/app/services/retrieval.py`
- **Actions:**
  - Add `search_with_filter(query, section_filter=None, part_filter=None)`
  - Modify SQL to add WHERE clause when filters provided:
    ```sql
    WHERE section_number = :section_filter
    ```
  - Combine with hybrid search logic
- **Effort:** 1-2 hours

### Step 2.3: Integrate Query Analysis in Chat
- **File:** `backend/app/routers/chat.py`
- **Actions:**
  - Before retrieval, call `query_analyzer.extract_citations(message)`
  - If citations found, pass as filter to retrieval
  - Log when filter is applied for debugging
- **Effort:** 1 hour

### Step 2.4: Add Part-Level Filtering
- **File:** `backend/app/services/query_analyzer.py`
- **Actions:**
  - Add `extract_part_references(query)` for queries like "Part 164 requirements"
  - Detect keywords: "Privacy Rule" → Part 164, "Security Rule" → Part 164 Subpart C
- **Effort:** 1 hour

**Phase 2 Deliverables:**
- [x] Citation extraction from queries
- [x] SQL filter integration
- [x] Automatic filter application
- [x] Part/Rule keyword mapping

**Testing Checkpoint:**
- Query "Quote §164.502(a)" should return ONLY chunks from section 164.502
- Query "Security Rule requirements" should filter to Part 164 Subpart C

---

## Phase 3: Semantic Chunking (P2 - Medium Priority)

**Goal:** Prevent list truncation and improve chunk coherence by respecting document structure.

### Step 3.1: Enhance List Detection in Parser
- **File:** `backend/app/services/pdf_parser.py`
- **Actions:**
  - Add regex patterns to detect list starts:
    - `^\s*\([a-z]\)` - matches "(a)", "(b)", etc.
    - `^\s*\(\d+\)` - matches "(1)", "(2)", etc.
    - `^\s*\(i+\)` - matches "(i)", "(ii)", etc.
  - Track "in_list" state during parsing
  - Don't break chunks mid-list (override size limit if in list)
- **Effort:** 2-3 hours

### Step 3.2: Add Paragraph Boundary Detection
- **File:** `backend/app/services/pdf_parser.py`
- **Actions:**
  - Detect paragraph breaks (double newlines, indentation changes)
  - Prefer breaking at paragraph boundaries over arbitrary character limits
  - Implement `find_best_break_point(text, target_size)` function
- **Effort:** 2 hours

### Step 3.3: Implement Sliding Window with Smart Breaks
- **File:** `backend/app/services/pdf_parser.py`
- **Actions:**
  - Replace fixed-size chunking with semantic-aware chunking
  - Priority order for break points:
    1. Section boundaries (highest priority - never break)
    2. Paragraph boundaries
    3. Sentence boundaries (`. ` followed by capital letter)
    4. Size limit (fallback only)
- **Effort:** 2-3 hours

### Step 3.4: Extract Paragraph References
- **File:** `backend/app/services/pdf_parser.py`
- **Actions:**
  - Populate `paragraph_reference` field (currently NULL)
  - Track current paragraph marker: "(a)(1)(i)" format
  - Store in chunk metadata for granular retrieval
- **Effort:** 2 hours

### Step 3.5: Add Chunk Validation
- **File:** `backend/app/services/pdf_parser.py`
- **Actions:**
  - Add post-processing validation:
    - Check no chunk starts mid-sentence
    - Check lists are complete (matching open/close markers)
    - Log warnings for potentially problematic chunks
- **Effort:** 1 hour

**Phase 3 Deliverables:**
- [x] List-aware chunking
- [x] Paragraph boundary detection
- [x] Smart break point selection
- [x] Paragraph reference extraction
- [x] Chunk validation

**Testing Checkpoint:**
- Re-ingest PDF and verify "Covered Entities" list is in single chunk
- Check no chunks end with "(a)" without "(b)" following in same chunk

---

## Phase 4: Re-Ranker Integration (P1.5 - High Priority, Not in Original Audit)

**Goal:** Add cross-encoder reranking to improve retrieval precision.

### Step 4.1: Add Cross-Encoder Dependency
- **File:** `backend/requirements.txt`
- **Action:** Ensure `sentence-transformers` supports cross-encoders (already installed)
- **Effort:** 5 minutes

### Step 4.2: Create Reranker Service
- **File:** `backend/app/services/reranker.py` (new)
- **Actions:**
  - Create `RerankerService` class with singleton pattern
  - Load cross-encoder model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
  - Implement `rerank(query, documents, top_k)`:
    - Score each (query, document) pair
    - Sort by cross-encoder score
    - Return top-k reranked results
- **Effort:** 1-2 hours

### Step 4.3: Integrate Reranker in Retrieval Pipeline
- **File:** `backend/app/services/retrieval.py`
- **Actions:**
  - Modify `hybrid_search()` to:
    1. Get top-20 from hybrid search (over-retrieve)
    2. Pass to reranker
    3. Return top-8 after reranking
  - Add config option to enable/disable reranking
- **Effort:** 1 hour

### Step 4.4: Add Configuration Options
- **File:** `backend/app/config.py`
- **Actions:**
  - Add `reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"`
  - Add `reranker_enabled: bool = True`
  - Add `retrieval_candidates: int = 20` (pre-rerank pool size)
- **Effort:** 30 minutes

**Phase 4 Deliverables:**
- [x] Cross-encoder reranker service
- [x] Pipeline integration
- [x] Configuration options

**Testing Checkpoint:**
- Compare retrieval quality with/without reranker on test queries
- Verify latency is acceptable (< 500ms added)

---

## Phase 5: Query Classification (P3 - Enhancement)

**Goal:** Detect query type to adjust retrieval strategy dynamically.

### Step 5.1: Define Query Types
- **File:** `backend/app/services/query_analyzer.py`
- **Actions:**
  - Define enum: `QueryType.BROAD`, `QueryType.SPECIFIC`, `QueryType.EXACT_QUOTE`
  - Add classification rules:
    - EXACT_QUOTE: Contains "quote", "exact text", "verbatim", section references
    - SPECIFIC: Contains specific terms, definitions, single concept
    - BROAD: "What are", "List all", "Overview", "Requirements"
- **Effort:** 1 hour

### Step 5.2: Implement Query Classifier
- **File:** `backend/app/services/query_analyzer.py`
- **Actions:**
  - Add `classify_query(query) -> QueryType` method
  - Use keyword matching + heuristics (question length, specificity signals)
  - Return classification with confidence score
- **Effort:** 1-2 hours

### Step 5.3: Adjust Retrieval Parameters by Query Type
- **File:** `backend/app/services/retrieval.py`
- **Actions:**
  - For BROAD queries:
    - Increase `top_k` to 12-15
    - Prefer parent chunks (larger context)
    - Weight vector search higher (semantic understanding)
  - For SPECIFIC queries:
    - Standard `top_k` of 8
    - Balance vector and BM25
  - For EXACT_QUOTE queries:
    - Apply section filter if citation found
    - Weight BM25 higher (exact matching)
    - Lower `top_k` to 5 (precision over recall)
- **Effort:** 2 hours

### Step 5.4: Add Query Type to Response (Optional)
- **File:** `backend/app/models.py`
- **Actions:**
  - Add `query_type: Optional[str]` to `ChatResponse`
  - Include for debugging/analytics
- **Effort:** 30 minutes

**Phase 5 Deliverables:**
- [x] Query type enum and classifier
- [x] Dynamic retrieval parameters
- [x] Optional response metadata

**Testing Checkpoint:**
- "What are HIPAA requirements?" → classified as BROAD, returns 12+ diverse chunks
- "Quote §164.502" → classified as EXACT_QUOTE, applies section filter

---

## Phase 6: Parent-Child Indexing (P2.5 - Advanced)

**Goal:** Create hierarchical index for better broad/specific query handling.

### Step 6.1: Add Parent Chunks Table
- **File:** `backend/app/database.py`
- **Actions:**
  - Create `parent_documents` table:
    - Same schema as `documents`
    - Larger chunks (entire sections/subparts)
    - Foreign key relationship to child chunks
  - Add `parent_id` column to `documents` table
- **Effort:** 1 hour

### Step 6.2: Generate Parent Chunks During Ingestion
- **File:** `backend/app/services/pdf_parser.py`
- **Actions:**
  - After generating child chunks, aggregate into parent chunks:
    - Group by section → create section-level parent
    - Group by subpart → create subpart-level parent
  - Store parent-child relationships
- **Effort:** 2-3 hours

### Step 6.3: Update Retrieval for Hierarchical Search
- **File:** `backend/app/services/retrieval.py`
- **Actions:**
  - For BROAD queries: search parent chunks first
  - For SPECIFIC queries: search child chunks
  - Implement `expand_to_children(parent_id)` for drill-down
- **Effort:** 2 hours

### Step 6.4: Update Ingestion Router
- **File:** `backend/app/routers/ingest.py`
- **Actions:**
  - Modify ingestion to populate both tables
  - Ensure parent embeddings are generated
- **Effort:** 1 hour

**Phase 6 Deliverables:**
- [x] Parent documents table
- [x] Parent chunk generation
- [x] Hierarchical retrieval logic

**Testing Checkpoint:**
- "What is the Security Rule?" → returns subpart-level parent chunk
- "Device media re-use requirements" → returns specific child chunk

---

## Implementation Timeline Summary

| Phase | Priority | Components | Estimated Effort |
|-------|----------|------------|------------------|
| **Phase 1** | P0 Critical | Hybrid Search (BM25 + RRF) | 6-8 hours |
| **Phase 2** | P1 High | Metadata Filtering | 4-5 hours |
| **Phase 3** | P2 Medium | Semantic Chunking | 9-11 hours |
| **Phase 4** | P1.5 High | Re-Ranker | 3-4 hours |
| **Phase 5** | P3 Low | Query Classification | 4-6 hours |
| **Phase 6** | P2.5 Medium | Parent-Child Indexing | 6-7 hours |

**Total Estimated Effort:** 32-41 hours

---

## Recommended Implementation Order

1. **Phase 1 (Hybrid Search)** - Biggest impact, enables exact matching
2. **Phase 2 (Metadata Filtering)** - Quick win, leverages existing metadata
3. **Phase 4 (Re-Ranker)** - Improves all queries with minimal code changes
4. **Phase 3 (Semantic Chunking)** - Requires re-ingestion, do before production
5. **Phase 5 (Query Classification)** - Polish and optimization
6. **Phase 6 (Parent-Child)** - Advanced feature, implement if needed

---

## Testing Strategy

### Unit Tests (per phase)
- BM25 tokenization and scoring
- RRF fusion correctness
- Citation extraction regex
- Chunk boundary detection
- Query classification accuracy

### Integration Tests
- End-to-end chat flow with hybrid search
- Ingestion → BM25 rebuild → search
- Filter application with various query formats

### Regression Tests (from audit)
- Q1: "Main requirements of HIPAA" → returns comprehensive overview
- Q2: "Definition of PHI" → returns exact definition chunk
- Q3: "Quote §164.502(a)" → returns exact section text

---

## Configuration Summary

New config options to add to `config.py`:

```python
# Hybrid Search
bm25_enabled: bool = True
vector_weight: float = 0.5  # 0.0 = BM25 only, 1.0 = vector only
rrf_k: int = 60  # RRF constant

# Reranker
reranker_enabled: bool = True
reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
retrieval_candidates: int = 20  # Pre-rerank pool size

# Query Analysis
auto_filter_citations: bool = True
query_classification_enabled: bool = True

# Chunking
respect_list_boundaries: bool = True
max_list_chunk_size: int = 2000  # Override limit for lists
```

---

## Rollback Plan

Each phase is independent and can be disabled via config:
- `bm25_enabled = False` → falls back to vector-only
- `reranker_enabled = False` → skips reranking step
- `auto_filter_citations = False` → no metadata filtering

Database schema changes (Phase 6) require migration rollback script.
