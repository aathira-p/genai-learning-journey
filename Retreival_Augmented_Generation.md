# Retrieval-Augmented Generation (RAG)

_Reference:_ A comprehensive guide to understanding Retrieval-Augmented Generation, a technique that combines information retrieval with generative models to produce more accurate, contextually grounded responses.

---

## What is RAG?

**Definition:**

Retrieval-Augmented Generation (RAG) is a machine learning framework that enhances large language models (LLMs) by retrieving relevant documents or passages from a knowledge base and using them as context for generating responses. Instead of relying solely on the model's training data, RAG dynamically fetches relevant information to ground the generation process.

**Why RAG?**

Traditional LLMs have limitations:
- **Knowledge cutoff**: Training data becomes outdated
- **Hallucinations**: Models generate plausible but incorrect information
- **Domain-specific gaps**: Limited knowledge about specific domains or proprietary data
- **Lack of citations**: Difficult to trace information sources

RAG addresses these by:
- Grounding responses in retrieved documents
- Reducing hallucinations through factual references
- Enabling knowledge over custom or proprietary data
- Providing verifiable sources for claims

---

## The RAG Pipeline: Step-by-Step

### **Step 1: Data Ingestion and Preprocessing**

**What happens:**
Raw documents from various sources (PDFs, websites, databases, etc.) are collected and prepared for storage in a retrieval system.

**Key processes:**
1. **Document collection**: Gather all available documents for your knowledge base
2. **Text extraction**: Convert PDFs, images, or structured data into plain text
3. **Cleaning**: Remove noise, formatting artifacts, special characters
4. **Normalization**: Standardize text (lowercase, remove extra whitespace, etc.)
5. **Deduplication**: Remove duplicate or near-duplicate documents



**Caveats & Nuances:**

- **Data quality matters**: Garbage in, garbage out. Poorly formatted or noisy documents will reduce retrieval quality
- **Encoding issues**: PDFs with mixed encoding (UTF-8, Latin-1, etc.) can cause text extraction failures
- **Semantic meaning loss**: Over-aggressive cleaning (removing punctuation, lowercase conversion) can lose nuance
- **Handling structured vs unstructured data**: Tables, lists, and hierarchical data need special handling
- **Language mixing**: Multilingual documents may need language detection and separate processing

---

### **Step 2: Text Chunking/Segmentation**

**What happens:**
Documents are split into smaller, manageable pieces (chunks) suitable for embedding and retrieval.

**Why chunking is necessary:**

- **Embedding limitations**: Most embedding models have maximum token limits (e.g., 512-8192 tokens)
- **Granularity**: Retrieving entire documents is often too broad; specific sections are more useful
- **Context window**: LLMs have finite context windows (e.g., 4K, 8K, 128K tokens)
- **Relevance precision**: Smaller chunks are more semantically coherent and relevant

**Chunking Strategies:**

#### 1. **Fixed-Size Chunking**
Split documents into chunks of fixed token/character count.


**Pros:** Simple, predictable, easy to implement  
**Cons:** May cut in the middle of sentences or ideas

#### 2. **Overlapping Chunks**
Chunks overlap to preserve context at boundaries.

```
Chunk 1: [Token 0-512]
Chunk 2: [Token 463-975]      # 50 tokens overlap
Chunk 3: [Token 926-1438]
```

**Pros:** Prevents loss of context at chunk boundaries  
**Cons:** Increased storage and processing overhead; potential redundancy in retrieval

#### 3. **Semantic/Sentence-Based Chunking**
Split at sentence or paragraph boundaries to preserve meaning.


**Pros:** Preserves semantic coherence  
**Cons:** Variable chunk sizes; harder to implement correctly

#### 4. **Structure-Aware Chunking**
Respect document structure (headers, sections, paragraphs).



**Pros:** Preserves document structure; enables hierarchical retrieval  
**Cons:** Requires understanding document format; complex implementation

#### 5. **Recursive Chunking**
Split documents hierarchically, starting with large chunks and recursively splitting based on a splitting strategy.



**Pros:** Intelligent splitting respecting document structure  
**Cons:** More complex; requires tuning splitters

**Critical Caveats & Nuances:**

- **Chunk size tradeoff**: 
  - **Too small** (e.g., 64 tokens): Insufficient context, fragmented information
  - **Too large** (e.g., 2048 tokens): May exceed embedding model limits, less precise retrieval
  - **Sweet spot**: Usually 256-1024 tokens depending on use case and embedding model
  
- **Overlap amount**: 
  - **No overlap**: Risk of losing context at boundaries
  - **High overlap** (50%+): Redundancy increases storage and retrieval time
  - **Recommended**: 10-20% overlap for sentence-based; 5-10% for fixed-size
  
- **Information loss at boundaries**: Critical information might be split across chunks, making individual chunks less useful
  
- **Semantic fragmentation**: A chunk might not contain enough context for the embedding model to capture full meaning
  
- **Language-specific issues**: Some languages (e.g., East Asian languages) don't use spaces; splitting strategies must adapt
  
- **Table and list handling**: Fixed chunking may break tables or lists mid-row/item, destroying readability
  
- **Metadata preservation**: Chunk origin (source document, section, page number) should be preserved for citation

---

### **Step 3: Embedding Generation**

**What happens:**
Each chunk is converted into a dense vector representation using an embedding model.



**Embedding Model Selection:**

| Model | Dimension | Strengths | Weaknesses | Best For |
|-------|-----------|-----------|-----------|----------|
| OpenAI ada-002 | 1536 | Well-trained, production-ready | API-dependent, cost per query | Production systems with budget |
| BGE (BAAI) | 768 | Strong retrieval performance, open-source | Requires self-hosting | RAG, semantic search |
| Sentence-BERT | 384-768 | Fast, lightweight, versatile | Moderate performance | Speed-critical applications |
| GPT-4 Embeddings | 3072 | State-of-the-art quality | Expensive, API-dependent | High-accuracy requirements |
| Voyage AI | 1024 | Domain-optimized variants available | Proprietary, cost | Domain-specific RAG |
| LLaMA Embeddings | 4096 | Large capacity, open-source | Computationally expensive | Custom fine-tuning |

**Critical Caveats & Nuances:**

- **Embedding model consistency**: 
  - Must use the **same embedding model** for chunks and queries
  - Different models produce incompatible vector spaces
  - Changing embedding models requires re-embedding entire corpus

- **Semantic drift**: Embedding models capture different aspects of semantics:
  - Some emphasize syntactic similarity (word order, grammar)
  - Others emphasize semantic similarity (meaning)
  - Choose based on your retrieval priorities

- **Language-specific embeddings**: 
  - Multilingual models (e.g., multilingual-e5) handle multiple languages
  - Language-specific models (e.g., paraphrase-xlm-r-multilingual-v1) may perform better
  - Mixing languages can degrade performance

- **Cost and latency**:
  - API-based embeddings (OpenAI, Cohere) incur per-query costs
  - Self-hosted models have upfront computational cost but zero marginal cost
  - Embedding generation can be a bottleneck in high-throughput systems

- **Dimensionality vs performance**:
  - Higher dimensions = better expressive power but slower retrieval
  - Dimensionality reduction (e.g., PCA) can speed up retrieval with minimal loss

- **Temporal degradation**: Embeddings from different time periods may have different characteristics if training data distributions shift

---

### **Step 4: Vector Storage and Indexing**

**What happens:**
Embeddings are stored in a vector database or search index for fast retrieval.

**Storage Options:**

#### 1. **Vector Databases**
Purpose-built systems optimized for vector similarity search.



**Popular Vector DBs:**
- **Weaviate**: Open-source, flexible, GraphQL interface
- **Pinecone**: Fully managed, serverless, high-scale
- **Qdrant**: High performance, similarity search focused
- **Milvus**: Open-source, scalable, supports multiple distance metrics
- **Chroma**: Lightweight, embedding-first, good for prototyping

**Pros:** 
- Optimized for similarity search
- Scalable to millions of vectors
- Built-in similarity metrics (cosine, L2, inner product)
- Often support metadata filtering and hybrid search

**Cons:** 
- Additional infrastructure to maintain
- Data synchronization challenges
- Cost at scale

#### 2. **Traditional Search + Embeddings (Hybrid)**
Use traditional search engines (Elasticsearch, Solr) alongside vector stores.


**Pros:** 
- Mature, well-understood technology
- Flexible querying
- Cost-effective for smaller scales
- Better for keyword search

**Cons:** 
- Vector search support is newer
- Less optimized than dedicated vector DBs

#### 3. **In-Memory Stores (for small-scale)**
FAISS (Facebook AI Similarity Search) or similar for prototyping.

**Pros:** 
- Simple, no external dependencies
- Fast for moderate-sized datasets
- Good for prototyping

**Cons:** 
- Entire index must fit in memory
- Not suitable for large-scale deployments
- No persistence by default

**Indexing Strategies:**

1. **Flat Index**: Store all vectors, brute-force search (slow for large-scale)
2. **IVF (Inverted File)**: Partition vectors into clusters, search relevant clusters (faster, approximate)
3. **HNSW (Hierarchical Navigable Small World)**: Graph-based indexing (very fast, good quality)
4. **Product Quantization**: Compress vectors for memory efficiency
5. **Hierarchical clustering**: Multi-level indexing for billion-scale datasets

**Critical Caveats & Nuances:**

- **Index stale data problem**: 
  - Embeddings don't auto-update if source documents change
  - Need versioning strategy and update mechanisms
  - Reindexing entire corpus can be expensive

- **Approximate search tradeoff**:
  - **Exact search**: Slow but guaranteed to find most similar
  - **Approximate search**: Fast but may miss relevant documents
  - HNSW and IVF provide good balance

- **Metadata filtering overhead**:
  - Filtering by metadata (e.g., date range, document type) reduces search space
  - Can improve retrieval speed but requires careful index design
  - Poor filter selectivity can negate performance gains

- **Scale considerations**:
  - Small-scale (< 100K vectors): In-memory FAISS sufficient
  - Medium-scale (100K - 10M vectors): Managed vector DBs (Pinecone, Weaviate)
  - Large-scale (> 10M vectors): Distributed systems (Milvus, Vespa)

- **Vector compression**: Quantizing vectors to int8 or binary reduces memory but degrades recall

- **Concurrency and consistency**: Ensure thread-safe updates; CAP theorem applies to distributed systems

---

### **Step 5: Query Embedding**

**What happens:**
User queries are converted into embeddings using the **same embedding model** as the corpus.


**Critical Caveats & Nuances:**

- **Embedding consistency**: 
  - **MUST use the same model** as corpus embeddings
  - Different models produce incompatible vector spaces
  - Version mismatches can occur with model updates

- **Query formulation**:
  - Longer, more descriptive queries often retrieve better results
  - Query context matters: "What is X?" vs "Explain X" may have different intent
  - Queries with multiple aspects may retrieve documents for only one aspect

- **Query preprocessing**:
  - Queries may benefit from cleaning (removing stop words, stemming)
  - However, embeddings capture semantic meaning, so aggressive preprocessing may hurt
  - Typically, minimal preprocessing is best

- **Ambiguous queries**:
  - "bank" (financial institution vs riverbank) may retrieve both
  - Lack of context makes disambiguation difficult
  - May need query expansion or multi-turn clarification

- **Language mismatch**:
  - Multilingual models required if corpus and query are in different languages
  - Translation can be an alternative but introduces additional error

---

### **Step 6: Similarity Search and Ranking**

**What happens:**
The query embedding is compared against all stored chunk embeddings to find the most relevant documents.


**Similarity Metrics:**

| Metric | Formula | Properties | Use Case |
|--------|---------|-----------|----------|
| **Cosine Similarity** | cos(θ) = (A·B)/(‖A‖‖B‖) | Invariant to magnitude, range [-1, 1] | Default for embeddings |
| **Euclidean (L2) Distance** | √(Σ(aᵢ-bᵢ)²) | Geometric distance, considers magnitude | When magnitude matters |
| **Dot Product** | A·B | Fast on normalized vectors | Optimized systems |


**Top-K Retrieval:**

**What is top-k?**

Retrieve the k most similar chunks to the query instead of searching the entire corpus.


**Critical Caveats for Top-K:**

- **Optimal k value**:
  - **k too small** (k=1): Limited context, single document may be wrong
  - **k too large** (k=50): Noise from irrelevant documents, context window overflow, slower generation
  - **Sweet spot**: k=3-10 for most use cases; depends on chunk size and context window
  - **Dynamic k**: Retrieve based on context window availability and relevance threshold

  
- **Relevance threshold problem**: 
  - Absolute similarity scores vary by embedding model
  - A score of 0.8 might be excellent for one model, poor for another
  - Percentile-based ranking is more robust

- **Cold start problem**: 
  - New or rare queries might not match any documents well
  - Fallback strategies needed (expand query, use BM25, ask for clarification)

- **Redundancy**: 
  - Top-k results often contain semantically similar chunks
  - Diversifying results can reduce redundancy
  - Tradeoff: Diversity vs relevance

---

### **Step 7: Optional - Hybrid Search**

**What is Hybrid Search?**

Combining vector search (semantic) with traditional keyword search (BM25/TF-IDF) to leverage both approaches.

**Why Hybrid?**

- **Vector search strengths**: Captures semantic meaning, synonyms, paraphrasing
- **Vector search weaknesses**: Poor for exact phrase matching, domain-specific terminology, rare entities
- **Keyword search strengths**: Exact matches, rare entities, domain-specific terms
- **Keyword search weaknesses**: Misses synonyms, paraphrasing, semantic variations


**Fusion Strategies:**

1. **Reciprocal Rank Fusion (RRF)**: Combine ranks from different rankers
   ```python
   # For each document, compute: 1/(k + rank)
   rrf_score = 1.0 / (60 + bm25_rank) + 1.0 / (60 + vector_rank)
   ```

2. **Weighted Sum**: Simple weighted combination (as shown above)

3. **Learning-to-rank**: Train ML model to combine multiple signals

4. **Reranking**: Use vector search to pre-filter, then rerank with different metric

**Critical Caveats & Nuances:**

- **Score normalization**:
  - BM25 and vector similarity scores are on different scales
  - Raw combination gives disproportionate weight to one signal
  - Min-max scaling, z-score normalization, or RRF are solutions

- **Parameter tuning**:
  - Weight (alpha) between keyword and semantic is crucial
  - No universal optimal value; depends on domain and use case
  - A/B testing or cross-validation recommended

- **Computational overhead**:
  - Hybrid search is slower than single-approach search
  - Both BM25 and vector indexes needed (2x storage)
  - Suitable when accuracy is prioritized over latency

- **Empty or low-recall results**:
  - If BM25 returns no results (e.g., misspelled query), vector search provides fallback
  - Inverse: Vector search fails on exact technical terms; BM25 catches these

- **Corpus-specific tuning**:
  - Short text (tweets, titles): Vector search may be sufficient
  - Long documents (research papers): Hybrid approach often better
  - Domain-specific terminology: Weighted toward keyword search

---

### **Step 8: Result Reranking (Optional)**

**What is Reranking?**

Re-scoring and re-ordering retrieved results using a more sophisticated (but slower) ranking model.

**Why Rerank?**

Initial retrieval is fast but approximate. Reranking refines results for higher quality.

**Common Reranking Approaches:**

1. **Cross-Encoder Models**: Score (query, document) pairs directly
   

2. **LLM-based Reranking**: Use LLM to score relevance
  

3. **Domain-specific Scoring**: Custom heuristics
   

**Critical Caveats & Nuances:**

- **Computational cost**: 
  - Reranking adds latency, especially with LLMs
  - Only practical if initial top-k is small (k=10-100)
  - For large retrievals, may not be feasible

- **Model selection**:
  - Cross-encoders are faster than LLM-based reranking
  - Larger cross-encoders are more accurate but slower
  - LLM-based reranking is flexible but expensive

- **Diminishing returns**:
  - Marginal improvement if initial retrieval is already good
  - Worthwhile if initial retrieval accuracy is poor

- **Contextual sensitivity**:
  - Reranker may favor certain document styles or lengths
  - A long document might score higher due to more text
  - Normalize by length if needed

---

### **Step 9: Context Preparation**

**What happens:**
Retrieved chunks are formatted and prepared as context for the LLM prompt.


**Critical Caveats & Nuances:**

- **Context ordering**:
  - Position bias: LLMs pay more attention to beginning and end of context
  - Most relevant documents should be first and last
  - Middle documents may be overlooked ("lost in the middle" problem)
  - Solution: Reorder by relevance or use middle-document boosts

- **Context window overflow**:
  - If total tokens exceed LLM's context limit, truncation occurs
  - Truncation may remove important information
  - Solution: Dynamic chunk selection or summarization

- **Context format**:
  - Clear formatting helps LLM parse multiple documents
  - Separators, section markers, source attribution essential
  - Bad formatting → LLM confusion and errors

- **Source attribution**:
  - Include document source in context for traceability
  - Helps LLM cite sources in response
  - Critical for transparency and fact-checking

- **Irrelevant context injection**:
  - Poor retrieval includes irrelevant chunks in context
  - LLM may incorporate irrelevant information into response
  - Wastes context tokens on noise

- **Conflicting information**:
  - Retrieved documents may contain contradictory information
  - LLM must reconcile or highlight disagreement
  - May choose to amplify whichever is mentioned first

---

### **Step 10: Generation**

**What happens:**
The LLM generates a response using the retrieved context and the user query.


**Critical Caveats & Nuances:**

- **Instruction adherence**:
  - System prompt must clearly instruct LLM to use retrieved context
  - "Answer based on provided documents" vs "Use your knowledge" gives different results
  - Be explicit about citation expectations

- **Hallucination despite context**:
  - LLM may still generate unsupported claims even with relevant context
  - Context doesn't guarantee factuality; LLM may ignore context
  - Solution: Temperature=0 for more deterministic behavior, explicit constraints

- **Context confusion**:
  - If context contradicts query assumptions, LLM may be confused
  - May try to reconcile contradictions rather than clearly state the issue
  - Clear instructions about handling contradictions help

- **Length and verbosity**:
  - `max_tokens` controls response length
  - Too small: Important information truncated
  - Too large: Verbose, repetitive responses
  - Adjust based on use case requirements

- **Temperature and randomness**:
  - `temperature=0`: Deterministic, safe, best for factual tasks
  - `temperature=1.0`: More creative but less reproducible
  - For RAG: Use low temperature (0.1-0.3) to stay grounded in context

- **Token usage tracking**:
  - Monitor total tokens used (context + generation)
  - Affects cost and quality of retrieval-generation balance
  - May need to reduce context chunks if token budget exceeded

---

## Advanced RAG Techniques

### **Query Expansion**

**Problem:** User queries are often short and ambiguous, missing important context.

**Solution:** Expand queries with related terms before searching.



**Caveats:**
- Query expansion increases computational cost (multiple searches)
- Poor expansions introduce noise and hurt precision
- Works well for complex multi-faceted queries

### **Iterative Refinement (Multi-Turn RAG)**

**Problem:** Single retrieval-generation pass may miss complex aspects.

**Solution:** Multiple refinement passes based on intermediate results.

**Caveats:**
- Multiple LLM calls increase latency and cost
- Risk of infinite loops or contradictory refinements
- Requires stopping criteria

### **Hypothetical Document Embeddings (HyDE)**

**Problem:** Query embeddings may not align well with document embeddings.

**Solution:** Generate hypothetical relevant documents, embed them, use for search.



**Caveats:**
- Hypothetical documents may introduce bias
- Requires accurate LLM generation
- Adds latency (extra LLM call)
- Works well for question-answering tasks

### **Metadata Filtering**

**Problem:** Retrieved documents may include irrelevant categories (old dates, wrong domain).

**Solution:** Filter documents by metadata before/after retrieval.


**Caveats:**
- Strict filters may reduce recall (fewer results retrieved)
- Metadata must be accurate and up-to-date
- Multi-condition filters are complex to manage

### **Dense Passage Retrieval (DPR)**

**Problem:** Single embedding model may not capture all relevance aspects.

**Solution:** Use specially trained retriever model (dense passage retrieval).

**How DPR works:**
- Separate encoder for queries and passages
- Trained with contrastive learning (similar queries near similar documents, dissimilar far apart)
- Often outperforms generic embeddings on specific domains

**Caveats:**
- Requires fine-tuning or domain-adapted models
- Two separate models increase memory and latency
- Training data-dependent (may not transfer to new domains)

---

## Use-Case Specific Nuances

### **1. Customer Support / FAQ RAG**

**Characteristics:**
- Frequent, repetitive queries
- Need for quick, accurate answers
- Citation and traceability critical

**Optimization Strategies:**
- Use FAQ documents as primary knowledge base
- Chunk by Q&A pairs, not arbitrary sections
- Low k (k=1-3) often sufficient
- Include confidence scores in response
- Use exact phrase matching (BM25) heavily

**Caveats:**
- FAQ might become outdated quickly; versioning essential
- Customer queries often don't match exact FAQ wording; use query expansion
- Responsibility for wrong answers is high; use LLM guardrails

---

### **2. Scientific/Technical Document QA**

**Characteristics:**
- Dense, structured information (tables, equations, citations)
- High precision required (hallucinations costly)
- Cross-references and hierarchical relationships important

**Optimization Strategies:**
- Preserve document structure during chunking (sections, subsections)
- Use larger k (k=5-10) to provide comprehensive context
- Employ reranking for precision
- Consider hierarchical retrieval (retrieve section → subsection → paragraph)
- Use metadata filtering by document type, domain, publication date

**Caveats:**
- Tables and equations often get corrupted in text extraction; handle specially
- Citations may be mis-extracted; verify manually
- Complex reasoning across multiple documents may require iterative refinement

---

### **3. Legal/Compliance Document Search**

**Characteristics:**
- Precision over recall (missed clauses are costly)
- Exact phrase matching essential (contracts are precise)
- Metadata critical (version, effective date, jurisdiction)

**Optimization Strategies:**
- Heavy use of BM25 for exact phrase matching
- Metadata filtering by jurisdiction, document type, effective date
- Low k (k=3-5) to reduce irrelevant results
- Explainability paramount; include full document citations
- Use reranking to verify relevance before presenting to user

**Caveats:**
- Pure semantic search may miss relevant clauses with exact phrasing
- Synonyms may lead to false positives (e.g., "shall" vs "must")
- Versioning and updates must be extremely careful

---

### **4. Code Documentation / API Documentation**

**Characteristics:**
- Highly structured (functions, parameters, examples)
- Exact technical accuracy required
- Code snippets essential

**Optimization Strategies:**
- Chunk by function/class/method, preserving examples
- Use code-specific embedding models if available
- Include metadata: function signature, parameters, return type
- Lower k sufficient (k=3-5) due to specificity
- Use syntax highlighting in retrieved context

**Caveats:**
- Code snippets in documents may become outdated with new versions
- API signatures must be exact; hallucinations are obvious to users
- Multimodal documents (code + text + diagrams) require special handling

---

### **5. Multi-Lingual RAG**

**Characteristics:**
- Queries and documents in multiple languages
- Cross-lingual search may be needed

**Optimization Strategies:**
- Use multilingual embedding models (multilingual-e5, LaBSE)
- Separate indexes by language for performance, or use single multilingual index
- Language-specific metadata for filtering
- Consider query translation if searching cross-lingual

**Caveats:**
- Embedding quality varies significantly across languages
- Some language pairs have poor translation quality
- Language detection errors can propagate through pipeline

---

### **6. Real-Time/Streaming Data RAG**

**Characteristics:**
- Knowledge base changes constantly (news, social media)
- Temporal relevance important
- Freshness critical

**Optimization Strategies:**
- Implement continuous indexing pipeline
- Metadata filtering by recency (prefer recent documents)
- Separate "hot" (recent) and "cold" (archive) indexes for performance
- Use message queues for decoupling ingestion from retrieval
- Consider time-decay scoring (older docs score lower)

**Caveats:**
- Index staleness: retrieving outdated information
- Indexing lag: delay between data generation and retrieval availability
- Storage growth: continuous indexing increases costs
- Deduplication complex for streaming data

---

## Evaluation and Optimization

### **Key Metrics**

| Metric | Definition | Good Value | Notes |
|--------|-----------|-----------|-------|
| **Recall@k** | % relevant docs in top-k | > 0.8 | Did we retrieve what we needed? |
| **Precision@k** | % of top-k that are relevant | > 0.7 | How much noise is there? |
| **MRR (Mean Reciprocal Rank)** | Average rank of first relevant doc | > 0.8 | How soon do we find answers? |
| **NDCG (Normalized DCG)** | Ranking quality (accounts for position) | > 0.7 | Position-aware relevance |
| **Latency** | Time to retrieve top-k | < 100ms | Real-time feasibility |
| **Generation Quality** | LLM output accuracy (manual eval) | > 0.8 (F1, BLEU) | Does final answer help user? |

### **Optimization Checklist**

- [ ] **Chunking**: Tested different chunk sizes (256, 512, 1024, 2048)
- [ ] **Embeddings**: Tried multiple embedding models for your domain
- [ ] **Top-k**: Experimented with k=3, 5, 10, 20; found optimal value
- [ ] **Hybrid search**: A/B tested vs pure vector search
- [ ] **Metadata filtering**: Assessed impact of filtering on recall/precision
- [ ] **Reranking**: Evaluated reranker impact on final quality
- [ ] **Query expansion**: Tested on ambiguous/short queries
- [ ] **Context formatting**: Verified LLM correctly parses context
- [ ] **System prompt**: Tuned instructions for citation and grounding

---

## Common RAG Pitfalls and Solutions

| Pitfall | Cause | Solution |
|---------|-------|----------|
| **Low retrieval recall** | Poor chunking, wrong embedding model | Evaluate chunk size, try domain-specific embeddings |
| **High latency** | Large k, expensive embedding model | Reduce k, use lightweight embeddings, implement caching |
| **Hallucinations persist** | LLM ignores context, weak instructions | Use temperature=0, explicit constraints, stronger system prompt |
| **Lost-in-the-middle problem** | Context ordering biases LLM | Reorder by relevance, use dense prompt formatting |
| **Stale knowledge base** | Outdated documents not refreshed | Implement version control, auto-refresh, metadata timestamps |
| **High cost** | Many API calls (embeddings, LLM) | Batch process, cache results, use cheaper models |
| **Poor multilingual support** | Monolingual embeddings | Use multilingual embedding models |
| **Conflicting retrieved docs** | Multiple contradictory sources | Ask LLM to synthesize or highlight disagreement |

---

## RAG Architecture Patterns

### **Pattern 1: Simple RAG**
Query → Retrieve → Generate

**Best for:** Straightforward QA, small knowledge bases  
**Complexity:** Low  
**Latency:** Low

### **Pattern 2: Hierarchical RAG**
Query → Retrieve sections → Retrieve subsections → Generate

**Best for:** Large structured documents  
**Complexity:** Medium  
**Latency:** Medium

### **Pattern 3: Agentic RAG**
Query → Agent decides → Retrieve, Search, or Reason → Iterate → Generate

**Best for:** Complex multi-step reasoning  
**Complexity:** High  
**Latency:** High

### **Pattern 4: Ensemble RAG**
Query → Multiple retrievers (dense, sparse, keyword) → Fuse results → Generate

**Best for:** High-quality requirements, diverse data  
**Complexity:** High  
**Latency:** Medium-High

---

## Key Takeaways

| Aspect | Best Practice |
|--------|---------------|
| **Chunking** | Balance size (256-1024 tokens), use overlaps, respect structure |
| **Embeddings** | Use domain-adapted models, maintain consistency across pipeline |
| **Retrieval** | Combine semantic + keyword search (hybrid), tune k=5-10 |
| **Top-K** | Don't assume larger k is better; test on validation set |
| **Context** | Order by relevance, use clear formatting, include sources |
| **Generation** | Use explicit instructions, low temperature, include guardrails |
| **Evaluation** | Measure recall/precision/latency, not just LLM output quality |
| **Debugging** | Check each step independently; errors compound |
