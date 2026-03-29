# Tokens and Embeddings 
---

## Tokenization

**What is Tokenization?**

Tokenization is the process of breaking down text into smaller units called tokens. These tokens can be words, subwords, characters, or bytes, depending on the tokenization strategy used.

**How does a tokenizer break down text?**

Popular tokenization methods include:
- **Byte Pair Encoding (BPE)**: Widely used by GPT models. It iteratively merges the most frequent pairs of bytes or characters.
- **WordPiece**: Used by BERT. It builds a vocabulary by greedily adding the longest subsequences.

The tokenization process involves three main steps:
1. **Choose a tokenization method** (BPE, WordPiece, SentencePiece, etc.)
2. **Make design choices** such as vocabulary size and special tokens to use (e.g., `[CLS]`, `[SEP]`, `[UNK]`)
3. **Train the tokenizer** on a specific dataset to establish the best vocabulary for representing that data

### Word Versus Subword Versus Character Versus Byte Tokens

#### 1. Word Tokens
**Limitations:**
- **Out-of-vocabulary (OOV) problem**: The tokenizer may be unable to handle new words that enter the dataset after training. Every unknown word is typically mapped to a generic `[UNK]` token, losing information.
- **Vocabulary bloat**: Languages with many morphological variations (e.g., "apology," "apologize," "apologetic") require separate tokens, leading to large vocabularies with minimal semantic differences.

**Example:**
```
Vocabulary: ["the", "cat", "apology", "apologize", "apologetic", ...]
Text: "The cat offered its apology."
Tokens: ["the", "cat", "offered", "[UNK]", "its", "apology", "."]
```

#### 2. Subword Tokens
**Advantages:**
- Can represent new words by breaking them down into smaller, more common units (morphemes or character sequences)
- Reduces vocabulary size while maintaining expressiveness
- Tokens like 'apolog-' appear across related words, capturing semantic relationships

**Example:**
```
Vocabulary: ["the", "cat", "apolog", "y", "ize", "etic", "offered", ...]
Text: "The cat offered its apology."
Tokens: ["the", "cat", "offered", "its", "apolog", "y", "."]
New word "apologetic" → ["apolog", "etic"]
```

#### 3. Character Tokens
**Limitations:**
- The sequence length becomes very long, as every character is a separate token
- The model must process significantly more tokens, increasing computational cost
- Harder for the model to learn meaningful patterns from individual characters
- Can lead to memory and performance issues

**Example:**
```
Vocabulary: ["a", "b", "c", ..., "z", ".", " ", ...]
Text: "Cat"
Tokens: ["C", "a", "t"]  (3 tokens for a single word)
```

#### 4. Byte Tokens
**Advantages:**
- Can represent any text without OOV problems, as any character can be decomposed into bytes
- Particularly effective in **multilingual datasets** where character sets vary widely
- Minimal vocabulary size (typically 256 bytes in UTF-8)
- No need for language-specific preprocessing

**Tradeoffs:**
- Sequence lengths can be longer than subword tokenization
- Less intuitive for humans to interpret

---

### Special Note on Hybrid Approaches

Some subword tokenizers use a **hybrid approach**, combining multiple tokenization strategies:
- They primarily use subword tokens for efficiency
- They include **bytes as fallback tokens** in their vocabulary as the final building block
- When encountering characters they cannot represent as subwords, they fall back to bytes

This ensures complete coverage without OOV tokens while maintaining efficiency.

**Example:** GPT-2's byte-level BPE uses this approach to handle any Unicode character gracefully.

---

## Embeddings

**What is an Embedding?**

An embedding is a dense numerical representation of a token, word, or document in a continuous vector space. Instead of treating text as discrete symbols, embeddings capture semantic meaning in the form of vectors of floating-point numbers.

**Why Do We Need Embeddings?**

Models cannot directly process text or tokens as discrete symbols. Embeddings provide:
- **Semantic representation**: Numerically capture meaning and relationships between tokens
- **Similarity measurement**: Calculate how similar two tokens or documents are using distance metrics
- **Dimensionality reduction**: Compress high-dimensional sparse representations into compact dense vectors
- **Transfer learning**: Pre-trained embeddings allow models to leverage learned representations from large corpora

### Embedding Dimensions

**What is embedding dimension?**

The embedding dimension is the size of the vector representing each token. Common dimensions include:
- **Small models**: 256-512 dimensions
- **Standard models**: 768 dimensions (used in BERT)
- **Large models**: 1024-4096 dimensions
- **State-of-the-art models**: Up to 12,288 dimensions (GPT-4)

**Tradeoff:**
- **Higher dimensions** = More expressive, captures finer details, but requires more memory and computation
- **Lower dimensions** = More efficient but may lose important semantic information

### How Embeddings Are Created

#### 1. Token Embeddings
Each token in the vocabulary is assigned a unique embedding vector, typically initialized randomly and learned during training.

```
Token: "cat"
Token ID: 2847
Embedding: [0.234, -0.891, 0.156, ..., 0.423]  (768-dimensional)
```

#### 2. Positional Embeddings
In transformers, positional embeddings encode the position of tokens in a sequence, allowing the model to understand word order.

```
Position 0: [0.1, 0.2, 0.3, ...]
Position 1: [0.15, 0.25, 0.35, ...]
...
```

#### 3. Contextual Embeddings
Models like BERT and GPT generate different embeddings for the same token based on its context, capturing nuanced meaning.

```
Sentence 1: "I went to the bank to withdraw money."
"bank" embedding: [0.234, -0.891, 0.156, ..., 0.423]

Sentence 2: "We sat on the bank of the river."
"bank" embedding: [0.100, -0.450, 0.300, ..., 0.521]  (different!)
```

### Popular Embedding Models

| Model | Dimension | Method | Use Case |
|-------|-----------|--------|----------|
| Word2Vec | 300 | Skip-gram / CBOW | Static embeddings for general text |
| GloVe | 300 | Matrix factorization | General text, competitive with Word2Vec |
| FastText | 300 | Subword-based | Better for rare words and morphology |
| BERT | 768 | Contextual (Transformer) | Classification, NER, semantic similarity |
| Sentence-BERT | 384-768 | Contextual sentence-level | Semantic search, clustering |
| OpenAI Ada | 1536 | Large-scale training | Production embeddings via API |
| BGE (Baai General Embedding) | 768 | Domain-agnostic | RAG, retrieval tasks |

### Embedding Distance Metrics

**How do we measure similarity between embeddings?**

Common metrics used to compare embeddings:

1. **Cosine Similarity** (most common)
   - Measures the angle between two vectors
   - Range: -1 (opposite) to 1 (identical)
   - Formula: `cos(θ) = (A · B) / (||A|| × ||B||)`

2. **Euclidean Distance**
   - Measures the straight-line distance between two points
   - Lower values indicate higher similarity
   - Formula: `√((x₂-x₁)² + (y₂-y₁)² + ...)`

3. **Dot Product**
   - Simple inner product of two vectors
   - Used when embeddings are normalized

**Example:**
```
Embedding 1 ("king"): [0.2, -0.5, 0.8, ...]
Embedding 2 ("queen"): [0.21, -0.48, 0.79, ...]
Cosine Similarity: 0.99  (very similar!)

Embedding 3 ("apple"): [0.1, 0.9, -0.3, ...]
Cosine Similarity with "king": 0.42  (less similar)
```

---



## Key Takeaways

| Concept | Key Point |
|---------|-----------|
| **Tokenization** | Converts raw text into discrete, manageable units (tokens) |
| **Subword Tokenization** | Balances vocabulary size with expressiveness; handles OOV gracefully |
| **Embeddings** | Map tokens/documents to dense vectors that capture semantic meaning |
| **Contextual Embeddings** | Same token can have different embeddings based on context |
| **Embedding Dimension** | Trade-off between expressiveness and computational efficiency |
| **Similarity Metrics** | Cosine similarity is the standard for comparing embeddings |

-