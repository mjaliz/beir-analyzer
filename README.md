# BEIR Analyzer

A comprehensive tool for analyzing and evaluating BEIR (Benchmarking Information Retrieval) datasets using vector embeddings and Qdrant storage.

## Features

- **Corpus Indexing**: Index document collections using state-of-the-art embedding models
- **Search & Evaluation**: Search queries against indexed corpus and evaluate retrieval performance
- **False Positive Detection**: Identify and record documents that are retrieved but not relevant
- **Comprehensive Metrics**: Calculate precision, recall, and F1-score for retrieval evaluation
- **Flexible Configuration**: Configurable thresholds, top-k results, and output formats

## Installation

```bash
# Install dependencies using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

## Usage

### 1. Index Your Corpus

First, index your document corpus using an embedding model:

```python
from src.main import index_copurs

# Index corpus with BGE-M3 model
index_copurs(model_name="BAAI/bge-m3")
```

### 2. Search and Evaluate

Run search and evaluation to find false positives and calculate metrics:

```python
from src.main import search_and_evaluate

# Run evaluation
results = search_and_evaluate(
    model_name="BAAI/bge-m3",
    score_threshold=0.5,  # Minimum similarity score
    top_k=10,             # Top documents to retrieve per query
    output_file="data/false_positives.jsonl"
)

print(f"Precision: {results['precision']:.4f}")
print(f"Recall: {results['recall']:.4f}")
print(f"F1-Score: {results['f1_score']:.4f}")
```

### 3. Run Complete Example

```bash
python evaluate_example.py
```

## Data Format

### Input Files

- **corpus.jsonl**: Document collection
  ```json
  {"_id": "doc_1", "text": "Document content", "title": "Document title"}
  ```

- **query.jsonl**: Search queries
  ```json
  {"_id": "q_1", "text": "Search query text"}
  ```

- **qrels.tsv**: Relevance judgments (tab-separated)
  ```
  q_id    doc_id  score
  q_1     doc_1   1
  q_1     doc_2   0
  ```

### Output Files

- **false_positives.jsonl**: Retrieved but non-relevant documents
  ```json
  {
    "query_id": "q_1",
    "query_text": "Search query",
    "doc_id": "doc_123",
    "doc_text": "Document content",
    "similarity_score": 0.75,
    "relevant_docs_count": 5,
    "retrieved_docs_count": 10,
    "true_positives_count": 3
  }
  ```

## Configuration

Configure the system using environment variables or modify `src/config.py`:

```python
# Search configuration
SEARCH_TOP_K=10
SEARCH_SCORE_THRESHOLD=0.5

# Data paths
CORPUS_PATH=data/beir_data/corpus.jsonl
QUERY_PATH=data/beir_data/query.jsonl
QRELS_PATH=data/beir_data/qrels.tsv
FALSE_POSITIVES_OUTPUT=data/false_positives.jsonl

# Qdrant configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

## Evaluation Metrics

The system calculates standard information retrieval metrics:

- **Precision**: Ratio of relevant documents among retrieved documents
- **Recall**: Ratio of relevant documents that were retrieved
- **F1-Score**: Harmonic mean of precision and recall

## Architecture

```
├── src/
│   ├── main.py           # Main indexing and evaluation functions
│   ├── config.py         # Configuration management
│   ├── embedding.py      # Embedding model wrapper
│   ├── models.py         # Data models (Pydantic)
│   └── qdrant_storage.py # Qdrant vector database interface
├── data/
│   └── beir_data/        # BEIR dataset files
└── evaluate_example.py   # Example usage script
```

## Requirements

- Python 3.8+
- Qdrant vector database
- GPU recommended for embedding models

## License

MIT License
