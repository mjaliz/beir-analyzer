from pathlib import Path
from typing import Generator, Dict, List, Set
import json

import pandas as pd
from loguru import logger

from qdrant_storage import QdrantStorage
from src.config import config
from src.embedding import Embedder
from src.models import Corpus, Query, corups_list_adaptor

DATA_DIR = Path(__file__).parent.parent / "data"


def load_qrels(qrels_path):
    return pd.read_csv(qrels_path, delimiter="\t")


def load_corpus(corpus_path) -> Generator[Corpus, None, None]:
    """Load the corpus from the data directory."""
    logger.info("Loading corpus")
    with open(corpus_path, "r") as f:
        for line in f:
            yield Corpus.model_validate_json(line.strip())


def load_query(qurey_path) -> Generator[Query, None, None]:
    with open(qurey_path, "r") as f:
        for line in f:
            yield Query.model_validate_json(line.strip())


def prepare_corpus(corpus_loader):
    # Load qrels once
    qrels_df = load_qrels(config.data.qrels_path)
    
    # Pre-load all queries into a dictionary for O(1) lookup
    logger.info("Pre-loading queries for efficient lookup...")
    query_dict = {}
    for q in load_query(config.data.query_path):
        query_dict[q.id] = q.text
    logger.info(f"Loaded {len(query_dict)} queries into memory")
    
    # Process each corpus document
    for c in corpus_loader:
        # Find the qrel entry for this document
        qrel = qrels_df.loc[qrels_df["doc"] == c.id]
        
        if qrel.empty:
            logger.warning(f"No qrel found for document {c.id}, skipping...")
            continue
            
        q_id = qrel["q"].values[0]
        qrel_score = qrel["score"].values[0]
        
        # Fast O(1) lookup for query text
        q_text = query_dict.get(q_id)
        if q_text is None:
            logger.warning(f"Query {q_id} not found for document {c.id}, skipping...")
            continue
            
        yield Corpus(
            **c.model_dump(exclude_none=True, by_alias=True),
            q_id=q_id,
            qrel_score=qrel_score,
            q_text=q_text,
        )


def embed_generator(embedder: Embedder, corpus_generator):
    batch_size = 300
    batch = []
    for c in corpus_generator:
        batch.append(c)
        if len(batch) >= batch_size:
            embeddings = embedder.embed([d.text for d in batch])
            for doc, embedding in zip(batch, embeddings):
                payload = doc.model_dump(exclude=set(["title"]))
                yield {
                    "vector": embedding,
                    "payload": payload,
                }
            batch = []
    
    # Process remaining documents in the final batch
    if batch:
        embeddings = embedder.embed([d.text for d in batch])
        for doc, embedding in zip(batch, embeddings):
            payload = doc.model_dump(exclude=set(["title"]))
            yield {
                "vector": embedding,
                "payload": payload,
            }


def index_copurs(model_name: str):
    embedder = Embedder(model_name=model_name)
    qdrant_storage = QdrantStorage()
    collection_name = model_name.replace("/", "-")
    qdrant_storage.create_collection(collection_name=collection_name)
    qdrant_storage.index_corpus(
        collection_name=collection_name,
        embedding_generator=embed_generator(
            embedder=embedder,
            corpus_generator=prepare_corpus(load_corpus(config.data.corpus_path)),
        ),
    )


def load_relevant_docs(qrels_path: str) -> Dict[str, Set[str]]:
    """Load relevant documents for each query from qrels.tsv"""
    qrels_df = pd.read_csv(qrels_path, delimiter="\t")
    # Filter only relevant documents (score > 0)
    relevant_qrels = qrels_df[qrels_df["score"] > 0]
    
    relevant_docs = {}
    for _, row in relevant_qrels.iterrows():
        query_id = row["q"]
        doc_id = row["doc"]
        
        if query_id not in relevant_docs:
            relevant_docs[query_id] = set()
        relevant_docs[query_id].add(doc_id)
    
    return relevant_docs


def search_and_evaluate(
    model_name: str,
    score_threshold: float = None,
    top_k: int = None,
    output_file: str = None
):
    """
    Search for each query and evaluate retrieved documents against relevant documents.
    Records false positives (retrieved but not relevant) in output file.
    
    Args:
        model_name: Name of the embedding model used for indexing
        score_threshold: Minimum similarity score for retrieved documents
        top_k: Maximum number of documents to retrieve per query
        output_file: Path to save false positives results
    """
    # Use config defaults if not provided
    if score_threshold is None:
        score_threshold = config.search.score_threshold
    if top_k is None:
        top_k = config.search.top_k
    if output_file is None:
        output_file = config.data.false_positives_output
    
    logger.info(f"Starting search and evaluation with model: {model_name}")
    logger.info(f"Score threshold: {score_threshold}, Top-K: {top_k}")
    
    # Initialize components
    embedder = Embedder(model_name=model_name)
    qdrant_storage = QdrantStorage()
    collection_name = model_name.replace("/", "-")
    
    # Load relevant documents from qrels
    relevant_docs = load_relevant_docs(config.data.qrels_path)
    logger.info(f"Loaded relevant documents for {len(relevant_docs)} queries")
    
    # Statistics tracking
    total_queries = 0
    total_retrieved = 0
    total_relevant_retrieved = 0
    total_false_positives = 0
    false_positives_data = []
    
    # Process each query
    logger.info("Processing queries...")
    for query in load_query(config.data.query_path):
        total_queries += 1
        query_id = query.id
        query_text = query.text
        
        # Get relevant documents for this query
        query_relevant_docs = relevant_docs.get(query_id, set())
        
        # Embed the query
        query_embedding = embedder.embed([query_text], show_progress_bar=False)[0]
        
        # Search in Qdrant
        search_results = qdrant_storage.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            score_threshold=score_threshold,
            limit=top_k
        )
        
        total_retrieved += len(search_results)
        
        # Evaluate results
        retrieved_doc_ids = set()
        for result in search_results:
            doc_id = result.payload.get("id")
            if doc_id:
                retrieved_doc_ids.add(doc_id)
        
        # Find true positives and false positives
        true_positives = retrieved_doc_ids.intersection(query_relevant_docs)
        false_positives = retrieved_doc_ids - query_relevant_docs
        
        total_relevant_retrieved += len(true_positives)
        total_false_positives += len(false_positives)
        
        # Record false positives
        for fp_doc_id in false_positives:
            # Find the document details from search results
            fp_doc_data = None
            for result in search_results:
                if result.payload.get("id") == fp_doc_id:
                    fp_doc_data = {
                        "query_id": query_id,
                        "query_text": query_text,
                        "doc_id": fp_doc_id,
                        "doc_text": result.payload.get("text", ""),
                        "doc_q_id": result.payload.get("q_id", ""),
                        "doc_q_text": result.payload.get("q_text", ""),
                        "qrel_score": result.payload.get("qrel_score", ""),
                        "similarity_score": result.score,
                        "relevant_docs_count": len(query_relevant_docs),
                        "retrieved_docs_count": len(retrieved_doc_ids),
                        "true_positives_count": len(true_positives)
                    }
                    break
            
            if fp_doc_data:
                false_positives_data.append(fp_doc_data)
        
        # Log progress every 100 queries
        if total_queries % 100 == 0:
            logger.info(f"Processed {total_queries} queries...")
    
    # Calculate evaluation metrics
    precision = total_relevant_retrieved / total_retrieved if total_retrieved > 0 else 0
    recall_denominator = sum(len(docs) for docs in relevant_docs.values())
    recall = total_relevant_retrieved / recall_denominator if recall_denominator > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Save false positives to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for fp_data in false_positives_data:
            f.write(json.dumps(fp_data, ensure_ascii=False) + '\n')
    
    # Log final statistics
    logger.info("=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Model: {model_name}")
    logger.info(f"Score threshold: {score_threshold}")
    logger.info(f"Top-K: {top_k}")
    logger.info(f"Total queries processed: {total_queries}")
    logger.info(f"Total documents retrieved: {total_retrieved}")
    logger.info(f"Total relevant documents retrieved: {total_relevant_retrieved}")
    logger.info(f"Total false positives: {total_false_positives}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1_score:.4f}")
    logger.info(f"False positives saved to: {output_path}")
    logger.info("=" * 50)
    
    return {
        "total_queries": total_queries,
        "total_retrieved": total_retrieved,
        "total_relevant_retrieved": total_relevant_retrieved,
        "total_false_positives": total_false_positives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "false_positives_file": str(output_path)
    }


if __name__ == "__main__":
    model_name = "BAAI/bge-m3"
    
    # First index the corpus (comment out if already indexed)
    index_copurs(model_name=model_name)
    
    # Run search and evaluation
    results = search_and_evaluate(
        model_name=model_name,
        score_threshold=0.5,  # Adjust threshold as needed
        top_k=10,  # Retrieve top 10 documents per query
        output_file="data/false_positives.jsonl"  # Output file for false positives
    )
    
    logger.info("Evaluation completed!")
    logger.info(f"Results: {results}")
