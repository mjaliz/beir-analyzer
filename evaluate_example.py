#!/usr/bin/env python3
"""
Example script demonstrating how to use the search and evaluation functionality.
"""

from src.main import search_and_evaluate, index_copurs
from loguru import logger

def main():
    """Run search and evaluation example"""
    model_name = "BAAI/bge-m3"
    
    logger.info("Starting BEIR evaluation example")
    
    # Step 1: Index the corpus (run this first if not already indexed)
    logger.info("Step 1: Indexing corpus...")
    # Uncomment the line below if you haven't indexed the corpus yet
    # index_copurs(model_name=model_name)
    
    # Step 2: Run search and evaluation
    logger.info("Step 2: Running search and evaluation...")
    results = search_and_evaluate(
        model_name=model_name,
        score_threshold=0.3,  # Lower threshold to capture more documents
        top_k=20,             # Retrieve top 20 documents per query
        # output_file will be auto-generated: data/false_positives_BAAI-bge-m3.jsonl
    )
    
    # Step 3: Display results
    logger.info("Step 3: Evaluation completed!")
    logger.info(f"Final Results:")
    logger.info(f"  - Precision: {results['precision']:.4f}")
    logger.info(f"  - Recall: {results['recall']:.4f}")
    logger.info(f"  - F1-Score: {results['f1_score']:.4f}")
    logger.info(f"  - False Positives: {results['total_false_positives']}")
    logger.info(f"  - False Positives File: {results['false_positives_file']}")

if __name__ == "__main__":
    main()
