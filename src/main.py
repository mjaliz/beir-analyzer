from pathlib import Path
from typing import Generator

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
    qrels_df = load_qrels(config.data.qrels_path)
    for c in corpus_loader:
        qrel = qrels_df.loc[qrels_df["doc"] == c.id]
        q_id = qrel["q"].values[0]
        qrel_score = qrel["score"].values[0]
        q_text = None
        for q in load_query(config.data.query_path):
            if q.id == q_id:
                q_text = q.text
                break
        if q_text is None:
            raise ValueError("pair query not found")
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
                payload = c.model_dump(exclude=set(["title"]))
                yield {
                    "vector": embedding,
                    "payload": payload,
                }
            batch = []
        if batch:
            embeddings = embedder.embed([d.text for d in batch])
            for doc, embedding in zip(batch, embeddings):
                payload = c.model_dump(exclude=set(["title"]))
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


if __name__ == "__main__":
    index_copurs(model_name="BAAI/bge-m3")
