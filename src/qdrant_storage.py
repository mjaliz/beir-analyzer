import socket
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    ScoredPoint,
    SearchRequest,
    VectorParams,
)
from tqdm import tqdm


class QdrantStorage:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        api_key: str | None = None,
        https: bool = False,
        max_retries: int = 3,
        retry_delay: int = 5,
    ):
        self.host = host
        self.port = port
        self.api_key = api_key
        self.https = https
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client: QdrantClient | None = None

        self._connect_with_retry()

    def _connect_with_retry(self):
        """Connect to Qdrant with retry logic"""
        for attempt in range(self.max_retries):
            try:
                logger.info(
                    f"Attempting to connect to Qdrant at {self.host}:{self.port} (attempt {attempt + 1}/{self.max_retries})"
                )

                self.client = QdrantClient(
                    host=self.host,
                    port=self.port,
                    api_key=self.api_key,
                    https=self.https,
                    timeout=10,  # 10 second timeout for connection
                )

                # Test the connection by getting collections
                _ = self.client.get_collections()
                logger.info(
                    f"Successfully connected to Qdrant at {self.host}:{self.port}"
                )
                return

            except (
                socket.error,
                socket.gaierror,
                ResponseHandlingException,
                Exception,
            ) as e:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Failed to connect to Qdrant (attempt {attempt + 1}/{self.max_retries}): {e}"
                    )
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(
                        f"Failed to connect to Qdrant after {self.max_retries} attempts"
                    )
                    logger.error(
                        f"Please ensure Qdrant is running at {self.host}:{self.port}"
                    )
                    logger.error(
                        "You can start Qdrant with: docker run -p 6333:6333 qdrant/qdrant"
                    )
                    raise ConnectionError(
                        f"Cannot connect to Qdrant at {self.host}:{self.port}: {e}"
                    )

    def _execute_with_retry(self, func, *args, **kwargs):
        """Execute a function with retry logic"""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except (socket.error, socket.gaierror, ResponseHandlingException) as e:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Operation failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                    )
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                    # Try to reconnect
                    self._connect_with_retry()
                else:
                    logger.error(
                        f"Operation failed after {self.max_retries} attempts: {e}"
                    )
                    raise

    def create_collection(self, collection_name: str) -> bool | None:
        if self.client.collection_exists(collection_name):
            return None
        return self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )

    def index_corpus(self, collection_name: str, embedding_generator):
        points = []
        batch_size = 300
        total_indexed = 0

        for doc_embedding in tqdm(
            embedding_generator, desc=f"Indexing to {collection_name}"
        ):
            point = PointStruct(
                id=str(uuid4()),
                vector=doc_embedding["vector"],
                payload=doc_embedding["payload"],
            )
            points.append(point)

            if len(points) >= batch_size:
                self._execute_with_retry(
                    self.client.upsert, collection_name=collection_name, points=points
                )
                total_indexed += len(points)
                points = []

        # Index remaining points
        if points:
            self._execute_with_retry(
                self.client.upsert, collection_name=collection_name, points=points
            )
            total_indexed += len(points)

        logger.info(
            f"Indexed {total_indexed} documents to collection {collection_name}"
        )

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        score_threshold: Optional[float] = None,
    ) -> List[ScoredPoint]:
        search_params = {
            "collection_name": collection_name,
            "query_vector": query_vector,
            "with_payload": True,
        }

        if score_threshold is not None:
            search_params["score_threshold"] = score_threshold

        results = self._execute_with_retry(self.client.search, **search_params)
        return results
