"""
embedder.py
-----------
Converts text chunks into vector embeddings using OpenAI.

What is an embedding?
    An embedding is a list of numbers (a vector) that represents the
    *meaning* of a piece of text. Text with similar meanings produces
    similar vectors. This is what makes semantic search possible —
    instead of matching keywords, we match meaning.

    Example:
        "cardiac medication"  → [0.021, -0.847, 0.332, ...]
        "heart drug"          → [0.019, -0.851, 0.341, ...]
        (very similar vectors — similar meaning)

        "quarterly earnings"  → [-0.442, 0.201, -0.887, ...]
        (very different vector — unrelated meaning)

How it fits in the pipeline:
    document_processor.py  →  chunker.py  →  [embedder.py]  →  indexer.py

Usage:
    embedder = Embedder()
    vector = embedder.embed_text("What is the data retention policy?")
    embedded_chunks = embedder.embed_chunks(chunks)
"""

import logging
import os
import time
from dataclasses import dataclass, field

from openai import OpenAI

from src.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072
MAX_BATCH_SIZE = 16


@dataclass
class EmbeddedChunk:
    """
    A chunk of text paired with its embedding vector.

    This is what gets stored in the vector index — both the original
    text (so we can show it to the user as a citation) and the vector
    (so we can do semantic search against it).
    """
    text: str
    vector: list[float]
    metadata: dict = field(default_factory=dict)

    @property
    def dimensions(self) -> int:
        """Number of dimensions in the embedding vector."""
        return len(self.vector)


class Embedder:
    """
    Converts text into embedding vectors using OpenAI.

    Key concepts:
        - Use the SAME model for both ingestion and querying.
          If you embed chunks with model A and questions with model B,
          the vectors won't be comparable and search won't work.

        - Chunks are processed in batches to stay within API rate
          limits and reduce the number of API calls.

    Args:
        model:      OpenAI embedding model. Defaults to text-embedding-3-large.
        batch_size: How many texts to embed per API call. Default is 16.
    """

    def __init__(
        self,
        model: str = EMBEDDING_MODEL,
        batch_size: int = MAX_BATCH_SIZE,
    ):
        self.model = model
        self.batch_size = min(batch_size, MAX_BATCH_SIZE)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set. "
                "Add it to your .env file."
            )

        self._client = OpenAI(api_key=api_key)

        logger.info(
            "Embedder initialised",
            extra={"model": self.model, "batch_size": self.batch_size},
        )

    def embed_text(self, text: str) -> list[float]:
        """
        Embed a single piece of text and return its vector.

        Used at QUERY TIME — when a user types a question, embed it
        here so we can compare it against stored chunk vectors.

        Args:
            text: The text to embed. Must be non-empty.

        Returns:
            A list of 3072 floats representing the meaning of the text.

        Example:
            vector = embedder.embed_text("What is the refund policy?")
            # vector → [0.021, -0.847, 0.332, ...] — 3072 numbers
        """
        if not text or not text.strip():
            raise ValueError("text cannot be empty")

        results = self._embed_batch([text.strip()])
        return results[0]

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """
        Embed a list of chunks and return EmbeddedChunk objects.

        Used at INGESTION TIME — after the document processor has
        split a document into chunks, embed all of them so they can
        be stored in the vector index.

        Args:
            chunks: List of Chunk objects from the chunker.

        Returns:
            List of EmbeddedChunk objects — text + vector + metadata.

        Example:
            chunks = chunker.chunk(text, metadata={"source": "policy.pdf"})
            embedded = embedder.embed_chunks(chunks)
            # embedded[0].text    → the chunk text
            # embedded[0].vector  → [0.021, -0.847, ...]
            # embedded[0].metadata → {"source": "policy.pdf", ...}
        """
        if not chunks:
            logger.warning("embed_chunks called with empty list")
            return []

        total_batches = -(-len(chunks) // self.batch_size)
        logger.info(
            "Embedding chunks",
            extra={
                "chunk_count": len(chunks),
                "batches_needed": total_batches,
            },
        )

        embedded_chunks: list[EmbeddedChunk] = []
        start_time = time.monotonic()

        for batch_num, batch in enumerate(self._make_batches(chunks), start=1):
            texts = [chunk.text for chunk in batch]
            print(f"  embedding batch {batch_num}/{total_batches} ({len(texts)} chunks)...")

            vectors = self._embed_batch(texts)

            for chunk, vector in zip(batch, vectors):
                embedded_chunks.append(
                    EmbeddedChunk(
                        text=chunk.text,
                        vector=vector,
                        metadata=chunk.metadata,
                    )
                )

            if batch_num < total_batches:
                time.sleep(0.1)

        elapsed = round(time.monotonic() - start_time, 2)
        print(f"  done — {len(embedded_chunks)} chunks embedded in {elapsed}s")

        return embedded_chunks

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Call the OpenAI embeddings API for a batch of texts."""
        response = self._client.embeddings.create(
            input=texts,
            model=self.model,
        )
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]

    def _make_batches(self, chunks: list[Chunk]):
        """
        Split chunks into smaller batches for API calls.

        Python generator — yields one batch at a time without
        loading everything into memory at once.

        Example with batch_size=3 and 7 chunks:
            batch 1 → chunks[0], chunks[1], chunks[2]
            batch 2 → chunks[3], chunks[4], chunks[5]
            batch 3 → chunks[6]
        """
        for i in range(0, len(chunks), self.batch_size):
            yield chunks[i : i + self.batch_size]