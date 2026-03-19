"""
retriever.py
------------
Searches ChromaDB for chunks relevant to a user question.

What does the retriever do?
    When a user asks a question, we can't send the entire document
    library to GPT — it's too expensive and too slow. Instead we:

        1. Convert the question into a vector (using the embedder)
        2. Search ChromaDB for the chunks with the most similar vectors
        3. Return just those top chunks to be sent to GPT

    This is called "semantic search" — we're searching by meaning,
    not by keywords. "heart drug" will find chunks about "cardiac
    medication" even though the words are different.

How it fits in the pipeline:
    indexer.py → [retriever.py] → prompt_builder.py

Usage:
    retriever = Retriever()

    # Find the 5 most relevant chunks for a question
    results = retriever.retrieve("What is the data retention policy?")
    for r in results:
        print(r.text)
        print(f"Score: {r.score}")
        print(f"Source: {r.metadata['source']}")
"""

import logging
import os
from dataclasses import dataclass, field

import chromadb
from chromadb.config import Settings
from openai import OpenAI

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "./chroma_db"
COLLECTION_NAME = "rag_chunks"
EMBEDDING_MODEL = "text-embedding-3-large"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """
    A single chunk returned by the retriever.

    Contains the text, how relevant it is (score),
    and where it came from (metadata).

    Score is between 0 and 1:
        1.0 = perfect match
        0.9 = very relevant
        0.7 = somewhat relevant
        below 0.5 = probably not relevant
    """
    text: str
    score: float
    metadata: dict = field(default_factory=dict)

    @property
    def source(self) -> str:
        """Filename this chunk came from."""
        return self.metadata.get("source", "unknown")

    @property
    def page(self) -> int:
        """Page number this chunk came from."""
        return self.metadata.get("page", 0)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class Retriever:
    """
    Searches ChromaDB for chunks relevant to a user question.

    How similarity search works:
        Every chunk in ChromaDB has a vector — a list of 3072 numbers
        representing its meaning. When you ask a question, we convert
        it to a vector too, then find the chunks whose vectors are
        mathematically closest to the question vector.

        "Closest" is measured using cosine similarity:
            - 1.0 = identical meaning
            - 0.9 = very similar
            - 0.5 = somewhat related
            - 0.0 = completely unrelated

    Args:
        db_path:        Path to ChromaDB database. Must match Indexer path.
        collection_name: ChromaDB collection name. Must match Indexer name.
        top_k:          How many chunks to return. Default 5.
        min_score:      Minimum similarity score. Chunks below this
                        threshold are filtered out. Default 0.5.
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        collection_name: str = COLLECTION_NAME,
        top_k: int = 5,
        min_score: float = 0.5,
    ):
        self.top_k = top_k
        self.min_score = min_score

        # Connect to the same ChromaDB the indexer wrote to
        self._chroma = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._chroma.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # OpenAI client for embedding the question
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not set. Add it to your .env file."
            )
        self._openai = OpenAI(api_key=api_key)

        logger.info(
            "Retriever initialised",
            extra={
                "top_k": top_k,
                "min_score": min_score,
                "chunks_in_index": self._collection.count(),
            },
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def retrieve(self, question: str) -> list[RetrievalResult]:
        """
        Find the most relevant chunks for a question.

        Steps:
            1. Embed the question into a vector
            2. Search ChromaDB for similar vectors
            3. Filter out results below min_score
            4. Return as RetrievalResult objects

        Args:
            question: The user's question as a plain string.

        Returns:
            List of RetrievalResult objects ordered by relevance
            (most relevant first). May be empty if nothing is relevant.

        Example:
            results = retriever.retrieve("What is the refund policy?")
            for r in results:
                print(f"Score: {r.score:.2f} | Source: {r.source}")
                print(r.text[:200])
                print("---")
        """
        if not question or not question.strip():
            raise ValueError("question cannot be empty")

        if self._collection.count() == 0:
            logger.warning("Retrieval attempted on empty index")
            print("  warning: index is empty — no chunks to search")
            return []

        # Step 1 — embed the question
        print(f"  searching for: '{question[:60]}...' " if len(question) > 60
              else f"  searching for: '{question}'")

        query_vector = self._embed_question(question)

        # Step 2 — search ChromaDB
        # n_results is how many candidates to fetch before filtering
        # We fetch more than top_k so we have candidates to filter
        raw_results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=min(self.top_k * 2, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        # Step 3 — parse and filter results
        results = self._parse_results(raw_results)

        print(f"  found {len(results)} relevant chunks "
              f"(from {self._collection.count()} total)")

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _embed_question(self, question: str) -> list[float]:
        """
        Convert the question to a vector using OpenAI.

        IMPORTANT: Must use the same model used during indexing.
        If you indexed with text-embedding-3-large, you must
        query with text-embedding-3-large too.
        """
        response = self._openai.embeddings.create(
            input=question.strip(),
            model=EMBEDDING_MODEL,
        )
        return response.data[0].embedding

    def _parse_results(self, raw: dict) -> list[RetrievalResult]:
        """
        Convert ChromaDB raw output into RetrievalResult objects.

        ChromaDB returns distances (lower = more similar).
        We convert to similarity scores (higher = more similar)
        so they're more intuitive: 1.0 = perfect, 0.0 = unrelated.

        ChromaDB cosine distance → similarity score:
            similarity = 1 - distance
        """
        results = []

        # ChromaDB wraps everything in an extra list — [0] unwraps it
        ids = raw.get("ids", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        for doc, meta, distance in zip(documents, metadatas, distances):
            # Convert distance to similarity score
            score = round(1 - distance, 4)

            # Filter out low-relevance results
            if score < self.min_score:
                continue

            results.append(
                RetrievalResult(
                    text=doc,
                    score=score,
                    metadata=meta or {},
                )
            )

        # Sort by score descending — most relevant first
        results.sort(key=lambda r: r.score, reverse=True)

        # Return only top_k results
        return results[:self.top_k]
        