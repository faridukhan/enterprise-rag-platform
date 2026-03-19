"""
indexer.py
----------
Stores embedded chunks into ChromaDB — a local vector database.

What is a vector database?
    A regular database stores rows of text you search with keywords.
    A vector database stores lists of numbers (vectors) you search
    by similarity. "Find me the 5 chunks whose meaning is closest
    to this question" — that's what a vector database does.

    ChromaDB runs entirely on your laptop in a local folder.
    No cloud account, no API key, no cost. Perfect for development.

How it fits in the pipeline:
    embedder.py → [indexer.py] → retriever.py

    The indexer takes EmbeddedChunk objects from the embedder
    and stores them in ChromaDB. The retriever later searches
    that same database to find relevant chunks.

Usage:
    indexer = Indexer()

    # Store chunks (run once per document)
    indexer.index(embedded_chunks)

    # Check how many chunks are stored
    print(indexer.count())

    # Clear the index (start fresh)
    indexer.clear()
"""

import logging
import os
from pathlib import Path

import chromadb
from chromadb.config import Settings

from src.ingestion.embedder import EmbeddedChunk

logger = logging.getLogger(__name__)

# Where ChromaDB stores its data on your laptop
DEFAULT_DB_PATH = "./chroma_db"

# Name of the collection inside ChromaDB
# Think of a collection like a table in a regular database
COLLECTION_NAME = "rag_chunks"


class Indexer:
    """
    Stores and manages embedded chunks in ChromaDB.

    What is a collection?
        ChromaDB organises data into collections — similar to tables
        in a relational database. All your document chunks live in
        one collection called "rag_chunks".

        Each item in the collection has three parts:
          - id:        a unique identifier for the chunk
          - embedding: the vector (list of numbers)
          - document:  the original text
          - metadata:  source filename, page number, etc.

    Args:
        db_path:         Where to store the ChromaDB files on disk.
                         Defaults to ./chroma_db in your project folder.
        collection_name: Name of the ChromaDB collection to use.
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        collection_name: str = COLLECTION_NAME,
    ):
        self.db_path = db_path
        self.collection_name = collection_name

        # Create the database folder if it doesn't exist
        Path(db_path).mkdir(parents=True, exist_ok=True)

        # Connect to ChromaDB — creates a new database or opens existing one
        self._client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create the collection
        # get_or_create means: use existing collection if it exists,
        # otherwise create a new empty one
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            # cosine similarity is the standard way to measure how
            # similar two vectors are — value between 0 and 1
            # where 1 means identical meaning
        )

        logger.info(
            "Indexer initialised",
            extra={
                "db_path": db_path,
                "collection": collection_name,
                "existing_chunks": self._collection.count(),
            },
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def index(self, embedded_chunks: list[EmbeddedChunk]) -> int:
        """
        Store a list of embedded chunks in ChromaDB.

        Each chunk is stored with:
          - A unique ID (source filename + chunk index)
          - Its vector embedding
          - Its original text
          - Its metadata (source, page, etc.)

        Args:
            embedded_chunks: List of EmbeddedChunk objects from embedder.py

        Returns:
            Number of chunks successfully stored.

        Example:
            chunks = chunker.chunk(text, metadata={"source": "policy.pdf"})
            embedded = embedder.embed_chunks(chunks)
            stored = indexer.index(embedded)
            print(f"Stored {stored} chunks")
        """
        if not embedded_chunks:
            logger.warning("index() called with empty list")
            return 0

        # Build the four lists ChromaDB needs
        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for chunk in embedded_chunks:
            # Create a unique ID from source file + chunk index
            # This prevents duplicates if you index the same file twice
            source = chunk.metadata.get("source", "unknown")
            chunk_idx = chunk.metadata.get("chunk_index", 0)
            chunk_id = f"{source}__chunk_{chunk_idx}"

            ids.append(chunk_id)
            embeddings.append(chunk.vector)
            documents.append(chunk.text)
            metadatas.append(chunk.metadata)

        # upsert = insert if new, update if already exists
        # This means you can safely re-index a document without duplicates
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        stored = len(embedded_chunks)
        logger.info(
            "Chunks indexed",
            extra={
                "chunks_stored": stored,
                "total_in_index": self._collection.count(),
            },
        )

        print(f"  indexed {stored} chunks — total in database: {self._collection.count()}")
        return stored

    def count(self) -> int:
        """
        Return the total number of chunks currently in the index.

        Example:
            print(f"Index contains {indexer.count()} chunks")
        """
        return self._collection.count()

    def clear(self) -> None:
        """
        Delete all chunks from the index and start fresh.

        Useful during development when you want to re-index
        your documents from scratch.

        Example:
            indexer.clear()
            print(indexer.count())  # prints 0
        """
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Index cleared")
        print("  index cleared — 0 chunks remaining")

    def peek(self, n: int = 3) -> list[dict]:
        """
        Show the first N chunks stored in the index.

        Useful for verifying that indexing worked correctly.
        Shows the text and metadata of each chunk.

        Args:
            n: Number of chunks to show. Default 3.

        Example:
            for item in indexer.peek(3):
                print(item["text"][:100])
                print(item["metadata"])
                print("---")
        """
        results = self._collection.peek(limit=n)
        chunks = []
        for i in range(len(results["ids"])):
            chunks.append({
                "id": results["ids"][i],
                "text": results["documents"][i],
                "metadata": results["metadatas"][i],
            })
        return chunks