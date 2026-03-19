"""
document_processor.py
---------------------
Orchestrates the document ingestion pipeline for the Enterprise RAG Platform.

Takes a file path (PDF or Word), routes it to the correct reader,
extracts clean text, runs it through the semantic chunker, and returns
a list of chunks ready for embedding and indexing.

Usage:
    processor = DocumentProcessor()

    # Process a single file
    chunks = processor.process("docs/policy.pdf")
    for chunk in chunks:
        print(chunk.text, chunk.metadata)

    # Process an entire folder
    all_chunks = processor.process_folder("docs/")
    print(f"Total chunks: {len(all_chunks)}")
"""

import logging
import os
from pathlib import Path

from src.ingestion.chunker import Chunk, SemanticChunker
from src.ingestion.pdf_reader import PdfReader
from src.ingestion.word_reader import WordReader

logger = logging.getLogger(__name__)

# Supported file extensions and their readers
SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "word",
    ".doc": "word",
}


class DocumentProcessor:
    """
    Entry point for the document ingestion pipeline.

    Responsibilities:
      - Detect file type and route to the correct reader
      - Extract clean text with page/section metadata
      - Run text through the semantic chunker
      - Return chunks enriched with source metadata

    Args:
        max_tokens:     Maximum tokens per chunk (passed to chunker).
        overlap_tokens: Overlap tokens between chunks (passed to chunker).
    """

    def __init__(
        self,
        max_tokens: int = 400,
        overlap_tokens: int = 50,
    ):
        self.chunker = SemanticChunker(
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
        self._pdf_reader = PdfReader()
        self._word_reader = WordReader()

        logger.info(
            "DocumentProcessor initialised",
            extra={
                "max_tokens": max_tokens,
                "overlap_tokens": overlap_tokens,
            },
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process(self, file_path: str | Path) -> list[Chunk]:
        """
        Process a single document into chunks.

        Args:
            file_path: Path to a PDF or Word file.

        Returns:
            List of Chunk objects ready for embedding and indexing.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError:        If the file type is not supported.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        extension = path.suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: '{extension}'. "
                f"Supported types: {list(SUPPORTED_EXTENSIONS.keys())}"
            )

        logger.info("Processing document", extra={"file": path.name})

        # Extract text + page metadata from the file
        pages = self._extract_pages(path, extension)

        if not pages:
            logger.warning("No text extracted from document", extra={"file": path.name})
            return []

        # Chunk each page separately so chunk boundaries respect page breaks
        all_chunks: list[Chunk] = []
        for page_num, page_text in pages:
            if not page_text.strip():
                continue

            metadata = {
                "source": path.name,
                "file_path": str(path),
                "file_type": extension.lstrip("."),
                "page": page_num,
                "total_pages": len(pages),
            }

            page_chunks = self.chunker.chunk(page_text, metadata=metadata)
            all_chunks.extend(page_chunks)

        # Re-index chunks sequentially across the whole document
        for i, chunk in enumerate(all_chunks):
            chunk.chunk_index = i
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(all_chunks)

        logger.info(
            "Document processed",
            extra={
                "file": path.name,
                "pages": len(pages),
                "chunks": len(all_chunks),
            },
        )

        return all_chunks

    def process_folder(
        self,
        folder_path: str | Path,
        recursive: bool = False,
    ) -> list[Chunk]:
        """
        Process all supported documents in a folder.

        Args:
            folder_path: Path to the folder containing documents.
            recursive:   If True, also process files in subfolders.

        Returns:
            Combined list of chunks from all processed documents.
        """
        folder = Path(folder_path)

        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")

        if not folder.is_dir():
            raise ValueError(f"Path is not a folder: {folder}")

        # Find all supported files
        pattern = "**/*" if recursive else "*"
        files = [
            f for f in folder.glob(pattern)
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        if not files:
            logger.warning(
                "No supported files found in folder",
                extra={"folder": str(folder), "recursive": recursive},
            )
            return []

        logger.info(
            "Processing folder",
            extra={"folder": str(folder), "file_count": len(files)},
        )

        all_chunks: list[Chunk] = []
        failed: list[str] = []

        for file in sorted(files):
            try:
                chunks = self.process(file)
                all_chunks.extend(chunks)
                print(f"  processed: {file.name} → {len(chunks)} chunks")
            except Exception as e:
                logger.error(
                    "Failed to process file",
                    extra={"file": file.name, "error": str(e)},
                )
                failed.append(file.name)
                print(f"  failed:    {file.name} — {e}")

        logger.info(
            "Folder processing complete",
            extra={
                "total_chunks": len(all_chunks),
                "files_processed": len(files) - len(failed),
                "files_failed": len(failed),
            },
        )

        if failed:
            print(f"\nFailed files ({len(failed)}): {', '.join(failed)}")

        return all_chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_pages(
        self,
        path: Path,
        extension: str,
    ) -> list[tuple[int, str]]:
        """
        Route to the correct reader and return a list of (page_num, text) tuples.
        Page numbers are 1-indexed.
        """
        file_type = SUPPORTED_EXTENSIONS[extension]

        if file_type == "pdf":
            return self._pdf_reader.read(path)
        elif file_type == "word":
            return self._word_reader.read(path)
        else:
            raise ValueError(f"No reader for file type: {file_type}")
