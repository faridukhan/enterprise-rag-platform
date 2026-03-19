"""
pdf_reader.py
-------------
Extracts text from PDF files page by page using pypdf.

Returns a list of (page_number, text) tuples so the document
processor can attach accurate page metadata to every chunk —
essential for citation tracing in the RAG pipeline.

Usage:
    reader = PdfReader()
    pages = reader.read("policy.pdf")
    for page_num, text in pages:
        print(f"Page {page_num}: {len(text)} chars")
"""

import logging
from pathlib import Path

import pypdf

logger = logging.getLogger(__name__)


class PdfReader:
    """
    Reads text from PDF files using pypdf.

    Handles:
      - Multi-page PDFs (returns one entry per page)
      - PDFs with mixed text and empty pages (empty pages are skipped)
      - Basic text cleaning (normalise whitespace, remove null bytes)

    Limitations:
      - Does not OCR scanned PDFs (image-only pages return empty text)
      - Does not extract tables as structured data
      - Complex layouts (multi-column) may have word-order issues

    For scanned PDFs, Azure Document Intelligence (AI service) gives
    much better results — see the architecture doc for that upgrade path.
    """

    def read(self, file_path: str | Path) -> list[tuple[int, str]]:
        """
        Extract text from a PDF file, one entry per page.

        Args:
            file_path: Path to the PDF file.

        Returns:
            List of (page_number, text) tuples. Page numbers are 1-indexed.
            Empty pages are included with empty strings so page numbering
            stays accurate for citation purposes.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError:        If the file cannot be read as a PDF.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        logger.debug("Reading PDF", extra={"file": path.name})

        try:
            reader = pypdf.PdfReader(str(path))
        except Exception as e:
            raise ValueError(f"Could not open PDF '{path.name}': {e}") from e

        total_pages = len(reader.pages)
        pages: list[tuple[int, str]] = []
        empty_count = 0

        for i, page in enumerate(reader.pages):
            page_num = i + 1  # 1-indexed for human-readable citations

            try:
                raw_text = page.extract_text() or ""
                clean_text = self._clean(raw_text)
            except Exception as e:
                logger.warning(
                    "Failed to extract text from page",
                    extra={"file": path.name, "page": page_num, "error": str(e)},
                )
                clean_text = ""

            if not clean_text.strip():
                empty_count += 1

            pages.append((page_num, clean_text))

        logger.info(
            "PDF read complete",
            extra={
                "file": path.name,
                "total_pages": total_pages,
                "empty_pages": empty_count,
                "pages_with_text": total_pages - empty_count,
            },
        )

        return pages

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean(text: str) -> str:
        """
        Basic text cleaning for PDF output.

        pypdf sometimes produces:
          - Null bytes (\x00) in certain PDFs
          - Excessive whitespace from layout spacing
          - Hyphenated line breaks (word- \nbreaks)
        """
        if not text:
            return ""

        # Remove null bytes
        text = text.replace("\x00", "")

        # Rejoin hyphenated line breaks e.g. "govern-\nance" → "governance"
        import re
        text = re.sub(r"-\n(\w)", r"\1", text)

        # Normalise multiple spaces to single space
        text = re.sub(r"[ \t]+", " ", text)

        # Normalise multiple newlines to max two
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()
