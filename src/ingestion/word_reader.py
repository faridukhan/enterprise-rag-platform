"""
word_reader.py
--------------
Extracts text from Word (.docx) files using python-docx.

Word documents don't have pages in the same way PDFs do — they reflow
based on font size and margins. This reader uses headings and sections
as logical "page" boundaries instead, which gives more semantically
meaningful chunk boundaries than arbitrary page breaks would.

Usage:
    reader = WordReader()
    pages = reader.read("report.docx")
    for section_num, text in pages:
        print(f"Section {section_num}: {len(text)} chars")
"""

import logging
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn

logger = logging.getLogger(__name__)

# Heading styles that mark a new logical section
HEADING_STYLES = {
    "Heading 1",
    "Heading 2",
    "Heading 3",
    "Title",
    "Subtitle",
}


class WordReader:
    """
    Reads text from Word (.docx) files using python-docx.

    Strategy:
        Splits the document into logical sections at Heading 1 / Heading 2
        boundaries. Each section becomes one "page" entry — this gives the
        chunker better input than raw continuous text would, because heading
        boundaries are almost always semantic boundaries too.

        Falls back to treating the whole document as one section if no
        headings are present.

    Handles:
      - Multi-section documents with headings
      - Tables (extracted as tab-separated rows)
      - Documents with no headings (treated as single section)
      - Paragraph-level text cleaning
    """

    def read(self, file_path: str | Path) -> list[tuple[int, str]]:
        """
        Extract text from a Word file, split by logical sections.

        Args:
            file_path: Path to the .docx file.

        Returns:
            List of (section_number, text) tuples. Section numbers
            are 1-indexed. For documents without headings, returns
            a single entry with all text.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError:        If the file cannot be read as a Word document.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Word file not found: {path}")

        logger.debug("Reading Word document", extra={"file": path.name})

        try:
            doc = Document(str(path))
        except Exception as e:
            raise ValueError(
                f"Could not open Word document '{path.name}': {e}"
            ) from e

        sections = self._split_into_sections(doc)

        logger.info(
            "Word document read complete",
            extra={
                "file": path.name,
                "sections": len(sections),
                "total_chars": sum(len(t) for _, t in sections),
            },
        )

        return sections

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split_into_sections(
        self,
        doc: Document,
    ) -> list[tuple[int, str]]:
        """
        Walk the document and split at Heading 1 / Heading 2 boundaries.
        Returns list of (section_number, text) tuples.
        """
        sections: list[tuple[int, str]] = []
        current_lines: list[str] = []
        section_num = 1

        for element in self._iter_block_elements(doc):
            element_type, text = element

            if not text.strip():
                continue

            if element_type == "heading":
                # Flush the current section before starting a new one
                if current_lines:
                    section_text = "\n".join(current_lines).strip()
                    if section_text:
                        sections.append((section_num, section_text))
                        section_num += 1
                    current_lines = []

            current_lines.append(text)

        # Flush the final section
        if current_lines:
            section_text = "\n".join(current_lines).strip()
            if section_text:
                sections.append((section_num, section_text))

        # If no sections were found, return everything as one section
        if not sections:
            all_text = self._extract_all_text(doc)
            if all_text:
                sections = [(1, all_text)]

        return sections

    def _iter_block_elements(
        self,
        doc: Document,
    ):
        """
        Yield (element_type, text) for every block element in the document.
        element_type is "heading", "paragraph", or "table".
        """
        for element in doc.element.body:
            tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag

            if tag == "p":
                # Paragraph
                para_text = "".join(
                    node.text or ""
                    for node in element.iter()
                    if node.tag.endswith("}t")
                ).strip()

                if not para_text:
                    continue

                # Check if this paragraph uses a heading style
                style_elem = element.find(f".//{qn('w:pStyle')}")
                style_name = ""
                if style_elem is not None:
                    style_name = style_elem.get(qn("w:val"), "")

                is_heading = any(
                    h.replace(" ", "").lower() in style_name.replace(" ", "").lower()
                    for h in ["Heading1", "Heading2", "Title"]
                )

                yield ("heading" if is_heading else "paragraph", para_text)

            elif tag == "tbl":
                # Table — extract as tab-separated rows
                table_text = self._extract_table_text(element)
                if table_text:
                    yield ("paragraph", table_text)

    @staticmethod
    def _extract_table_text(table_element) -> str:
        """Extract table content as newline-separated rows of tab-separated cells."""
        rows = []
        for row in table_element.findall(f".//{qn('w:tr')}"):
            cells = []
            for cell in row.findall(f".//{qn('w:tc')}"):
                cell_text = "".join(
                    node.text or ""
                    for node in cell.iter()
                    if node.tag.endswith("}t")
                ).strip()
                cells.append(cell_text)
            if any(cells):
                rows.append("\t".join(cells))
        return "\n".join(rows)

    @staticmethod
    def _extract_all_text(doc: Document) -> str:
        """Fallback: extract all paragraph text as a single string."""
        lines = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                lines.append(text)
        return "\n".join(lines)
