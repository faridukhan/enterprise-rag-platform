"""
prompt_builder.py
-----------------
Assembles retrieved chunks and a user question into a prompt for GPT.

What does the prompt builder do?
    After the retriever finds the most relevant chunks, we can't just
    send them directly to GPT. We need to structure them into a clear,
    well-formatted prompt that tells GPT:

        1. HOW to behave (system instruction)
        2. WHAT to use as reference (the retrieved chunks)
        3. WHAT to answer (the user's question)

    Think of it like writing a briefing note for a consultant:
        "Here are the relevant documents. Based ONLY on these,
         please answer the following question: ..."

    This structure is what prevents hallucination — by telling GPT
    to answer only from the provided context, it can't make things up.

How it fits in the pipeline:
    retriever.py → [prompt_builder.py] → llm_client.py

Usage:
    builder = PromptBuilder()
    prompt = builder.build(
        question="What is the data retention policy?",
        chunks=retrieval_results,
    )
    print(prompt.system)    # the system instruction
    print(prompt.user)      # context + question combined
"""

import logging
from dataclasses import dataclass

from src.ingestion.retriever import RetrievalResult

logger = logging.getLogger(__name__)

# Default system instruction — tells GPT how to behave
DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that answers questions
based strictly on the provided context documents.

Rules you must follow:
- Answer ONLY using information from the context provided below
- If the context does not contain enough information to answer, say:
  "I don't have enough information in the provided documents to answer this."
- Always cite which document your answer comes from
- Be concise and accurate
- Do not make up information or use knowledge outside the provided context"""


@dataclass
class BuiltPrompt:
    """
    A fully assembled prompt ready to send to GPT.

    Contains two parts:
        system: Instructions telling GPT how to behave
        user:   The context chunks + the user's question

    These map directly to the messages format OpenAI expects:
        [
            {"role": "system", "content": prompt.system},
            {"role": "user",   "content": prompt.user}
        ]
    """
    system: str
    user: str
    question: str
    chunks_used: int
    sources: list[str]

    def to_messages(self) -> list[dict]:
        """
        Convert to the messages format expected by OpenAI.

        This is what you pass directly to llm_client.complete():
            response = client.complete(messages=prompt.to_messages())
        """
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]

    def summary(self) -> str:
        """Print a summary of what was assembled."""
        return (
            f"Question: {self.question}\n"
            f"Chunks used: {self.chunks_used}\n"
            f"Sources: {', '.join(self.sources)}\n"
            f"Prompt length: {len(self.user)} chars"
        )


class PromptBuilder:
    """
    Assembles retrieved chunks and a question into a structured prompt.

    Design decisions:
        - Context comes BEFORE the question in the prompt.
          Research shows GPT performs better when it reads the
          context first, then sees the question.

        - Each chunk is labelled with its source and page number.
          This makes it easy for GPT to cite where its answer
          came from.

        - We include the relevance score in the chunk header.
          This helps GPT weight more relevant chunks higher.

        - There is a max_context_chars limit to prevent prompts
          from getting too long and expensive.

    Args:
        system_prompt:    The system instruction for GPT.
                          Controls how GPT behaves and what rules it follows.
        max_context_chars: Maximum characters of context to include.
                          Prevents prompts from getting too long/expensive.
                          Default 8000 chars ≈ ~2000 tokens of context.
    """

    def __init__(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_context_chars: int = 8000,
    ):
        self.system_prompt = system_prompt
        self.max_context_chars = max_context_chars

    def build(
        self,
        question: str,
        chunks: list[RetrievalResult],
    ) -> BuiltPrompt:
        """
        Assemble chunks and question into a structured prompt.

        Args:
            question: The user's question as a plain string.
            chunks:   List of RetrievalResult objects from the retriever.

        Returns:
            BuiltPrompt ready to send to llm_client.complete()

        Example:
            results = retriever.retrieve("What is the refund policy?")
            prompt = builder.build(
                question="What is the refund policy?",
                chunks=results,
            )
            response = client.complete(messages=prompt.to_messages())
            print(response.content)
        """
        if not question or not question.strip():
            raise ValueError("question cannot be empty")

        if not chunks:
            # No relevant chunks found — tell GPT there's no context
            user_message = (
                "No relevant context was found in the documents.\n\n"
                f"Question: {question}"
            )
            logger.warning("Building prompt with no chunks")
            return BuiltPrompt(
                system=self.system_prompt,
                user=user_message,
                question=question,
                chunks_used=0,
                sources=[],
            )

        # Build the context section
        context_parts = []
        total_chars = 0
        sources_used = []
        chunks_used = 0

        for i, chunk in enumerate(chunks, start=1):
            # Format each chunk with a clear header
            chunk_header = (
                f"[Document {i} | Source: {chunk.source} | "
                f"Page: {chunk.page} | Relevance: {chunk.score:.2f}]"
            )
            chunk_text = f"{chunk_header}\n{chunk.text}"

            # Check if adding this chunk would exceed our limit
            if total_chars + len(chunk_text) > self.max_context_chars:
                logger.debug(
                    "Context limit reached",
                    extra={"chunks_included": chunks_used, "limit": self.max_context_chars},
                )
                break

            context_parts.append(chunk_text)
            total_chars += len(chunk_text)
            chunks_used += 1

            if chunk.source not in sources_used:
                sources_used.append(chunk.source)

        # Assemble the full user message
        context_block = "\n\n".join(context_parts)

        user_message = (
            "CONTEXT DOCUMENTS:\n"
            "==================\n"
            f"{context_block}\n\n"
            "==================\n"
            f"QUESTION: {question}\n\n"
            "Please answer the question based only on the context documents above. "
            "Cite the document number and source in your answer."
        )

        logger.info(
            "Prompt assembled",
            extra={
                "question_length": len(question),
                "chunks_used": chunks_used,
                "context_chars": total_chars,
                "sources": sources_used,
            },
        )

        print(f"  prompt built — {chunks_used} chunks, "
              f"{total_chars} context chars, "
              f"sources: {', '.join(sources_used)}")

        return BuiltPrompt(
            system=self.system_prompt,
            user=user_message,
            question=question,
            chunks_used=chunks_used,
            sources=sources_used,
        )