"""Knowledge document chunking adapters."""

from declaration_qa.indexing.chunking.chonkie_chunker import (
    ChonkieChunkingError,
    StructuredElement,
    build_chonkie_chunk_specs,
    build_fallback_chunk_specs,
    build_markdown_chunk_specs,
)

__all__ = [
    "ChonkieChunkingError",
    "StructuredElement",
    "build_chonkie_chunk_specs",
    "build_fallback_chunk_specs",
    "build_markdown_chunk_specs",
]
