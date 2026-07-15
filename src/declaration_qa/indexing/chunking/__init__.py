"""知识文档分块适配器。"""

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
