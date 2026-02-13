"""
=============================================================================
HUMMINGBIRD-LEA - Text Chunking
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
Text chunking strategies for preparing documents for embedding.

Good chunking is critical for RAG quality:
- Chunks should be semantically coherent
- Overlap helps preserve context at boundaries
- Size should match embedding model limits
=============================================================================
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Callable
from enum import Enum

from .loaders import Document, DocumentChunk, DocumentMetadata

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class ChunkingStrategy(Enum):
    """Available chunking strategies"""
    FIXED_SIZE = "fixed_size"           # Fixed character count
    SENTENCE = "sentence"               # Sentence-based
    PARAGRAPH = "paragraph"             # Paragraph-based
    SEMANTIC = "semantic"               # Semantic boundaries (headers, etc.)
    RECURSIVE = "recursive"             # Recursive splitting


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ChunkingConfig:
    """Configuration for text chunking"""
    chunk_size: int = 512              # Target chunk size in characters
    chunk_overlap: int = 50            # Overlap between chunks
    min_chunk_size: int = 100          # Minimum chunk size
    max_chunk_size: int = 1000         # Maximum chunk size
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    separator: str = "\n\n"            # Primary separator

    # For nomic-embed-text, 512 tokens is a good size
    # ~4 chars per token, so 512 chars is reasonable


# =============================================================================
# Base Chunker
# =============================================================================

class BaseChunker(ABC):
    """Abstract base class for text chunkers"""

    @abstractmethod
    def chunk(self, document: Document) -> List[DocumentChunk]:
        """
        Split a document into chunks.

        Args:
            document: The document to chunk

        Returns:
            List of DocumentChunk objects
        """
        pass


# =============================================================================
# Fixed Size Chunker
# =============================================================================

class FixedSizeChunker(BaseChunker):
    """
    Simple fixed-size chunking with overlap.
    Fast but may split mid-sentence.
    """

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()

    def chunk(self, document: Document) -> List[DocumentChunk]:
        """Split document into fixed-size chunks"""
        text = document.content
        chunks = []

        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.config.chunk_size

            # Don't exceed text length
            if end > len(text):
                end = len(text)

            chunk_text = text[start:end].strip()

            if len(chunk_text) >= self.config.min_chunk_size:
                chunks.append(DocumentChunk(
                    content=chunk_text,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                    metadata=document.metadata,
                ))
                chunk_index += 1

            # Move start with overlap
            start = end - self.config.chunk_overlap

            # Prevent infinite loop
            if start >= len(text) - self.config.min_chunk_size:
                break

        logger.info(f"Created {len(chunks)} fixed-size chunks from {document.metadata.filename}")
        return chunks


# =============================================================================
# Sentence Chunker
# =============================================================================

class SentenceChunker(BaseChunker):
    """
    Chunk by sentences, grouping until target size is reached.
    Better semantic coherence than fixed size.
    """

    # Sentence ending patterns
    SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s+')

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()

    def chunk(self, document: Document) -> List[DocumentChunk]:
        """Split document by sentences"""
        text = document.content

        # Split into sentences
        sentences = self.SENTENCE_ENDINGS.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = []
        current_size = 0
        chunk_start = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            # If adding this sentence exceeds max, save current chunk
            if current_size + sentence_size > self.config.max_chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunk_end = chunk_start + len(chunk_text)

                chunks.append(DocumentChunk(
                    content=chunk_text,
                    chunk_index=chunk_index,
                    start_char=chunk_start,
                    end_char=chunk_end,
                    metadata=document.metadata,
                ))
                chunk_index += 1

                # Handle overlap - keep last sentence(s)
                overlap_sentences = []
                overlap_size = 0
                for s in reversed(current_chunk):
                    if overlap_size + len(s) < self.config.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_size += len(s)
                    else:
                        break

                current_chunk = overlap_sentences
                current_size = overlap_size
                chunk_start = chunk_end - overlap_size

            current_chunk.append(sentence)
            current_size += sentence_size

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) >= self.config.min_chunk_size:
                chunks.append(DocumentChunk(
                    content=chunk_text,
                    chunk_index=chunk_index,
                    start_char=chunk_start,
                    end_char=chunk_start + len(chunk_text),
                    metadata=document.metadata,
                ))

        logger.info(f"Created {len(chunks)} sentence-based chunks from {document.metadata.filename}")
        return chunks


# =============================================================================
# Paragraph Chunker
# =============================================================================

class ParagraphChunker(BaseChunker):
    """
    Chunk by paragraphs (double newlines).
    Good for well-structured documents.
    """

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()

    def chunk(self, document: Document) -> List[DocumentChunk]:
        """Split document by paragraphs"""
        text = document.content

        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        current_chunk = []
        current_size = 0
        chunk_start = 0
        chunk_index = 0

        for para in paragraphs:
            para_size = len(para)

            # If single paragraph exceeds max, split it further
            if para_size > self.config.max_chunk_size:
                # Save current chunk if any
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunks.append(DocumentChunk(
                        content=chunk_text,
                        chunk_index=chunk_index,
                        start_char=chunk_start,
                        end_char=chunk_start + len(chunk_text),
                        metadata=document.metadata,
                    ))
                    chunk_index += 1
                    current_chunk = []
                    current_size = 0

                # Split large paragraph with sentence chunker
                sentence_chunker = SentenceChunker(self.config)
                temp_doc = Document(content=para, metadata=document.metadata)
                para_chunks = sentence_chunker.chunk(temp_doc)

                for pc in para_chunks:
                    pc.chunk_index = chunk_index
                    chunks.append(pc)
                    chunk_index += 1

                chunk_start += para_size + 2  # +2 for \n\n
                continue

            # If adding this paragraph exceeds max, save current chunk
            if current_size + para_size > self.config.max_chunk_size and current_chunk:
                chunk_text = "\n\n".join(current_chunk)

                chunks.append(DocumentChunk(
                    content=chunk_text,
                    chunk_index=chunk_index,
                    start_char=chunk_start,
                    end_char=chunk_start + len(chunk_text),
                    metadata=document.metadata,
                ))
                chunk_index += 1
                chunk_start += len(chunk_text) + 2
                current_chunk = []
                current_size = 0

            current_chunk.append(para)
            current_size += para_size

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            if len(chunk_text) >= self.config.min_chunk_size:
                chunks.append(DocumentChunk(
                    content=chunk_text,
                    chunk_index=chunk_index,
                    start_char=chunk_start,
                    end_char=chunk_start + len(chunk_text),
                    metadata=document.metadata,
                ))

        logger.info(f"Created {len(chunks)} paragraph-based chunks from {document.metadata.filename}")
        return chunks


# =============================================================================
# Recursive Chunker (Recommended)
# =============================================================================

class RecursiveChunker(BaseChunker):
    """
    Recursive text splitting - tries progressively smaller separators.
    This is the recommended chunker for most use cases.

    Order of separators tried:
    1. Double newline (paragraphs)
    2. Single newline
    3. Sentence endings
    4. Spaces (words)
    5. Characters (last resort)
    """

    # Separators in order of preference (most semantic to least)
    SEPARATORS = [
        "\n\n",     # Paragraphs
        "\n",       # Lines
        ". ",       # Sentences
        "! ",       # Exclamations
        "? ",       # Questions
        "; ",       # Semicolons
        ", ",       # Commas
        " ",        # Words
        "",         # Characters
    ]

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()

    def chunk(self, document: Document) -> List[DocumentChunk]:
        """Recursively split document"""
        chunks = self._split_text(document.content, self.SEPARATORS)

        # Create DocumentChunk objects
        result = []
        char_pos = 0

        for i, chunk_text in enumerate(chunks):
            if len(chunk_text) >= self.config.min_chunk_size:
                result.append(DocumentChunk(
                    content=chunk_text,
                    chunk_index=i,
                    start_char=char_pos,
                    end_char=char_pos + len(chunk_text),
                    metadata=document.metadata,
                ))
            char_pos += len(chunk_text)

        logger.info(f"Created {len(result)} recursive chunks from {document.metadata.filename}")
        return result

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using separators"""
        if not text:
            return []

        # If text is small enough, return as is
        if len(text) <= self.config.chunk_size:
            return [text.strip()] if text.strip() else []

        # Try each separator
        for sep in separators:
            if sep == "":
                # Last resort: split by characters
                return self._split_by_chars(text)

            if sep in text:
                splits = text.split(sep)

                # Merge small splits
                merged = self._merge_splits(splits, sep)

                # Recursively process any chunks that are still too large
                result = []
                remaining_separators = separators[separators.index(sep) + 1:]

                for chunk in merged:
                    if len(chunk) > self.config.max_chunk_size:
                        result.extend(self._split_text(chunk, remaining_separators))
                    elif chunk.strip():
                        result.append(chunk.strip())

                return result

        # No separator found, split by characters
        return self._split_by_chars(text)

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge small splits back together"""
        result = []
        current = []
        current_size = 0

        for split in splits:
            split = split.strip()
            if not split:
                continue

            split_size = len(split)

            # If adding this split exceeds target, save current
            if current_size + split_size > self.config.chunk_size and current:
                result.append(separator.join(current))

                # Keep some overlap
                overlap_items = []
                overlap_size = 0
                for item in reversed(current):
                    if overlap_size + len(item) < self.config.chunk_overlap:
                        overlap_items.insert(0, item)
                        overlap_size += len(item)
                    else:
                        break

                current = overlap_items
                current_size = overlap_size

            current.append(split)
            current_size += split_size

        if current:
            result.append(separator.join(current))

        return result

    def _split_by_chars(self, text: str) -> List[str]:
        """Split text by character count (last resort)"""
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + self.config.chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - self.config.chunk_overlap

            if start >= len(text):
                break

        return chunks


# =============================================================================
# Chunker Factory
# =============================================================================

def get_chunker(
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
    config: Optional[ChunkingConfig] = None,
) -> BaseChunker:
    """
    Get a chunker for the specified strategy.

    Args:
        strategy: The chunking strategy to use
        config: Optional chunking configuration

    Returns:
        Appropriate chunker instance
    """
    chunkers = {
        ChunkingStrategy.FIXED_SIZE: FixedSizeChunker,
        ChunkingStrategy.SENTENCE: SentenceChunker,
        ChunkingStrategy.PARAGRAPH: ParagraphChunker,
        ChunkingStrategy.RECURSIVE: RecursiveChunker,
    }

    chunker_class = chunkers.get(strategy, RecursiveChunker)
    return chunker_class(config)


def chunk_document(
    document: Document,
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
    config: Optional[ChunkingConfig] = None,
) -> List[DocumentChunk]:
    """
    Convenience function to chunk a document.

    Args:
        document: The document to chunk
        strategy: Chunking strategy to use
        config: Optional configuration

    Returns:
        List of DocumentChunk objects
    """
    chunker = get_chunker(strategy, config)
    return chunker.chunk(document)
