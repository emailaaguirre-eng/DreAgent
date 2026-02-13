"""
=============================================================================
HUMMINGBIRD-LEA - RAG Agent Mixin
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
Mixin class that adds RAG capabilities to agents.

This mixin enables agents to:
- Automatically search the knowledge base for relevant context
- Include retrieved knowledge in their responses
- Cite sources from the knowledge base
=============================================================================
"""

import logging
from typing import Optional, List, Dict, Any

from .engine import RAGEngine, RAGContext, get_rag_engine

logger = logging.getLogger(__name__)


class RAGMixin:
    """
    Mixin that adds RAG capabilities to agents.

    Usage:
        class LeaAgent(BaseAgent, RAGMixin):
            def __init__(self):
                super().__init__()
                self.init_rag()

            async def process(self, message, context=None):
                # Get RAG context
                rag_context = await self.get_rag_context(message)

                # Include in processing
                enhanced_context = self.enhance_context_with_rag(context, rag_context)
                return await super().process(message, enhanced_context)
    """

    # Configuration
    rag_enabled: bool = True
    rag_auto_search: bool = True        # Auto-search for every query
    rag_min_confidence: float = 0.3     # Minimum confidence to include RAG
    rag_max_chunks: int = 3             # Maximum chunks to include

    def init_rag(self, engine: Optional[RAGEngine] = None):
        """Initialize RAG capabilities"""
        self._rag_engine = engine or get_rag_engine()
        self._last_rag_context: Optional[RAGContext] = None
        logger.info(f"RAG initialized for agent: {getattr(self, 'name', 'unknown')}")

    @property
    def rag_engine(self) -> RAGEngine:
        """Get the RAG engine"""
        if not hasattr(self, '_rag_engine') or self._rag_engine is None:
            self._rag_engine = get_rag_engine()
        return self._rag_engine

    async def get_rag_context(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> Optional[RAGContext]:
        """
        Get RAG context for a query.

        Args:
            query: The user's question
            top_k: Number of chunks to retrieve

        Returns:
            RAGContext or None if no relevant context found
        """
        if not self.rag_enabled:
            return None

        try:
            top_k = top_k or self.rag_max_chunks
            context = await self.rag_engine.get_context(query, top_k=top_k)

            # Check if we have useful context
            if not context.context_text:
                logger.debug(f"No RAG context found for query: {query[:50]}...")
                return None

            # Check minimum relevance
            if context.relevance_scores:
                avg_score = sum(context.relevance_scores) / len(context.relevance_scores)
                if avg_score < self.rag_min_confidence:
                    logger.debug(
                        f"RAG context below confidence threshold: {avg_score:.2f} < {self.rag_min_confidence}"
                    )
                    return None

            self._last_rag_context = context
            logger.info(
                f"RAG context retrieved: {context.chunks_used} chunks, "
                f"sources: {context.sources}"
            )
            return context

        except Exception as e:
            logger.error(f"Error getting RAG context: {e}")
            return None

    async def search_knowledge(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search the knowledge base directly.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of search results
        """
        try:
            results = await self.rag_engine.search(query, top_k=top_k)
            return [
                {
                    "content": r.content,
                    "score": r.score,
                    "source": r.source,
                    "filename": r.filename,
                }
                for r in results.results
            ]
        except Exception as e:
            logger.error(f"Knowledge search error: {e}")
            return []

    def format_rag_context_for_prompt(
        self,
        rag_context: RAGContext,
    ) -> str:
        """
        Format RAG context for inclusion in the system prompt.

        Args:
            rag_context: The retrieved context

        Returns:
            Formatted string for prompt
        """
        if not rag_context or not rag_context.context_text:
            return ""

        return rag_context.format_for_prompt()

    def get_last_rag_sources(self) -> List[str]:
        """Get sources from the last RAG query"""
        if self._last_rag_context:
            return self._last_rag_context.sources
        return []

    def get_rag_stats(self) -> Dict[str, Any]:
        """Get RAG statistics"""
        return self.rag_engine.get_stats()


class RAGEnhancedPromptBuilder:
    """
    Helper to build RAG-enhanced prompts.

    Usage:
        builder = RAGEnhancedPromptBuilder()
        enhanced_prompt = builder.build(
            base_prompt="You are a helpful assistant",
            rag_context=context,
            user_query=query,
        )
    """

    CONTEXT_TEMPLATE = """
## Knowledge Base Context

The following information was retrieved from the knowledge base and may help answer the user's question:

{context}

---
Sources consulted: {sources}

Guidelines for using this context:
- Use this information to enhance your response when relevant
- Always prioritize accuracy - if the context doesn't fully answer the question, say so
- Cite the source when directly using information from the context
- If the context contradicts your knowledge, mention both perspectives
"""

    def build(
        self,
        base_prompt: str,
        rag_context: Optional[RAGContext],
        user_query: str,
    ) -> str:
        """
        Build an enhanced prompt with RAG context.

        Args:
            base_prompt: The base system prompt
            rag_context: Retrieved context (can be None)
            user_query: The user's question

        Returns:
            Enhanced prompt string
        """
        if not rag_context or not rag_context.context_text:
            return base_prompt

        context_section = self.CONTEXT_TEMPLATE.format(
            context=rag_context.context_text,
            sources=", ".join(rag_context.sources) if rag_context.sources else "internal knowledge base",
        )

        return f"{base_prompt}\n\n{context_section}"


# =============================================================================
# Helper Functions
# =============================================================================

def create_rag_system_prompt_addition(rag_context: RAGContext) -> str:
    """
    Create a system prompt addition with RAG context.

    Args:
        rag_context: The retrieved context

    Returns:
        Formatted prompt addition
    """
    if not rag_context or not rag_context.context_text:
        return ""

    return rag_context.format_for_prompt()


async def should_use_rag(query: str) -> bool:
    """
    Determine if a query would benefit from RAG.

    Simple heuristic based on query characteristics.

    Args:
        query: The user's query

    Returns:
        True if RAG should be used
    """
    # Short greetings don't need RAG
    if len(query.split()) < 3:
        return False

    # Questions typically benefit from RAG
    question_indicators = ["what", "how", "why", "when", "where", "who", "which", "?"]
    query_lower = query.lower()

    if any(indicator in query_lower for indicator in question_indicators):
        return True

    # Information-seeking keywords
    info_keywords = [
        "tell me", "explain", "describe", "information",
        "about", "regarding", "concerning", "details",
        "find", "search", "look up",
    ]

    if any(keyword in query_lower for keyword in info_keywords):
        return True

    return False
