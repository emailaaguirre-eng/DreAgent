// DreAgent Cloud - RAG Query
// Copyright (c) 2026 B&D Servicing LLC - All Rights Reserved
// Powered by CoDre-X™

import { supabase } from '@/lib/db/supabase';
import { generateEmbedding } from './embeddings';
import type { SearchResult } from './types';

/**
 * Search knowledge base using semantic similarity
 */
export async function searchKnowledge(
  query: string,
  userId?: string,
  options: {
    threshold?: number;
    limit?: number;
  } = {}
): Promise<SearchResult[]> {
  const { threshold = 0.7, limit = 5 } = options;

  // Generate embedding for query
  const queryEmbedding = await generateEmbedding(query);

  // Search using pgvector
  const { data, error } = await supabase.rpc('search_knowledge', {
    query_embedding: queryEmbedding,
    match_threshold: threshold,
    match_count: limit,
    filter_user_id: userId || null,
  });

  if (error) {
    console.error('Knowledge search error:', error);
    throw error;
  }

  return (data || []).map((row: {
    id: string;
    title: string;
    content: string;
    metadata: Record<string, unknown>;
    similarity: number;
  }) => ({
    id: row.id,
    title: row.title,
    content: row.content,
    metadata: row.metadata,
    similarity: row.similarity,
  }));
}

/**
 * Build RAG context from search results
 */
export function buildRagContext(results: SearchResult[]): string {
  if (results.length === 0) return '';

  const contextParts = results.map((r, i) => {
    const source = r.metadata.source || r.title;
    return `### Source ${i + 1}: ${source}\n${r.content}`;
  });

  return contextParts.join('\n\n---\n\n');
}

/**
 * Get relevant context for a user query
 */
export async function getRelevantContext(
  query: string,
  userId?: string
): Promise<string> {
  // Skip RAG if Supabase isn't configured
  if (!process.env.NEXT_PUBLIC_SUPABASE_URL || !process.env.SUPABASE_SERVICE_ROLE_KEY) {
    return '';
  }

  try {
    const results = await searchKnowledge(query, userId, {
      threshold: 0.72,
      limit: 3,
    });

    return buildRagContext(results);
  } catch (error) {
    console.error('Error getting RAG context:', error);
    return '';
  }
}
