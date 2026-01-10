// DreAgent Cloud - Model Configuration
// Copyright (c) 2026 B&D Servicing LLC - All Rights Reserved
// Powered by CoDre-X™

import { openai } from '@ai-sdk/openai';

export const models = {
  // Fast responses - cost optimized
  fast: openai('gpt-4o-mini'),
  
  // Balanced - good for most tasks
  balanced: openai('gpt-4o'),
  
  // Deep reasoning - complex analysis
  deep: openai('gpt-4-turbo'),
  
  // Embeddings for RAG
  embedding: 'text-embedding-3-small',
} as const;

export type ModelTier = keyof typeof models;

export const EMBEDDING_DIMENSIONS = 1536;
