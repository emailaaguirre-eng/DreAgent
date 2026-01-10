// DreAgent Cloud - RAG Types
// Copyright (c) 2026 B&D Servicing LLC - All Rights Reserved
// Powered by CoDre-X™

export interface DocumentChunk {
  id?: string;
  title: string;
  content: string;
  chunkIndex: number;
  metadata: DocumentMetadata;
}

export interface DocumentMetadata {
  source?: string;
  fileType?: string;
  pageNumber?: number;
  totalPages?: number;
  createdAt?: string;
  [key: string]: unknown;
}

export interface SearchResult {
  id: string;
  title: string;
  content: string;
  metadata: DocumentMetadata;
  similarity: number;
}

export interface IngestResult {
  success: boolean;
  documentId?: string;
  chunksCreated: number;
  error?: string;
}
