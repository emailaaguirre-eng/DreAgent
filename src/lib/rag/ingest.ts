// DreAgent Cloud - Document Ingestion
// Copyright (c) 2026 B&D Servicing LLC - All Rights Reserved
// Powered by CoDre-X™

import { supabase } from '@/lib/db/supabase';
import { generateEmbeddings } from './embeddings';
import type { DocumentChunk, DocumentMetadata, IngestResult } from './types';

const CHUNK_SIZE = 1000;       // Characters per chunk
const CHUNK_OVERLAP = 200;     // Overlap between chunks

/**
 * Split text into overlapping chunks
 */
export function chunkText(
  text: string,
  chunkSize = CHUNK_SIZE,
  overlap = CHUNK_OVERLAP
): string[] {
  const chunks: string[] = [];
  let start = 0;

  while (start < text.length) {
    // Find a good break point (end of sentence or paragraph)
    let end = start + chunkSize;
    
    if (end < text.length) {
      // Look for sentence end
      const sentenceEnd = text.lastIndexOf('.', end);
      const paragraphEnd = text.lastIndexOf('\n\n', end);
      
      if (paragraphEnd > start + chunkSize / 2) {
        end = paragraphEnd;
      } else if (sentenceEnd > start + chunkSize / 2) {
        end = sentenceEnd + 1;
      }
    }

    chunks.push(text.slice(start, end).trim());
    start = end - overlap;
  }

  return chunks.filter(c => c.length > 50); // Filter tiny chunks
}

/**
 * Ingest a document into the knowledge base
 */
export async function ingestDocument(
  userId: string,
  title: string,
  content: string,
  metadata: DocumentMetadata = {}
): Promise<IngestResult> {
  try {
    // Split into chunks
    const textChunks = chunkText(content);
    
    if (textChunks.length === 0) {
      return { success: false, chunksCreated: 0, error: 'No valid content to ingest' };
    }

    // Generate embeddings for all chunks
    const embeddings = await generateEmbeddings(textChunks);

    // Prepare documents for insertion
    const documents = textChunks.map((chunk, index) => ({
      user_id: userId,
      title: `${title} (Part ${index + 1})`,
      content: chunk,
      chunk_index: index,
      embedding: embeddings[index],
      metadata: {
        ...metadata,
        originalTitle: title,
        totalChunks: textChunks.length,
      },
    }));

    // Insert into Supabase
    const { data, error } = await supabase
      .from('knowledge_documents')
      .insert(documents)
      .select('id');

    if (error) {
      console.error('Ingest error:', error);
      return { success: false, chunksCreated: 0, error: error.message };
    }

    return {
      success: true,
      documentId: data?.[0]?.id,
      chunksCreated: documents.length,
    };
  } catch (error) {
    console.error('Document ingestion failed:', error);
    return {
      success: false,
      chunksCreated: 0,
      error: error instanceof Error ? error.message : 'Unknown error',
    };
  }
}

/**
 * Delete a document and all its chunks
 */
export async function deleteDocument(documentId: string): Promise<boolean> {
  const { error } = await supabase
    .from('knowledge_documents')
    .delete()
    .eq('id', documentId);

  return !error;
}

/**
 * List documents for a user
 */
export async function listDocuments(userId: string) {
  const { data, error } = await supabase
    .from('knowledge_documents')
    .select('id, title, metadata, created_at')
    .eq('user_id', userId)
    .eq('chunk_index', 0) // Only get first chunk (represents document)
    .order('created_at', { ascending: false });

  if (error) throw error;
  return data;
}
