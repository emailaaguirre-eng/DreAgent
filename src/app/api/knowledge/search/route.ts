// DreAgent Cloud - Knowledge Search API
// Copyright (c) 2026 B&D Servicing LLC - All Rights Reserved
// Powered by CoDre-X™

import { NextRequest, NextResponse } from 'next/server';
import { searchKnowledge } from '@/lib/rag/query';

export const runtime = 'nodejs';

// POST - Semantic search
export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const {
      query,
      userId,
      threshold,
      limit,
    } = body as {
      query: string;
      userId?: string;
      threshold?: number;
      limit?: number;
    };

    if (!query) {
      return NextResponse.json(
        { error: 'query required' },
        { status: 400 }
      );
    }

    const results = await searchKnowledge(query, userId, {
      threshold: threshold || 0.7,
      limit: limit || 5,
    });

    return NextResponse.json({
      query,
      results,
      count: results.length,
    });
  } catch (error) {
    console.error('Knowledge search error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Search failed' },
      { status: 500 }
    );
  }
}
