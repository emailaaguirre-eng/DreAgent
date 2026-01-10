// DreAgent Cloud - Conversations API
// Copyright (c) 2026 B&D Servicing LLC - All Rights Reserved
// Powered by CoDre-X™

import { NextRequest, NextResponse } from 'next/server';
import {
  saveConversation,
  getConversation,
  getUserConversations,
  deleteConversation,
  type Message,
} from '@/lib/db/supabase';

export const runtime = 'nodejs';

// GET - Fetch conversation(s)
export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const conversationId = searchParams.get('id');
    const userId = req.headers.get('x-user-id') || searchParams.get('userId');

    if (conversationId) {
      const conversation = await getConversation(conversationId);
      return NextResponse.json({ conversation });
    }

    if (userId) {
      const limit = parseInt(searchParams.get('limit') || '50');
      const conversations = await getUserConversations(userId, limit);
      return NextResponse.json({ conversations });
    }

    return NextResponse.json(
      { error: 'userId or id required' },
      { status: 400 }
    );
  } catch (error) {
    console.error('Conversations GET error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to fetch' },
      { status: 500 }
    );
  }
}

// POST - Save/update conversation
export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const {
      conversationId,
      userId,
      title,
      messages,
      mode,
    } = body as {
      conversationId: string;
      userId: string;
      title: string;
      messages: Message[];
      mode: string;
    };

    if (!userId || !conversationId) {
      return NextResponse.json(
        { error: 'userId and conversationId required' },
        { status: 400 }
      );
    }

    const conversation = await saveConversation(
      userId,
      conversationId,
      title || 'New Conversation',
      messages || [],
      mode || 'general'
    );

    return NextResponse.json({ conversation, status: 'saved' });
  } catch (error) {
    console.error('Conversations POST error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to save' },
      { status: 500 }
    );
  }
}

// DELETE - Remove conversation
export async function DELETE(req: NextRequest) {
  try {
    const body = await req.json();
    const { conversationId } = body as { conversationId: string };

    if (!conversationId) {
      return NextResponse.json(
        { error: 'conversationId required' },
        { status: 400 }
      );
    }

    await deleteConversation(conversationId);
    return NextResponse.json({ status: 'deleted' });
  } catch (error) {
    console.error('Conversations DELETE error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to delete' },
      { status: 500 }
    );
  }
}
