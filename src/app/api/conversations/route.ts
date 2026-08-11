// DreAgent Cloud - Conversations API
// Copyright (c) 2026 B&D Servicing LLC - All Rights Reserved
// Powered by CoDre-X™

import { NextRequest, NextResponse } from 'next/server';
import {
  ownerAuthErrorResponse,
  requireTrustedOwner,
} from '@/lib/auth/owner-session';
import {
  saveConversation,
  getConversation,
  getUserConversations,
  deleteConversation,
  type Message,
} from '@/lib/db/supabase';

export const runtime = 'nodejs';

// GET - Fetch conversation(s) for trusted owner only
export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const conversationId = searchParams.get('id');
    const clientUserId =
      req.headers.get('x-user-id') || searchParams.get('userId');

    const trusted = requireTrustedOwner(req, {
      untrustedClientUserId: clientUserId,
    });
    if (!trusted.ok) {
      return ownerAuthErrorResponse(trusted);
    }

    if (conversationId) {
      const conversation = await getConversation(conversationId);
      if (!conversation || conversation.user_id !== trusted.ownerId) {
        return NextResponse.json(
          { error: 'Conversation not found for owner' },
          { status: 404 }
        );
      }
      return NextResponse.json({
        conversation,
        ownerId: trusted.ownerId,
        identitySource: trusted.source,
      });
    }

    const limit = parseInt(searchParams.get('limit') || '50');
    const conversations = await getUserConversations(trusted.ownerId, limit);
    return NextResponse.json({
      conversations,
      ownerId: trusted.ownerId,
      identitySource: trusted.source,
    });
  } catch (error) {
    console.error('Conversations GET error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to fetch' },
      { status: 500 }
    );
  }
}

// POST - Save/update conversation (bound to trusted owner)
export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const {
      conversationId,
      userId: clientUserId,
      title,
      messages,
      mode,
    } = body as {
      conversationId: string;
      userId?: string;
      title: string;
      messages: Message[];
      mode: string;
    };

    const trusted = requireTrustedOwner(req, {
      untrustedClientUserId: clientUserId,
    });
    if (!trusted.ok) {
      return ownerAuthErrorResponse(trusted);
    }

    if (!conversationId) {
      return NextResponse.json(
        { error: 'conversationId required' },
        { status: 400 }
      );
    }

    const conversation = await saveConversation(
      trusted.ownerId,
      conversationId,
      title || 'New Conversation',
      messages || [],
      mode || 'general'
    );

    return NextResponse.json({
      conversation,
      status: 'saved',
      ownerId: trusted.ownerId,
      identitySource: trusted.source,
    });
  } catch (error) {
    console.error('Conversations POST error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to save' },
      { status: 500 }
    );
  }
}

// DELETE - Remove conversation (owner session + ownership check)
export async function DELETE(req: NextRequest) {
  try {
    const trusted = requireTrustedOwner(req);
    if (!trusted.ok) {
      return ownerAuthErrorResponse(trusted);
    }

    const body = await req.json();
    const { conversationId } = body as { conversationId: string };

    if (!conversationId) {
      return NextResponse.json(
        { error: 'conversationId required' },
        { status: 400 }
      );
    }

    const existing = await getConversation(conversationId).catch(() => null);
    if (!existing || existing.user_id !== trusted.ownerId) {
      return NextResponse.json(
        { error: 'Conversation not found for owner' },
        { status: 404 }
      );
    }

    await deleteConversation(conversationId);
    return NextResponse.json({
      status: 'deleted',
      ownerId: trusted.ownerId,
      identitySource: trusted.source,
    });
  } catch (error) {
    console.error('Conversations DELETE error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to delete' },
      { status: 500 }
    );
  }
}
