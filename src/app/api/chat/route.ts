// DreAgent Cloud - Streaming Chat API
// Copyright (c) 2026 B&D Servicing LLC - All Rights Reserved
// Powered by CoDre-X™

import { streamText } from 'ai';
import { openai } from '@ai-sdk/openai';
import { NextRequest } from 'next/server';
import { getSystemPrompt, getModeConfig, type AgentMode } from '@/lib/ai/prompts';
import { getRelevantContext } from '@/lib/rag/query';

export const runtime = 'nodejs'; // Use Node.js for full OpenAI SDK support
export const maxDuration = 60;   // Vercel Pro limit

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const {
      messages,
      mode = 'general',
      userId,
      enableRag = true,
    } = body as {
      messages: { role: 'user' | 'assistant'; content: string }[];
      mode?: AgentMode;
      userId?: string;
      enableRag?: boolean;
    };

    if (!messages || messages.length === 0) {
      return new Response(
        JSON.stringify({ error: 'No messages provided' }),
        { status: 400, headers: { 'Content-Type': 'application/json' } }
      );
    }

    // Get mode configuration
    const modeConfig = getModeConfig(mode);

    // Get RAG context if enabled
    let ragContext = '';
    if (enableRag) {
      const lastUserMessage = messages.filter(m => m.role === 'user').pop();
      if (lastUserMessage) {
        ragContext = await getRelevantContext(lastUserMessage.content, userId);
      }
    }

    // Build system prompt with RAG context
    const systemPrompt = getSystemPrompt(mode, ragContext);

    // Stream the response using Vercel AI SDK
    const result = await streamText({
      model: openai(modeConfig.model),
      system: systemPrompt,
      messages,
      temperature: modeConfig.temperature,
      maxTokens: 4096,
    });

    // Return streaming response
    return result.toDataStreamResponse();

  } catch (error) {
    console.error('Chat API error:', error);
    return new Response(
      JSON.stringify({ 
        error: error instanceof Error ? error.message : 'Internal server error' 
      }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    );
  }
}
