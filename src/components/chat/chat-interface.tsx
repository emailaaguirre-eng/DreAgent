// DreAgent Cloud - Chat Interface
// Copyright (c) 2026 B&D Servicing LLC - All Rights Reserved
// Powered by CoDre-X™

'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { useChat } from 'ai/react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Sparkles, RotateCcw, FileDown } from 'lucide-react';
import { ModeSelector } from './mode-selector';
import { MessageBubble, TypingIndicator } from './message-bubble';
import { VoiceInput } from './voice-input';
import { type AgentMode } from '@/lib/ai/prompts';
import { cn } from '@/lib/utils';

export function ChatInterface() {
  const [mode, setMode] = useState<AgentMode>('general');
  const [userId, setUserId] = useState<string>('');
  const [isDownloadingReport, setIsDownloadingReport] = useState(false);
  const [downloadStatus, setDownloadStatus] = useState<string>('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const formRef = useRef<HTMLFormElement>(null);

  useEffect(() => {
    const storageKey = 'dreagent_user_id';
    const existing = window.localStorage.getItem(storageKey);
    if (existing) {
      setUserId(existing);
      return;
    }

    const newUserId = `user-${crypto.randomUUID()}`;
    window.localStorage.setItem(storageKey, newUserId);
    setUserId(newUserId);
  }, []);

  const {
    messages,
    input,
    setInput,
    handleInputChange,
    handleSubmit,
    isLoading,
    reload,
  } = useChat({
    api: '/api/chat',
    body: { mode, enableRag: true, userId },
    headers: userId ? { 'x-user-id': userId } : undefined,
  });

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Auto-resize textarea
  const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    handleInputChange(e);
    e.target.style.height = 'auto';
    e.target.style.height = `${Math.min(e.target.scrollHeight, 200)}px`;
  };

  // Handle keyboard shortcuts
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (input.trim() && !isLoading) {
        formRef.current?.requestSubmit();
      }
    }
  };

  // Voice input callback
  const handleVoiceTranscript = useCallback((text: string) => {
    setInput(text);
    inputRef.current?.focus();
  }, [setInput]);

  const downloadExecutiveReport = useCallback(async (includeCalendar: boolean) => {
    if (!userId) {
      setDownloadStatus('User session not ready yet. Try again in a moment.');
      return;
    }

    try {
      setIsDownloadingReport(true);
      setDownloadStatus('');

      const params = new URLSearchParams({
        userId,
        folder: 'inbox',
        limit: '200',
        include_calendar: String(includeCalendar),
        days_behind: '30',
        days_ahead: '30',
      });
      const url = `/api/outlook/email-history?${params.toString()}`;
      const response = await fetch(url);

      if (!response.ok) {
        const errorPayload = await response.json().catch(() => null);
        throw new Error(
          errorPayload?.error || 'Unable to generate report file right now.'
        );
      }

      const blob = await response.blob();
      const downloadUrl = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = downloadUrl;

      const fallbackName = includeCalendar
        ? `outlook-history-${new Date().toISOString().slice(0, 10)}.csv`
        : `email-history-${new Date().toISOString().slice(0, 10)}.csv`;
      const disposition = response.headers.get('content-disposition');
      const filenameMatch = disposition?.match(/filename=\"([^\"]+)\"/i);
      a.download = filenameMatch?.[1] || fallbackName;

      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(downloadUrl);

      setDownloadStatus('Report downloaded successfully.');
    } catch (error) {
      setDownloadStatus(
        error instanceof Error ? error.message : 'Report download failed.'
      );
    } finally {
      setIsDownloadingReport(false);
    }
  }, [userId]);

  return (
    <div className="flex flex-col h-screen max-h-screen">
      {/* Header */}
      <header className="flex-shrink-0 px-4 py-4 border-b border-white/5">
        <div className="max-w-4xl mx-auto">
          {/* Logo and title */}
          <div className="flex items-center justify-center gap-3 mb-4">
            <motion.div
              className="w-10 h-10 rounded-xl bg-gradient-to-br from-brand-400 to-brand-600 flex items-center justify-center"
              animate={{ rotate: [0, 5, -5, 0] }}
              transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
            >
              <Sparkles className="w-5 h-5 text-white" />
            </motion.div>
            <div>
              <h1 className="text-xl font-semibold text-text-primary">
                DreAgent
              </h1>
              <p className="text-xs text-text-muted">
                Powered by CoDre-X™
              </p>
            </div>
          </div>

          {/* Mode selector */}
          <ModeSelector currentMode={mode} onModeChange={setMode} />
        </div>
      </header>

      {/* Messages area */}
      <main className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-4xl mx-auto space-y-4">
          {messages.length === 0 ? (
            <EmptyState mode={mode} />
          ) : (
            <AnimatePresence mode="popLayout">
              {messages.map((message) => (
                <MessageBubble
                  key={message.id}
                  role={message.role as 'user' | 'assistant'}
                  content={message.content}
                  isStreaming={isLoading && message.id === messages[messages.length - 1]?.id}
                />
              ))}
            </AnimatePresence>
          )}

          {isLoading && messages[messages.length - 1]?.role === 'user' && (
            <TypingIndicator />
          )}

          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* Input area */}
      <footer className="flex-shrink-0 px-4 py-4 border-t border-white/5 bg-surface-900/50 backdrop-blur">
        <form
          ref={formRef}
          onSubmit={handleSubmit}
          className="max-w-4xl mx-auto"
        >
          <div className="flex items-end gap-3">
            {/* Voice input */}
            <VoiceInput
              onTranscript={handleVoiceTranscript}
              disabled={isLoading}
            />

            {/* Text input */}
            <div className="flex-1 relative">
              <textarea
                ref={inputRef}
                value={input}
                onChange={handleTextareaChange}
                onKeyDown={handleKeyDown}
                placeholder="Ask me anything..."
                rows={1}
                disabled={isLoading}
                className={cn(
                  'w-full px-4 py-3 pr-12 rounded-2xl resize-none',
                  'bg-surface-700/50 border border-white/10',
                  'text-text-primary placeholder:text-text-muted',
                  'focus:outline-none focus:ring-2 focus:ring-brand-500/50 focus:border-brand-500',
                  'disabled:opacity-50 disabled:cursor-not-allowed',
                  'transition-all duration-200'
                )}
                style={{ maxHeight: '200px' }}
              />
            </div>

            {/* Send button */}
            <motion.button
              type="submit"
              disabled={!input.trim() || isLoading}
              className={cn(
                'p-3 rounded-full transition-all duration-200',
                'bg-brand-500 text-white',
                'hover:bg-brand-600',
                'disabled:opacity-50 disabled:cursor-not-allowed',
                'glow-brand-soft'
              )}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Send className="w-5 h-5" />
            </motion.button>

            {/* Reload button (when there are messages) */}
            {messages.length > 0 && (
              <motion.button
                type="button"
                onClick={() => reload()}
                disabled={isLoading}
                className={cn(
                  'p-3 rounded-full transition-all duration-200',
                  'bg-surface-700 text-text-secondary',
                  'hover:bg-surface-700/80 hover:text-text-primary',
                  'disabled:opacity-50 disabled:cursor-not-allowed'
                )}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <RotateCcw className="w-5 h-5" />
              </motion.button>
            )}
          </div>

          <p className="text-center text-xs text-text-muted mt-2">
            Press Enter to send • Shift+Enter for new line
          </p>

          {mode === 'executive' && (
            <div className="mt-3 flex flex-wrap gap-2 justify-center">
              <button
                type="button"
                onClick={() => downloadExecutiveReport(false)}
                disabled={isDownloadingReport}
                className={cn(
                  'px-3 py-2 rounded-lg text-xs font-medium',
                  'bg-surface-700/70 text-text-secondary border border-white/10',
                  'hover:bg-surface-700 hover:text-text-primary',
                  'disabled:opacity-50 disabled:cursor-not-allowed',
                  'inline-flex items-center gap-1.5 transition-colors duration-200'
                )}
              >
                <FileDown className="w-3.5 h-3.5" />
                Download Email CSV
              </button>

              <button
                type="button"
                onClick={() => downloadExecutiveReport(true)}
                disabled={isDownloadingReport}
                className={cn(
                  'px-3 py-2 rounded-lg text-xs font-medium',
                  'bg-surface-700/70 text-text-secondary border border-white/10',
                  'hover:bg-surface-700 hover:text-text-primary',
                  'disabled:opacity-50 disabled:cursor-not-allowed',
                  'inline-flex items-center gap-1.5 transition-colors duration-200'
                )}
              >
                <FileDown className="w-3.5 h-3.5" />
                Download Email + Calendar CSV
              </button>
            </div>
          )}

          {downloadStatus && (
            <p className="text-center text-xs text-text-muted mt-2">
              {downloadStatus}
            </p>
          )}
        </form>
      </footer>
    </div>
  );
}

function EmptyState({ mode }: { mode: AgentMode }) {
  const assistantNameByMode: Record<AgentMode, string> = {
    general: 'Grant',
    'it-support': 'Chiquis',
    executive: 'Lea',
    legal: 'Lea',
    finance: 'Lea',
    research: 'Lea',
    incentives: 'Lea',
  };

  const suggestions = {
    general: [
      'What can you help me with?',
      'Summarize my last 5 emails',
      'What meetings do I have today?',
    ],
    'it-support': [
      'Debug this Python error...',
      'How do I set up a Vercel deployment?',
      'Review my code for security issues',
    ],
    executive: [
      'Draft a follow-up email for the client meeting',
      'Prepare talking points for tomorrow\'s presentation',
      'Organize my calendar for next week',
    ],
    legal: [
      'Explain this contract clause',
      'What are the key terms in an NDA?',
      'Draft a cease and desist template',
    ],
    finance: [
      'Explain the tax implications of...',
      'Help me analyze this balance sheet',
      'What deductions can I claim for my home office?',
    ],
    research: [
      'Explain quantum computing in simple terms',
      'What are the latest trends in AI?',
      'Compare React vs Vue for my project',
    ],
    incentives: [
      'What are the requirements for the R&D tax credit?',
      'Help me complete this incentive application',
      'What documentation do I need for WOTC?',
    ],
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex flex-col items-center justify-center min-h-[50vh] text-center px-4"
    >
      <motion.div
        className="w-20 h-20 rounded-2xl bg-gradient-to-br from-brand-400 to-brand-600 flex items-center justify-center mb-6"
        animate={{ 
          rotate: [0, 5, -5, 0],
          scale: [1, 1.02, 1],
        }}
        transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
      >
        <Sparkles className="w-10 h-10 text-white" />
      </motion.div>

      <h2 className="text-2xl font-semibold text-text-primary mb-2">
        How can I help you today?
      </h2>
      <p className="text-text-secondary mb-8 max-w-md">
        I&apos;m {assistantNameByMode[mode]}, your AI assistant. Ask me anything or try one of these suggestions:
      </p>

      <div className="flex flex-wrap gap-2 justify-center max-w-lg">
        {suggestions[mode].map((suggestion, i) => (
          <motion.button
            key={i}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1 }}
            className={cn(
              'px-4 py-2 rounded-full text-sm',
              'bg-surface-700/50 text-text-secondary',
              'hover:bg-surface-700 hover:text-text-primary',
              'border border-white/5 transition-all duration-200'
            )}
          >
            {suggestion}
          </motion.button>
        ))}
      </div>
    </motion.div>
  );
}
