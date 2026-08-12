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
import { normalizeToVisibleMode, type AgentMode } from '@/lib/ai/prompts';
import { cn } from '@/lib/utils';

const MODE_STORAGE_KEY = 'dreagent_last_mode';
/** Legacy UI preference only — NEVER treated as trusted owner identity by the server. */
const CLIENT_UI_ID_PREF_KEY = 'dreagent_user_id';

const MODE_CAPABILITIES: Partial<Record<AgentMode, string[]>> = {
  general: [
    'Incentives & economic development',
    'Calm triage',
    'Switch to Lea for day-to-day',
  ],
  'it-support': ['Debugging', 'Code review', 'Cloud & sysadmin'],
  executive: [
    'Plan & prioritize',
    'Drafts & follow-ups',
    'Project framing',
    'Briefings',
    'Mail when connected',
  ],
};

export function ChatInterface() {
  const [mode, setMode] = useState<AgentMode>('executive');
  // Client-only UI preference id; not sent as trusted identity for sensitive APIs.
  const [, setClientUiId] = useState<string>('');
  const [isDownloadingReport, setIsDownloadingReport] = useState(false);
  const [downloadStatus, setDownloadStatus] = useState<string>('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const formRef = useRef<HTMLFormElement>(null);

  useEffect(() => {
    // Keep legacy localStorage id for UI/debug continuity only — not auth.
    const existingUser = window.localStorage.getItem(CLIENT_UI_ID_PREF_KEY);
    if (existingUser) {
      setClientUiId(existingUser);
    } else {
      const newUserId = `ui-${crypto.randomUUID()}`;
      window.localStorage.setItem(CLIENT_UI_ID_PREF_KEY, newUserId);
      setClientUiId(newUserId);
    }

    const storedMode = window.localStorage.getItem(MODE_STORAGE_KEY);
    const next = normalizeToVisibleMode(storedMode);
    setMode(next);
    window.localStorage.setItem(MODE_STORAGE_KEY, next);
  }, []);

  const handleModeChange = useCallback((next: AgentMode) => {
    const resolved = normalizeToVisibleMode(next);
    setMode(resolved);
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(MODE_STORAGE_KEY, resolved);
    }
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
    // Do not send client userId as body/header identity; cookies handle owner session.
    body: { mode, enableRag: true },
    credentials: 'same-origin',
  });

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    handleInputChange(e);
    e.target.style.height = 'auto';
    e.target.style.height = `${Math.min(e.target.scrollHeight, 200)}px`;
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (input.trim() && !isLoading) {
        formRef.current?.requestSubmit();
      }
    }
  };

  const handleVoiceTranscript = useCallback(
    (text: string) => {
      setInput(text);
      inputRef.current?.focus();
    },
    [setInput]
  );

  const applySuggestion = useCallback(
    (text: string) => {
      setInput(text);
      inputRef.current?.focus();
    },
    [setInput]
  );

  const downloadExecutiveReport = useCallback(
    async (includeCalendar: boolean) => {
      try {
        setIsDownloadingReport(true);
        setDownloadStatus('');

        // Identity is server-side owner session (or Bearer), not client userId.
        const params = new URLSearchParams({
          folder: 'inbox',
          limit: '200',
          include_calendar: String(includeCalendar),
          days_behind: '30',
          days_ahead: '30',
        });
        const url = `/api/outlook/email-history?${params.toString()}`;
        const response = await fetch(url, { credentials: 'same-origin' });

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
    },
    []
  );

  return (
    <div className="flex flex-col h-screen max-h-screen">
      <header className="flex-shrink-0 px-4 py-4 border-b border-white/10">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center justify-center gap-3 mb-4">
            <motion.div
              className="w-10 h-10 rounded-xl bg-gradient-to-br from-brand-400 to-brand-600 flex items-center justify-center"
              animate={{ rotate: [0, 5, -5, 0] }}
              transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
            >
              <Sparkles className="w-5 h-5 text-white" />
            </motion.div>
            <div>
              <h1 className="text-xl font-semibold text-text-primary tracking-tight">
                DreAgent
              </h1>
              <p className="text-xs text-text-secondary">Powered by CoDre-X™</p>
            </div>
          </div>

          <ModeSelector currentMode={mode} onModeChange={handleModeChange} />

          <div className="mt-3 flex flex-wrap gap-2 justify-center">
            {(MODE_CAPABILITIES[mode] ?? MODE_CAPABILITIES.executive ?? []).map(
              (cap) => (
                <span
                  key={cap}
                  className="px-2.5 py-1 rounded-full text-[11px] font-medium bg-surface-700 text-text-secondary border border-white/15"
                >
                  {cap}
                </span>
              )
            )}
          </div>
        </div>
      </header>

      <main className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-4xl mx-auto space-y-4">
          {messages.length === 0 ? (
            <EmptyState mode={mode} onSuggest={applySuggestion} />
          ) : (
            <AnimatePresence mode="popLayout">
              {messages.map((message) => (
                <MessageBubble
                  key={message.id}
                  role={message.role as 'user' | 'assistant'}
                  content={message.content}
                  isStreaming={
                    isLoading &&
                    message.id === messages[messages.length - 1]?.id
                  }
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

      <footer className="flex-shrink-0 px-4 py-3 pb-8 border-t border-white/10 bg-surface-800/80 backdrop-blur-sm">
        <form ref={formRef} onSubmit={handleSubmit} className="max-w-4xl mx-auto">
          <div className="flex items-end gap-3">
            <VoiceInput
              onTranscript={handleVoiceTranscript}
              disabled={isLoading}
            />

            <div className="flex-1 relative">
              <textarea
                ref={inputRef}
                value={input}
                onChange={handleTextareaChange}
                onKeyDown={handleKeyDown}
                placeholder={
                  mode === 'executive'
                    ? 'Tell Lea what is on your plate — plans, drafts, or where to start…'
                    : 'Ask me anything...'
                }
                rows={1}
                disabled={isLoading}
                className={cn(
                  'w-full px-4 py-3 pr-12 rounded-2xl resize-none',
                  'bg-surface-700 border border-white/15',
                  'text-text-primary placeholder:text-text-muted',
                  'focus:outline-none focus:ring-2 focus:ring-brand-500/50 focus:border-brand-500',
                  'disabled:opacity-50 disabled:cursor-not-allowed',
                  'transition-all duration-200'
                )}
                style={{ maxHeight: '200px' }}
              />
            </div>

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

            {messages.length > 0 && (
              <motion.button
                type="button"
                onClick={() => reload()}
                disabled={isLoading}
                className={cn(
                  'p-3 rounded-full transition-all duration-200',
                  'bg-surface-700 text-text-secondary border border-white/15',
                  'hover:bg-surface-700/90 hover:text-text-primary',
                  'disabled:opacity-50 disabled:cursor-not-allowed'
                )}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <RotateCcw className="w-5 h-5" />
              </motion.button>
            )}
          </div>

          <p className="text-center text-xs text-text-secondary mt-2">
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
                  'bg-surface-700 text-text-secondary border border-white/15',
                  'hover:bg-surface-700/90 hover:text-text-primary hover:border-white/25',
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
                  'bg-surface-700 text-text-secondary border border-white/15',
                  'hover:bg-surface-700/90 hover:text-text-primary hover:border-white/25',
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
            <p className="text-center text-xs text-text-secondary mt-2">
              {downloadStatus}
            </p>
          )}
        </form>
      </footer>
    </div>
  );
}

function EmptyState({
  mode,
  onSuggest,
}: {
  mode: AgentMode;
  onSuggest: (text: string) => void;
}) {
  const introByMode: Partial<Record<AgentMode, string>> = {
    general:
      "I'm Grant, optional incentives and economic-development specialist. Switch to Lea for day-to-day executive help, or Chiquis for coding/IT.",
    'it-support':
      "I'm Chiquis, optional IT and coding specialist. Switch to Lea for planning, drafting, and mail/calendar support.",
    executive:
      "I'm Lea, your executive assistant. I help you organize what matters, draft the next message, and take one clear step at a time — with connected mail/calendar when available. I can help with planning, drafting, research, and organization even without a live inbox.",
  };

  const suggestions: Partial<Record<AgentMode, string[]>> = {
    general: [
      'What incentive programs should I review?',
      'Help me frame an economic development checklist',
      'When should I switch to Lea?',
    ],
    'it-support': [
      'Debug this Python error...',
      'How do I set up a Vercel deployment?',
      'Review my code for security issues',
    ],
    executive: [
      "I'm overwhelmed — help me sort Now / Next / Later",
      'Frame this as a project and give me one first move',
      'Draft a follow-up email to the client about next steps',
      'Build a short morning briefing outline for today',
      'Check my inbox for the last 7 days',
      'What meetings do I have this week?',
    ],
  };

  const modeSuggestions =
    suggestions[mode] ?? suggestions.executive ?? [];
  const intro =
    introByMode[mode] ?? introByMode.executive ?? 'How can I help?';
  const isLea = mode === 'executive';

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

      <h2 className="text-2xl font-semibold text-text-primary mb-2 tracking-tight">
        {isLea ? 'Good to see you — where should we start?' : 'How can I help you today?'}
      </h2>
      <p className="text-text-secondary mb-4 max-w-lg leading-relaxed">
        {intro}
      </p>
      {isLea && (
        <p className="text-text-muted text-sm mb-8 max-w-md leading-relaxed">
          Prefer a gentle start? Pick a suggestion, or just tell me what is on your mind.
        </p>
      )}
      {!isLea && (
        <p className="text-text-secondary mb-8 max-w-lg leading-relaxed">
          Try one of these suggestions:
        </p>
      )}

      <div className="flex flex-wrap gap-2 justify-center max-w-xl">
        {modeSuggestions.map((suggestion, i) => (
          <motion.button
            key={i}
            type="button"
            onClick={() => onSuggest(suggestion)}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1 }}
            className={cn(
              'px-4 py-2 rounded-full text-sm font-medium',
              'bg-surface-700 text-text-primary',
              'hover:bg-surface-700/90 hover:border-brand-400/40',
              'border border-white/15 transition-all duration-200'
            )}
          >
            {suggestion}
          </motion.button>
        ))}
      </div>
    </motion.div>
  );
}
