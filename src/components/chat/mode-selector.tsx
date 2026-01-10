// DreAgent Cloud - Mode Selector
// Copyright (c) 2026 B&D Servicing LLC - All Rights Reserved
// Powered by CoDre-X™

'use client';

import { motion } from 'framer-motion';
import { MODES, type AgentMode } from '@/lib/ai/prompts';
import { cn } from '@/lib/utils';

interface ModeSelectorProps {
  currentMode: AgentMode;
  onModeChange: (mode: AgentMode) => void;
}

export function ModeSelector({ currentMode, onModeChange }: ModeSelectorProps) {
  const modes = Object.values(MODES);

  return (
    <div className="flex flex-wrap gap-2 justify-center">
      {modes.map((mode) => (
        <motion.button
          key={mode.id}
          onClick={() => onModeChange(mode.id)}
          className={cn(
            'px-4 py-2 rounded-full text-sm font-medium transition-all duration-200',
            'flex items-center gap-2',
            currentMode === mode.id
              ? 'bg-brand-500 text-white shadow-lg'
              : 'bg-surface-700/50 text-text-secondary hover:bg-surface-700 hover:text-text-primary'
          )}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          style={
            currentMode === mode.id
              ? { boxShadow: '0 4px 15px rgba(205, 111, 77, 0.3)' }
              : {}
          }
        >
          <span>{mode.icon}</span>
          <span className="hidden sm:inline">{mode.name}</span>
        </motion.button>
      ))}
    </div>
  );
}
