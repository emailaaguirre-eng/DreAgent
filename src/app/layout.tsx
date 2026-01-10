// DreAgent Cloud - Root Layout
// Copyright (c) 2026 B&D Servicing LLC - All Rights Reserved
// Powered by CoDre-X™

import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'DreAgent Cloud | AI Assistant',
  description: 'Intelligent AI assistant powered by CoDre-X™',
  keywords: ['AI', 'assistant', 'chat', 'productivity'],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="gradient-bg grid-overlay min-h-screen">
        {children}
        
        {/* Footer branding */}
        <footer className="fixed bottom-0 left-0 right-0 py-2 text-center text-xs text-text-muted/50 pointer-events-none">
          Powered by CoDre-X™ © 2026 B&D Servicing LLC
        </footer>
      </body>
    </html>
  );
}
