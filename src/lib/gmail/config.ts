// DreAgent - Gmail read configuration status (M3 foundation)
// Copyright (c) 2026 B&D Servicing LLC - All Rights Reserved
// Powered by CoDre-X™
//
// Does not load or write .env files. Reports whether operators have set
// expected env names for a future Gmail OAuth read path.

/** Env names required before a live Gmail OAuth client can be built (operators set these; never commit values). */
export const GMAIL_OAUTH_ENV_NAMES = [
  'GMAIL_CLIENT_ID',
  'GMAIL_CLIENT_SECRET',
] as const;

/** Optional env names for a future Gmail OAuth redirect / app URL. */
export const GMAIL_OAUTH_OPTIONAL_ENV_NAMES = [
  'GMAIL_REDIRECT_URI',
  'NEXT_PUBLIC_APP_URL',
] as const;

export type GmailConfigStatus = {
  /** True when both client id and secret env vars are non-empty. */
  oauthClientConfigured: boolean;
  missingRequiredEnv: string[];
  presentOptionalEnv: string[];
  /**
   * Live Gmail mailbox read is not available in this M3 foundation slice.
   * Always false until OAuth + token store + Gmail API list are implemented.
   */
  liveReadAvailable: boolean;
  reason: string;
};

function envPresent(name: string): boolean {
  return Boolean(process.env[name]?.trim());
}

/**
 * Inspect Gmail-related env presence without inventing credentials or touching .env files.
 */
export function getGmailConfigStatus(): GmailConfigStatus {
  const missingRequiredEnv = GMAIL_OAUTH_ENV_NAMES.filter((name) => !envPresent(name));
  const presentOptionalEnv = GMAIL_OAUTH_OPTIONAL_ENV_NAMES.filter((name) =>
    envPresent(name)
  );
  const oauthClientConfigured = missingRequiredEnv.length === 0;

  return {
    oauthClientConfigured,
    missingRequiredEnv: [...missingRequiredEnv],
    presentOptionalEnv: [...presentOptionalEnv],
    liveReadAvailable: false,
    reason: oauthClientConfigured
      ? 'Gmail OAuth client env appears set, but Gmail OAuth token store and Gmail API read client are not implemented yet. Live inbox read remains unavailable.'
      : `Gmail read is not available. Missing env: ${missingRequiredEnv.join(', ') || '(none)'}. OAuth token store and Gmail API client are also not implemented yet.`,
  };
}

export function isGmailLiveReadAvailable(): boolean {
  return getGmailConfigStatus().liveReadAvailable;
}
