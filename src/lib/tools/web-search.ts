// DreAgent Cloud - Web Search Tool
// Copyright (c) 2026 B&D Servicing LLC - All Rights Reserved
// Powered by CoDre-X™

interface SearchResult {
  title: string;
  link: string;
  snippet: string;
}

interface SearchResponse {
  results: SearchResult[];
  query: string;
}

const RELIABLE_DOMAINS = [
  'reuters.com',
  'apnews.com',
  'associatedpress.com',
  'nature.com',
  'science.org',
  'nejm.org',
  'thelancet.com',
  'jamanetwork.com',
  'who.int',
  'cdc.gov',
  'nih.gov',
  'nasa.gov',
  'noaa.gov',
  'sec.gov',
  'irs.gov',
  'federalreserve.gov',
  'europa.eu',
  'oecd.org',
  'imf.org',
  'worldbank.org',
];

function getHostname(url: string): string {
  try {
    return new URL(url).hostname.toLowerCase();
  } catch {
    return '';
  }
}

function isReliableDomain(hostname: string): boolean {
  if (!hostname) return false;
  if (
    hostname.endsWith('.gov') ||
    hostname.endsWith('.edu') ||
    hostname.endsWith('.int') ||
    hostname.endsWith('.org')
  ) {
    // Allow institutional domains by default but still keep explicit list for key news/science sources.
    return true;
  }

  return RELIABLE_DOMAINS.some(
    (domain) => hostname === domain || hostname.endsWith(`.${domain}`)
  );
}

/**
 * Search the web using SerpAPI
 */
export async function webSearch(query: string): Promise<SearchResponse> {
  const apiKey = process.env.SERPAPI_API_KEY;
  
  if (!apiKey) {
    console.warn('SERPAPI_API_KEY not set, web search disabled');
    return { results: [], query };
  }

  try {
    const params = new URLSearchParams({
      q: query,
      api_key: apiKey,
      engine: 'google',
      num: '5',
    });

    const response = await fetch(
      `https://serpapi.com/search.json?${params.toString()}`
    );

    if (!response.ok) {
      throw new Error(`SerpAPI error: ${response.status}`);
    }

    const data = await response.json();

    const results: SearchResult[] = (data.organic_results || [])
      .slice(0, 5)
      .map((r: { title: string; link: string; snippet: string }) => ({
        title: r.title,
        link: r.link,
        snippet: r.snippet,
      }));

    return { results, query };
  } catch (error) {
    console.error('Web search error:', error);
    return { results: [], query };
  }
}

export function filterReliableSearchResults(
  response: SearchResponse,
  maxResults = 5
): SearchResponse {
  const reliable = response.results.filter((result) => {
    const hostname = getHostname(result.link);
    return isReliableDomain(hostname);
  });

  return {
    query: response.query,
    results: reliable.slice(0, maxResults),
  };
}

/**
 * Format search results for AI context
 */
export function formatSearchResults(response: SearchResponse): string {
  if (response.results.length === 0) {
    return '';
  }

  const formatted = response.results
    .map((r, i) => `${i + 1}. **${r.title}**\n   ${r.snippet}\n   Source: ${r.link}`)
    .join('\n\n');

  return `## Web Search Results for "${response.query}"\n\n${formatted}`;
}
