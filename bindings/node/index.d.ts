export type State = string | Record<string, unknown> | unknown[];
export interface QueryResult {
  query: string; matched: boolean; handoff: boolean;
  stage: 'exact' | 'substring' | 'stem' | 'fuzzy' | 'hub' | 'empty_memory';
  confidence: number; term_coverage: number;
  hits: { id: string; label: string; level: string; block: string; score: number; text?: string }[];
  blocks: { name: string; members: string[]; size: number }[];
}
export class Bodi {
  constructor(config?: string | Record<string, unknown>);
  call(method: string, args?: Record<string, unknown>): any;
  callAsync(method: string, args?: Record<string, unknown>): Promise<any>;
  ingest(text: string, opts?: { format?: 'auto' | 'atf' | 'entity_tags' | 'text'; source?: string }): any;
  ingestBatch(docs: { text: string; format?: string; source?: string }[]): any;
  query(q: string, k?: number): QueryResult;
  traverse(start: string, hops?: number, limit?: number): any;
  context(q: string, k?: number, maxChars?: number): { context: string; matched: boolean; handoff: boolean };
  density(): any; ensureDensity(): any; fastmemorySearch(q: string): any[];
  snapshot(): any; restore(snapshot: any): any; clear(): void; stats(): any;
  loadLaya(dir: string, options?: Record<string, unknown>): void;
  loadEmbedder(dir: string, options?: Record<string, unknown>): void;
  embedText(texts: string[]): Promise<number[][]>;
  learn(states: State[], questions: Record<string, unknown>, labels: Record<string, unknown>[]): Promise<any>;
  forget(): void;
  predict(state: State, questions: Record<string, unknown>): Promise<any>;
  decide(state: State, questions: Record<string, unknown>, options?: Record<string, unknown> | null): Promise<any>;
  decideBatch(states: State[], questions: Record<string, unknown>, options?: Record<string, unknown> | null): Promise<any[]>;
  decideWithMemory(state: State, questions: Record<string, unknown>, opts?: { query?: string; k?: number; maxChars?: number; options?: Record<string, unknown> | null }): Promise<any>;
}
export function version(): string;
