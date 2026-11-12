// Build-time data layer. Everything on the site comes from the repository itself:
// models/<id>/model.yaml + ratchet.yaml, docs/backlog.yaml and docs/*.md. Nothing is duplicated.
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';
import yaml from 'js-yaml';

export const REPO_ROOT = resolve(process.cwd(), '..');
export const REPO_URL = 'https://github.com/dogum/ML-Ratchet';

export type Verdict = 'win' | 'loss' | 'tie' | 'mixed' | 'info';
export type Decision = 'kept' | 'rejected' | 'released';

export interface Baseline { name: string; kind: 'classical' | 'dumb' | 'published'; value: number | string }
export interface ExamRow {
  metric: string; better: 'lower' | 'higher' | 'closer'; unit?: string;
  model: number | string; baselines: Baseline[]; verdict: Verdict; note?: string;
}
export interface Card {
  id: string; name: string; version: string; status: 'released' | 'partial' | 'in-progress' | 'idea';
  released?: string | Date | null; order?: number; domain: string; tags?: string[]; accent: string;
  tagline: string; question: string; summary: string; disclaimer?: string;
  reads: string; returns: string[];
  physics: { summary: string; simulator: string[]; checks: string[] };
  noise?: string; network?: string;
  data: { name: string; url: string; licence: string; role: string }[];
  split: { dev: string; exam: string };
  exam: { summary: string; results: ExamRow[]; coverage?: { output: string; actual: number }[] };
  wins?: string[]; losses?: string[]; limits: string[]; help_wanted?: string[]; next?: string[];
  usage: { install: string; demo: string; python?: string };
  files: { inference: string[]; weights: string[]; demo: string; examples?: string; card: string; lab?: string; notebook?: string };
  demo_seconds?: number;
}
export interface Rung { id: string; change: string; sim?: number; decision: Decision; why: string; metrics?: (number | string)[] }
export interface Ladder {
  name: string; objective: string; rule: string; margin: number; dev_gate?: string;
  frozen_sets?: string[]; scored_on?: string; note?: string; columns_source?: string; columns?: string[]; rungs: Rung[];
}
export interface Model extends Card {
  ladders: Ladder[];
  hasLab: boolean;
  hasInstrument: boolean;
  labBytes: number;
  weightBytes: number;
  tally: Record<Verdict, number>;
  rungCount: number;
  rejectedCount: number;
}

const read = <T>(p: string): T => yaml.load(readFileSync(p, 'utf-8')) as T;

let cache: Model[] | null = null;

export function getModels(): Model[] {
  if (cache) return cache;
  const dir = join(REPO_ROOT, 'models');
  const out: Model[] = [];
  for (const id of readdirSync(dir)) {
    const p = join(dir, id);
    if (!existsSync(join(p, 'model.yaml'))) continue;
    const card = read<Card>(join(p, 'model.yaml'));
    if (card.status === 'in-progress' || card.status === 'idea') continue;
    const lad = existsSync(join(p, 'ratchet.yaml')) ? read<{ ladders: Ladder[] }>(join(p, 'ratchet.yaml')) : { ladders: [] };
    const lab = join(p, 'lab', 'index.html');
    const tally: Record<Verdict, number> = { win: 0, loss: 0, tie: 0, mixed: 0, info: 0 };
    for (const r of card.exam.results) tally[r.verdict]++;
    const rungs = lad.ladders.flatMap((l) => l.rungs);
    out.push({
      ...card,
      ladders: lad.ladders,
      hasLab: existsSync(lab),
      hasInstrument: existsSync(join(REPO_ROOT, 'site', 'public', 'instruments', id, 'reference.json')),
      labBytes: existsSync(lab) ? statSync(lab).size : 0,
      weightBytes: card.files.weights.reduce((s, w) => s + (existsSync(join(p, w)) ? statSync(join(p, w)).size : 0), 0),
      tally,
      rungCount: rungs.length,
      rejectedCount: rungs.filter((r) => r.decision === 'rejected').length,
    });
  }
  cache = out.sort((a, b) => (a.order ?? 99) - (b.order ?? 99));
  return cache;
}

export const getModel = (id: string) => getModels().find((m) => m.id === id);

export function collectionStats() {
  const ms = getModels();
  const sum = (f: (m: Model) => number) => ms.reduce((s, m) => s + f(m), 0);
  return {
    models: ms.length,
    rungs: sum((m) => m.rungCount),
    rejected: sum((m) => m.rejectedCount),
    wins: sum((m) => m.tally.win),
    losses: sum((m) => m.tally.loss),
    ties: sum((m) => m.tally.tie + m.tally.mixed),
    domains: [...new Set(ms.map((m) => m.domain))],
    realTrainingLabels: 0,
  };
}

export interface Idea {
  id: string; name: string; domain: string; question: string; physics: string;
  data: string; data_url: string; exam: string; status: string; note?: string;
}
export const getBacklog = (): Idea[] => read<{ ideas: Idea[] }>(join(REPO_ROOT, 'docs', 'backlog.yaml')).ideas;

// ---- formatting --------------------------------------------------------------------------------
export function fmt(v: number | string, unit?: string): string {
  const s = typeof v === 'number' ? (Number.isInteger(v) ? v.toLocaleString('en-US') : String(v)) : v;
  if (!unit) return s;
  return unit === '%' ? `${s}%` : `${s} ${unit === 'deg' ? '°' : unit}`.replace(' °', '°');
}

export const mb = (b: number) => `${(b / 1048576).toFixed(1)} MB`;

/** "Black-hole events found (O1/O2)" style rows can't be plotted; numeric rows can. */
export function numericRow(r: ExamRow): { model: number; baselines: { b: Baseline; v: number }[] } | null {
  if (typeof r.model !== 'number') return null;
  const bs = r.baselines.filter((b) => typeof b.value === 'number').map((b) => ({ b, v: b.value as number }));
  if (!bs.length) return null;
  return { model: r.model, baselines: bs };
}

export const ghFile = (path: string) => `${REPO_URL}/blob/main/${path}`;
export const ghTree = (path: string) => `${REPO_URL}/tree/main/${path}`;
