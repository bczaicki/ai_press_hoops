import type { EventStatus, HoopsEvent, Narrative, Source, TeamRef } from '../api/types'
import { parseClock } from './format'

export type Outcome = 'made' | 'missed' | 'attempt' | 'other'

const SHOT_TYPES = new Set(['made_shot', 'missed_shot', 'shot_attempt', 'three_pointer', 'free_throw'])

export function isShot(e: HoopsEvent): boolean {
  return SHOT_TYPES.has(e.type) || e.shot != null
}

export function outcomeOf(e: HoopsEvent): Outcome {
  if (e.type === 'made_shot') return 'made'
  if (e.type === 'missed_shot') return 'missed'
  if (SHOT_TYPES.has(e.type)) {
    if (e.shot?.result === 'made') return 'made'
    if (e.shot?.result === 'missed') return 'missed'
    return 'attempt'
  }
  return 'other'
}

export function keyTime(e: HoopsEvent): number {
  return e.t_key_s ?? e.t_start_s
}

const KIND: Record<string, string> = {
  jumper: 'jumper',
  layup: 'layup',
  dunk: 'dunk',
  hook: 'hook shot',
  tip: 'tip-in',
  free_throw: 'free throw',
}

const TYPE_LABELS: Record<string, string> = {
  rebound: 'Rebound',
  assist: 'Assist',
  turnover: 'Turnover',
  steal: 'Steal',
  block: 'Block',
  foul: 'Foul',
  possession_change: 'Possession change',
  timeout: 'Timeout',
  other: 'Other',
  free_throw: 'Free throw',
}

/** "Made 3PT jumper", "Missed layup", "Shot attempt", "Rebound" */
export function eventTitle(e: HoopsEvent): string {
  const o = outcomeOf(e)
  if (o === 'other') return TYPE_LABELS[e.type] ?? e.type.replace(/_/g, ' ')
  const kind = e.shot?.kind && e.shot.kind !== 'unknown' ? KIND[e.shot.kind] : 'shot'
  const three = e.shot?.points === 3 ? '3PT ' : ''
  if (o === 'made') return `Made ${three}${kind}`
  if (o === 'missed') return `Missed ${three}${kind}`
  return `${three}${kind === 'shot' ? 'Shot' : kind[0].toUpperCase() + kind.slice(1)} attempt`.trim()
}

export const OUTCOME_LABEL: Record<Outcome, string> = { made: 'Made', missed: 'Missed', attempt: 'Unclear', other: 'Play' }

export const SOURCE_LABEL: Record<Source, string> = {
  cv: 'CV',
  rule: 'Rules',
  system_one: 'System One',
  vlm: 'VLM',
  human: 'Human',
  import: 'Import',
}

export const SOURCE_HELP: Record<Source, string> = {
  cv: 'Computer vision (detector + tracker) found this on its own.',
  rule: 'A deterministic rule derived this from other events.',
  system_one: 'A fast System One model made a typed, calibrated call.',
  vlm: 'A vision-language model reviewed the frames and decided.',
  human: 'You added or reviewed this.',
  import: 'Imported from another run or label file.',
}

export const STATUS_LABEL: Record<EventStatus, string> = {
  provisional: 'Provisional',
  confirmed: 'Confirmed',
  needs_review: 'Needs review',
  rejected: 'Rejected',
}

export const STATUS_HELP: Record<EventStatus, string> = {
  provisional: 'Leaning one way, but not past the confidence bar.',
  confirmed: 'Confident call (or reviewed).',
  needs_review: 'Too close to call. Watch it and accept or reject.',
  rejected: 'Dropped as a false alarm.',
}

// ------------------------------------------------------------------ teams

/** Validated team palette (dataviz six-checks on the navy surface): slot 1 blue, slot 2 magenta, rest gray. */
const TEAM_SLOTS = ['var(--color-team-a)', 'var(--color-team-b)']
export const TEAM_OTHER = 'var(--color-team-other)'

/** Color follows the entity: teams keep their slot by team_id order, never by rank. */
export function teamColor(teamId: string | null | undefined, teams: TeamRef[]): string {
  if (!teamId) return TEAM_OTHER
  const ordered = [...teams].map((t) => t.team_id).sort()
  const i = ordered.indexOf(teamId)
  return i >= 0 && i < TEAM_SLOTS.length ? TEAM_SLOTS[i] : TEAM_OTHER
}

export function teamName(teamId: string | null | undefined, teams: TeamRef[]): string {
  if (!teamId) return 'Unknown team'
  const t = teams.find((x) => x.team_id === teamId)
  return t?.name ?? `Team ${teamId}`
}

// ------------------------------------------------------------------ summary narrative

export interface KeyMomentView {
  t_s: number | null
  ref: string
  description: string
}

export interface SummaryView {
  headline: string | null
  body: string
  keyMoments: KeyMomentView[]
  caveats: string[]
  unsupported: string[]
}

interface SummaryParams {
  headline?: string
  narrative?: string
  key_moments?: { ref: string; description: string; t_s?: number | null }[]
  caveats?: string[]
}

/**
 * The canonical Narrative only has `text`. ClipSummary extras (headline, key_moments, caveats) are
 * read from provenance.params.summary when the backend provides them; otherwise derived from text.
 */
export function toSummaryView(n: Narrative, events: HoopsEvent[]): SummaryView {
  let params = (n.provenance?.params?.summary ?? null) as SummaryParams | null
  let body = n.text.trim()
  // Structured parsing can fail server-side, leaving raw JSON in `text`.
  if (!params && body.startsWith('{')) {
    try {
      const j = JSON.parse(body) as SummaryParams
      if (j && typeof j.narrative === 'string') {
        params = j
        body = j.narrative
      }
    } catch {
      /* not JSON after all */
    }
  }
  let headline = params?.headline ?? null
  const h = /^#{1,4}\s+(.+)\n+/.exec(body)
  if (h) {
    headline = headline ?? h[1].trim()
    body = body.slice(h[0].length)
  }
  let keyMoments: KeyMomentView[]
  if (params?.key_moments?.length) {
    keyMoments = params.key_moments.map((k) => ({ ref: k.ref, description: k.description, t_s: k.t_s ?? parseClock(k.ref) }))
  } else {
    keyMoments = n.cited_times_s.map((t) => {
      const near = events.reduce<HoopsEvent | null>((best, e) => {
        const d = Math.abs((e.t_key_s ?? e.t_start_s) - t)
        return d < 2 && (!best || d < Math.abs((best.t_key_s ?? best.t_start_s) - t)) ? e : best
      }, null)
      return { t_s: t, ref: '', description: near ? (near.description ?? eventTitle(near)) : 'Cited moment' }
    })
  }
  return { headline, body, keyMoments, caveats: params?.caveats ?? [], unsupported: n.unsupported_claims }
}
