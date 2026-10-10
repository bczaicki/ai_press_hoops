import type { HoopsEvent, TeamRef } from '../api/types'
import { eventTitle, keyTime, STATUS_LABEL, teamName } from './events'

const ts = (t: number) => {
  const h = Math.floor(t / 3600)
  const m = Math.floor((t % 3600) / 60)
  const s = t - h * 3600 - m * 60
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${s.toFixed(3).padStart(6, '0')}`
}

const clean = (s: string) => s.replace(/-->/g, '→').replace(/[\r\n]+/g, ' ')

/**
 * WebVTT built from the CURRENT event list. The server writes events.vtt once at the end of a run
 * and doesn't regenerate it after review / add / recompute, so the player uses this instead.
 */
export function eventsToVtt(events: HoopsEvent[], teams: TeamRef[]): string {
  const cues = events
    .filter((e) => e.status !== 'rejected')
    .map((e) => {
      const start = Math.max(0, keyTime(e) - 0.6)
      const end = Math.max(start + 1.8, (e.t_end_s ?? keyTime(e)) + 0.8)
      const status = e.status === 'confirmed' ? '' : ` (${STATUS_LABEL[e.status].toLowerCase()})`
      return { start, end, text: clean(`${eventTitle(e)} · ${teamName(e.team_id, teams)}${status}`) }
    })
    .sort((a, b) => a.start - b.start)
  // Avoid overlapping cues stacking on screen.
  for (let i = 0; i < cues.length - 1; i++) cues[i].end = Math.min(cues[i].end, cues[i + 1].start)
  return ['WEBVTT', '', ...cues.flatMap((c, i) => [String(i + 1), `${ts(c.start)} --> ${ts(Math.max(c.start + 0.3, c.end))}`, c.text, ''])].join('\n')
}
