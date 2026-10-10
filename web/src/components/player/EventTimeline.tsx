import { Check, CircleDashed, X } from 'lucide-react'
import { useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from 'react'

import type { HoopsEvent, TeamRef } from '../../api/types'
import { eventTitle, isShot, keyTime, outcomeOf, SOURCE_LABEL, STATUS_LABEL, teamColor, teamName, type Outcome } from '../../lib/events'
import { cn, fmtClock } from '../../lib/format'
import { Tip } from '../ui/tip'

const STATUS_STYLE: Record<Outcome, { fill: string; ring: string; ink: string }> = {
  made: { fill: 'bg-made', ring: 'border-made', ink: 'text-made-text' },
  missed: { fill: 'bg-miss', ring: 'border-miss', ink: 'text-miss-text' },
  attempt: { fill: 'bg-attempt', ring: 'border-attempt', ink: 'text-attempt' },
  other: { fill: 'bg-team-other', ring: 'border-team-other', ink: 'text-fg-2' },
}

/**
 * The marker glyph: outcome = icon + color (never color alone), status = fill.
 * confirmed = filled disk, provisional = hollow ring, needs_review = hollow ring + pulse.
 */
export function EventGlyph({ e, size = 'md' }: { e: HoopsEvent; size?: 'sm' | 'md' }) {
  const o = outcomeOf(e)
  const st = STATUS_STYLE[o]
  const filled = e.status === 'confirmed'
  const Icon = o === 'made' ? Check : o === 'missed' ? X : o === 'attempt' ? CircleDashed : null
  const box = size === 'sm' ? 'size-4' : 'size-5'
  const icon = size === 'sm' ? 'size-2.5' : 'size-3'
  return (
    <span className={cn('relative grid shrink-0 place-items-center rounded-full', box)}>
      {e.status === 'needs_review' && <span className={cn('absolute inset-0 animate-ring-pulse rounded-full border-2', st.ring)} aria-hidden />}
      <span className={cn('relative grid place-items-center rounded-full border-2', box, filled ? cn(st.fill, st.ring) : cn('bg-ink-900', st.ring), e.status === 'rejected' && 'opacity-40')}>
        {Icon ? <Icon className={cn(icon, filled ? 'text-ink-950' : st.ink)} strokeWidth={3.5} aria-hidden /> : <span className={cn('size-1.5 rounded-full', filled ? 'bg-ink-950' : 'bg-fg-2')} />}
      </span>
    </span>
  )
}

function tickStep(duration: number): number {
  for (const s of [5, 10, 15, 30, 60, 120, 300, 600]) if (duration / s <= 8) return s
  return 900
}

export function EventTimeline({
  events,
  teams,
  duration,
  time,
  onSeek,
}: {
  events: HoopsEvent[]
  teams: TeamRef[]
  duration: number
  time: number
  onSeek: (t: number) => void
}) {
  const track = useRef<HTMLDivElement>(null)
  const [hover, setHover] = useState<number | null>(null)
  const [dragging, setDragging] = useState(false)
  const d = Math.max(0.001, duration)

  // Greedy lanes so close-together markers don't overlap (percent-of-width spacing).
  const placed = useMemo(() => {
    const items = events.filter((e) => e.status !== 'rejected').map((e) => ({ e, pct: (keyTime(e) / d) * 100, lane: 0 }))
    items.sort((a, b) => a.pct - b.pct)
    const lastInLane: number[] = []
    for (const it of items) {
      let lane = 0
      while (lastInLane[lane] != null && it.pct - lastInLane[lane] < 2.4) lane++
      it.lane = Math.min(lane, 2)
      lastInLane[it.lane] = it.pct
    }
    return items
  }, [events, d])
  const lanes = Math.max(1, ...placed.map((p) => p.lane + 1))

  const tFromX = (clientX: number) => {
    const r = track.current?.getBoundingClientRect()
    if (!r) return 0
    return Math.max(0, Math.min(d, ((clientX - r.left) / r.width) * d))
  }
  const onPointer = (e: PointerEvent<HTMLDivElement>, kind: 'down' | 'move' | 'up') => {
    const t = tFromX(e.clientX)
    if (kind === 'down') {
      e.currentTarget.setPointerCapture(e.pointerId)
      setDragging(true)
      onSeek(t)
    } else if (kind === 'move') {
      setHover(t)
      if (dragging) onSeek(t)
    } else {
      setDragging(false)
    }
  }
  const onKey = (e: KeyboardEvent) => {
    const step = e.shiftKey ? 10 : 5
    if (e.key === 'ArrowLeft') onSeek(Math.max(0, time - step))
    else if (e.key === 'ArrowRight') onSeek(Math.min(d, time + step))
    else if (e.key === 'Home') onSeek(0)
    else if (e.key === 'End') onSeek(d)
    else if (e.key === 'PageDown' || e.key === ']') {
      const next = placed.find((p) => keyTime(p.e) > time + 0.5)
      if (next) onSeek(Math.max(0, keyTime(next.e) - 2))
    } else if (e.key === 'PageUp' || e.key === '[') {
      const prev = [...placed].reverse().find((p) => keyTime(p.e) < time - 2.5)
      if (prev) onSeek(Math.max(0, keyTime(prev.e) - 2))
    } else return
    e.preventDefault()
  }

  const step = tickStep(d)
  const ticks = Array.from({ length: Math.floor(d / step) + 1 }, (_, i) => i * step)
  const pct = (time / d) * 100

  return (
    <div className="select-none">
      {/* marker lanes */}
      <div className="relative mx-2.5" style={{ height: lanes * 22 + 6 }}>
        {placed.map(({ e, pct: p, lane }) => {
          const o = outcomeOf(e)
          const label = `${fmtClock(keyTime(e))}, ${eventTitle(e)}, ${teamName(e.team_id, teams)}, ${STATUS_LABEL[e.status].toLowerCase()}`
          return (
            <Tip
              key={e.event_id}
              content={
                <span className="block">
                  <span className="block font-semibold text-fg">
                    {fmtClock(keyTime(e))} · {eventTitle(e)}
                  </span>
                  <span className="block text-fg-2">
                    {teamName(e.team_id, teams)} · {STATUS_LABEL[e.status]} · {SOURCE_LABEL[e.provenance.source]} {Math.round(e.confidence * 100)}%
                  </span>
                </span>
              }
            >
              <button
                type="button"
                aria-label={`${label}. Jump to it.`}
                onClick={() => onSeek(Math.max(0, keyTime(e) - 2))}
                className="group absolute flex -translate-x-1/2 flex-col items-center gap-0.5 rounded-full outline-none focus-visible:ring-2 focus-visible:ring-flame-400"
                style={{ left: `${p}%`, bottom: lane * 22 + 2 }}
              >
                <span className={cn('transition-transform group-hover:scale-125', !isShot(e) && 'scale-75')}>
                  <EventGlyph e={e} />
                </span>
                {o !== 'other' && <span className="h-[3px] w-3 rounded-full" style={{ background: teamColor(e.team_id, teams) }} aria-hidden />}
              </button>
            </Tip>
          )
        })}
      </div>

      {/* scrub track */}
      <div
        ref={track}
        role="slider"
        tabIndex={0}
        aria-label="Seek. Arrow keys move 5 seconds; [ and ] jump between events."
        aria-valuemin={0}
        aria-valuemax={Math.round(d)}
        aria-valuenow={Math.round(time)}
        aria-valuetext={`${fmtClock(time)} of ${fmtClock(d)}`}
        onKeyDown={onKey}
        onPointerDown={(e) => onPointer(e, 'down')}
        onPointerMove={(e) => onPointer(e, 'move')}
        onPointerUp={(e) => onPointer(e, 'up')}
        onPointerLeave={() => setHover(null)}
        className="group relative mx-2.5 h-6 cursor-pointer touch-none rounded-full outline-none focus-visible:ring-2 focus-visible:ring-flame-400"
      >
        <div className="absolute inset-x-0 top-1/2 h-1.5 -translate-y-1/2 overflow-hidden rounded-full bg-white/10 transition-[height] group-hover:h-2">
          <div className="h-full rounded-full bg-gradient-to-r from-flame-600 to-flame-400" style={{ width: `${pct}%` }} />
        </div>
        {placed.map(({ e, pct: p }) => (
          <span key={e.event_id} aria-hidden className="absolute top-1/2 h-2.5 w-px -translate-y-1/2 bg-white/35" style={{ left: `${p}%` }} />
        ))}
        <div className="pointer-events-none absolute top-1/2 size-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-ink-950 bg-white shadow-[0_0_0_4px_rgb(255_122_26/0.35)]" style={{ left: `${pct}%` }} aria-hidden />
        {hover != null && (
          <div className="pointer-events-none absolute -top-7 -translate-x-1/2 rounded-md bg-ink-800 px-1.5 py-0.5 text-[0.7rem] font-semibold text-fg tnum shadow" style={{ left: `${(hover / d) * 100}%` }} aria-hidden>
            {fmtClock(hover)}
          </div>
        )}
      </div>

      {/* ticks */}
      <div className="relative mx-2.5 mt-0.5 h-4 text-[0.65rem] text-fg-3 tnum" aria-hidden>
        {ticks.map((t) => (
          <span key={t} className="absolute -translate-x-1/2" style={{ left: `${(t / d) * 100}%` }}>
            {fmtClock(t)}
          </span>
        ))}
      </div>
    </div>
  )
}

/** Legend for the timeline and court map: shape + color + label (never color alone). */
export function OutcomeLegend({ className }: { className?: string }) {
  const sample = (status: HoopsEvent['status'], type: HoopsEvent['type']) => ({ status, type, shot: null }) as unknown as HoopsEvent
  return (
    <ul className={cn('flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs text-fg-2', className)}>
      <li className="flex items-center gap-1.5">
        <EventGlyph e={sample('confirmed', 'made_shot')} size="sm" /> Made
      </li>
      <li className="flex items-center gap-1.5">
        <EventGlyph e={sample('confirmed', 'missed_shot')} size="sm" /> Missed
      </li>
      <li className="flex items-center gap-1.5">
        <EventGlyph e={sample('confirmed', 'shot_attempt')} size="sm" /> Unclear
      </li>
      <li className="flex items-center gap-1.5 text-fg-3">
        <EventGlyph e={sample('provisional', 'made_shot')} size="sm" /> Hollow = provisional
      </li>
      <li className="flex items-center gap-1.5 text-fg-3">
        <EventGlyph e={sample('needs_review', 'shot_attempt')} size="sm" /> Pulsing = needs review
      </li>
    </ul>
  )
}
