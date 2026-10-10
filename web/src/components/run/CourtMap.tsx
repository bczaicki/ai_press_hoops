import { Map as MapIcon, Table2 } from 'lucide-react'
import { useState } from 'react'

import type { HoopsEvent, RunView } from '../../api/types'
import { eventTitle, keyTime, OUTCOME_LABEL, outcomeOf, teamName, type Outcome } from '../../lib/events'
import { cn, fmtClock } from '../../lib/format'
import { usePlayer } from '../player/PlayerContext'
import { Poster } from '../Poster'
import { EmptyState, OutcomeIcon } from '../ui/misc'
import { Segmented } from '../ui/segmented'

/** Half court in feet, filmed basket at the top, camera-left on the left (matches the calibration diagram). */
const L = 94
const W = 50
const HALF = L / 2
const RIM_FROM_BASELINE = 5.25
const OUT: Record<Outcome, string> = { made: '#0ca30c', missed: '#d03b3b', attempt: '#fab219', other: '#8a94a9' }
const SURFACE = '#0d1424'

interface Shot {
  e: HoopsEvent
  sx: number
  sy: number
  folded: boolean
}

function toCourt(e: HoopsEvent): Shot | null {
  const x = e.shot?.court_x_ft
  const y = e.shot?.court_y_ft
  if (x == null || y == null) return null
  const folded = x < 0
  const fx = folded ? -x : x
  const fy = folded ? -y : y
  return { e, sx: W / 2 - fy, sy: HALF - fx, folded }
}

function CourtLines() {
  return (
    <g fill="none" stroke="#e8c08f" strokeOpacity="0.5" strokeWidth="0.22">
      <rect x="0" y="0" width={W} height={HALF} />
      <rect x={W / 2 - 8} y="0" width="16" height="19" />
      <rect x={W / 2 - 6} y="0" width="12" height="19" strokeOpacity="0.25" />
      <path d={`M${W / 2 - 6} 19 A 6 6 0 0 0 ${W / 2 + 6} 19`} />
      <path d={`M${W / 2 - 6} 19 A 6 6 0 0 1 ${W / 2 + 6} 19`} strokeOpacity="0.25" />
      <path d={`M${W / 2 - 22} 0 V 14 A 23.75 23.75 0 0 0 ${W / 2 + 22} 14 V 0`} />
      <path d={`M${W / 2 - 4} ${RIM_FROM_BASELINE} A 4 4 0 0 0 ${W / 2 + 4} ${RIM_FROM_BASELINE}`} strokeOpacity="0.35" />
      <line x1={W / 2 - 3} y1="4" x2={W / 2 + 3} y2="4" strokeOpacity="0.9" strokeWidth="0.35" />
      <circle cx={W / 2} cy={RIM_FROM_BASELINE} r="0.75" stroke="#ff7a1a" strokeOpacity="0.9" strokeWidth="0.3" />
      <path d={`M${W / 2 - 6} ${HALF} A 6 6 0 0 1 ${W / 2 + 6} ${HALF}`} />
    </g>
  )
}

function ShotMark({ s, onSeek, onHover }: { s: Shot; onSeek: () => void; onHover: (v: boolean) => void }) {
  const o = outcomeOf(s.e)
  const color = OUT[o]
  const r = 0.95
  const label = `${fmtClock(keyTime(s.e))}, ${eventTitle(s.e)}. Jump to it.`
  return (
    <g
      role="button"
      tabIndex={0}
      aria-label={label}
      onClick={onSeek}
      onKeyDown={(ev) => {
        if (ev.key === 'Enter' || ev.key === ' ') {
          ev.preventDefault()
          onSeek()
        }
      }}
      onPointerEnter={() => onHover(true)}
      onPointerLeave={() => onHover(false)}
      onFocus={() => onHover(true)}
      onBlur={() => onHover(false)}
      className="cursor-pointer outline-none [&:focus-visible>circle:first-child]:stroke-flame-400"
    >
      {/* hit target, bigger than the mark */}
      <circle cx={s.sx} cy={s.sy} r="2.3" fill="transparent" stroke="transparent" strokeWidth="0.3" />
      {o === 'made' && <circle cx={s.sx} cy={s.sy} r={r} fill={color} stroke={SURFACE} strokeWidth="0.45" />}
      {o === 'missed' && (
        <g stroke={color} strokeWidth="0.42" strokeLinecap="round">
          <line x1={s.sx - r} y1={s.sy - r} x2={s.sx + r} y2={s.sy + r} />
          <line x1={s.sx - r} y1={s.sy + r} x2={s.sx + r} y2={s.sy - r} />
        </g>
      )}
      {o === 'attempt' && <circle cx={s.sx} cy={s.sy} r={r * 0.9} fill={SURFACE} stroke={color} strokeWidth="0.4" />}
      {s.e.status === 'needs_review' && <circle cx={s.sx} cy={s.sy} r={r + 0.9} fill="none" stroke={OUT.attempt} strokeWidth="0.18" strokeDasharray="0.5 0.4" />}
    </g>
  )
}

export function CourtMap({ run, events }: { run: RunView; events: HoopsEvent[] }) {
  const player = usePlayer()
  const [team, setTeam] = useState<string>('all')
  const [table, setTable] = useState(false)
  const [hover, setHover] = useState<Shot | null>(null)
  const shots = events.filter((e) => e.shot && e.status !== 'rejected' && outcomeOf(e) !== 'other').filter((e) => team === 'all' || e.team_id === team)
  const onCourt = shots.map(toCourt).filter((s): s is Shot => !!s)
  const imageSpace = shots.filter((e) => e.shot?.image_x != null && e.shot?.image_y != null)
  const seek = (e: HoopsEvent) => player.seek(Math.max(0, keyTime(e) - 2), { play: true })
  const anyFolded = onCourt.some((s) => s.folded)
  const mode: 'court' | 'png' | 'image' | 'none' = onCourt.length ? 'court' : run.urls.court_maps.length ? 'png' : imageSpace.length ? 'image' : 'none'

  return (
    <section aria-labelledby="court-h" className="glass flex h-full flex-col overflow-hidden rounded-[var(--radius-card)]">
      <div className="flex flex-wrap items-center gap-2 px-4 pt-4 pb-3 sm:px-5">
        <div className="mr-auto">
          <div className="eyebrow">Where the shots came from</div>
          <h2 id="court-h" className="font-display text-2xl font-bold tracking-wide text-fg uppercase">
            Shot chart
          </h2>
        </div>
        {mode === 'court' && (
          <button type="button" onClick={() => setTable((t) => !t)} aria-pressed={table} className={cn('inline-flex h-8 items-center gap-1.5 rounded-lg border px-2.5 text-xs font-semibold', table ? 'border-flame-500/60 text-fg' : 'border-line-strong text-fg-2 hover:text-fg')}>
            {table ? <MapIcon className="size-3.5" /> : <Table2 className="size-3.5" />}
            {table ? 'Chart' : 'Table'}
          </button>
        )}
      </div>
      {run.teams.length > 1 && mode !== 'png' && mode !== 'none' && (
        <div className="px-4 pb-3 sm:px-5">
          <Segmented size="sm" label="Team" value={team} onChange={setTeam} options={[{ key: 'all', label: 'Both teams' }, ...run.teams.map((t) => ({ key: t.team_id, label: teamName(t.team_id, run.teams) }))]} />
        </div>
      )}

      <div className="relative flex-1 px-4 pb-4 sm:px-5">
        {mode === 'court' && !table && (
          <div className="relative mx-auto max-w-md">
            <svg viewBox={`-1 -1 ${W + 2} ${HALF + 2}`} className="block w-full" role="group" aria-label={`Shot chart with ${onCourt.length} shots`}>
              <rect x="-1" y="-1" width={W + 2} height={HALF + 2} rx="1" fill="#101830" />
              <CourtLines />
              {onCourt.map((s) => (
                <ShotMark key={s.e.event_id} s={s} onSeek={() => seek(s.e)} onHover={(v) => setHover(v ? s : null)} />
              ))}
            </svg>
            {hover && (
              <div className="pointer-events-none absolute z-10 w-max max-w-56 -translate-x-1/2 -translate-y-[calc(100%+10px)] rounded-lg border border-line-strong bg-ink-800/95 px-2.5 py-1.5 text-xs shadow-xl" style={{ left: `${((hover.sx + 1) / (W + 2)) * 100}%`, top: `${((hover.sy + 1) / (HALF + 2)) * 100}%` }}>
                <div className="font-semibold text-fg">
                  {OUTCOME_LABEL[outcomeOf(hover.e)]} · {fmtClock(keyTime(hover.e))}
                </div>
                <div className="text-fg-2">
                  {eventTitle(hover.e)} · {teamName(hover.e.team_id, run.teams)}
                </div>
              </div>
            )}
          </div>
        )}
        {mode === 'court' && table && (
          <div className="overflow-x-auto scrollbar-thin">
            <table className="w-full text-left text-sm">
              <caption className="sr-only">Shots with court position</caption>
              <thead className="text-xs text-fg-3">
                <tr>
                  <th className="py-1.5 font-semibold">Time</th>
                  <th className="py-1.5 font-semibold">Team</th>
                  <th className="py-1.5 font-semibold">Result</th>
                  <th className="py-1.5 text-right font-semibold">From rim</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-line tnum">
                {onCourt.map((s) => {
                  const dx = s.sx - W / 2
                  const dy = s.sy - RIM_FROM_BASELINE
                  return (
                    <tr key={s.e.event_id}>
                      <td className="py-1.5">
                        <button type="button" className="font-mono text-flame-300" onClick={() => seek(s.e)}>
                          {fmtClock(keyTime(s.e))}
                        </button>
                      </td>
                      <td className="py-1.5 text-fg-2">{teamName(s.e.team_id, run.teams)}</td>
                      <td className="py-1.5">
                        <span className="inline-flex items-center gap-1.5 text-fg">
                          <OutcomeIcon outcome={outcomeOf(s.e)} /> {eventTitle(s.e)}
                        </span>
                      </td>
                      <td className="py-1.5 text-right text-fg-2">{Math.round(Math.hypot(dx, dy))} ft</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
        {mode === 'png' && (
          <div className="space-y-3">
            {run.urls.court_maps.map((u) => (
              <img key={u} src={u} alt="Shot chart rendered by the server" className="w-full rounded-xl border border-line" loading="lazy" />
            ))}
          </div>
        )}
        {mode === 'image' && (
          <div>
            <Poster src={`/media/clips/${encodeURIComponent(run.clip.clip_id)}/poster.jpg`} alt="" className="rounded-xl">
              <svg viewBox={`0 0 ${run.clip.proxy.width} ${run.clip.proxy.height}`} className="absolute inset-0 h-full w-full" preserveAspectRatio="none" aria-label="Shot spots in image space">
                {imageSpace.map((e) => {
                  const o = outcomeOf(e)
                  return <circle key={e.event_id} cx={e.shot!.image_x!} cy={e.shot!.image_y!} r={run.clip.proxy.width / 110} fill={o === 'made' ? OUT.made : SURFACE} stroke={OUT[o]} strokeWidth={run.clip.proxy.width / 400} />
                })}
              </svg>
            </Poster>
            <p className="mt-2 text-xs text-fg-3">
              <b className="text-fg-2">Image space:</b> dots mark where shooters were in the frame, not true court positions. Calibrate the court (fixed camera) for a real shot chart.
            </p>
          </div>
        )}
        {mode === 'none' && <EmptyState icon={<MapIcon className="size-6" />} title="No shot spots" body={run.outputs_requested.includes('court_map') ? 'No shots had a usable position.' : 'The court map was off for this run.'} />}
      </div>

      {mode === 'court' && (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-line px-4 py-2.5 text-xs text-fg-2 sm:px-5">
          <span className="flex items-center gap-1.5">
            <svg viewBox="-1.2 -1.2 2.4 2.4" className="size-3" aria-hidden>
              <circle r="1" fill={OUT.made} />
            </svg>
            Made
          </span>
          <span className="flex items-center gap-1.5">
            <svg viewBox="-1.2 -1.2 2.4 2.4" className="size-3" aria-hidden>
              <g stroke={OUT.missed} strokeWidth="0.45" strokeLinecap="round">
                <line x1="-0.9" y1="-0.9" x2="0.9" y2="0.9" />
                <line x1="-0.9" y1="0.9" x2="0.9" y2="-0.9" />
              </g>
            </svg>
            Missed
          </span>
          <span className="flex items-center gap-1.5">
            <svg viewBox="-1.2 -1.2 2.4 2.4" className="size-3" aria-hidden>
              <circle r="0.85" fill="none" stroke={OUT.attempt} strokeWidth="0.4" />
            </svg>
            Unclear
          </span>
          <span className="ml-auto text-fg-3">{onCourt.length} shots{anyFolded ? ' · far-end shots folded onto this half' : ''}</span>
        </div>
      )}
    </section>
  )
}
