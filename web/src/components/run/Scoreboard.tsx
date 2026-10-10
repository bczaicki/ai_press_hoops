import { AnimatePresence, motion } from 'framer-motion'
import { ChevronRight } from 'lucide-react'

import type { HoopsEvent, StatLine, TeamRef, TeamSummary } from '../../api/types'
import { outcomeOf, teamColor, teamName } from '../../lib/events'
import { cn, fmtPct } from '../../lib/format'
import { Tip } from '../ui/tip'

/** Broadcast-style number flip when a value changes (e.g. after Strict ↔ Eager). */
export function Ticker({ value, className }: { value: string | number; className?: string }) {
  return (
    <span className={cn('relative inline-flex overflow-hidden align-bottom', className)}>
      <AnimatePresence initial={false} mode="popLayout">
        <motion.span key={String(value)} initial={{ y: '-70%', opacity: 0 }} animate={{ y: 0, opacity: 1 }} exit={{ y: '70%', opacity: 0 }} transition={{ type: 'spring', stiffness: 520, damping: 36 }} className="inline-block">
          {value}
        </motion.span>
      </AnimatePresence>
    </span>
  )
}

function Stat({ label, value, sub, hint, big }: { label: string; value: string | number; sub?: string; hint: string; big?: boolean }) {
  return (
    <Tip content={hint}>
      <div tabIndex={0} className="min-w-0 rounded-lg px-1 text-center outline-none focus-visible:ring-2 focus-visible:ring-flame-400">
        <div className="eyebrow text-[0.62rem] @3xl:text-[0.7rem]">{label}</div>
        <div className={cn('numeral mt-0.5 whitespace-nowrap text-fg', big ? 'text-5xl @3xl:text-6xl' : 'text-3xl @3xl:text-4xl')}>
          <Ticker value={value} />
          {sub && <span className="ml-0.5 text-[0.55em] text-fg-3">{sub}</span>}
        </div>
      </div>
    </Tip>
  )
}

const emptyLine = (): StatLine => ({ pts: 0, fga: 0, fgm: 0, tpa: 0, tpm: 0, fta: 0, ftm: 0, reb: 0, ast: 0, stl: 0, blk: 0, tov: 0, pf: 0 })

/** No team split (e.g. a solo practice clip): one line computed from the events themselves. */
function overall(events: HoopsEvent[]): TeamSummary {
  const s = emptyLine()
  for (const e of events) {
    if (e.status === 'rejected') continue
    const o = outcomeOf(e)
    if (e.type === 'rebound') s.reb++
    if (o === 'other') continue
    s.fga++
    if (e.shot?.points === 3) s.tpa++
    if (o === 'made') {
      s.fgm++
      if (e.shot?.points === 3) s.tpm++
      s.pts += e.shot?.points ?? 2
    }
  }
  return { team: { team_id: '_all', name: 'All shots', color_hex: null }, stats: s, derived_from_event_ids: [] }
}

export function Scoreboard({
  teams,
  summaries,
  events,
  onReview,
}: {
  teams: TeamRef[]
  summaries: TeamSummary[]
  events: HoopsEvent[]
  onReview: () => void
}) {
  const needsReview = events.filter((e) => e.status === 'needs_review').length
  const provisional = events.filter((e) => e.status === 'provisional').length
  const hasActivity = (st: StatLine) => st.fga > 0 || st.reb > 0 || st.pts > 0
  const named = new Set(teams.map((t) => t.team_id))
  const base: TeamSummary[] = summaries.length ? summaries : teams.map((t) => ({ team: t, stats: emptyLine(), derived_from_event_ids: [] }))
  // If no named team got credit for anything (the team split didn't attribute shots), one honest line beats "0/0, 0/0, Unknown 1/1".
  const unattributed = events.length > 0 && !base.some((r) => named.has(r.team.team_id) && hasActivity(r.stats))
  const rows: TeamSummary[] = !base.length || unattributed ? [overall(events)] : base
  const allTeams = rows.map((r) => r.team)
  const leader = rows.length === 2 && rows[0].stats.pts !== rows[1].stats.pts ? (rows[0].stats.pts > rows[1].stats.pts ? rows[0].team.team_id : rows[1].team.team_id) : null

  return (
    <section aria-label="Scoreboard" className="glass @container overflow-hidden rounded-[var(--radius-card)]">
      <div className="divide-y divide-line">
        {rows.map((r) => {
          const color = r.team.team_id === '_all' ? 'var(--color-team-other)' : teamColor(r.team.team_id, allTeams)
          const s = r.stats
          return (
            <div
              key={r.team.team_id}
              className="relative grid grid-cols-[minmax(0,1fr)_repeat(2,auto)] items-center gap-3 py-3 pr-3 pl-5 @lg:grid-cols-[minmax(0,1fr)_repeat(3,minmax(4.5rem,auto))] @3xl:grid-cols-[minmax(0,14rem)_repeat(5,minmax(0,1fr))] @3xl:gap-2 @3xl:pr-5 @3xl:pl-6"
            >
              <span aria-hidden className="absolute inset-y-0 left-0 w-1.5" style={{ background: color }} />
              <div className="flex min-w-0 items-center gap-2.5">
                {r.team.color_hex && (
                  <Tip content={`Jersey color as seen: ${r.team.color_hex}`}>
                    <span tabIndex={0} className="size-4 shrink-0 rounded-full ring-2 ring-white/25 outline-none focus-visible:ring-flame-400" style={{ background: r.team.color_hex }} aria-label="Jersey color" />
                  </Tip>
                )}
                <div className="min-w-0">
                  <div className="truncate font-display text-xl font-bold tracking-wide text-fg uppercase @3xl:text-2xl">{teamName(r.team.team_id, allTeams)}</div>
                  {leader === r.team.team_id && <div className="eyebrow text-[0.6rem] text-flame-300">Leading</div>}
                </div>
              </div>
              <Stat label="FG" value={`${s.fgm}/${s.fga}`} hint="Field goals made / attempted" />
              <div className="hidden @lg:block">
                <Stat label="FG%" value={s.fga ? fmtPct(s.fgm / s.fga) : '—'} hint="Field-goal percentage" />
              </div>
              <div className="hidden @3xl:block">
                <Stat label="3PT" value={`${s.tpm}/${s.tpa}`} hint="Threes made / attempted" />
              </div>
              <div className="hidden @3xl:block">
                <Stat label="REB" value={s.reb} hint="Rebounds (when detected)" />
              </div>
              <Stat label="PTS" value={s.pts} hint="Points from made shots" big />
            </div>
          )
        })}
      </div>
      {unattributed && teams.length > 0 && (
        <p className="border-t border-line px-5 py-2.5 text-xs text-fg-3 sm:px-6">Teams couldn't be told apart in this clip (similar jerseys or too few frames), so shots are counted together.</p>
      )}
      {(needsReview > 0 || provisional > 0) && (
        <button type="button" onClick={onReview} className="flex w-full items-center gap-2 border-t border-line bg-attempt/[0.06] px-5 py-2.5 text-left text-sm text-fg-2 hover:bg-attempt/[0.1] sm:px-6">
          <span className="relative flex size-2 shrink-0">
            <span className="absolute inset-0 animate-ring-pulse rounded-full bg-attempt" />
            <span className="relative size-2 rounded-full bg-attempt" />
          </span>
          <span className="min-w-0 flex-1">
            {provisional > 0 && <>Includes {provisional} provisional call{provisional === 1 ? '' : 's'}. </>}
            {needsReview > 0 && (
              <b className="text-fg">
                {needsReview} close call{needsReview === 1 ? '' : 's'} need{needsReview === 1 ? 's' : ''} your eyes.
              </b>
            )}
          </span>
          <ChevronRight className="size-4 shrink-0 text-fg-3" aria-hidden />
        </button>
      )}
    </section>
  )
}
