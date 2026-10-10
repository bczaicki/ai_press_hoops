import { AnimatePresence, motion } from 'framer-motion'
import { Bot, Check, Cpu, Plus, ScanEye, Sparkles, User, X, type LucideIcon } from 'lucide-react'
import { forwardRef, useState } from 'react'

import type { AddableEventType, HoopsEvent, Source, TeamRef, ThresholdPreset } from '../../api/types'
import { eventTitle, isShot, keyTime, OUTCOME_LABEL, outcomeOf, SOURCE_HELP, SOURCE_LABEL, STATUS_LABEL, teamColor, teamName } from '../../lib/events'
import { cn, fmtClock } from '../../lib/format'
import { EventGlyph } from '../player/EventTimeline'
import { usePlayer } from '../player/PlayerContext'
import { PRESET_COPY } from '../Pickers'
import { Button } from '../ui/button'
import { Badge, EmptyState, OutcomeIcon, StatusBadge } from '../ui/misc'
import { Segmented } from '../ui/segmented'
import { Tip } from '../ui/tip'

export type LogFilter = 'all' | 'shots' | 'review'

const SOURCE_ICON: Record<Source, LucideIcon> = { cv: ScanEye, rule: Cpu, system_one: Bot, vlm: Sparkles, human: User, import: Cpu }

function SourceChip({ e }: { e: HoopsEvent }) {
  const src = e.provenance.source
  const Icon = SOURCE_ICON[src] ?? Cpu
  const before = (e.provenance.params?.before ?? null) as { status: string; confidence?: number } | null
  return (
    <Tip
      content={
        <span className="block">
          <span className="block">{SOURCE_HELP[src]}</span>
          {before && (
            <span className="mt-1 block text-fg-2">
              Before AI review: {STATUS_LABEL[before.status as keyof typeof STATUS_LABEL] ?? before.status}
              {before.confidence != null ? ` at ${Math.round(before.confidence * 100)}%` : ''}
            </span>
          )}
          <span className="mt-1 block font-mono text-[0.7rem] text-fg-3">{e.provenance.producer}</span>
        </span>
      }
    >
      <span tabIndex={0} className="inline-flex h-6 items-center gap-1 rounded-md border border-line bg-white/[0.04] px-1.5 text-[0.7rem] font-semibold whitespace-nowrap text-fg-2 outline-none focus-visible:ring-2 focus-visible:ring-flame-400">
        <Icon className="size-3" aria-hidden /> {SOURCE_LABEL[src]}
      </span>
    </Tip>
  )
}

function Confidence({ value, kind }: { value: number; kind: string }) {
  return (
    <Tip content={`Confidence ${Math.round(value * 100)}% (${kind.replace('_', ' ')})`}>
      <span tabIndex={0} className="flex items-center gap-2 outline-none focus-visible:ring-2 focus-visible:ring-flame-400" aria-label={`Confidence ${Math.round(value * 100)} percent`}>
        <span className="h-1.5 w-14 overflow-hidden rounded-full bg-white/10">
          <span className="block h-full rounded-full bg-fg-2" style={{ width: `${Math.round(value * 100)}%` }} />
        </span>
        <span className="w-8 text-right text-xs text-fg-2 tnum">{Math.round(value * 100)}%</span>
      </span>
    </Tip>
  )
}

export const EventLog = forwardRef<HTMLDivElement, {
  events: HoopsEvent[]
  teams: TeamRef[]
  presets: ThresholdPreset[]
  preset: string
  recomputing: boolean
  changed: Set<string>
  filter: LogFilter
  onFilter: (f: LogFilter) => void
  onPreset: (k: string) => void
  onReview: (id: string, action: 'accept' | 'reject') => Promise<void>
  onAdd: (t: number, type: AddableEventType, teamId: string | null) => Promise<void>
  busyIds: Set<string>
}>(function EventLog({ events, teams, presets, preset, recomputing, changed, filter, onFilter, onPreset, onReview, onAdd, busyIds }, ref) {
  const player = usePlayer()
  const [adding, setAdding] = useState<{ t: number; type: AddableEventType; team: string | null } | null>(null)
  const [saving, setSaving] = useState(false)
  const needsReview = events.filter((e) => e.status === 'needs_review').length
  const shotCount = events.filter((e) => isShot(e) && e.status !== 'rejected').length
  const shown = events.filter((e) => (filter === 'shots' ? isShot(e) : filter === 'review' ? e.status === 'needs_review' || e.status === 'provisional' : true))
  const now = player.time

  return (
    <section ref={ref} aria-labelledby="log-h" className="glass scroll-mt-24 overflow-hidden rounded-[var(--radius-card)]">
      <div className="flex flex-wrap items-center gap-3 px-4 pt-4 pb-3 sm:px-5">
        <div className="mr-auto">
          <div className="eyebrow">Every call, with receipts</div>
          <h2 id="log-h" className="font-display text-2xl font-bold tracking-wide text-fg uppercase">
            Shot log
          </h2>
        </div>
        <Button size="sm" variant="secondary" icon={<Plus className="size-4" />} onClick={() => setAdding({ t: player.getTime(), type: 'made_shot', team: teams[0]?.team_id ?? null })}>
          Add missed shot
        </Button>
      </div>

      <div className="flex flex-wrap items-center gap-x-4 gap-y-3 border-y border-line bg-white/[0.02] px-4 py-3 sm:px-5">
        <Segmented
          size="sm"
          label="Filter events"
          value={filter}
          onChange={onFilter}
          options={[
            { key: 'all', label: `All ${events.length}` },
            { key: 'shots', label: 'Shots' },
            { key: 'review', label: needsReview ? `To review ${needsReview}` : 'To review' },
          ]}
        />
        <div className="flex items-center gap-2">
          <Tip content={<span>How sure System One must be before a shot counts. Changing it re-applies stored scores instantly. Nothing is re-analyzed.</span>}>
            <span tabIndex={0} className="text-xs font-semibold text-fg-3 underline decoration-dotted underline-offset-2 outline-none">
              Strictness
            </span>
          </Tip>
          <Segmented size="sm" label="Strictness" value={preset} onChange={onPreset} disabled={recomputing} options={presets.map((p) => ({ key: p.key, label: p.label, title: PRESET_COPY[p.key]?.title }))} />
          {recomputing && <span className="size-4 animate-spin rounded-full border-2 border-white/20 border-t-flame-400" aria-label="Recomputing" />}
        </div>
      </div>

      <AnimatePresence initial={false}>
        {adding && (
          <motion.div initial={{ height: 0 }} animate={{ height: 'auto' }} exit={{ height: 0 }} className="overflow-hidden">
            <form
              className="flex flex-wrap items-end gap-3 border-b border-line bg-flame-500/[0.05] px-4 py-4 sm:px-5"
              onSubmit={async (ev) => {
                ev.preventDefault()
                setSaving(true)
                try {
                  await onAdd(adding.t, adding.type, adding.team)
                  setAdding(null)
                } finally {
                  setSaving(false)
                }
              }}
            >
              <div>
                <div className="mb-1 text-xs font-semibold text-fg-2">At</div>
                <div className="flex items-center gap-1.5">
                  <span className="numeral text-3xl text-fg tnum">{fmtClock(adding.t, true)}</span>
                  <Button size="sm" variant="ghost" onClick={() => setAdding({ ...adding, t: player.getTime() })}>
                    Use video time
                  </Button>
                </div>
              </div>
              <div>
                <div className="mb-1 text-xs font-semibold text-fg-2">Result</div>
                <Segmented<AddableEventType>
                  size="sm"
                  label="Result"
                  value={adding.type}
                  onChange={(type) => setAdding({ ...adding, type })}
                  options={[
                    { key: 'made_shot', label: 'Made' },
                    { key: 'missed_shot', label: 'Missed' },
                    { key: 'shot_attempt', label: 'Not sure' },
                  ]}
                />
              </div>
              <div>
                <div className="mb-1 text-xs font-semibold text-fg-2">Team</div>
                <Segmented
                  size="sm"
                  label="Team"
                  value={adding.team ?? '__none'}
                  onChange={(k) => setAdding({ ...adding, team: k === '__none' ? null : k })}
                  options={[...teams.map((t) => ({ key: t.team_id, label: teamName(t.team_id, teams) })), { key: '__none', label: '?' }]}
                />
              </div>
              <div className="ml-auto flex gap-2">
                <Button size="sm" variant="ghost" onClick={() => setAdding(null)}>
                  Cancel
                </Button>
                <Button size="sm" variant="primary" type="submit" loading={saving}>
                  Add shot
                </Button>
              </div>
            </form>
          </motion.div>
        )}
      </AnimatePresence>

      {shown.length === 0 ? (
        <EmptyState
          icon={<Check className="size-6" />}
          title={filter === 'review' ? 'Nothing to review' : 'No events'}
          body={filter === 'review' ? 'Every call is confident or already reviewed. Nice.' : 'No shots were found. Try Eager, or add one yourself.'}
        />
      ) : (
        <ul className="divide-y divide-line" aria-label="Events">
          <AnimatePresence initial={false}>
            {shown.map((e) => {
              const o = outcomeOf(e)
              const t = keyTime(e)
              const active = now >= e.t_start_s - 0.3 && now <= (e.t_end_s ?? e.t_start_s + 1.5) + 0.3
              const reviewed = e.review !== 'unreviewed'
              const human = e.provenance.source === 'human'
              const busy = busyIds.has(e.event_id)
              return (
                <motion.li
                  key={e.event_id}
                  layout="position"
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ duration: 0.25 }}
                  className={cn('relative', e.status === 'rejected' && 'opacity-55')}
                >
                  {changed.has(e.event_id) && <motion.span aria-hidden className="absolute inset-0 bg-flame-500/15" initial={{ opacity: 1 }} animate={{ opacity: 0 }} transition={{ duration: 1.8, delay: 0.3 }} />}
                  {active && <span aria-hidden className="absolute inset-y-0 left-0 w-1 bg-flame-500" />}
                  <div className="relative grid grid-cols-[3.4rem_minmax(0,1fr)_auto] items-center gap-x-3 gap-y-1.5 px-4 py-3 sm:px-5">
                    <button type="button" onClick={() => player.seek(Math.max(0, t - 2), { play: true })} className="rounded-md text-left font-mono text-[0.95rem] font-semibold text-flame-300 tnum hover:text-flame-200" aria-label={`Play from ${fmtClock(t)}`}>
                      {fmtClock(t)}
                    </button>
                    <button type="button" onClick={() => player.seek(Math.max(0, t - 2), { play: true })} className="flex min-w-0 items-center gap-2.5 rounded-md text-left">
                      <EventGlyph e={e} />
                      <span className="min-w-0">
                        <span className={cn('block truncate font-semibold text-fg', e.status === 'rejected' && 'line-through')}>{eventTitle(e)}</span>
                        {e.description && e.description.trim().toLowerCase() !== eventTitle(e).toLowerCase() && <span className="block truncate text-xs text-fg-3">{e.description}</span>}
                      </span>
                    </button>
                    {/* actions (always visible; on phones they sit at the row end) */}
                    <div className="flex items-center justify-end gap-1">
                      {human ? (
                        <span className="text-xs text-fg-3">Yours</span>
                      ) : reviewed ? (
                        <span className="text-xs text-fg-3">{e.review === 'accepted' ? 'Accepted' : e.review === 'rejected' ? 'Rejected' : 'Edited'}</span>
                      ) : (
                        <>
                          <Tip content="Looks right">
                            <button type="button" disabled={busy} onClick={() => onReview(e.event_id, 'accept')} className="grid size-8 place-items-center rounded-lg border border-line text-fg-2 hover:border-made/60 hover:bg-made/15 hover:text-fg disabled:opacity-40" aria-label={`Accept ${eventTitle(e)} at ${fmtClock(t)}`}>
                              <Check className="size-4" strokeWidth={3} />
                            </button>
                          </Tip>
                          <Tip content="Wrong call or not a shot">
                            <button type="button" disabled={busy} onClick={() => onReview(e.event_id, 'reject')} className="grid size-8 place-items-center rounded-lg border border-line text-fg-2 hover:border-miss/60 hover:bg-miss/15 hover:text-fg disabled:opacity-40" aria-label={`Reject ${eventTitle(e)} at ${fmtClock(t)}`}>
                              <X className="size-4" strokeWidth={3} />
                            </button>
                          </Tip>
                        </>
                      )}
                    </div>
                    <div className="col-span-2 col-start-2 flex flex-wrap items-center gap-x-3 gap-y-1.5">
                      <span className="flex min-w-0 items-center gap-1.5 text-sm text-fg-2">
                        <span className="size-2.5 shrink-0 rounded-full" style={{ background: teamColor(e.team_id, teams) }} aria-hidden />
                        <span className="truncate">{teamName(e.team_id, teams)}</span>
                      </span>
                      <span className="flex items-center gap-1.5">
                        <StatusBadge status={e.status} />
                      </span>
                      <span>
                        <SourceChip e={e} />
                      </span>
                      <span className="flex items-center">
                        {o !== 'other' ? (
                          <Confidence value={e.confidence} kind={e.confidence_kind} />
                        ) : (
                          <Badge tone="muted" icon={<OutcomeIcon outcome={o} />}>
                            {OUTCOME_LABEL[o]}
                          </Badge>
                        )}
                      </span>
                    </div>
                  </div>
                </motion.li>
              )
            })}
          </AnimatePresence>
        </ul>
      )}
      {filter === 'all' && shotCount > 0 && shotCount <= 2 && (
        <div className="flex items-start gap-3 border-t border-line bg-white/[0.02] px-4 py-3.5 text-sm text-fg-2 sm:px-5">
          <Sparkles className="mt-0.5 size-4 shrink-0 text-flame-400" aria-hidden />
          <p>
            {shotCount === 1 ? 'One shot in this clip.' : 'Two shots in this clip.'} Short clips often have just a play or two. Saw one the analysis missed? Pause on it and use <b className="text-fg">Add missed shot</b>. Or try <b className="text-fg">Eager</b> to surface borderline attempts.
          </p>
        </div>
      )}
    </section>
  )
})
