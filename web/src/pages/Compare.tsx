import { ArrowLeftRight, GitCompareArrows } from 'lucide-react'
import { useEffect, useMemo } from 'react'
import { useSearchParams } from 'react-router'

import { api, isMockMedia, posterForClip } from '../api'
import type { CompareResponse, HoopsEvent, RunSummary, RunView } from '../api/types'
import { PlayerProvider, usePlayer } from '../components/player/PlayerContext'
import { EventGlyph } from '../components/player/EventTimeline'
import { VideoPlayer } from '../components/player/VideoPlayer'
import { Scoreboard } from '../components/run/Scoreboard'
import { ButtonLink } from '../components/ui/button'
import { Badge, Card, EmptyState, ErrorPanel, Select, Skeleton } from '../components/ui/misc'
import { eventTitle, isShot, keyTime, teamName } from '../lib/events'
import { cn, fmtClock, fmtDateTime, fmtPct, fmtUsd } from '../lib/format'
import { useAsync } from '../lib/hooks'

function runLabel(r: RunSummary): string {
  const ai = r.providers.length ? r.providers.join('+') : 'no AI'
  return `${r.filename ?? r.clip_id} · ${fmtDateTime(r.created_at)} · ${ai} · ${r.n_events} events`
}

const family = (e: HoopsEvent) => (isShot(e) ? 'shot' : e.type)

/** Same rule as the server: same family, within ±1 s, nearest first. Returns partner ids and whether results differ. */
function matchEvents(a: HoopsEvent[], b: HoopsEvent[]) {
  const live = (xs: HoopsEvent[]) => xs.filter((e) => e.status !== 'rejected')
  const A = live(a)
  const B = live(b)
  const used = new Set<string>()
  const partner = new Map<string, string>()
  for (const x of A) {
    let best: HoopsEvent | null = null
    for (const y of B) {
      if (used.has(y.event_id) || family(y) !== family(x)) continue
      const d = Math.abs(keyTime(y) - keyTime(x))
      if (d <= 1 && (!best || d < Math.abs(keyTime(best) - keyTime(x)))) best = y
    }
    if (best) {
      used.add(best.event_id)
      partner.set(x.event_id, best.event_id)
      partner.set(best.event_id, x.event_id)
    }
  }
  return partner
}

function Side({ label, run, partner, other }: { label: 'A' | 'B'; run: RunView; partner: Map<string, string>; other: RunView }) {
  const decider = run.pipeline.find((s) => s.stage === 'decide')
  const preset = (decider?.params?.preset as string | undefined) ?? undefined
  const sources = [...(run.urls.annotated ? [{ key: 'annotated', label: 'Annotated', url: run.urls.annotated }] : []), { key: 'original', label: 'Original', url: run.urls.proxy || run.urls.master }]
  const poster = run.urls.poster && !isMockMedia(run.urls.poster) ? run.urls.poster : posterForClip(run.clip.clip_id)
  return (
    <PlayerProvider>
      <div className="min-w-0 space-y-4">
        <Card className="flex flex-wrap items-center gap-2 px-4 py-3">
          <span className="numeral grid size-9 place-items-center rounded-xl bg-flame-500 text-xl text-ink-950">{label}</span>
          <div className="min-w-0 flex-1">
            <div className="truncate font-semibold text-fg">{run.clip.original_filename ?? run.run_id}</div>
            <div className="flex flex-wrap gap-1.5 pt-1">
              {decider && <Badge>System One: {decider.impl}</Badge>}
              {preset && <Badge tone="muted">{preset}</Badge>}
              <Badge tone={run.vlm_calls.length ? 'flame' : 'muted'}>{run.vlm_calls[0]?.model ?? 'No AI'}</Badge>
              <Badge tone="muted">{fmtUsd(run.metrics.cost_usd)}</Badge>
            </div>
          </div>
          <ButtonLink to={`/runs/${encodeURIComponent(run.run_id)}`} size="sm" variant="ghost">
            Open
          </ButtonLink>
        </Card>
        <VideoPlayer sources={sources} captionsUrl={null} poster={poster} fallbackDuration={run.clip.proxy.duration_s} events={run.events} teams={run.teams} />
        <Scoreboard teams={run.teams} summaries={run.team_summaries} events={run.events} onReview={() => {}} />
        <SideEvents run={run} partner={partner} other={other} />
      </div>
    </PlayerProvider>
  )
}

function SideEvents({ run, partner, other }: { run: RunView; partner: Map<string, string>; other: RunView }) {
  const player = usePlayer()
  const otherById = new Map(other.events.map((e) => [e.event_id, e]))
  const list = run.events.filter((e) => e.status !== 'rejected')
  return (
    <Card className="overflow-hidden">
      <div className="px-4 pt-3.5 pb-2 font-display text-lg font-bold tracking-wide text-fg uppercase">Events</div>
      {list.length === 0 ? (
        <EmptyState title="No events" />
      ) : (
        <ul className="divide-y divide-line">
          {list.map((e) => {
            const pid = partner.get(e.event_id)
            const p = pid ? otherById.get(pid) : undefined
            const differs = !!p && (p.type !== e.type || p.shot?.result !== e.shot?.result)
            return (
              <li key={e.event_id}>
                <button type="button" onClick={() => player.seek(Math.max(0, keyTime(e) - 2), { play: true })} className="flex w-full items-center gap-3 px-4 py-2.5 text-left hover:bg-white/[0.03]">
                  <span className="w-10 font-mono text-sm text-flame-300 tnum">{fmtClock(keyTime(e))}</span>
                  <EventGlyph e={e} size="sm" />
                  <span className="min-w-0 flex-1 truncate text-sm text-fg">
                    {eventTitle(e)} <span className="text-fg-3">· {teamName(e.team_id, run.teams)}</span>
                  </span>
                  {!p ? <Badge tone="attempt">Only here</Badge> : differs ? <Badge tone="miss">Different call</Badge> : <Badge tone="muted">Both</Badge>}
                </button>
              </li>
            )
          })}
        </ul>
      )}
    </Card>
  )
}

function Kpi({ label, value, sub }: { label: string; value: string | number; sub: string }) {
  return (
    <div className="glass rounded-2xl px-4 py-3.5">
      <div className="eyebrow text-[0.65rem]">{label}</div>
      <div className="numeral mt-1 text-5xl text-fg">{value}</div>
      <div className="mt-0.5 text-xs text-fg-3">{sub}</div>
    </div>
  )
}

export default function ComparePage() {
  const [params, setParams] = useSearchParams()
  const runs = useAsync(() => api.runs(), [])
  const list = useMemo(() => (runs.data ?? []).filter((r) => r.status === 'done'), [runs.data])
  const a = params.get('a') ?? ''
  const b = params.get('b') ?? ''

  // Sensible defaults: A = given or newest; B = another run of the same clip, else the next newest.
  useEffect(() => {
    if (!list.length) return
    const A = a || list[0].run_id
    const aClip = list.find((r) => r.run_id === A)?.clip_id
    const B = b || list.find((r) => r.run_id !== A && r.clip_id === aClip)?.run_id || list.find((r) => r.run_id !== A)?.run_id || ''
    if (A !== a || B !== b) setParams({ a: A, ...(B ? { b: B } : {}) }, { replace: true })
  }, [list, a, b, setParams])

  const cmp = useAsync<CompareResponse | null>(() => (a && b && a !== b ? api.compare(a, b) : Promise.resolve(null)), [a, b])
  const partner = useMemo(() => (cmp.data ? matchEvents(cmp.data.a.events, cmp.data.b.events) : new Map<string, string>()), [cmp.data])
  const sameClip = cmp.data && cmp.data.a.clip.clip_id === cmp.data.b.clip.clip_id

  return (
    <div className="space-y-6">
      <div>
        <div className="eyebrow mb-1 text-flame-300">Head to head</div>
        <h1 className="font-display text-5xl leading-none font-bold tracking-wide text-fg uppercase">Compare runs</h1>
        <p className="mt-2 max-w-2xl text-fg-2">Run the same clip with different settings (another AI, another decider, Strict vs Eager), then see where they agree.</p>
      </div>

      {runs.error ? (
        <ErrorPanel message={runs.error} />
      ) : runs.loading && !runs.data ? (
        <Skeleton className="h-24" />
      ) : list.length < 2 ? (
        <Card>
          <EmptyState icon={<GitCompareArrows className="size-6" />} title="Need two runs" body="Analyze a clip twice with different settings, then come back here to compare them." action={<ButtonLink to="/" variant="primary">Go to clips</ButtonLink>} />
        </Card>
      ) : (
        <Card className="grid items-end gap-3 p-4 sm:grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)]">
          <div className="min-w-0">
            <label htmlFor="cmp-a" className="mb-1.5 block text-xs font-semibold text-fg-2">
              Run A
            </label>
            <Select id="cmp-a" value={a} onChange={(e) => setParams({ a: e.target.value, ...(b ? { b } : {}) })}>
              {list.map((r) => (
                <option key={r.run_id} value={r.run_id}>
                  {runLabel(r)}
                </option>
              ))}
            </Select>
          </div>
          <button type="button" onClick={() => setParams({ a: b, b: a })} className="mx-auto grid size-10 place-items-center rounded-xl border border-line-strong text-fg-2 hover:text-fg" aria-label="Swap A and B">
            <ArrowLeftRight className="size-4" />
          </button>
          <div className="min-w-0">
            <label htmlFor="cmp-b" className="mb-1.5 block text-xs font-semibold text-fg-2">
              Run B
            </label>
            <Select id="cmp-b" value={b} onChange={(e) => setParams({ a, b: e.target.value })}>
              {list.map((r) => (
                <option key={r.run_id} value={r.run_id} disabled={r.run_id === a}>
                  {runLabel(r)}
                </option>
              ))}
            </Select>
          </div>
        </Card>
      )}

      {cmp.error && <ErrorPanel message={cmp.error} />}
      {cmp.loading && a && b && !cmp.data && <Skeleton className="h-96" />}
      {cmp.data && (
        <div className={cn('space-y-6 transition-opacity', cmp.loading && 'opacity-60')}>
          {!sameClip && <p className="rounded-xl border border-attempt/30 bg-attempt/[0.06] p-3 text-sm text-fg-2">These runs are on different clips, so agreement numbers mostly measure the clips, not the settings.</p>}
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <Kpi label="Matched" value={cmp.data.agreement.matched} sub="found by both (±1 s)" />
            <Kpi label="Only in A" value={cmp.data.agreement.only_a} sub="B missed or dropped these" />
            <Kpi label="Only in B" value={cmp.data.agreement.only_b} sub="A missed or dropped these" />
            <Kpi label="Same result" value={cmp.data.agreement.outcome_agreement == null ? '—' : fmtPct(cmp.data.agreement.outcome_agreement)} sub="of matched shots: made vs missed" />
          </div>
          <div className="grid gap-6 lg:grid-cols-2">
            <Side label="A" run={cmp.data.a} partner={partner} other={cmp.data.b} />
            <Side label="B" run={cmp.data.b} partner={partner} other={cmp.data.a} />
          </div>
        </div>
      )}
    </div>
  )
}
