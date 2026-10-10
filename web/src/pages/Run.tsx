import { ArrowLeft, Clock3, Download, GitCompareArrows, MessageCircleQuestion, RotateCcw, Sparkles } from 'lucide-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate, useParams, useSearchParams } from 'react-router'

import { api, errorMessage, isMock, isMockMedia, posterForClip } from '../api'
import { useJob } from '../api/jobs'
import type { AddableEventType, HoopsEvent, RunView, TeamSummary } from '../api/types'
import { useConfig } from '../components/ConfigContext'
import { PlayerProvider } from '../components/player/PlayerContext'
import { VideoPlayer, type Source } from '../components/player/VideoPlayer'
import { AskPanel } from '../components/run/AskPanel'
import { CourtMap } from '../components/run/CourtMap'
import { EventLog, type LogFilter } from '../components/run/EventLog'
import { JobProgress } from '../components/run/JobProgress'
import { Scoreboard } from '../components/run/Scoreboard'
import { SummaryCard } from '../components/run/SummaryCard'
import { UnderTheHood } from '../components/run/UnderTheHood'
import { Button, ButtonLink, buttonClass } from '../components/ui/button'
import { Badge, ErrorPanel, Skeleton } from '../components/ui/misc'
import { useToast } from '../components/ui/toast'
import { eventTitle } from '../lib/events'
import { fmtDateTime, fmtUsd, footageLabel } from '../lib/format'
import { useAsync } from '../lib/hooks'
import { eventsToVtt } from '../lib/vtt'

function initialPreset(run: RunView): string {
  const fromPipeline = run.pipeline.find((s) => s.stage === 'decide')?.params?.preset
  if (typeof fromPipeline === 'string') return fromPipeline
  const counts = new Map<string, number>()
  for (const e of run.events) {
    const p = e.features?.preset
    if (typeof p === 'string') counts.set(p, (counts.get(p) ?? 0) + 1)
  }
  const top = [...counts.entries()].sort((a, b) => b[1] - a[1])[0]
  return top?.[0] ?? 'balanced'
}

export default function RunPage() {
  const { runId = '' } = useParams()
  const [params] = useSearchParams()
  const navigate = useNavigate()
  const toast = useToast()
  const { reload: reloadConfig } = useConfig()
  const jobId = params.get('job')
  const { job, transport, error: jobError } = useJob(jobId)
  const jobActive = !!jobId && !jobError && (!job || job.status === 'running' || job.status === 'queued')
  const [nonce, setNonce] = useState(0)
  const run = useAsync(() => api.run(runId), [runId, nonce])
  const announced = useRef(false)

  // When the job finishes, load the full run and drop ?job (jobs live in server memory only).
  useEffect(() => {
    if (job?.status !== 'done' || announced.current) return
    announced.current = true
    setNonce((n) => n + 1)
    reloadConfig()
    toast({ title: 'Breakdown ready', body: 'Scroll down for the shot log, recap, and court map.', tone: 'success' })
    navigate(`/runs/${encodeURIComponent(runId)}`, { replace: true })
  }, [job?.status, runId, navigate, reloadConfig, toast])

  // A run that's still "running" without a job to follow: poll it.
  const stillRunning = !jobId && run.data?.status === 'running'
  useEffect(() => {
    if (!stillRunning) return
    const t = setInterval(() => setNonce((n) => n + 1), 4000)
    return () => clearInterval(t)
  }, [stillRunning])

  const title = run.data?.clip.original_filename ?? 'Your breakdown'

  if (jobId && (jobActive || job?.status === 'failed' || job?.status === 'cancelled')) {
    return (
      <div className="space-y-4">
        <Link to="/" className="inline-flex items-center gap-1.5 text-sm font-semibold text-fg-2 hover:text-fg">
          <ArrowLeft className="size-4" aria-hidden /> Home
        </Link>
        <JobProgress
          job={job}
          transport={transport}
          title={title}
          onCancel={async () => {
            try {
              await api.cancelJob(jobId)
            } catch (err) {
              toast({ title: "Couldn't cancel", body: errorMessage(err), tone: 'error' })
            }
          }}
        />
        {job?.status === 'failed' && (
          <ErrorPanel
            title="The analysis stopped"
            message={job.error ?? 'Something went wrong on the server.'}
            action={
              job.clip_id && (
                <ButtonLink to={`/clips/${encodeURIComponent(job.clip_id)}`} size="sm" icon={<RotateCcw className="size-3.5" />}>
                  Try different settings
                </ButtonLink>
              )
            }
          />
        )}
        {job?.status === 'cancelled' && job.clip_id && (
          <ButtonLink to={`/clips/${encodeURIComponent(job.clip_id)}`} variant="primary" icon={<RotateCcw className="size-4" />}>
            Start over with this clip
          </ButtonLink>
        )}
      </div>
    )
  }

  if (run.error && !run.data) {
    return (
      <div className="mx-auto max-w-xl py-10">
        <ErrorPanel
          title={run.status === 404 ? 'Run not found' : "Couldn't load this breakdown"}
          message={run.status === 404 ? 'It may still be waiting in line, or the link is wrong.' : run.error}
          action={
            <div className="flex gap-2">
              <Button size="sm" onClick={run.reload}>
                Try again
              </Button>
              <ButtonLink size="sm" variant="ghost" to="/">
                Home
              </ButtonLink>
            </div>
          }
        />
        {jobError && <p className="mt-3 text-sm text-fg-3">Live progress isn't available: {jobError}</p>}
      </div>
    )
  }

  if (!run.data) return <DashboardSkeleton />

  return (
    <PlayerProvider>
      <Dashboard key={run.data.run_id} run={run.data} stillRunning={stillRunning} onReloadConfig={reloadConfig} />
    </PlayerProvider>
  )
}

function DashboardSkeleton() {
  return (
    <div className="space-y-5" aria-busy>
      <Skeleton className="h-12 w-80" />
      <Skeleton className="h-28" />
      <div className="grid gap-5 lg:grid-cols-12">
        <Skeleton className="aspect-video lg:col-span-8" />
        <Skeleton className="h-80 lg:col-span-4 lg:h-auto" />
      </div>
    </div>
  )
}

function Dashboard({ run, stillRunning, onReloadConfig }: { run: RunView; stillRunning: boolean; onReloadConfig: () => void }) {
  const toast = useToast()
  const { config } = useConfig()
  const [events, setEvents] = useState<HoopsEvent[]>(run.events)
  const [summaries, setSummaries] = useState<TeamSummary[]>(run.team_summaries)
  const [preset, setPreset] = useState(() => initialPreset(run))
  const [recomputing, setRecomputing] = useState(false)
  const [changed, setChanged] = useState<Set<string>>(new Set())
  const [busyIds, setBusyIds] = useState<Set<string>>(new Set())
  const [filter, setFilter] = useState<LogFilter>('all')
  const [askOpen, setAskOpen] = useState(false)
  const logRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    setEvents(run.events)
    setSummaries(run.team_summaries)
  }, [run])

  const flash = (ids: string[]) => {
    setChanged(new Set(ids))
    setTimeout(() => setChanged(new Set()), 2600)
  }

  const refreshStats = useCallback(async () => {
    try {
      const fresh = await api.run(run.run_id)
      setSummaries(fresh.team_summaries)
    } catch {
      /* stats refresh is best-effort */
    }
  }, [run.run_id])

  const onPreset = async (k: string) => {
    if (k === preset) return
    const before = new Map(events.map((e) => [e.event_id, e]))
    setRecomputing(true)
    try {
      const res = await api.recompute(run.run_id, k)
      const diff = res.events.filter((e) => {
        const b = before.get(e.event_id)
        return !b || b.type !== e.type || b.status !== e.status
      })
      setEvents(res.events)
      setSummaries(res.team_summaries)
      setPreset(k)
      flash(diff.map((e) => e.event_id))
      const label = config?.threshold_presets.find((p) => p.key === k)?.label ?? k
      toast({
        title: `${label} applied`,
        body: diff.length ? `${diff.length} call${diff.length === 1 ? '' : 's'} changed: ${diff.slice(0, 3).map(eventTitle).join(', ')}${diff.length > 3 ? '…' : ''}` : 'No calls changed. Every call was already clear of both bars.',
        tone: 'success',
      })
    } catch (err) {
      toast({ title: "Couldn't re-apply thresholds", body: errorMessage(err), tone: 'error' })
    } finally {
      setRecomputing(false)
    }
  }

  const onReview = async (id: string, action: 'accept' | 'reject') => {
    setBusyIds((s) => new Set(s).add(id))
    try {
      const ev = await api.review(run.run_id, id, { action, note: null })
      setEvents((es) => es.map((e) => (e.event_id === id ? ev : e)))
      flash([id])
      void refreshStats()
      toast({ title: action === 'accept' ? 'Call accepted' : 'Call rejected', body: `${eventTitle(ev)}. Stats updated.`, tone: 'success' })
    } catch (err) {
      toast({ title: "Couldn't save your review", body: errorMessage(err), tone: 'error' })
    } finally {
      setBusyIds((s) => {
        const n = new Set(s)
        n.delete(id)
        return n
      })
    }
  }

  const onAdd = async (t: number, type: AddableEventType, teamId: string | null) => {
    try {
      const ev = await api.addEvent(run.run_id, { t_s: Math.round(t * 100) / 100, type, team_id: teamId })
      setEvents((es) => [...es.filter((e) => e.event_id !== ev.event_id), ev].sort((a, b) => a.t_start_s - b.t_start_s))
      flash([ev.event_id])
      void refreshStats()
      toast({ title: 'Shot added', body: `${eventTitle(ev)}. Thanks, that helps tune the detector.`, tone: 'success' })
    } catch (err) {
      toast({ title: "Couldn't add that shot", body: errorMessage(err), tone: 'error' })
      throw err
    }
  }

  // Captions from the current events (the server's events.vtt isn't regenerated after edits).
  const captionsUrl = useMemo(() => {
    if (!events.length) return run.urls.captions
    return URL.createObjectURL(new Blob([eventsToVtt(events, run.teams)], { type: 'text/vtt' }))
  }, [events, run.teams, run.urls.captions])
  useEffect(() => () => {
    if (captionsUrl?.startsWith('blob:')) URL.revokeObjectURL(captionsUrl)
  }, [captionsUrl])

  const sources: Source[] = [
    ...(run.urls.annotated ? [{ key: 'annotated', label: 'Annotated', url: run.urls.annotated }] : []),
    { key: 'original', label: 'Original', url: run.urls.proxy || run.urls.master },
  ]
  const poster = run.urls.poster && !isMockMedia(run.urls.poster) ? run.urls.poster : posterForClip(run.clip.clip_id)
  const provider = run.vlm_calls[0]
  const decider = run.pipeline.find((s) => s.stage === 'decide')?.impl

  const download = () => {
    if (!isMock) return
    const blob = new Blob([JSON.stringify({ ...run, events, team_summaries: summaries }, null, 2)], { type: 'application/json' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `run-${run.run_id}.json`
    a.click()
    setTimeout(() => URL.revokeObjectURL(a.href), 2000)
  }

  return (
    <div className="space-y-5 pb-20 md:pb-0">
      {/* header */}
      <div className="flex flex-wrap items-end gap-x-6 gap-y-3">
        <div className="min-w-0 flex-1">
          <Link to="/" className="mb-1.5 inline-flex items-center gap-1.5 text-sm font-semibold text-fg-2 hover:text-fg">
            <ArrowLeft className="size-4" aria-hidden /> Home
          </Link>
          <h1 className="truncate font-display text-4xl leading-none font-bold tracking-wide text-fg uppercase sm:text-5xl">{run.clip.original_filename ?? run.run_id}</h1>
          <div className="mt-2 flex flex-wrap items-center gap-1.5">
            <Badge icon={<Clock3 className="size-3" aria-hidden />}>{fmtDateTime(run.created_at)}</Badge>
            <Badge>{footageLabel(run.clip.footage_type)}</Badge>
            {decider && <Badge>System One: {decider}</Badge>}
            <Badge tone={provider ? 'flame' : 'muted'} icon={<Sparkles className="size-3" aria-hidden />}>
              {provider ? provider.model : 'No AI'}
            </Badge>
            <Badge tone="muted">{fmtUsd(run.metrics.cost_usd)}</Badge>
            {stillRunning && <Badge tone="attempt">Still running or interrupted</Badge>}
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button variant="primary" icon={<MessageCircleQuestion className="size-4" />} onClick={() => setAskOpen(true)}>
            Ask the AI
          </Button>
          {isMock ? (
            <Button icon={<Download className="size-4" />} onClick={download}>
              Download
            </Button>
          ) : (
            <a href={run.urls.download} download className={buttonClass('secondary', 'md')}>
              <Download className="size-4" aria-hidden /> Download
            </a>
          )}
          <ButtonLink to={`/compare?a=${encodeURIComponent(run.run_id)}`} icon={<GitCompareArrows className="size-4" />}>
            Compare
          </ButtonLink>
          <ButtonLink to={`/clips/${encodeURIComponent(run.clip.clip_id)}`} variant="ghost" icon={<RotateCcw className="size-4" />}>
            Re-run
          </ButtonLink>
        </div>
      </div>

      {stillRunning && (
        <ErrorPanel
          title="This breakdown hasn't finished"
          message="It may still be running, or it stopped when the server restarted (unfinished runs don't resume). This page checks again every few seconds. You can also run it again."
        />
      )}

      <Scoreboard
        teams={run.teams}
        summaries={summaries}
        events={events}
        onReview={() => {
          setFilter('review')
          logRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
        }}
      />

      <div className="grid gap-5 lg:grid-cols-12">
        <div className="min-w-0 lg:col-span-8">
          <VideoPlayer sources={sources} captionsUrl={captionsUrl} poster={poster} fallbackDuration={run.clip.proxy.duration_s || run.clip.master.duration_s} events={events} teams={run.teams} />
        </div>
        <div className="relative min-h-[22rem] min-w-0 lg:col-span-4">
          <div className="lg:absolute lg:inset-0">
            <SummaryCard run={{ ...run, events }} />
          </div>
        </div>
      </div>

      <div className="grid gap-5 lg:grid-cols-12">
        <div className="min-w-0 lg:col-span-7">
          <EventLog
            ref={logRef}
            events={events}
            teams={run.teams}
            presets={config?.threshold_presets ?? []}
            preset={preset}
            recomputing={recomputing}
            changed={changed}
            filter={filter}
            onFilter={setFilter}
            onPreset={(k) => void onPreset(k)}
            onReview={onReview}
            onAdd={onAdd}
            busyIds={busyIds}
          />
        </div>
        <div className="min-w-0 lg:col-span-5">
          <div className="lg:sticky lg:top-24">
            <CourtMap run={run} events={events} />
          </div>
        </div>
      </div>

      <UnderTheHood run={{ ...run, events }} events={events} config={config} preset={preset} />

      {/* phone: floating Ask */}
      <button
        type="button"
        onClick={() => setAskOpen(true)}
        className="fixed right-4 bottom-[max(1rem,env(safe-area-inset-bottom))] z-30 inline-flex h-14 items-center gap-2 rounded-full bg-flame-500 px-5 font-display text-lg font-bold tracking-wide text-ink-950 uppercase shadow-[0_16px_40px_-10px_rgb(255_122_26/0.8)] md:hidden"
      >
        <MessageCircleQuestion className="size-5" aria-hidden /> Ask
      </button>

      <AskPanel open={askOpen} onOpenChange={setAskOpen} run={{ ...run, events }} config={config} onSpent={onReloadConfig} />
    </div>
  )
}
