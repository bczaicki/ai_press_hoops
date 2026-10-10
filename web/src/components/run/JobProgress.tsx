import { AnimatePresence, motion } from 'framer-motion'
import { Check, CircleSlash, Loader2, Radio, RefreshCw, X } from 'lucide-react'
import { useEffect, useState } from 'react'

import type { Transport } from '../../api/jobs'
import { STAGES, type Job, type JobStage } from '../../api/types'
import { cn, fmtElapsed } from '../../lib/format'
import { useNow } from '../../lib/hooks'
import { Button } from '../ui/button'
import { Card } from '../ui/misc'

/** Plain-English "what's happening" copy for the long waits. */
const EXPLAIN: Record<string, string> = {
  ingest: 'Converting your video to a browser-friendly format and making a lighter 720p copy to analyze.',
  detect: 'A detection model looks at the frames and boxes every player, the ball, and the rim. This is the slowest step.',
  track: 'Detections are linked frame to frame so each player keeps one ID, then split into teams by jersey color.',
  shots: 'The ball path near the rim is checked for anything that looks like a shot.',
  decide: 'A fast System One model gives each candidate a typed verdict with a confidence. Sure calls go straight in.',
  adjudicate: 'Close calls get a second look: the AI sees a few frames around each one and says made, missed, or unclear.',
  summarize: 'The AI writes the recap from the event log, then every sentence is checked against what was actually detected.',
  court: 'Shot spots are projected onto a court diagram.',
  render: 'Drawing the overlays onto your video and encoding it so every browser can play it.',
}

function normalizeStages(job: Job): JobStage[] {
  if (job.stages?.length) return job.stages
  // Older servers may omit stages: derive them from `stage`.
  const idx = STAGES.findIndex((s) => s.key === job.stage)
  return STAGES.map((s, i) => ({ key: s.key, label: s.label, status: i < idx ? 'done' : i === idx ? 'running' : 'pending', progress: i < idx ? 1 : i === idx ? job.progress : 0 }))
}

function StageIcon({ status, progress }: { status: JobStage['status']; progress: number }) {
  if (status === 'done')
    return (
      <span className="grid size-7 place-items-center rounded-full bg-flame-500 text-ink-950">
        <Check className="size-4" strokeWidth={3.5} aria-hidden />
      </span>
    )
  if (status === 'running') {
    const r = 12
    const c = 2 * Math.PI * r
    return (
      <span className="relative grid size-7 place-items-center">
        <svg viewBox="0 0 28 28" className="absolute inset-0 -rotate-90" aria-hidden>
          <circle cx="14" cy="14" r={r} fill="none" stroke="rgb(255 255 255 / 0.12)" strokeWidth="2.5" />
          <circle cx="14" cy="14" r={r} fill="none" stroke="var(--color-flame-400)" strokeWidth="2.5" strokeLinecap="round" strokeDasharray={c} strokeDashoffset={c * (1 - Math.max(0.04, Math.min(1, progress)))} className="transition-[stroke-dashoffset] duration-500" />
        </svg>
        <span className="size-2 animate-pulse rounded-full bg-flame-400" />
      </span>
    )
  }
  if (status === 'failed')
    return (
      <span className="grid size-7 place-items-center rounded-full bg-miss text-white">
        <X className="size-4" strokeWidth={3} aria-hidden />
      </span>
    )
  if (status === 'skipped' || status === 'cancelled')
    return (
      <span className="grid size-7 place-items-center rounded-full border border-dashed border-fg-3/60 text-fg-3">
        <CircleSlash className="size-3.5" aria-hidden />
      </span>
    )
  return <span className="size-7 rounded-full border-2 border-white/15" aria-hidden />
}

export function JobProgress({
  job,
  transport,
  title,
  onCancel,
}: {
  job: Job | null
  transport: Transport | null
  title: string
  onCancel: () => Promise<void>
}) {
  const running = !job || job.status === 'running' || job.status === 'queued'
  const now = useNow(1000, running)
  const [confirm, setConfirm] = useState(false)
  const [cancelling, setCancelling] = useState(false)
  useEffect(() => {
    if (!confirm) return
    const t = setTimeout(() => setConfirm(false), 3500)
    return () => clearTimeout(t)
  }, [confirm])

  const started = job?.started_at ? new Date(job.started_at).getTime() : null
  const ended = job?.finished_at ? new Date(job.finished_at).getTime() : null
  const elapsed = started ? ((ended ?? now) - started) / 1000 : 0
  const p = job?.progress ?? 0
  const eta = p > 0.06 && running ? (elapsed * (1 - p)) / p : null
  const stages = job ? normalizeStages(job) : STAGES.map((s) => ({ ...s, status: 'pending' as const, progress: 0 }))
  const current = stages.find((s) => s.status === 'running')

  return (
    <Card className="overflow-hidden">
      <div className="relative border-b border-line px-5 pt-5 pb-4 sm:px-7">
        <div aria-hidden className="pointer-events-none absolute inset-x-0 top-0 h-28 bg-[radial-gradient(50%_100%_at_30%_0%,rgb(255_122_26/0.16),transparent)]" />
        <div className="relative flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="eyebrow flex items-center gap-2 text-flame-300">
              {running ? (
                <>
                  <span className="relative flex size-2">
                    <span className="absolute inset-0 animate-ping rounded-full bg-flame-400 opacity-75" />
                    <span className="relative size-2 rounded-full bg-flame-500" />
                  </span>
                  {job?.status === 'queued' ? 'Waiting in line' : 'Analyzing'}
                </>
              ) : job?.status === 'failed' ? (
                'Analysis failed'
              ) : job?.status === 'cancelled' ? (
                'Cancelled'
              ) : (
                'Done'
              )}
            </div>
            <h1 className="mt-1 truncate font-display text-3xl font-bold tracking-wide text-fg uppercase sm:text-4xl">{title}</h1>
          </div>
          <div className="text-right">
            <div className="numeral text-5xl text-fg tnum">{fmtElapsed(elapsed)}</div>
            <div className="text-xs text-fg-3">{eta != null ? `about ${eta < 60 ? `${Math.max(5, Math.round(eta / 5) * 5)} s` : `${Math.round(eta / 60)} min`} left` : 'elapsed'}</div>
          </div>
        </div>
        <div className="relative mt-4 h-2.5 overflow-hidden rounded-full bg-white/[0.07]" role="progressbar" aria-label="Overall progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(p * 100)}>
          <motion.div className="h-full rounded-full bg-[linear-gradient(90deg,var(--color-flame-600),var(--color-flame-400))]" animate={{ width: `${Math.max(2, p * 100)}%` }} transition={{ ease: 'easeOut', duration: 0.5 }} />
        </div>
        <div className="relative mt-2 flex flex-wrap items-center justify-between gap-2 text-sm">
          <p className="min-w-0 text-fg-2" aria-live="polite">
            <span className="font-semibold text-fg tnum">{Math.floor(p * 100)}%</span> · {job?.message ?? 'Connecting…'}
          </p>
          <span className="inline-flex items-center gap-1.5 text-xs text-fg-3" title={transport === 'poll' ? 'Live updates were blocked, so this page checks every second.' : 'Live updates'}>
            {transport === 'poll' ? <RefreshCw className="size-3" aria-hidden /> : <Radio className="size-3" aria-hidden />}
            {transport === 'poll' ? 'Checking every second' : transport === 'sse' ? 'Live' : 'Connecting'}
          </span>
        </div>
      </div>

      <div className="grid gap-6 px-5 py-5 sm:px-7 lg:grid-cols-[minmax(0,1fr)_18rem]">
        <ol className="space-y-1" aria-label="Pipeline stages">
          {stages.map((s, i) => {
            const isLast = i === stages.length - 1
            const active = s.status === 'running'
            return (
              <li key={s.key} className="relative flex gap-3.5" aria-current={active ? 'step' : undefined}>
                {!isLast && <span aria-hidden className={cn('absolute top-8 bottom-[-0.25rem] left-[13px] w-0.5 rounded-full', s.status === 'done' ? 'bg-flame-500/60' : 'bg-white/10')} />}
                <StageIcon status={s.status} progress={s.progress} />
                <div className={cn('min-w-0 flex-1 pb-3', s.status === 'skipped' && 'opacity-55')}>
                  <div className="flex items-baseline justify-between gap-3">
                    <span className={cn('font-semibold', active ? 'text-fg' : s.status === 'done' ? 'text-fg-2' : 'text-fg-3')}>{s.label}</span>
                    <span className="shrink-0 text-xs text-fg-3 tnum">
                      {s.status === 'running' ? `${Math.round(s.progress * 100)}%` : s.status === 'skipped' ? 'skipped' : s.status === 'done' ? 'done' : s.status === 'failed' ? 'failed' : ''}
                    </span>
                  </div>
                  <AnimatePresence initial={false}>
                    {active && (
                      <motion.div initial={{ height: 0 }} animate={{ height: 'auto' }} exit={{ height: 0 }} className="overflow-hidden">
                        <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-flame-500/15">
                          <div className="h-full rounded-full bg-flame-500 transition-[width] duration-500" style={{ width: `${Math.max(3, s.progress * 100)}%` }} />
                        </div>
                        {job?.message && <p className="mt-1.5 text-sm text-fg-2">{job.message}</p>}
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              </li>
            )
          })}
        </ol>

        <aside className="space-y-4">
          <div className="rounded-2xl border border-line bg-white/[0.03] p-4">
            <div className="eyebrow mb-1.5">What's happening</div>
            <p className="text-sm leading-relaxed text-fg-2">{current ? EXPLAIN[current.key] ?? current.label : running ? 'Getting started…' : 'Wrapping up.'}</p>
          </div>
          {running && (
            <Button
              variant={confirm ? 'danger' : 'secondary'}
              className="w-full"
              loading={cancelling}
              icon={!cancelling && (confirm ? <X className="size-4" /> : <Loader2 className="size-4" />)}
              onClick={async () => {
                if (!confirm) return setConfirm(true)
                setCancelling(true)
                try {
                  await onCancel()
                } finally {
                  setCancelling(false)
                  setConfirm(false)
                }
              }}
            >
              {confirm ? 'Tap again to cancel' : 'Cancel analysis'}
            </Button>
          )}
          <p className="text-xs text-fg-3">You can leave this page. The analysis keeps going, and it will show up under Recent breakdowns.</p>
        </aside>
      </div>
    </Card>
  )
}
