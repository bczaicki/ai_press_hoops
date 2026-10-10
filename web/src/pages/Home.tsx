import { ChevronRight, Clapperboard, Clock3, Cpu, Map as MapIcon, MessageSquareText, ScanEye, Sparkles, Table2 } from 'lucide-react'
import type { ReactNode } from 'react'
import { Link } from 'react-router'

import { api, posterForClip } from '../api'
import type { ClipCard, RunSummary } from '../api/types'
import { Poster } from '../components/Poster'
import { UploadZone } from '../components/UploadZone'
import { Badge, EmptyState, ErrorPanel, Skeleton } from '../components/ui/misc'
import { Button } from '../components/ui/button'
import { fmtClock, fmtRelative, fmtUsd, footageLabel, plural } from '../lib/format'
import { useAsync } from '../lib/hooks'

const PERKS = [
  { icon: ScanEye, label: 'Annotated video' },
  { icon: Table2, label: 'Shot log + stats' },
  { icon: MessageSquareText, label: 'AI recap & Q&A' },
  { icon: MapIcon, label: 'Court map' },
]

export function Home() {
  const clips = useAsync(() => api.clips(), [])
  const runs = useAsync(() => api.runs(), [])
  const posterByClip = new Map((clips.data ?? []).map((c) => [c.clip_id, c.poster_url]))

  return (
    <div className="space-y-12">
      <section className="grid items-center gap-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)] lg:gap-10">
        <div className="min-w-0">
          <div className="eyebrow mb-2 text-flame-300">Tonight's film session</div>
          <h1 className="font-display text-[2.6rem] leading-[0.95] font-bold tracking-wide text-fg uppercase sm:text-6xl">
            Upload a clip. <span className="text-flame-500">Get the breakdown.</span>
          </h1>
          <p className="mt-3 max-w-xl text-base text-fg-2 sm:text-lg">
            Computer vision finds the players, ball, and rim. Fast System One models call each shot. An AI you choose explains what happened, and you can ask it anything.
          </p>
          <ul className="mt-4 flex flex-wrap gap-2">
            {PERKS.map(({ icon: Icon, label }) => (
              <li key={label} className="inline-flex items-center gap-1.5 rounded-full border border-line bg-white/[0.04] px-3 py-1.5 text-[0.8rem] font-semibold text-fg-2">
                <Icon className="size-3.5 text-flame-400" aria-hidden /> {label}
              </li>
            ))}
          </ul>
        </div>
        <div className="min-w-0">
          <UploadZone />
        </div>
      </section>

      <section aria-labelledby="runs-h">
        <SectionHead id="runs-h" title="Recent breakdowns" icon={<Sparkles className="size-5 text-flame-400" />} count={runs.data?.length} />
        {runs.error ? (
          <ErrorPanel message={runs.error} action={<Button size="sm" onClick={runs.reload}>Try again</Button>} />
        ) : runs.loading && !runs.data ? (
          <GridSkeleton />
        ) : runs.data?.length ? (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {runs.data.slice(0, 9).map((r) => (
              <RunTile key={r.run_id} run={r} poster={posterByClip.get(r.clip_id) ?? posterForClip(r.clip_id)} />
            ))}
          </div>
        ) : (
          <div className="glass rounded-[var(--radius-card)]">
            <EmptyState icon={<Cpu className="size-6" />} title="No breakdowns yet" body="Upload a clip above, pick what you want, and hit Analyze. Your results will show up here." />
          </div>
        )}
      </section>

      <section aria-labelledby="clips-h">
        <SectionHead id="clips-h" title="Your clips" icon={<Clapperboard className="size-5 text-flame-400" />} count={clips.data?.length} />
        {clips.error ? (
          <ErrorPanel message={clips.error} action={<Button size="sm" onClick={clips.reload}>Try again</Button>} />
        ) : clips.loading && !clips.data ? (
          <GridSkeleton />
        ) : clips.data?.length ? (
          <div className="grid grid-cols-2 gap-3 sm:gap-4 md:grid-cols-3 lg:grid-cols-4">
            {clips.data.map((c) => (
              <ClipTile key={c.clip_id} clip={c} />
            ))}
          </div>
        ) : (
          <div className="glass rounded-[var(--radius-card)]">
            <EmptyState icon={<Clapperboard className="size-6" />} title="No clips yet" body="Clips you upload are kept here so you can analyze them again with different settings." />
          </div>
        )}
      </section>
    </div>
  )
}

function SectionHead({ id, title, icon, count }: { id: string; title: string; icon: ReactNode; count?: number }) {
  return (
    <div className="mb-4 flex items-end justify-between gap-3">
      <h2 id={id} className="flex items-center gap-2 font-display text-2xl font-bold tracking-wide text-fg uppercase">
        {icon}
        {title}
      </h2>
      {count != null && count > 0 && <span className="text-sm text-fg-3">{plural(count, 'total', 'total')}</span>}
    </div>
  )
}

function GridSkeleton() {
  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {[0, 1, 2].map((i) => (
        <div key={i} className="glass overflow-hidden rounded-[var(--radius-card)]">
          <Skeleton className="aspect-video rounded-none" />
          <div className="space-y-2 p-4">
            <Skeleton className="h-4 w-2/3" />
            <Skeleton className="h-3 w-1/3" />
          </div>
        </div>
      ))}
    </div>
  )
}

function RunTile({ run, poster }: { run: RunSummary; poster: string }) {
  const statusTone = run.status === 'done' ? 'made' : run.status === 'running' ? 'flame' : run.status === 'failed' ? 'miss' : 'muted'
  return (
    <div className="min-w-0">
      <Link to={`/runs/${run.run_id}`} className="group glass block overflow-hidden rounded-[var(--radius-card)] transition hover:border-flame-500/40">
        <Poster src={poster} alt="">
          <div className="absolute top-3 left-3">
            <Badge tone={statusTone}>{run.status === 'done' ? 'Ready' : run.status === 'running' ? 'Analyzing' : run.status === 'failed' ? 'Failed' : run.status}</Badge>
          </div>
          <div className="absolute right-3 bottom-3 left-3 flex items-end justify-between gap-2">
            <div className="min-w-0">
              <div className="truncate font-semibold text-fg">{run.filename ?? run.clip_id}</div>
              <div className="text-xs text-fg-2">{fmtRelative(run.created_at)}</div>
            </div>
            <div className="text-right">
              <div className="numeral text-4xl text-fg">{run.n_events}</div>
              <div className="eyebrow text-[0.6rem]">events</div>
            </div>
          </div>
        </Poster>
        <div className="flex items-center gap-2 px-4 py-3 text-xs text-fg-2">
          <span className="truncate">{run.providers.length ? run.providers.join(' + ') : 'No AI'}</span>
          <span className="text-fg-3">·</span>
          <span className="tnum">{fmtUsd(run.cost_usd)}</span>
          <span className="text-fg-3">·</span>
          <span className="truncate">{footageLabel(run.footage_type)}</span>
          <ChevronRight className="ml-auto size-4 shrink-0 text-fg-3 transition group-hover:translate-x-0.5 group-hover:text-flame-400" aria-hidden />
        </div>
      </Link>
    </div>
  )
}

function ClipTile({ clip }: { clip: ClipCard }) {
  return (
    <div className="min-w-0">
      <Link to={`/clips/${clip.clip_id}`} className="group glass block overflow-hidden rounded-2xl transition hover:border-flame-500/40" aria-label={`Analyze ${clip.original_filename ?? 'clip'}`}>
        <Poster src={clip.poster_url} alt="">
          <span className="absolute right-2 bottom-2 inline-flex items-center gap-1 rounded-md bg-ink-950/80 px-1.5 py-0.5 text-[0.7rem] font-semibold text-fg tnum">
            <Clock3 className="size-3" aria-hidden />
            {fmtClock(clip.duration_s)}
          </span>
        </Poster>
        <div className="px-3 py-2.5">
          <div className="truncate text-sm font-semibold text-fg">{clip.original_filename ?? clip.clip_id}</div>
          <div className="truncate text-xs text-fg-3">
            {footageLabel(clip.footage_type)} · {fmtRelative(clip.created_at)}
          </div>
        </div>
      </Link>
    </div>
  )
}
