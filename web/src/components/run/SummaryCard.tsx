import { AlertTriangle, Info, RotateCcw, Sparkles } from 'lucide-react'

import type { RunView } from '../../api/types'
import { toSummaryView } from '../../lib/events'
import { fmtClock } from '../../lib/format'
import { Markdown } from '../../lib/markdown'
import { usePlayer } from '../player/PlayerContext'
import { ButtonLink } from '../ui/button'
import { Badge, EmptyState } from '../ui/misc'
import { Tip } from '../ui/tip'

function producerLabel(producer: string): string {
  // "anthropic:claude-opus-5-5" -> "claude-opus-5-5"
  const parts = producer.split(':')
  return parts.length > 1 ? parts.slice(1).join(':') : producer
}

export function SummaryCard({ run }: { run: RunView }) {
  const player = usePlayer()
  const narrative = run.narratives.find((n) => n.kind !== 'qa') ?? null
  const seek = (t: number) => player.seek(Math.max(0, t - 2), { play: true })

  if (!narrative) {
    const requested = run.outputs_requested.includes('summary')
    return (
      <section className="glass flex h-full flex-col rounded-[var(--radius-card)]" aria-label="AI recap">
        <EmptyState
          className="my-auto"
          icon={<Sparkles className="size-6" />}
          title={requested ? "The recap didn't come through" : 'No AI recap this time'}
          body={requested ? 'The AI step failed or was skipped. Check Under the hood for details.' : 'This run used computer vision and System One only. Run it again with an AI provider to get a written recap and Q&A.'}
          action={
            <ButtonLink to={`/clips/${encodeURIComponent(run.clip.clip_id)}`} size="sm" icon={<RotateCcw className="size-3.5" />}>
              Run again with AI
            </ButtonLink>
          }
        />
      </section>
    )
  }

  const view = toSummaryView(narrative, run.events)
  return (
    <section className="glass flex h-full min-h-0 flex-col overflow-hidden rounded-[var(--radius-card)]" aria-labelledby="recap-h">
      <div className="relative px-5 pt-5 pb-1">
        <div aria-hidden className="pointer-events-none absolute inset-x-0 top-0 h-28 bg-[radial-gradient(70%_100%_at_0%_0%,rgb(255_122_26/0.14),transparent)]" />
        <div className="relative flex flex-wrap items-center gap-2">
          <span className="eyebrow flex items-center gap-1.5 text-flame-300">
            <Sparkles className="size-3.5" aria-hidden /> AI recap
          </span>
          <Badge tone="muted" className="font-mono">
            {producerLabel(narrative.provenance.producer)}
          </Badge>
          {view.unsupported.length > 0 && (
            <Tip content="System One checked each sentence against the event log. Underlined sentences couldn't be matched.">
              <span tabIndex={0} className="outline-none">
                <Badge tone="attempt" icon={<AlertTriangle className="size-3" aria-hidden />}>
                  {view.unsupported.length} unverified
                </Badge>
              </span>
            </Tip>
          )}
        </div>
        {view.headline && (
          <h2 id="recap-h" className="relative mt-2 font-display text-[1.7rem] leading-[1.05] font-bold tracking-wide text-fg uppercase sm:text-3xl">
            {view.headline}
          </h2>
        )}
      </div>
      <div className="min-h-0 flex-1 space-y-5 overflow-y-auto px-5 pt-3 pb-8 scrollbar-thin [mask-image:linear-gradient(to_bottom,black_calc(100%-2.5rem),transparent)]">
        <Markdown text={view.body} onTime={seek} flags={view.unsupported} />

        {view.keyMoments.length > 0 && (
          <div>
            <div className="eyebrow mb-2">Key moments</div>
            <ul className="space-y-1.5">
              {view.keyMoments.map((k, i) => (
                <li key={i}>
                  <button
                    type="button"
                    disabled={k.t_s == null}
                    onClick={() => k.t_s != null && seek(k.t_s)}
                    className="group flex w-full items-center gap-3 rounded-xl border border-line bg-white/[0.025] px-3 py-2 text-left transition hover:border-flame-500/50 hover:bg-flame-500/[0.06] disabled:cursor-default disabled:hover:border-line disabled:hover:bg-white/[0.025]"
                  >
                    <span className="rounded-md bg-flame-500/15 px-1.5 py-0.5 font-mono text-xs font-semibold text-flame-300 tnum group-hover:bg-flame-500/25">{k.t_s != null ? fmtClock(k.t_s) : k.ref}</span>
                    <span className="min-w-0 flex-1 text-sm text-fg-2 group-hover:text-fg">{k.description}</span>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}

        {view.caveats.length > 0 && (
          <div className="rounded-xl border border-line bg-white/[0.02] p-3">
            <div className="eyebrow mb-1.5 flex items-center gap-1.5">
              <Info className="size-3.5" aria-hidden /> Caveats
            </div>
            <ul className="space-y-1 text-[0.82rem] text-fg-2">
              {view.caveats.map((c, i) => (
                <li key={i}>{c}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </section>
  )
}
