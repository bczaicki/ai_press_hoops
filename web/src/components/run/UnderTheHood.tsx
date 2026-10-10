import { AlertTriangle, Bot, ChevronDown, Cpu, HardDrive, Layers, ShieldCheck, Sparkles, type LucideIcon } from 'lucide-react'
import { useRef, useState, type KeyboardEvent, type ReactNode } from 'react'

import type { AppConfig, DecisionAnswer, DecisionRecord, HoopsEvent, RunView, ThresholdPreset } from '../../api/types'
import { keyTime } from '../../lib/events'
import { cn, fmtClock, fmtElapsed, fmtMs, fmtTokens, fmtUsd } from '../../lib/format'
import { usePlayer } from '../player/PlayerContext'
import { Badge, EmptyState } from '../ui/misc'

const PURPOSE: Record<DecisionRecord['purpose'], string> = {
  triage: 'Footage triage',
  event_typing: 'Shot call',
  escalation: 'Escalation',
  qa_route: 'Question routing',
  claim_check: 'Claim check',
}

const ROUTED: Record<string, { label: string; tone: 'neutral' | 'muted' | 'attempt' }> = {
  accept: { label: 'Auto-accepted', tone: 'neutral' },
  reject: { label: 'Auto-rejected', tone: 'muted' },
  escalate: { label: 'Escalated', tone: 'attempt' },
}

/** Emphasis bars: the chosen option is bright, the rest recede. One neutral hue (no categorical color). */
function ProbBars({ ans }: { ans: DecisionAnswer }) {
  const entries = Object.entries(ans.probabilities ?? {}).sort((a, b) => b[1] - a[1])
  const top = entries.slice(0, 5)
  const rest = entries.slice(5).reduce((s, [, v]) => s + v, 0)
  if (rest > 0.005) top.push(['other', rest])
  return (
    <ul className="space-y-1.5">
      {top.map(([k, v]) => {
        const chosen = String(ans.value) === k
        return (
          <li key={k} className="grid grid-cols-[7.5rem_minmax(0,1fr)_3rem] items-center gap-2 text-xs">
            <span className={cn('truncate', chosen ? 'font-semibold text-fg' : 'text-fg-3')}>{k.replace(/_/g, ' ')}</span>
            <span className="h-2 rounded-sm bg-white/[0.05]">
              <span className={cn('block h-full rounded-sm', chosen ? 'bg-fg' : 'bg-white/25')} style={{ width: `${Math.max(1, v * 100)}%` }} />
            </span>
            <span className={cn('text-right tnum', chosen ? 'text-fg' : 'text-fg-3')}>{Math.round(v * 100)}%</span>
          </li>
        )
      })}
    </ul>
  )
}

/** Noul answers return only p: show it against the accept / reject bars of the current preset. */
function NoulBar({ p, preset }: { p: number; preset: ThresholdPreset | undefined }) {
  const accept = preset?.accept ?? 0.85
  const reject = preset?.reject ?? 0.15
  const verdict = p >= accept ? 'past the accept bar' : p <= reject ? 'under the reject bar' : 'between the bars, so it escalates'
  return (
    <div>
      <div className="relative h-2.5 rounded-sm bg-white/[0.05]">
        <span className="absolute inset-y-0 left-0 rounded-sm bg-fg" style={{ width: `${p * 100}%` }} />
        <span className="absolute -inset-y-1 w-px bg-miss-text" style={{ left: `${reject * 100}%` }} aria-hidden />
        <span className="absolute -inset-y-1 w-px bg-made-text" style={{ left: `${accept * 100}%` }} aria-hidden />
      </div>
      <div className="mt-1 flex justify-between text-[0.7rem] text-fg-3 tnum">
        <span>
          p = <b className="text-fg">{p.toFixed(2)}</b>, {verdict}
        </span>
        <span>
          reject ≤ {reject} · accept ≥ {accept}
        </span>
      </div>
    </div>
  )
}

function DecisionCard({ d, events, preset }: { d: DecisionRecord; events: HoopsEvent[]; preset: ThresholdPreset | undefined }) {
  const player = usePlayer()
  const ev = d.subject_id ? events.find((e) => e.event_id === d.subject_id) : undefined
  const routed = d.routed ? ROUTED[d.routed] : null
  return (
    <li className="rounded-2xl border border-line bg-white/[0.02] p-4">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-semibold text-fg">{PURPOSE[d.purpose] ?? d.purpose}</span>
        {ev && (
          <button type="button" onClick={() => player.seek(Math.max(0, keyTime(ev) - 2), { play: true })} className="rounded-md bg-flame-500/15 px-1.5 py-0.5 font-mono text-xs font-semibold text-flame-300 hover:bg-flame-500/25">
            {fmtClock(keyTime(ev))}
          </button>
        )}
        {routed && <Badge tone={routed.tone}>{routed.label}</Badge>}
        {d.error && <Badge tone="miss">Error</Badge>}
        <span className="ml-auto text-xs text-fg-3 tnum">
          {d.decider} · {fmtMs(d.latency_ms)} · {fmtUsd(d.cost_usd ?? 0)}
        </span>
      </div>
      {d.error && <p className="mt-2 text-sm text-miss-text">{d.error}</p>}
      <div className="mt-3 space-y-3.5">
        {Object.entries(d.answers).map(([name, ans]) => (
          <div key={name}>
            <div className="mb-1.5 flex items-baseline justify-between gap-2 text-xs">
              <span className="text-fg-2">{d.questions[name]?.instructions ?? name.replace(/_/g, ' ')}</span>
              {ans.confidence != null && <span className="shrink-0 text-fg-3 tnum">conf {Math.round(ans.confidence * 100)}%</span>}
            </div>
            {ans.kind === 'noul' ? <NoulBar p={Number(ans.value)} preset={preset} /> : ans.probabilities ? <ProbBars ans={ans} /> : <span className="text-sm font-semibold text-fg">{String(ans.value)}</span>}
          </div>
        ))}
      </div>
      <details className="group mt-3">
        <summary className="inline-flex cursor-pointer list-none items-center gap-1 text-xs font-semibold text-fg-3 hover:text-fg-2">
          <ChevronDown className="size-3.5 transition-transform group-open:rotate-180" aria-hidden /> What System One saw
        </summary>
        {typeof d.state === 'string' ? (
          <p className="mt-2 rounded-lg bg-black/25 p-2.5 text-xs leading-relaxed text-fg-2">{d.state}</p>
        ) : (
          <dl className="mt-2 grid grid-cols-[minmax(0,10rem)_minmax(0,1fr)] gap-x-3 gap-y-1 rounded-lg bg-black/25 p-2.5 text-xs">
            {Object.entries(d.state).map(([k, v]) => (
              <div key={k} className="contents">
                <dt className="truncate text-fg-3">{k.replace(/_/g, ' ')}</dt>
                <dd className="break-words text-fg-2">{typeof v === 'string' ? v : JSON.stringify(v)}</dd>
              </div>
            ))}
          </dl>
        )}
      </details>
    </li>
  )
}

type TabKey = 'decisions' | 'ai' | 'pipeline' | 'privacy' | 'warnings'

function Tile({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div className="rounded-2xl border border-line bg-white/[0.02] p-3.5">
      <div className="eyebrow text-[0.65rem]">{label}</div>
      <div className="numeral mt-1 text-3xl text-fg">{value}</div>
      {sub && <div className="mt-0.5 text-xs text-fg-3">{sub}</div>}
    </div>
  )
}

export function UnderTheHood({ run, events, config, preset }: { run: RunView; events: HoopsEvent[]; config: AppConfig | undefined; preset: string }) {
  const [open, setOpen] = useState(false)
  const [tab, setTab] = useState<TabKey>('decisions')
  const tabRefs = useRef<(HTMLButtonElement | null)[]>([])
  const presetObj = config?.threshold_presets.find((p) => p.key === preset)
  const providerByKey = new Map((config?.providers ?? []).map((p) => [p.key, p]))
  const tabs: { key: TabKey; label: string; icon: LucideIcon; count: number }[] = [
    { key: 'decisions', label: 'System One', icon: Bot, count: run.decisions.length },
    { key: 'ai', label: 'AI calls', icon: Sparkles, count: run.vlm_calls.length },
    { key: 'pipeline', label: 'Pipeline', icon: Layers, count: run.pipeline.length },
    { key: 'privacy', label: 'Privacy', icon: ShieldCheck, count: run.data_egress.length },
    { key: 'warnings', label: 'Warnings', icon: AlertTriangle, count: run.warnings.length },
  ]
  const onTabKey = (e: KeyboardEvent, i: number) => {
    const n = e.key === 'ArrowRight' ? (i + 1) % tabs.length : e.key === 'ArrowLeft' ? (i - 1 + tabs.length) % tabs.length : null
    if (n == null) return
    e.preventDefault()
    setTab(tabs[n].key)
    tabRefs.current[n]?.focus()
  }
  const aiCost = run.vlm_calls.reduce((s, c) => s + (c.cost_usd ?? 0), 0)
  const s1Cost = run.decisions.reduce((s, d) => s + (d.cost_usd ?? 0), 0)

  return (
    <section aria-labelledby="hood-h" className="glass overflow-hidden rounded-[var(--radius-card)]">
      <button type="button" aria-expanded={open} aria-controls="hood" onClick={() => setOpen((o) => !o)} className="flex w-full items-center gap-3 px-4 py-4 text-left sm:px-5">
        <Cpu className="size-5 text-fg-3" aria-hidden />
        <span className="min-w-0 flex-1">
          <span id="hood-h" className="block font-display text-2xl font-bold tracking-wide text-fg uppercase">
            Under the hood
          </span>
          <span className="block truncate text-sm text-fg-3">
            {run.decisions.length} System One decisions · {run.vlm_calls.length} AI calls · {fmtUsd(run.metrics.cost_usd)} total · {fmtElapsed(run.metrics.wall_s)} wall time
          </span>
        </span>
        <ChevronDown className={cn('size-5 shrink-0 text-fg-3 transition-transform', open && 'rotate-180')} aria-hidden />
      </button>
      {open && (
        <div id="hood" className="border-t border-line">
          <div role="tablist" aria-label="Details" className="flex gap-1 overflow-x-auto px-3 pt-3 scrollbar-thin sm:px-4">
            {tabs.map((t, i) => (
              <button
                key={t.key}
                ref={(el) => {
                  tabRefs.current[i] = el
                }}
                role="tab"
                id={`tab-${t.key}`}
                aria-selected={tab === t.key}
                aria-controls={`panel-${t.key}`}
                tabIndex={tab === t.key ? 0 : -1}
                onKeyDown={(e) => onTabKey(e, i)}
                onClick={() => setTab(t.key)}
                className={cn('flex h-9 shrink-0 items-center gap-1.5 rounded-xl px-3 text-sm font-semibold transition', tab === t.key ? 'bg-white/[0.09] text-fg' : 'text-fg-3 hover:text-fg')}
              >
                <t.icon className="size-4" aria-hidden />
                {t.label}
                <span className="rounded-md bg-white/[0.07] px-1.5 text-[0.7rem] text-fg-2 tnum">{t.count}</span>
              </button>
            ))}
          </div>
          <div role="tabpanel" id={`panel-${tab}`} aria-labelledby={`tab-${tab}`} className="p-4 sm:p-5">
            {tab === 'decisions' &&
              (run.decisions.length ? (
                <>
                  <p className="mb-3 text-sm text-fg-2">
                    Each decision is a typed question answered by a fast System One model. Confident answers are accepted or rejected on the spot; anything between the bars ({presetObj ? `${presetObj.label}: accept ≥ ${presetObj.accept}, reject ≤ ${presetObj.reject}` : 'per preset'}) escalates to the AI or to you.
                  </p>
                  <ul className="grid gap-3 lg:grid-cols-2">
                    {run.decisions.map((d) => (
                      <DecisionCard key={d.decision_id} d={d} events={events} preset={presetObj} />
                    ))}
                  </ul>
                  <p className="mt-3 text-xs text-fg-3 tnum">System One total: {fmtUsd(s1Cost)}</p>
                </>
              ) : (
                <EmptyState icon={<Bot className="size-6" />} title="No decisions" body="No candidate shots reached System One in this clip." />
              ))}

            {tab === 'ai' &&
              (run.vlm_calls.length ? (
                <div className="overflow-x-auto scrollbar-thin">
                  <table className="w-full min-w-[40rem] text-left text-sm">
                    <thead className="text-xs text-fg-3">
                      <tr className="border-b border-line">
                        <th className="py-2 pr-3 font-semibold">Purpose</th>
                        <th className="py-2 pr-3 font-semibold">Model</th>
                        <th className="py-2 pr-3 text-right font-semibold">Images</th>
                        <th className="py-2 pr-3 text-right font-semibold">Tokens in / out</th>
                        <th className="py-2 pr-3 text-right font-semibold">Latency</th>
                        <th className="py-2 text-right font-semibold">Cost</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-line tnum">
                      {run.vlm_calls.map((c) => (
                        <tr key={c.call_id}>
                          <td className="py-2 pr-3 text-fg capitalize">{c.purpose.replace('_', ' ')}</td>
                          <td className="py-2 pr-3">
                            <span className="text-fg-2">{providerByKey.get(c.provider)?.label.replace(/\s*\(.*\)$/, '') ?? c.provider}</span> <span className="font-mono text-xs text-fg-3">{c.model}</span>
                            {c.error && <span className="block text-xs text-miss-text">{c.error}</span>}
                          </td>
                          <td className="py-2 pr-3 text-right text-fg-2">{c.video_seconds ? `${c.video_seconds.toFixed(0)} s video` : c.n_images}</td>
                          <td className="py-2 pr-3 text-right text-fg-2">
                            {fmtTokens(c.input_tokens)} / {fmtTokens(c.output_tokens)}
                          </td>
                          <td className="py-2 pr-3 text-right text-fg-2">{fmtMs(c.latency_ms)}</td>
                          <td className="py-2 text-right text-fg">{fmtUsd(c.cost_usd ?? 0)}</td>
                        </tr>
                      ))}
                    </tbody>
                    <tfoot className="tnum">
                      <tr className="border-t border-line-strong">
                        <td className="py-2 font-semibold text-fg" colSpan={3}>
                          Total
                        </td>
                        <td className="py-2 pr-3 text-right text-fg-2">
                          {fmtTokens(run.metrics.tokens_in)} / {fmtTokens(run.metrics.tokens_out)}
                        </td>
                        <td />
                        <td className="py-2 text-right font-semibold text-fg">{fmtUsd(aiCost)}</td>
                      </tr>
                    </tfoot>
                  </table>
                </div>
              ) : (
                <EmptyState icon={<Sparkles className="size-6" />} title="No AI calls" body="This run didn't use a vision-language model." />
              ))}

            {tab === 'pipeline' && (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                  <Tile label="Wall time" value={fmtElapsed(run.metrics.wall_s)} sub="start to finish" />
                  <Tile label="Detection" value={run.metrics.detect_fps ? `${run.metrics.detect_fps.toFixed(0)}` : '—'} sub="frames per second" />
                  <Tile label="Tokens" value={fmtTokens(run.metrics.tokens_in + run.metrics.tokens_out)} sub={`${fmtTokens(run.metrics.tokens_in)} in · ${fmtTokens(run.metrics.tokens_out)} out`} />
                  <Tile label="Cost" value={fmtUsd(run.metrics.cost_usd)} sub={run.metrics.cost_estimate_usd != null ? `estimated ${fmtUsd(run.metrics.cost_estimate_usd)}` : undefined} />
                </div>
                <ul className="divide-y divide-line rounded-2xl border border-line">
                  {run.pipeline.map((s, i) => (
                    <li key={`${s.stage}-${i}`} className="grid gap-1 px-4 py-2.5 text-sm sm:grid-cols-[8rem_minmax(0,1fr)_auto] sm:items-center sm:gap-3">
                      <span className="font-semibold text-fg capitalize">{s.stage}</span>
                      <span className="min-w-0">
                        <span className="font-mono text-xs text-fg-2">{s.impl}</span>
                        {s.version && <span className="ml-2 text-xs text-fg-3">{s.version}</span>}
                        {Object.keys(s.params).length > 0 && <span className="block truncate font-mono text-[0.7rem] text-fg-3">{JSON.stringify(s.params)}</span>}
                      </span>
                      {s.license && <Badge tone="muted">{s.license}</Badge>}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {tab === 'privacy' &&
              (run.data_egress.length ? (
                <ul className="space-y-2">
                  {run.data_egress.map((k) => {
                    const p = providerByKey.get(k)
                    return (
                      <li key={k} className="flex items-start gap-3 rounded-2xl border border-line bg-white/[0.02] p-3.5">
                        {p?.local ? <HardDrive className="mt-0.5 size-5 text-made-text" aria-hidden /> : <ShieldCheck className="mt-0.5 size-5 text-flame-400" aria-hidden />}
                        <div>
                          <div className="font-semibold text-fg">{p?.local ? 'This computer' : (p?.sends_data_to ?? k)}</div>
                          <div className="text-sm text-fg-2">{p?.local ? `${p.label} ran locally. Nothing left this machine.` : `Sampled frames from this clip were sent to ${p?.label ?? k} for review and the recap.`}</div>
                        </div>
                      </li>
                    )
                  })}
                </ul>
              ) : (
                <EmptyState icon={<HardDrive className="size-6" />} title="Nothing left this computer" body="No frames or video were sent to any outside service for this run." />
              ))}

            {tab === 'warnings' &&
              (run.warnings.length ? (
                <ul className="space-y-2">
                  {run.warnings.map((w, i) => (
                    <li key={i} className="flex items-start gap-2.5 rounded-xl border border-attempt/25 bg-attempt/[0.05] p-3 text-sm text-fg-2">
                      <AlertTriangle className="mt-0.5 size-4 shrink-0 text-attempt" aria-hidden />
                      {w}
                    </li>
                  ))}
                </ul>
              ) : (
                <EmptyState icon={<ShieldCheck className="size-6" />} title="No warnings" body="Everything ran cleanly." />
              ))}
          </div>
        </div>
      )}
    </section>
  )
}
