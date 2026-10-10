import { AnimatePresence, motion } from 'framer-motion'
import { ArrowUp, Bot, RotateCcw, Sparkles, Trash2 } from 'lucide-react'
import { useEffect, useRef, useState, type KeyboardEvent } from 'react'

import { api, errorMessage } from '../../api'
import type { Answer, AnswerRoute, AppConfig, ChatTurn, RunView } from '../../api/types'
import { keyTime, outcomeOf } from '../../lib/events'
import { cn, fmtClock, fmtMs, fmtUsd } from '../../lib/format'
import { useMediaQuery } from '../../lib/hooks'
import { Markdown } from '../../lib/markdown'
import { usePlayer } from '../player/PlayerContext'
import { Select } from '../ui/misc'
import { Sheet } from '../ui/sheet'

type Turn = { role: 'user'; text: string } | { role: 'assistant'; answer: Answer } | { role: 'error'; text: string; question: string }

const ROUTE: Record<AnswerRoute, string> = {
  stats: 'Counted from the event log',
  find_moments: 'Found in the event log',
  explain_play: 'AI watched the play',
  summary: 'From the recap',
  unsupported: "Outside what this clip can answer",
}

function storageKey(runId: string) {
  return `hoops_ask_${runId}`
}

function load(runId: string): Turn[] {
  try {
    const raw = sessionStorage.getItem(storageKey(runId))
    return raw ? (JSON.parse(raw) as Turn[]) : []
  } catch {
    return []
  }
}

function suggestions(run: RunView): string[] {
  const out: string[] = []
  const shots = run.events.filter((e) => outcomeOf(e) !== 'other' && e.status !== 'rejected')
  if (run.teams.length >= 2) out.push('Who shot better?')
  else out.push('How many shots went in?')
  if (shots.some((e) => outcomeOf(e) === 'made')) out.push('Show me every made shot')
  const flagged = run.events.find((e) => e.status === 'needs_review')
  if (flagged) out.push(`Why was the ${fmtClock(keyTime(flagged))} shot flagged?`)
  else if (shots[0]) out.push(`What happened at ${fmtClock(keyTime(shots[0]))}?`)
  if (run.narratives.some((n) => n.kind !== 'qa')) out.push('Sum it up in two sentences')
  else out.push('What should we work on?')
  return out.slice(0, 4)
}

export function AskPanel({ open, onOpenChange, run, config, onSpent }: { open: boolean; onOpenChange: (v: boolean) => void; run: RunView; config: AppConfig | undefined; onSpent: () => void }) {
  const player = usePlayer()
  const desktop = useMediaQuery('(min-width: 768px)')
  const [turns, setTurns] = useState<Turn[]>(() => load(run.run_id))
  const [q, setQ] = useState('')
  const [busy, setBusy] = useState(false)
  const [provider, setProvider] = useState<string>('')
  const scroller = useRef<HTMLDivElement>(null)
  const input = useRef<HTMLTextAreaElement>(null)
  const runProviders = [...new Set(run.vlm_calls.map((c) => c.provider))]
  const available = (config?.providers ?? []).filter((p) => p.available)

  useEffect(() => {
    try {
      sessionStorage.setItem(storageKey(run.run_id), JSON.stringify(turns.slice(-30)))
    } catch {
      /* storage unavailable: history just won't persist */
    }
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: 'smooth' })
  }, [turns, run.run_id])

  async function send(question: string) {
    const text = question.trim()
    if (!text || busy) return
    const history: ChatTurn[] = turns.flatMap((t): ChatTurn[] => (t.role === 'user' ? [{ role: 'user', text: t.text }] : t.role === 'assistant' ? [{ role: 'assistant', text: t.answer.answer }] : []))
    setTurns((ts) => [...ts, { role: 'user', text }])
    setQ('')
    setBusy(true)
    try {
      const answer = await api.ask(run.run_id, { question: text, provider: provider || null, model: null, history })
      setTurns((ts) => [...ts, { role: 'assistant', answer }])
      if (answer.cost_usd > 0) onSpent()
    } catch (err) {
      setTurns((ts) => [...ts, { role: 'error', text: errorMessage(err), question: text }])
    } finally {
      setBusy(false)
      requestAnimationFrame(() => input.current?.focus())
    }
  }

  const jump = (t: number) => {
    player.seek(Math.max(0, t - 2), { play: true, reveal: true })
    if (!desktop) onOpenChange(false)
  }

  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault()
      void send(q)
    }
  }

  return (
    <Sheet
      open={open}
      onOpenChange={onOpenChange}
      title="Ask about this game"
      description="Stats come straight from the event log; anything visual goes to the AI."
      width="max-w-lg"
      footer={
        <div className="space-y-2">
          <div className="flex items-end gap-2">
            <label htmlFor="ask-q" className="sr-only">
              Your question
            </label>
            <textarea
              id="ask-q"
              ref={input}
              rows={1}
              value={q}
              onChange={(e) => setQ(e.target.value)}
              onKeyDown={onKey}
              placeholder="Ask anything about this clip…"
              className="max-h-36 min-h-11 flex-1 resize-none rounded-2xl border border-line-strong bg-ink-850 px-4 py-2.5 text-[0.95rem] text-fg placeholder:text-fg-3 outline-none focus:border-flame-500/70 focus:ring-2 focus:ring-flame-500/25"
            />
            <button type="button" onClick={() => void send(q)} disabled={!q.trim() || busy} className="grid size-11 shrink-0 place-items-center rounded-2xl bg-flame-500 text-ink-950 transition hover:bg-flame-400 disabled:opacity-40" aria-label="Send">
              <ArrowUp className="size-5" strokeWidth={2.6} />
            </button>
          </div>
          <div className="flex items-center gap-2 text-xs text-fg-3">
            <label htmlFor="ask-provider" className="shrink-0">
              Answer with
            </label>
            <div className="w-56">
              <Select id="ask-provider" value={provider} onChange={(e) => setProvider(e.target.value)} className="h-8 text-xs">
                <option value="">{runProviders.length ? `This run's AI (${runProviders[0]})` : 'Default'}</option>
                {available.map((p) => (
                  <option key={p.key} value={p.key}>
                    {p.label}
                  </option>
                ))}
              </Select>
            </div>
            {turns.length > 0 && (
              <button type="button" onClick={() => setTurns([])} className="ml-auto inline-flex items-center gap-1 rounded-lg px-2 py-1 hover:bg-white/5 hover:text-fg" aria-label="Clear conversation">
                <Trash2 className="size-3.5" /> Clear
              </button>
            )}
          </div>
        </div>
      }
    >
      <div ref={scroller} className="space-y-4 px-5 py-4" aria-live="polite">
        {turns.length === 0 && (
          <div className="space-y-4 py-2">
            <div className="flex items-center gap-3 rounded-2xl border border-line bg-white/[0.03] p-4">
              <span className="grid size-10 shrink-0 place-items-center rounded-xl bg-flame-500/15 text-flame-300">
                <Sparkles className="size-5" aria-hidden />
              </span>
              <p className="text-sm text-fg-2">Counting questions are answered instantly and for free from the shot log. Questions about how a play looked go to the AI, which watches those moments.</p>
            </div>
            <div>
              <div className="eyebrow mb-2">Try asking</div>
              <div className="flex flex-col gap-2">
                {suggestions(run).map((s) => (
                  <button key={s} type="button" onClick={() => void send(s)} className="rounded-2xl border border-line-strong bg-white/[0.03] px-4 py-2.5 text-left text-sm font-semibold text-fg transition hover:border-flame-500/50 hover:bg-flame-500/[0.06]">
                    {s}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
        <AnimatePresence initial={false}>
          {turns.map((t, i) => (
            <motion.div key={i} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.2 }}>
              {t.role === 'user' ? (
                <div className="ml-auto w-fit max-w-[85%] rounded-2xl rounded-br-md bg-flame-500/15 px-4 py-2.5 text-[0.95rem] text-fg">{t.text}</div>
              ) : t.role === 'error' ? (
                <div className="max-w-[92%] rounded-2xl rounded-bl-md border border-miss/40 bg-miss/[0.08] px-4 py-3 text-sm">
                  <div className="font-semibold text-fg">Couldn't answer that</div>
                  <div className="text-fg-2">{t.text}</div>
                  <button type="button" onClick={() => void send(t.question)} className="mt-2 inline-flex items-center gap-1 text-xs font-semibold text-flame-300 hover:text-flame-200">
                    <RotateCcw className="size-3.5" /> Try again
                  </button>
                </div>
              ) : (
                <div className="max-w-[94%]">
                  <div className="rounded-2xl rounded-bl-md border border-line bg-white/[0.04] px-4 py-3">
                    <Markdown text={t.answer.answer} onTime={jump} className="text-[0.92rem]" />
                    {t.answer.citations.length > 0 && (
                      <div className="mt-3 flex flex-wrap gap-1.5">
                        {t.answer.citations.map((c, j) => (
                          <button key={j} type="button" onClick={() => jump(c.t_s)} className="inline-flex items-center gap-1.5 rounded-lg border border-flame-500/30 bg-flame-500/10 px-2 py-1 text-xs font-semibold text-flame-200 transition hover:bg-flame-500/20">
                            <span className="font-mono tnum">{fmtClock(c.t_s)}</span>
                            <span className="max-w-48 truncate font-normal text-fg-2">{c.label.replace(/,?\s*\d+:\d{2}(\.\d+)?$/, '')}</span>
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                  <div className="mt-1.5 flex flex-wrap items-center gap-x-2 gap-y-0.5 px-1 text-[0.72rem] text-fg-3">
                    <span className="inline-flex items-center gap-1">
                      <Bot className="size-3" aria-hidden /> {ROUTE[t.answer.route] ?? t.answer.route}
                    </span>
                    <span>·</span>
                    <span>{t.answer.model ?? 'no AI needed'}</span>
                    <span>·</span>
                    <span className="tnum">{fmtUsd(t.answer.cost_usd)}</span>
                    <span>·</span>
                    <span className="tnum">{fmtMs(t.answer.latency_ms)}</span>
                    {t.answer.decision && (
                      <span className={cn('tnum')} title={`Routed by ${t.answer.decision.decider}`}>
                        · routed {Math.round(t.answer.decision.route_confidence * 100)}% sure
                      </span>
                    )}
                  </div>
                </div>
              )}
            </motion.div>
          ))}
        </AnimatePresence>
        {busy && (
          <div className="flex w-fit items-center gap-1.5 rounded-2xl rounded-bl-md border border-line bg-white/[0.04] px-4 py-3" aria-label="Thinking">
            {[0, 1, 2].map((i) => (
              <motion.span key={i} className="size-1.5 rounded-full bg-fg-2" animate={{ opacity: [0.25, 1, 0.25] }} transition={{ duration: 1.1, repeat: Infinity, delay: i * 0.18 }} />
            ))}
          </div>
        )}
      </div>
    </Sheet>
  )
}
