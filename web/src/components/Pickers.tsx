import { motion } from 'framer-motion'
import { Ban, Check, Cpu, Eye, HardDrive, Lock, Map as MapIcon, MessageSquareText, ScanEye, Table2, type LucideIcon } from 'lucide-react'
import type { ReactNode } from 'react'

import type { OutputKind, ProviderInfo, RegistryItem, ThresholdPreset } from '../api/types'
import { cn, fmtPct } from '../lib/format'
import { Badge } from './ui/misc'
import { Segmented } from './ui/segmented'
import { Tip } from './ui/tip'

// ------------------------------------------------------------------ outputs

const OUTPUT_META: Record<OutputKind, { icon: LucideIcon; blurb: string }> = {
  annotated_video: { icon: ScanEye, blurb: 'Your video with players, ball, and rim drawn on.' },
  events: { icon: Table2, blurb: 'Every shot with a time, result, and team, plus FG% and 3s.' },
  summary: { icon: MessageSquareText, blurb: 'A written recap, plus a chat to ask about the game.' },
  court_map: { icon: MapIcon, blurb: 'Where shots came from, drawn on a court.' },
}

export function OutputCards({
  options,
  value,
  onChange,
  disabledReason,
}: {
  options: { key: OutputKind; label: string }[]
  value: OutputKind[]
  onChange: (v: OutputKind[]) => void
  disabledReason: (k: OutputKind) => string | null
}) {
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2" role="group" aria-label="Outputs">
      {options.map((o) => {
        const meta = OUTPUT_META[o.key] ?? { icon: Cpu, blurb: '' }
        const reason = disabledReason(o.key)
        const on = value.includes(o.key) && !reason
        const Icon = meta.icon
        return (
          <Tip key={o.key} content={reason} disabled={!reason}>
            <button
              type="button"
              role="checkbox"
              aria-checked={on}
              aria-disabled={!!reason}
              onClick={() => {
                if (reason) return
                onChange(on ? value.filter((k) => k !== o.key) : [...value, o.key])
              }}
              className={cn(
                'group relative flex items-start gap-3.5 rounded-2xl border p-4 text-left transition-all duration-200',
                on ? 'border-flame-500/70 bg-flame-500/[0.09] shadow-[0_0_0_1px_rgb(255_122_26/0.25),0_16px_40px_-24px_rgb(255_122_26/0.6)]' : 'border-line-strong bg-white/[0.025] hover:border-white/25 hover:bg-white/[0.045]',
                reason && 'cursor-not-allowed opacity-50',
              )}
            >
              <span className={cn('grid size-11 shrink-0 place-items-center rounded-xl transition-colors', on ? 'bg-flame-500 text-ink-950' : 'bg-white/[0.06] text-fg-2')}>
                <Icon className="size-5.5" aria-hidden />
              </span>
              <span className="min-w-0 flex-1">
                <span className="block font-display text-lg leading-tight font-semibold tracking-wide text-fg uppercase">{o.label}</span>
                <span className="mt-0.5 block text-[0.8rem] leading-snug text-fg-2">{reason ?? meta.blurb}</span>
              </span>
              <span className={cn('grid size-5.5 shrink-0 place-items-center rounded-md border transition', on ? 'border-flame-400 bg-flame-500 text-ink-950' : 'border-line-strong')}>
                {on && <Check className="size-3.5" strokeWidth={3.5} aria-hidden />}
              </span>
            </button>
          </Tip>
        )
      })}
    </div>
  )
}

// ------------------------------------------------------------------ providers

const PROVIDER_ACCENT: Record<string, string> = {
  claude: 'from-[#d97757]/25',
  gemini: 'from-[#4f8df7]/25',
  openai: 'from-[#9aa4b5]/20',
  openrouter: 'from-[#7c6cf2]/20',
  ollama: 'from-[#34c759]/15',
}

export function ProviderCards({ providers, value, onChange }: { providers: ProviderInfo[]; value: string | null; onChange: (k: string | null) => void }) {
  const opts: (ProviderInfo | null)[] = [...providers, null]
  return (
    <div role="radiogroup" aria-label="AI provider" className="grid grid-cols-2 gap-2.5 sm:grid-cols-3">
      {opts.map((p) => {
        const key = p?.key ?? null
        const active = value === key
        const disabled = !!p && !p.available
        const card = (
          <button
            type="button"
            role="radio"
            aria-checked={active}
            aria-disabled={disabled}
            aria-describedby={disabled && p ? `why-${p.key}` : undefined}
            onClick={() => !disabled && onChange(key)}
            className={cn(
              'relative flex h-full min-h-[5.5rem] w-full flex-col items-start overflow-hidden rounded-2xl border p-3.5 text-left transition-all duration-200',
              active ? 'border-flame-500/70 bg-flame-500/[0.08] shadow-[0_0_0_1px_rgb(255_122_26/0.25)]' : 'border-line-strong bg-white/[0.025] hover:border-white/25',
              disabled && 'cursor-not-allowed opacity-45 hover:border-line-strong',
            )}
          >
            <span aria-hidden className={cn('pointer-events-none absolute inset-0 bg-gradient-to-br to-transparent opacity-80', p ? (PROVIDER_ACCENT[p.key] ?? 'from-white/10') : 'from-white/5')} />
            <span className="relative flex w-full items-start justify-between gap-2">
              <span className="font-semibold leading-tight text-fg">{p ? p.label.replace(/\s*\(.*\)$/, '') : 'No AI'}</span>
              {active && (
                <motion.span layoutId="prov-check" className="grid size-5 shrink-0 place-items-center rounded-full bg-flame-500 text-ink-950">
                  <Check className="size-3" strokeWidth={3.5} aria-hidden />
                </motion.span>
              )}
            </span>
            <span className="relative mt-1 text-xs text-fg-3">{p ? (p.label.match(/\((.*)\)$/)?.[1] ?? p.default_model) : 'Computer vision + System One only. Free.'}</span>
            <span className="relative mt-auto flex flex-wrap gap-1 pt-2">
              {p?.local && (
                <Badge tone="made" icon={<HardDrive className="size-3" aria-hidden />}>
                  Local
                </Badge>
              )}
              {p?.native_video && <Badge tone="info">Native video</Badge>}
              {disabled && (
                <Badge tone="muted" icon={<Ban className="size-3" aria-hidden />}>
                  Unavailable
                </Badge>
              )}
              {!p && (
                <Badge tone="muted" icon={<Lock className="size-3" aria-hidden />}>
                  Nothing leaves
                </Badge>
              )}
            </span>
            {disabled && p && (
              <span id={`why-${p.key}`} className="sr-only">
                Unavailable: {p.reason}
              </span>
            )}
          </button>
        )
        return (
          <Tip key={key ?? 'none'} content={disabled && p ? `Unavailable: ${p.reason ?? 'not configured'}` : null} disabled={!disabled}>
            {card}
          </Tip>
        )
      })}
    </div>
  )
}

// ------------------------------------------------------------------ deciders (System One)

export function deciderBlurb(d: RegistryItem): string {
  if (d.notes) return d.notes
  if (d.key === 'rules') return 'Deterministic thresholds. Free and instant.'
  if (d.key === 'jev') return 'Calibrated System One model.'
  if (d.key.startsWith('adapter:')) return 'LLM-backed stand-in with the same API.'
  return ''
}

export function DeciderList({ deciders, value, onChange }: { deciders: RegistryItem[]; value: string; onChange: (k: string) => void }) {
  return (
    <div role="radiogroup" aria-label="System One decider" className="grid gap-2 sm:grid-cols-2">
      {deciders.map((d) => {
        const active = d.key === value
        const disabled = !d.available
        return (
          <Tip key={d.key} content={disabled ? `Unavailable: ${d.reason ?? 'not configured'}` : null} disabled={!disabled}>
            <button
              type="button"
              role="radio"
              aria-checked={active}
              aria-disabled={disabled}
              onClick={() => !disabled && onChange(d.key)}
              className={cn(
                'flex items-start gap-3 rounded-xl border px-3.5 py-3 text-left transition',
                active ? 'border-flame-500/70 bg-flame-500/[0.08]' : 'border-line-strong bg-white/[0.02] hover:border-white/25',
                disabled && 'cursor-not-allowed opacity-45',
              )}
            >
              <span className={cn('mt-0.5 grid size-4.5 shrink-0 place-items-center rounded-full border-2', active ? 'border-flame-500' : 'border-fg-3')}>
                {active && <span className="size-2 rounded-full bg-flame-500" />}
              </span>
              <span className="min-w-0 flex-1">
                <span className="flex flex-wrap items-center gap-1.5">
                  <span className="text-sm font-semibold text-fg">{d.label}</span>
                  {d.supports_images && (
                    <Badge tone="info" icon={<Eye className="size-3" aria-hidden />}>
                      Sees images
                    </Badge>
                  )}
                </span>
                <span className="mt-0.5 block text-xs text-fg-3">{disabled ? d.reason : deciderBlurb(d)}</span>
              </span>
            </button>
          </Tip>
        )
      })}
    </div>
  )
}

// ------------------------------------------------------------------ thresholds

export const PRESET_COPY: Record<string, { title: string; body: string }> = {
  strict: { title: 'Fewer false alarms', body: 'Only counts shots it is very sure about. Close calls go to your review list instead of the box score.' },
  balanced: { title: 'Best for most clips', body: 'A middle ground. Confident calls go straight in, and borderline ones get a second look.' },
  eager: { title: 'Miss fewer shots', body: 'Counts borderline shots too. Catches more real makes, but expect a few phantom ones.' },
}

export function ThresholdControl({
  presets,
  value,
  onChange,
  disabled,
  compact,
  trailing,
}: {
  presets: ThresholdPreset[]
  value: string
  onChange: (k: string) => void
  disabled?: boolean
  compact?: boolean
  trailing?: ReactNode
}) {
  const current = presets.find((p) => p.key === value)
  const copy = PRESET_COPY[value]
  return (
    <div className="space-y-2.5">
      <div className="flex flex-wrap items-center gap-3">
        <Segmented label="How sure before counting a shot" value={value} onChange={onChange} disabled={disabled} size={compact ? 'sm' : 'md'} options={presets.map((p) => ({ key: p.key, label: p.label }))} />
        {trailing}
      </div>
      {!compact && current && (
        <p className="text-sm text-fg-2">
          <span className="font-semibold text-fg">{copy?.title ?? current.note}.</span> {copy?.body ?? current.note}{' '}
          <span className="text-fg-3 tnum">
            (auto-accept at {fmtPct(current.accept)}+, auto-reject at {fmtPct(current.reject)} or less)
          </span>
        </p>
      )}
    </div>
  )
}
