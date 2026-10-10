import { AlertTriangle, Check, CircleDashed, Loader2, X } from 'lucide-react'
import type { HTMLAttributes, ReactNode, SelectHTMLAttributes } from 'react'

import type { EventStatus } from '../../api/types'
import { cn } from '../../lib/format'
import { STATUS_HELP, STATUS_LABEL, type Outcome } from '../../lib/events'
import { Tip } from './tip'

export function Card({ className, children, ...rest }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn('glass rounded-[var(--radius-card)]', className)} {...rest}>
      {children}
    </div>
  )
}

export function CardHeader({ eyebrow, title, action, className, icon }: { eyebrow?: ReactNode; title: ReactNode; action?: ReactNode; className?: string; icon?: ReactNode }) {
  return (
    <div className={cn('flex items-start justify-between gap-3 px-5 pt-4 pb-3', className)}>
      <div className="min-w-0">
        {eyebrow && <div className="eyebrow mb-0.5">{eyebrow}</div>}
        <h2 className="flex items-center gap-2 font-display text-xl font-semibold uppercase tracking-wide text-fg">
          {icon}
          {title}
        </h2>
      </div>
      {action && <div className="flex shrink-0 items-center gap-2">{action}</div>}
    </div>
  )
}

type Tone = 'neutral' | 'flame' | 'made' | 'miss' | 'attempt' | 'info' | 'muted'

export function Badge({ tone = 'neutral', className, children, icon }: { tone?: Tone; className?: string; children: ReactNode; icon?: ReactNode }) {
  return (
    <span
      className={cn(
        'inline-flex h-6 shrink-0 items-center gap-1 rounded-full border px-2 text-[0.72rem] font-semibold leading-none whitespace-nowrap',
        tone === 'neutral' && 'border-line-strong bg-white/[0.05] text-fg-2',
        tone === 'muted' && 'border-line bg-transparent text-fg-3',
        tone === 'flame' && 'border-flame-500/40 bg-flame-500/12 text-flame-200',
        tone === 'made' && 'border-made/45 bg-made/12 text-fg',
        tone === 'miss' && 'border-miss/50 bg-miss/12 text-fg',
        tone === 'attempt' && 'border-attempt/45 bg-attempt/10 text-fg',
        tone === 'info' && 'border-team-a/45 bg-team-a/10 text-fg',
        className,
      )}
    >
      {icon}
      {children}
    </span>
  )
}

/** Outcome glyph: shape + color, so made/missed never depend on color alone (green/red collide for deutans). */
export function OutcomeIcon({ outcome, className }: { outcome: Outcome; className?: string }) {
  if (outcome === 'made') return <Check className={cn('size-3.5 text-made-text', className)} strokeWidth={3} aria-hidden />
  if (outcome === 'missed') return <X className={cn('size-3.5 text-miss-text', className)} strokeWidth={3} aria-hidden />
  if (outcome === 'attempt') return <CircleDashed className={cn('size-3.5 text-attempt', className)} strokeWidth={2.5} aria-hidden />
  return <span className={cn('size-2 rounded-full bg-fg-3', className)} aria-hidden />
}

export function StatusBadge({ status }: { status: EventStatus }) {
  const tone: Tone = status === 'confirmed' ? 'neutral' : status === 'needs_review' ? 'attempt' : status === 'rejected' ? 'muted' : 'muted'
  return (
    <Tip content={STATUS_HELP[status]}>
      <span tabIndex={0} className="inline-flex rounded-full outline-none focus-visible:ring-2 focus-visible:ring-flame-400">
        <Badge
          tone={tone}
          className={cn(status === 'rejected' && 'line-through')}
          icon={
            status === 'confirmed' ? (
              <span className="size-2 rounded-full bg-fg" aria-hidden />
            ) : status === 'needs_review' ? (
              <span className="relative flex size-2" aria-hidden>
                <span className="absolute inset-0 animate-ring-pulse rounded-full bg-attempt" />
                <span className="relative size-2 rounded-full bg-attempt" />
              </span>
            ) : (
              <span className="size-2 rounded-full border border-fg-3" aria-hidden />
            )
          }
        >
          {STATUS_LABEL[status]}
        </Badge>
      </span>
    </Tip>
  )
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn('skeleton', className)} aria-hidden />
}

export function Spinner({ className, label = 'Loading' }: { className?: string; label?: string }) {
  return (
    <span role="status" className="inline-flex items-center">
      <Loader2 className={cn('size-5 animate-spin text-flame-400', className)} aria-hidden />
      <span className="sr-only">{label}</span>
    </span>
  )
}

export function EmptyState({ icon, title, body, action, className }: { icon?: ReactNode; title: ReactNode; body?: ReactNode; action?: ReactNode; className?: string }) {
  return (
    <div className={cn('flex flex-col items-center justify-center gap-3 px-6 py-10 text-center', className)}>
      {icon && <div className="grid size-12 place-items-center rounded-2xl border border-line bg-white/[0.04] text-fg-3">{icon}</div>}
      <div className="font-display text-xl font-semibold uppercase tracking-wide text-fg">{title}</div>
      {body && <div className="max-w-md text-sm text-fg-2">{body}</div>}
      {action}
    </div>
  )
}

export function ErrorPanel({ title = "That didn't work", message, action }: { title?: string; message: string; action?: ReactNode }) {
  return (
    <div role="alert" className="flex items-start gap-3 rounded-2xl border border-miss/40 bg-miss/[0.08] p-4">
      <AlertTriangle className="mt-0.5 size-5 shrink-0 text-miss-text" aria-hidden />
      <div className="min-w-0 flex-1">
        <div className="font-semibold text-fg">{title}</div>
        <div className="mt-0.5 text-sm break-words text-fg-2">{message}</div>
        {action && <div className="mt-3">{action}</div>}
      </div>
    </div>
  )
}

/** Same-ramp meter: the track is a dim step of the fill's hue. */
export function Meter({ value, className, tone = 'team-a', label }: { value: number; className?: string; tone?: 'team-a' | 'flame' | 'attempt' | 'miss'; label?: string }) {
  const pct = Math.max(0, Math.min(1, value)) * 100
  const fill = tone === 'flame' ? 'bg-flame-500' : tone === 'attempt' ? 'bg-attempt' : tone === 'miss' ? 'bg-miss' : 'bg-team-a'
  const track = tone === 'flame' ? 'bg-flame-500/15' : tone === 'attempt' ? 'bg-attempt/15' : tone === 'miss' ? 'bg-miss/15' : 'bg-team-a/18'
  return (
    <div className={cn('h-1.5 w-full overflow-hidden rounded-full', track, className)} role="meter" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(pct)} aria-label={label}>
      <div className={cn('h-full rounded-full transition-[width] duration-500', fill)} style={{ width: `${pct}%` }} />
    </div>
  )
}

export function Field({ label, hint, children, htmlFor }: { label: ReactNode; hint?: ReactNode; children: ReactNode; htmlFor?: string }) {
  return (
    <div className="space-y-1.5">
      <label htmlFor={htmlFor} className="block text-[0.8rem] font-semibold text-fg-2">
        {label}
      </label>
      {children}
      {hint && <div className="text-xs text-fg-3">{hint}</div>}
    </div>
  )
}

export const inputClass =
  'h-10 w-full rounded-xl border border-line-strong bg-ink-900/70 px-3 text-sm text-fg placeholder:text-fg-3 outline-none transition focus:border-flame-500/70 focus:ring-2 focus:ring-flame-500/25 disabled:opacity-50'

export function Select({ className, children, ...rest }: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <div className="relative">
      <select className={cn(inputClass, 'appearance-none pr-9', className)} {...rest}>
        {children}
      </select>
      <svg className="pointer-events-none absolute top-1/2 right-3 size-4 -translate-y-1/2 text-fg-3" viewBox="0 0 16 16" fill="none" aria-hidden>
        <path d="M4 6l4 4 4-4" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </div>
  )
}

export function Kbd({ children }: { children: ReactNode }) {
  return <kbd className="rounded border border-line-strong bg-white/5 px-1.5 py-0.5 font-mono text-[0.7rem] text-fg-2">{children}</kbd>
}
