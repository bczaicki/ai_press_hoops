import { motion } from 'framer-motion'
import { useId, useRef, type KeyboardEvent, type ReactNode } from 'react'

import { cn } from '../../lib/format'

export interface SegOption<K extends string> {
  key: K
  label: ReactNode
  disabled?: boolean
  title?: string
}

/** Radio-group segmented control with arrow-key navigation and an animated thumb. */
export function Segmented<K extends string>({
  value,
  options,
  onChange,
  label,
  size = 'md',
  className,
  disabled,
}: {
  value: K
  options: SegOption<K>[]
  onChange: (k: K) => void
  label: string
  size?: 'sm' | 'md'
  className?: string
  disabled?: boolean
}) {
  const id = useId()
  const refs = useRef<(HTMLButtonElement | null)[]>([])
  const enabled = options.filter((o) => !o.disabled)
  const onKey = (e: KeyboardEvent) => {
    if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'].includes(e.key)) return
    e.preventDefault()
    const idx = enabled.findIndex((o) => o.key === value)
    let next = idx
    if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') next = (idx - 1 + enabled.length) % enabled.length
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') next = (idx + 1) % enabled.length
    if (e.key === 'Home') next = 0
    if (e.key === 'End') next = enabled.length - 1
    const opt = enabled[next]
    onChange(opt.key)
    refs.current[options.indexOf(opt)]?.focus()
  }
  return (
    <div
      role="radiogroup"
      aria-label={label}
      aria-disabled={disabled || undefined}
      onKeyDown={onKey}
      className={cn('relative inline-flex rounded-xl border border-line-strong bg-ink-900/70 p-1', disabled && 'opacity-50', className)}
    >
      {options.map((o, i) => {
        const active = o.key === value
        return (
          <button
            key={o.key}
            ref={(el) => {
              refs.current[i] = el
            }}
            type="button"
            role="radio"
            aria-checked={active}
            tabIndex={active ? 0 : -1}
            disabled={o.disabled || disabled}
            title={o.title}
            onClick={() => onChange(o.key)}
            className={cn(
              'relative z-10 flex-1 rounded-lg font-semibold whitespace-nowrap transition-colors',
              size === 'sm' ? 'h-7 px-2.5 text-xs' : 'h-8 px-3.5 text-sm',
              active ? 'text-ink-950' : 'text-fg-2 hover:text-fg',
            )}
          >
            {active && (
              <motion.span
                layoutId={`seg-${id}`}
                className="absolute inset-0 -z-10 rounded-lg bg-flame-500 shadow-[inset_0_1px_0_rgb(255_255_255/0.35)]"
                transition={{ type: 'spring', stiffness: 520, damping: 38 }}
              />
            )}
            {o.label}
          </button>
        )
      })}
    </div>
  )
}

export function Switch({ checked, onChange, label, disabled, id }: { checked: boolean; onChange: (v: boolean) => void; label: string; disabled?: boolean; id?: string }) {
  return (
    <button
      id={id}
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={cn(
        'relative inline-flex h-6 w-11 shrink-0 items-center rounded-full border transition-colors disabled:opacity-40',
        checked ? 'border-flame-400 bg-flame-500' : 'border-line-strong bg-ink-700',
      )}
    >
      <span className={cn('inline-block size-4.5 rounded-full bg-white shadow transition-transform', checked ? 'translate-x-[22px]' : 'translate-x-[3px]')} />
    </button>
  )
}
