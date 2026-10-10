import { Loader2 } from 'lucide-react'
import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from 'react'
import { Link, type LinkProps } from 'react-router'

import { cn } from '../../lib/format'

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger' | 'outline'
type Size = 'sm' | 'md' | 'lg' | 'icon'

export function buttonClass(variant: Variant = 'secondary', size: Size = 'md', extra?: string) {
  return cn(
    'relative inline-flex select-none items-center justify-center gap-2 whitespace-nowrap font-semibold transition-[background,color,box-shadow,transform,opacity] duration-150 active:translate-y-px disabled:pointer-events-none disabled:opacity-45',
    size === 'sm' && 'h-8 rounded-lg px-3 text-[0.8rem]',
    size === 'md' && 'h-10 rounded-xl px-4 text-sm',
    size === 'lg' && 'h-14 rounded-2xl px-7 font-display text-xl uppercase tracking-wide',
    size === 'icon' && 'h-9 w-9 rounded-xl',
    variant === 'primary' &&
      'bg-flame-500 text-ink-950 shadow-[0_10px_30px_-10px_rgb(255_122_26/0.7),inset_0_1px_0_rgb(255_255_255/0.35)] hover:bg-flame-400',
    variant === 'secondary' && 'border border-line-strong bg-white/[0.06] text-fg hover:bg-white/[0.1]',
    variant === 'outline' && 'border border-line-strong text-fg hover:border-flame-500/60 hover:text-flame-200',
    variant === 'ghost' && 'text-fg-2 hover:bg-white/[0.06] hover:text-fg',
    variant === 'danger' && 'border border-miss/50 bg-miss/15 text-miss-text hover:bg-miss/25',
    extra,
  )
}

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant
  size?: Size
  loading?: boolean
  icon?: ReactNode
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = 'secondary', size = 'md', loading, icon, className, children, disabled, type = 'button', ...rest },
  ref,
) {
  return (
    <button ref={ref} type={type} className={buttonClass(variant, size, className)} disabled={disabled || loading} aria-busy={loading || undefined} {...rest}>
      {loading ? <Loader2 className="size-4 animate-spin" aria-hidden /> : icon}
      {children}
    </button>
  )
})

export function ButtonLink({ variant = 'secondary', size = 'md', className, icon, children, ...rest }: LinkProps & { variant?: Variant; size?: Size; icon?: ReactNode }) {
  return (
    <Link className={buttonClass(variant, size, className)} {...rest}>
      {icon}
      {children}
    </Link>
  )
}
