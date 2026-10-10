import { cn } from '../lib/format'

export function BallMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 40 40" className={cn('size-9', className)} aria-hidden>
      <defs>
        <radialGradient id="ballg" cx="0.35" cy="0.3" r="0.8">
          <stop offset="0" stopColor="#ffb27a" />
          <stop offset="0.55" stopColor="#ff7a1a" />
          <stop offset="1" stopColor="#b84a06" />
        </radialGradient>
      </defs>
      <circle cx="20" cy="20" r="17" fill="url(#ballg)" />
      <g fill="none" stroke="#070b14" strokeWidth="2" strokeLinecap="round" opacity="0.85">
        <path d="M3 20h34" />
        <path d="M20 3v34" />
        <path d="M8 8c5 5 5 19 0 24" />
        <path d="M32 8c-5 5-5 19 0 24" />
      </g>
    </svg>
  )
}

export function Wordmark({ className }: { className?: string }) {
  return (
    <span className={cn('flex items-center gap-2.5', className)}>
      <BallMark />
      <span className="leading-none">
        <span className="block font-display text-[1.45rem] font-bold tracking-[0.06em] text-fg uppercase">
          Press<span className="text-flame-500">·</span>Hoops
        </span>
        <span className="block font-display text-[0.62rem] font-semibold tracking-[0.32em] text-fg-3 uppercase">AI film room</span>
      </span>
    </span>
  )
}

/** Faint court-line motif behind everything (decorative). */
export function CourtBackdrop() {
  return (
    <div aria-hidden className="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
      <div className="absolute inset-0 bg-[radial-gradient(1200px_600px_at_50%_-10%,rgb(255_122_26/0.13),transparent_60%),radial-gradient(900px_500px_at_100%_110%,rgb(57_135_229/0.08),transparent_60%)]" />
      <svg className="absolute top-1/2 left-1/2 h-[140vh] min-h-[900px] w-auto -translate-x-1/2 -translate-y-1/2 opacity-[0.05]" viewBox="0 0 500 940" fill="none" stroke="#e8c08f" strokeWidth="2.5">
        <rect x="10" y="10" width="480" height="920" rx="2" />
        <line x1="10" y1="470" x2="490" y2="470" />
        <circle cx="250" cy="470" r="60" />
        <circle cx="250" cy="470" r="20" />
        <rect x="170" y="10" width="160" height="190" />
        <rect x="170" y="740" width="160" height="190" />
        <circle cx="250" cy="200" r="60" />
        <circle cx="250" cy="740" r="60" />
        <path d="M40 10 V 150 A 237.5 237.5 0 0 0 460 150 V 10" />
        <path d="M40 930 V 790 A 237.5 237.5 0 0 1 460 790 V 930" />
      </svg>
      <div className="absolute inset-0 bg-[linear-gradient(180deg,transparent,rgb(7_11_20/0.6))]" />
    </div>
  )
}
