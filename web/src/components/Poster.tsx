import { useState, type ReactNode } from 'react'

import { cn } from '../lib/format'

/** Poster image with a court-pattern placeholder when the image is missing or still loading. */
export function Poster({ src, alt, className, children }: { src: string | null | undefined; alt: string; className?: string; children?: ReactNode }) {
  const [failed, setFailed] = useState(false)
  const [loaded, setLoaded] = useState(false)
  return (
    <div className={cn('relative aspect-video overflow-hidden bg-ink-800', className)}>
      <svg aria-hidden viewBox="0 0 160 90" className="absolute inset-0 h-full w-full text-wood-400/20" preserveAspectRatio="xMidYMid slice">
        <rect width="160" height="90" fill="#121a2c" />
        <g fill="none" stroke="currentColor" strokeWidth="1.2">
          <line x1="80" y1="0" x2="80" y2="90" />
          <circle cx="80" cy="45" r="14" />
          <path d="M0 18 H 22 A 27 27 0 0 1 22 72 H 0" />
          <path d="M160 18 H 138 A 27 27 0 0 0 138 72 H 160" />
        </g>
      </svg>
      {src && !failed && (
        <img
          src={src}
          alt={alt}
          loading="lazy"
          decoding="async"
          onLoad={() => setLoaded(true)}
          onError={() => setFailed(true)}
          className={cn('absolute inset-0 h-full w-full object-cover transition-[opacity,transform] duration-500 group-hover:scale-[1.03]', loaded ? 'opacity-100' : 'opacity-0')}
        />
      )}
      <div className="absolute inset-0 bg-[linear-gradient(180deg,transparent_40%,rgb(7_11_20/0.85))]" />
      {children}
    </div>
  )
}
