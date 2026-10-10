import { Check, Crosshair, RotateCcw, Undo2 } from 'lucide-react'
import { useRef, useState, type PointerEvent, type ReactNode } from 'react'

import type { CourtPoint } from '../api/types'
import { cn } from '../lib/format'
import { Button } from './ui/button'

export type Box = [number, number, number, number]
type Pt = [number, number]

const LOUPE = 120
const ZOOM = 3

/**
 * Click surface over the clip poster. Pointer positions are converted to PROXY pixels (the
 * coordinate space the API expects), using the proxy's width/height from ClipMeta.proxy.
 * A 3x loupe follows the pointer so the small rim can be hit precisely, including on touch.
 */
function PickSurface({
  poster,
  width,
  height,
  onPick,
  overlay,
  label,
}: {
  poster: string
  width: number
  height: number
  onPick: (p: Pt) => void
  overlay: (hover: Pt | null) => ReactNode
  label: string
}) {
  const ref = useRef<HTMLDivElement>(null)
  const [hover, setHover] = useState<{ p: Pt; px: Pt; touch: boolean } | null>(null)

  const locate = (e: PointerEvent): { p: Pt; px: Pt } | null => {
    const el = ref.current
    if (!el) return null
    const r = el.getBoundingClientRect()
    const fx = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width))
    const fy = Math.min(1, Math.max(0, (e.clientY - r.top) / r.height))
    return { p: [Math.round(fx * width), Math.round(fy * height)], px: [fx * r.width, fy * r.height] }
  }

  const rect = ref.current?.getBoundingClientRect()
  return (
    <div
      ref={ref}
      role="application"
      aria-label={label}
      className="relative w-full touch-none overflow-hidden rounded-xl border border-line-strong bg-ink-800 select-none"
      style={{ aspectRatio: `${width} / ${height}`, cursor: 'crosshair' }}
      onPointerMove={(e) => {
        const l = locate(e)
        if (l) setHover({ ...l, touch: e.pointerType !== 'mouse' })
      }}
      onPointerDown={(e) => {
        ;(e.target as HTMLElement).setPointerCapture?.(e.pointerId)
        const l = locate(e)
        if (l) setHover({ ...l, touch: e.pointerType !== 'mouse' })
      }}
      onPointerUp={(e) => {
        const l = locate(e)
        if (l) onPick(l.p)
        if (e.pointerType !== 'mouse') setHover(null)
      }}
      onPointerLeave={() => setHover(null)}
    >
      <img src={poster} alt="" draggable={false} className="pointer-events-none absolute inset-0 h-full w-full object-fill" />
      <svg viewBox={`0 0 ${width} ${height}`} className="pointer-events-none absolute inset-0 h-full w-full" preserveAspectRatio="none">
        {overlay(hover?.p ?? null)}
        {hover && (
          <g stroke="#fff" strokeOpacity="0.7" strokeWidth={width / 640}>
            <line x1={hover.p[0]} y1={0} x2={hover.p[0]} y2={height} strokeDasharray="none" opacity="0.35" />
            <line x1={0} y1={hover.p[1]} x2={width} y2={hover.p[1]} opacity="0.35" />
          </g>
        )}
      </svg>
      {hover && rect && (
        <div
          aria-hidden
          className="pointer-events-none absolute rounded-full border-2 border-white/80 shadow-[0_8px_24px_rgb(0_0_0/0.6)]"
          style={{
            width: LOUPE,
            height: LOUPE,
            left: Math.min(rect.width - LOUPE - 4, Math.max(4, hover.px[0] - LOUPE / 2)),
            top: hover.touch ? Math.max(4, hover.px[1] - LOUPE - 36) : hover.px[1] - LOUPE - 16 < 0 ? hover.px[1] + 20 : hover.px[1] - LOUPE - 16,
            backgroundImage: `url("${poster}")`,
            backgroundRepeat: 'no-repeat',
            backgroundSize: `${rect.width * ZOOM}px ${rect.height * ZOOM}px`,
            backgroundPosition: `${-(hover.px[0] * ZOOM - LOUPE / 2)}px ${-(hover.px[1] * ZOOM - LOUPE / 2)}px`,
          }}
        >
          <span className="absolute top-1/2 left-1/2 size-3 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-flame-400" />
        </div>
      )}
    </div>
  )
}

// ------------------------------------------------------------------ rim

export function RimPicker({ poster, width, height, value, onChange }: { poster: string; width: number; height: number; value: Box | null; onChange: (b: Box | null) => void }) {
  const [first, setFirst] = useState<Pt | null>(null)
  const sw = width / 400
  return (
    <div className="space-y-3">
      <PickSurface
        poster={poster}
        width={width}
        height={height}
        label="Rim picker. Click the top-left, then the bottom-right corner of the rim."
        onPick={(p) => {
          if (!first) {
            setFirst(p)
            return
          }
          const box: Box = [Math.min(first[0], p[0]), Math.min(first[1], p[1]), Math.max(first[0], p[0]), Math.max(first[1], p[1])]
          setFirst(null)
          if (box[2] - box[0] < 4 || box[3] - box[1] < 2) return
          onChange(box)
        }}
        overlay={(hover) => (
          <>
            {value && !first && (
              <g>
                <rect x={value[0]} y={value[1]} width={value[2] - value[0]} height={value[3] - value[1]} fill="rgb(255 122 26 / 0.18)" stroke="#ff9447" strokeWidth={sw} />
                <text x={value[0]} y={value[1] - sw * 3} fill="#ffb27a" fontSize={width / 55} fontWeight="700" fontFamily="Barlow Condensed, sans-serif" letterSpacing="2">
                  RIM
                </text>
              </g>
            )}
            {first && (
              <>
                <circle cx={first[0]} cy={first[1]} r={sw * 2.5} fill="#ff7a1a" stroke="#fff" strokeWidth={sw * 0.8} />
                {hover && (
                  <rect
                    x={Math.min(first[0], hover[0])}
                    y={Math.min(first[1], hover[1])}
                    width={Math.abs(hover[0] - first[0])}
                    height={Math.abs(hover[1] - first[1])}
                    fill="rgb(255 122 26 / 0.12)"
                    stroke="#ff9447"
                    strokeWidth={sw}
                    strokeDasharray={`${sw * 3} ${sw * 2}`}
                  />
                )}
              </>
            )}
          </>
        )}
      />
      <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
        <p className="text-fg-2" aria-live="polite">
          {first ? (
            <>
              <Crosshair className="mr-1 inline size-4 text-flame-400" aria-hidden />
              Now click the <b className="text-fg">bottom-right</b> corner of the rim.
            </>
          ) : value ? (
            <>
              <Check className="mr-1 inline size-4 text-made-text" aria-hidden />
              Rim marked <span className="text-fg-3 tnum">({value.join(', ')} px)</span>
            </>
          ) : (
            <>
              Click the <b className="text-fg">top-left</b> corner of the rim, then the <b className="text-fg">bottom-right</b>.
            </>
          )}
        </p>
        {(value || first) && (
          <Button
            size="sm"
            variant="ghost"
            icon={<RotateCcw className="size-3.5" />}
            onClick={() => {
              setFirst(null)
              onChange(null)
            }}
          >
            Clear
          </Button>
        )}
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ court calibration

export const COURT_LANDMARKS: { name: string; label: string; court: Pt }[] = [
  { name: 'baseline_left_corner', label: 'Baseline, left corner', court: [-47, -25] },
  { name: 'baseline_right_corner', label: 'Baseline, right corner', court: [-47, 25] },
  { name: 'lane_baseline_left', label: 'Paint, left side at baseline', court: [-47, -8] },
  { name: 'lane_baseline_right', label: 'Paint, right side at baseline', court: [-47, 8] },
  { name: 'free_throw_left', label: 'Free-throw line, left end', court: [-28, -8] },
  { name: 'free_throw_right', label: 'Free-throw line, right end', court: [-28, 8] },
  { name: 'halfcourt_left', label: 'Half-court line, left sideline', court: [0, -25] },
  { name: 'halfcourt_right', label: 'Half-court line, right sideline', court: [0, 25] },
]

function MiniCourt({ active, placed }: { active: string | null; placed: Set<string> }) {
  // Half court, baseline at the top; (x, y) feet → svg (y + 25, x + 47).
  return (
    <svg viewBox="-2 -2 54 51" className="h-32 w-auto shrink-0" aria-hidden>
      <g fill="none" stroke="#e8c08f" strokeOpacity="0.55" strokeWidth="0.6">
        <rect x="0" y="0" width="50" height="47" />
        <rect x="17" y="0" width="16" height="19" />
        <circle cx="25" cy="19" r="6" />
        <path d="M3 0 V 14 A 23.75 23.75 0 0 0 47 14 V 0" />
        <circle cx="25" cy="5.25" r="0.75" />
        <path d="M19 47 A 6 6 0 0 1 31 47" />
      </g>
      {COURT_LANDMARKS.map((l) => {
        const cx = l.court[1] + 25
        const cy = l.court[0] + 47
        const on = l.name === active
        return (
          <g key={l.name}>
            {on && <circle cx={cx} cy={cy} r="3.4" fill="#ff7a1a" opacity="0.3" />}
            <circle cx={cx} cy={cy} r={on ? 1.8 : 1.3} fill={placed.has(l.name) ? '#34c759' : on ? '#ff7a1a' : '#7d879c'} />
          </g>
        )
      })}
    </svg>
  )
}

export function CourtPointsPicker({ poster, width, height, value, onChange }: { poster: string; width: number; height: number; value: CourtPoint[]; onChange: (pts: CourtPoint[]) => void }) {
  const placed = new Set(value.map((p) => p.name))
  const nextUnplaced = COURT_LANDMARKS.find((l) => !placed.has(l.name))?.name ?? null
  const [active, setActive] = useState<string | null>(nextUnplaced)
  const sw = width / 400
  const current = active ?? nextUnplaced

  return (
    <div className="space-y-3">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start">
        <MiniCourt active={current} placed={placed} />
        <div className="min-w-0 flex-1">
          <p className="text-sm text-fg-2">
            Pick a spot below, then click exactly where it is in the picture. Use at least <b className="text-fg">4</b> spots you can see clearly. “Left” and “right” are as seen from the camera.
          </p>
          <div className="mt-2.5 flex flex-wrap gap-1.5" role="radiogroup" aria-label="Court landmark to place">
            {COURT_LANDMARKS.map((l, i) => {
              const on = l.name === current
              const done = placed.has(l.name)
              return (
                <button
                  key={l.name}
                  type="button"
                  role="radio"
                  aria-checked={on}
                  onClick={() => setActive(l.name)}
                  className={cn(
                    'inline-flex h-8 items-center gap-1.5 rounded-lg border px-2.5 text-xs font-semibold transition',
                    on ? 'border-flame-500 bg-flame-500/15 text-fg' : done ? 'border-made/40 bg-made/10 text-fg-2' : 'border-line-strong text-fg-2 hover:border-white/25',
                  )}
                >
                  <span className={cn('grid size-4 place-items-center rounded-full text-[0.6rem] text-ink-950', done ? 'bg-made-text' : on ? 'bg-flame-500' : 'bg-fg-3')}>{done ? <Check className="size-2.5" strokeWidth={4} /> : i + 1}</span>
                  {l.label}
                </button>
              )
            })}
          </div>
        </div>
      </div>
      <PickSurface
        poster={poster}
        width={width}
        height={height}
        label="Court calibration. Click where the selected landmark appears."
        onPick={(p) => {
          if (!current) return
          const next = [...value.filter((v) => v.name !== current), { name: current, x: p[0], y: p[1] }]
          onChange(next)
          const after = COURT_LANDMARKS.find((l) => l.name !== current && !next.some((v) => v.name === l.name))
          setActive(after?.name ?? null)
        }}
        overlay={() => (
          <>
            {value.length >= 2 && (
              <polyline
                points={COURT_LANDMARKS.filter((l) => placed.has(l.name))
                  .map((l) => value.find((v) => v.name === l.name)!)
                  .map((v) => `${v.x},${v.y}`)
                  .join(' ')}
                fill="none"
                stroke="#34c759"
                strokeOpacity="0.35"
                strokeWidth={sw * 0.8}
              />
            )}
            {value.map((v) => {
              const idx = COURT_LANDMARKS.findIndex((l) => l.name === v.name)
              return (
                <g key={v.name}>
                  <circle cx={v.x} cy={v.y} r={sw * 4} fill="#070b14" fillOpacity="0.7" stroke="#34c759" strokeWidth={sw} />
                  <text x={v.x} y={v.y + sw * 1.6} textAnchor="middle" fill="#fff" fontSize={sw * 4.6} fontWeight="700" fontFamily="Inter Variable, sans-serif">
                    {idx + 1}
                  </text>
                </g>
              )
            })}
          </>
        )}
      />
      <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
        <p className={cn('font-semibold', value.length >= 4 ? 'text-made-text' : 'text-fg-2')} aria-live="polite">
          {value.length} of 4+ spots placed{value.length >= 4 ? ', good to go' : ''}
        </p>
        <div className="flex gap-1">
          {value.length > 0 && (
            <Button size="sm" variant="ghost" icon={<Undo2 className="size-3.5" />} onClick={() => onChange(value.slice(0, -1))}>
              Undo
            </Button>
          )}
          {value.length > 0 && (
            <Button
              size="sm"
              variant="ghost"
              icon={<RotateCcw className="size-3.5" />}
              onClick={() => {
                onChange([])
                setActive(COURT_LANDMARKS[0].name)
              }}
            >
              Clear all
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}
