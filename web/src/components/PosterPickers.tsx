import { Check, Crosshair, RotateCcw, Undo2 } from 'lucide-react'
import { useRef, useState, type PointerEvent, type ReactNode } from 'react'

import type { CourtPoint, CourtSpec, RimSuggestion } from '../api/types'
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

// ------------------------------------------------------------------ rim suggestions

/** A zoomed crop of the poster around one suggested hoop, so a small rim is easy to recognize and tap. */
function HoopCrop({ poster, width, height, box }: { poster: string; width: number; height: number; box: RimSuggestion['box'] }) {
  const TW = 168
  const TH = 96
  const bw = box[2] - box[0]
  const bh = box[3] - box[1]
  // Crop ~3.2x the box, at the thumbnail's aspect ratio, clamped to the frame.
  let cw = Math.max(bw * 3.2, bh * 3.2 * (TW / TH), 90)
  let ch = cw * (TH / TW)
  cw = Math.min(cw, width)
  ch = Math.min(ch, height)
  const cx = Math.min(Math.max(0, (box[0] + box[2]) / 2 - cw / 2), width - cw)
  const cy = Math.min(Math.max(0, (box[1] + box[3]) / 2 - ch / 2), height - ch)
  const k = TW / cw
  return (
    <span
      aria-hidden
      className="relative block overflow-hidden rounded-lg bg-ink-800"
      style={{ width: TW, height: TH, backgroundImage: `url("${poster}")`, backgroundRepeat: 'no-repeat', backgroundSize: `${width * k}px ${height * k}px`, backgroundPosition: `${-cx * k}px ${-cy * k}px` }}
    >
      <span className="absolute rounded-[3px] border-2 border-flame-400" style={{ left: (box[0] - cx) * k - 3, top: (box[1] - cy) * k - 3, width: bw * k + 6, height: bh * k + 6 }} />
    </span>
  )
}

export function RimSuggestionPicker({
  poster,
  width,
  height,
  state,
  value,
  onChange,
  onDraw,
}: {
  poster: string
  width: number
  height: number
  state: { loading: boolean; error: string | null; data: RimSuggestion[] | null }
  value: Box | null
  onChange: (b: Box | null) => void
  onDraw: () => void
}) {
  const sw = width / 400
  const same = (a: Box | null, b: RimSuggestion['box']) => !!a && a.every((v, i) => Math.abs(v - b[i]) < 0.5)
  const pick = (b: RimSuggestion['box']) => onChange(same(value, b) ? null : [Math.round(b[0]), Math.round(b[1]), Math.round(b[2]), Math.round(b[3])])
  const list = state.data ?? []

  return (
    <div className="space-y-3">
      <div className="relative w-full overflow-hidden rounded-xl border border-line-strong bg-ink-800" style={{ aspectRatio: `${width} / ${height}` }}>
        <img src={poster} alt="" draggable={false} className="absolute inset-0 h-full w-full object-fill" />
        <svg viewBox={`0 0 ${width} ${height}`} className="absolute inset-0 h-full w-full" preserveAspectRatio="none">
          {list.map((s, i) => {
            const on = same(value, s.box)
            const [x1, y1, x2, y2] = s.box
            const pad = Math.max(28, (x2 - x1) * 0.6)
            return (
              <g key={i} className="cursor-pointer" onClick={() => pick(s.box)}>
                <rect x={x1 - pad} y={y1 - pad} width={x2 - x1 + pad * 2} height={y2 - y1 + pad * 2} fill="transparent" />
                <rect
                  x={x1 - sw * 2}
                  y={y1 - sw * 2}
                  width={x2 - x1 + sw * 4}
                  height={y2 - y1 + sw * 4}
                  rx={sw * 1.5}
                  fill={on ? 'rgb(255 122 26 / 0.22)' : 'rgb(7 11 20 / 0.15)'}
                  stroke={on ? '#ff9447' : '#ffffff'}
                  strokeOpacity={on ? 1 : 0.85}
                  strokeWidth={on ? sw * 1.3 : sw * 0.9}
                  strokeDasharray={on ? undefined : `${sw * 3} ${sw * 2}`}
                />
                <g transform={`translate(${x1 - sw * 2}, ${y1 - sw * 13})`}>
                  <rect width={sw * 17} height={sw * 9} rx={sw * 2} fill={on ? '#ff7a1a' : 'rgb(7 11 20 / 0.85)'} />
                  <text x={sw * 8.5} y={sw * 6.4} textAnchor="middle" fontSize={sw * 6} fontWeight="700" fill={on ? '#070b14' : '#fff'} fontFamily="Inter Variable, sans-serif">
                    {i + 1}
                  </text>
                </g>
              </g>
            )
          })}
        </svg>
        {state.loading && (
          <div className="absolute inset-0 grid place-items-center bg-ink-950/45">
            <div className="skeleton absolute inset-0 rounded-none opacity-60" />
            <span className="relative rounded-full bg-ink-950/85 px-3 py-1.5 text-xs font-semibold text-fg-2">Looking for hoops… the first look can take ~20 s</span>
          </div>
        )}
      </div>

      {state.loading ? (
        <div className="flex gap-2">
          <div className="skeleton h-[7.5rem] w-44" />
          <div className="skeleton h-[7.5rem] w-44" />
        </div>
      ) : list.length ? (
        <div role="radiogroup" aria-label="Which hoop is being played at?" className="flex flex-wrap gap-2.5">
          {list.map((s, i) => {
            const on = same(value, s.box)
            return (
              <button
                key={i}
                type="button"
                role="radio"
                aria-checked={on}
                onClick={() => pick(s.box)}
                className={cn('group rounded-xl border p-1.5 text-left transition', on ? 'border-flame-500 bg-flame-500/10 shadow-[0_0_0_1px_rgb(255_122_26/0.35)]' : 'border-line-strong bg-white/[0.02] hover:border-white/30')}
              >
                <HoopCrop poster={poster} width={width} height={height} box={s.box} />
                <span className="flex items-center justify-between gap-2 px-1 pt-1.5 pb-0.5 text-xs">
                  <span className="font-semibold text-fg">
                    Hoop {i + 1}
                    {i === 0 && <span className="ml-1 font-normal text-fg-3">best guess</span>}
                  </span>
                  <span className="text-fg-3 tnum">{Math.round(s.score * 100)}%</span>
                </span>
              </button>
            )
          })}
        </div>
      ) : (
        <p className="rounded-xl border border-line bg-white/[0.02] p-3 text-sm text-fg-2">
          {state.error ? <>Couldn't look for hoops ({state.error}). </> : <>No hoop spotted in this clip. </>}
          Draw it yourself, or skip this step and the analysis will pick the hoop the ball goes toward.
        </p>
      )}

      <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
        <p className="text-fg-2" aria-live="polite">
          {value ? (
            <>
              <Check className="mr-1 inline size-4 text-made-text" aria-hidden />
              Hoop set. You can tap it again to clear it.
            </>
          ) : list.length ? (
            <>
              Tap the hoop <b className="text-fg">being played at</b>.
            </>
          ) : null}
        </p>
        <Button size="sm" variant="ghost" icon={<Crosshair className="size-3.5" />} onClick={onDraw}>
          {list.length ? 'None of these? Draw it' : 'Draw it myself'}
        </Button>
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ court calibration

/** Used only when an older server sends no `courts` in /api/config. */
const FALLBACK_COURT: CourtSpec = {
  key: 'nba',
  label: 'NBA',
  length_ft: 94,
  width_ft: 50,
  landmarks: [
    { name: 'baseline_left_corner', label: 'Baseline left corner', x_ft: 47, y_ft: 25, filmed_half: true },
    { name: 'baseline_right_corner', label: 'Baseline right corner', x_ft: 47, y_ft: -25, filmed_half: true },
    { name: 'free_throw_left', label: 'Free-throw line, left end', x_ft: 28, y_ft: 8, filmed_half: true },
    { name: 'free_throw_right', label: 'Free-throw line, right end', x_ft: 28, y_ft: -8, filmed_half: true },
    { name: 'halfcourt_left', label: 'Half-court line, left sideline', x_ft: 0, y_ft: 25, filmed_half: true },
    { name: 'halfcourt_right', label: 'Half-court line, right sideline', x_ft: 0, y_ft: -25, filmed_half: true },
  ],
}

/**
 * Full court, filmed basket at the TOP, as the camera sees it (camera-left = diagram left).
 * Court feet (x toward the filmed basket, +y camera-left) -> svg (W/2 - y, L/2 - x).
 */
function MiniCourt({ court, next, placed }: { court: CourtSpec; next: string | null; placed: Set<string> }) {
  const L = court.length_ft
  const Wd = court.width_ft
  const hasFar = court.landmarks.some((l) => !l.filmed_half)
  const viewH = hasFar ? L : L / 2 + 4
  const toSvg = (x: number, y: number): [number, number] => [Wd / 2 - y, L / 2 - x]
  return (
    <svg viewBox={`-3 -3 ${Wd + 6} ${viewH + 8}`} className={cn('w-auto shrink-0 self-center sm:self-start', hasFar ? 'h-52' : 'h-36')} aria-hidden>
      <g fill="none" stroke="#e8c08f" strokeWidth="0.55">
        <rect x="0" y="0" width={Wd} height={L} strokeOpacity="0.6" />
        <line x1="0" y1={L / 2} x2={Wd} y2={L / 2} strokeOpacity="0.6" />
        <circle cx={Wd / 2} cy={L / 2} r="6" strokeOpacity="0.6" />
        <rect x={Wd / 2 - 8} y="0" width="16" height="19" strokeOpacity="0.6" />
        <circle cx={Wd / 2} cy="19" r="6" strokeOpacity="0.6" />
        <path d={`M${Wd / 2 - 22} 0 V 14 A 23.75 23.75 0 0 0 ${Wd / 2 + 22} 14 V 0`} strokeOpacity="0.6" />
        <circle cx={Wd / 2} cy="5.25" r="0.9" stroke="#ff7a1a" />
        <g strokeOpacity="0.25">
          <rect x={Wd / 2 - 8} y={L - 19} width="16" height="19" />
          <path d={`M${Wd / 2 - 22} ${L} V ${L - 14} A 23.75 23.75 0 0 1 ${Wd / 2 + 22} ${L - 14} V ${L}`} />
        </g>
      </g>
      <text x={Wd / 2} y={viewH + 3.6} textAnchor="middle" fontSize="2.8" fill="#7d879c" fontFamily="Inter Variable, sans-serif">
        {hasFar ? 'far end' : 'camera'}
      </text>
      {court.landmarks.filter((l) => !/(^|_)rim$/.test(l.name)).map((l) => {
        const [cx, cy] = toSvg(l.x_ft, l.y_ft)
        if (cy > viewH + 1) return null
        const isNext = l.name === next
        const done = placed.has(l.name)
        return (
          <g key={l.name}>
            {isNext && (
              <circle cx={cx} cy={cy} r="2" fill="none" stroke="#ff7a1a" strokeWidth="0.8">
                <animate attributeName="r" values="1.8;4.2;1.8" dur="1.6s" repeatCount="indefinite" />
                <animate attributeName="opacity" values="1;0.2;1" dur="1.6s" repeatCount="indefinite" />
              </circle>
            )}
            <circle cx={cx} cy={cy} r={isNext ? 1.6 : 1.15} fill={done ? '#34c759' : isNext ? '#ff7a1a' : l.filmed_half ? '#a3adc2' : '#56607a'} />
          </g>
        )
      })}
    </svg>
  )
}

export function CourtPointsPicker({
  poster,
  width,
  height,
  value,
  onChange,
  courts,
}: {
  poster: string
  width: number
  height: number
  value: CourtPoint[]
  onChange: (pts: CourtPoint[]) => void
  courts?: CourtSpec[]
}) {
  // The server calibrates with NBA geometry for now (RunRequest has no court field), so use NBA.
  const court = courts?.find((c) => c.key === 'nba') ?? courts?.[0] ?? FALLBACK_COURT
  // Rims are elevated, so they can't be floor landmarks. Filmed-half landmarks first: a half-court camera sees those.
  const landmarks = court.landmarks.filter((l) => !/(^|_)rim$/.test(l.name)).sort((a, b) => Number(b.filmed_half) - Number(a.filmed_half))
  const placed = new Set(value.map((p) => p.name))
  const firstOpen = landmarks.find((l) => !placed.has(l.name))?.name ?? null
  const [active, setActive] = useState<string | null>(firstOpen)
  const current = active && landmarks.some((l) => l.name === active) ? active : firstOpen
  const currentLabel = landmarks.find((l) => l.name === current)?.label
  const sw = width / 400

  return (
    <div className="space-y-3">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start">
        <MiniCourt court={court} next={current} placed={placed} />
        <div className="min-w-0 flex-1 space-y-3">
          <p className="text-sm text-fg-2" aria-live="polite">
            {current ? (
              <>
                Next: click <b className="text-fg">{currentLabel}</b> in the picture. It's the pulsing dot on the diagram.
              </>
            ) : (
              <>Every landmark is placed.</>
            )}{' '}
            Use at least <b className="text-fg">4</b> you can see clearly. “Left” means the camera's left.
          </p>
          <div className="flex flex-wrap gap-1.5" role="radiogroup" aria-label="Court landmark to place">
            {landmarks.map((l, i) => {
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
                    'inline-flex min-h-8 items-center gap-1.5 rounded-lg border px-2.5 py-1 text-left text-xs font-semibold transition',
                    on ? 'border-flame-500 bg-flame-500/15 text-fg' : done ? 'border-made/40 bg-made/10 text-fg-2' : 'border-line-strong text-fg-2 hover:border-white/25',
                    !l.filmed_half && !on && !done && 'opacity-60',
                  )}
                >
                  <span className={cn('grid size-4 shrink-0 place-items-center rounded-full text-[0.6rem] text-ink-950', done ? 'bg-made-text' : on ? 'bg-flame-500' : 'bg-fg-3')}>
                    {done ? <Check className="size-2.5" strokeWidth={4} /> : i + 1}
                  </span>
                  {l.label}
                  {!l.filmed_half && <span className="font-normal text-fg-3">(far end)</span>}
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
        label={`Court calibration. Click where ${currentLabel ?? 'the selected landmark'} appears.`}
        onPick={(p) => {
          if (!current) return
          const next = [...value.filter((v) => v.name !== current), { name: current, x: p[0], y: p[1] }]
          onChange(next)
          const after = landmarks.find((l) => l.name !== current && !next.some((v) => v.name === l.name))
          setActive(after?.name ?? null)
        }}
        overlay={() => (
          <>
            {value.map((v) => {
              const idx = landmarks.findIndex((l) => l.name === v.name)
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
          {value.length} of 4+ landmarks placed{value.length >= 4 ? ', good to go' : ''}
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
                setActive(landmarks[0]?.name ?? null)
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
