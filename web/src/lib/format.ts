export function cn(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(' ')
}

/** 75.4 -> "1:15" ; with tenths -> "1:15.4" */
export function fmtClock(t: number, tenths = false): string {
  if (!Number.isFinite(t) || t < 0) t = 0
  const m = Math.floor(t / 60)
  const s = t - m * 60
  const ss = tenths ? s.toFixed(1).padStart(4, '0') : Math.floor(s).toString().padStart(2, '0')
  return `${m}:${ss}`
}

/**
 * Parse a clock into seconds: "1:15", "00:12.40", "1:02:03", "75", "75.4s", or a ref with a clock
 * inside such as "S2#4 @ 00:12.40" (the server's resolved tile refs). Null when there's no clock.
 */
export function parseClock(ref: string): number | null {
  const s = ref.trim()
  const sec = /^(\d+(?:\.\d+)?)\s*s?$/.exec(s)
  if (sec) return Number(sec[1])
  const m = /(?<![\d.])(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)(?!\d)/.exec(s)
  if (!m) return null
  return (m[1] ? Number(m[1]) * 3600 : 0) + Number(m[2]) * 60 + Number(m[3])
}

export function fmtUsd(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return '—'
  if (v === 0) return '$0'
  if (v < 0.001) return '<$0.001'
  if (v < 0.1) return `$${v.toFixed(3)}`
  return `$${v.toFixed(2)}`
}

export function fmtPct(v: number | null | undefined, digits = 0): string {
  if (v == null || !Number.isFinite(v)) return '—'
  return `${(v * 100).toFixed(digits)}%`
}

export function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(0)} KB`
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(n < 10 * 1024 ** 2 ? 1 : 0)} MB`
  return `${(n / 1024 ** 3).toFixed(2)} GB`
}

export function fmtMs(ms: number | null | undefined): string {
  if (ms == null || !Number.isFinite(ms)) return '—'
  if (ms < 1000) return `${Math.round(ms)} ms`
  return `${(ms / 1000).toFixed(ms < 10_000 ? 1 : 0)} s`
}

export function fmtTokens(n: number | null | undefined): string {
  if (n == null) return '—'
  if (n < 1000) return `${n}`
  return `${(n / 1000).toFixed(n < 10_000 ? 1 : 0)}k`
}

export function fmtElapsed(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds))
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const r = s % 60
  return h ? `${h}:${m.toString().padStart(2, '0')}:${r.toString().padStart(2, '0')}` : `${m}:${r.toString().padStart(2, '0')}`
}

export function fmtRelative(iso: string | null | undefined): string {
  if (!iso) return ''
  const t = new Date(iso).getTime()
  if (!Number.isFinite(t)) return ''
  const diff = (Date.now() - t) / 1000
  if (diff < 45) return 'just now'
  if (diff < 3600) return `${Math.round(diff / 60)} min ago`
  if (diff < 86_400) return `${Math.round(diff / 3600)} h ago`
  if (diff < 7 * 86_400) return `${Math.round(diff / 86_400)} d ago`
  return new Date(t).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

export function fmtDateTime(iso: string | null | undefined): string {
  if (!iso) return ''
  const d = new Date(iso)
  return d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })
}

const FOOTAGE_LABELS: Record<string, string> = {
  unknown: 'Not sure yet',
  broadcast: 'TV broadcast',
  fixed_camera: 'Tripod / fixed camera',
  handheld: 'Handheld phone',
  highlights: 'Highlight reel',
  practice: 'Practice / drills',
  not_basketball: 'Not basketball',
}

export function footageLabel(key: string | null | undefined): string {
  if (!key) return 'Auto'
  return FOOTAGE_LABELS[key] ?? key.replace(/_/g, ' ')
}

export function plural(n: number, one: string, many = `${one}s`): string {
  return `${n} ${n === 1 ? one : many}`
}
