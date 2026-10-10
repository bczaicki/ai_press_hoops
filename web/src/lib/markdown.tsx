import type { ReactNode } from 'react'

import { Tip } from '../components/ui/tip'
import { cn, parseClock } from './format'

/**
 * A deliberately tiny, safe markdown renderer for model output. It only ever creates React nodes
 * (no innerHTML), allows http(s) links only, turns "m:ss" timestamps into seek chips, and
 * highlights sentences System One's claim check flagged as unsupported.
 */
interface Options {
  onTime?: (t: number) => void
  flags?: string[]
}

type Block =
  | { kind: 'h'; level: number; text: string }
  | { kind: 'p'; text: string }
  | { kind: 'ul' | 'ol'; items: string[] }
  | { kind: 'quote'; text: string }
  | { kind: 'code'; text: string }

function parseBlocks(src: string): Block[] {
  const lines = src.replace(/\r\n?/g, '\n').split('\n')
  const out: Block[] = []
  let para: string[] = []
  let list: { kind: 'ul' | 'ol'; items: string[] } | null = null
  let code: string[] | null = null
  const flushPara = () => {
    if (para.length) out.push({ kind: 'p', text: para.join(' ') })
    para = []
  }
  const flushList = () => {
    if (list) out.push(list)
    list = null
  }
  for (const raw of lines) {
    if (code) {
      if (/^```/.test(raw.trim())) {
        out.push({ kind: 'code', text: code.join('\n') })
        code = null
      } else code.push(raw)
      continue
    }
    if (/^```/.test(raw.trim())) {
      flushPara()
      flushList()
      code = []
      continue
    }
    const line = raw.trimEnd()
    if (!line.trim()) {
      flushPara()
      flushList()
      continue
    }
    const h = /^(#{1,4})\s+(.*)$/.exec(line)
    if (h) {
      flushPara()
      flushList()
      out.push({ kind: 'h', level: h[1].length, text: h[2] })
      continue
    }
    const ul = /^\s*[-*•]\s+(.*)$/.exec(line)
    const ol = /^\s*\d+[.)]\s+(.*)$/.exec(line)
    if (ul || ol) {
      flushPara()
      const kind = ul ? 'ul' : 'ol'
      if (!list || list.kind !== kind) {
        flushList()
        list = { kind, items: [] }
      }
      list.items.push((ul ?? ol)![1])
      continue
    }
    if (/^>\s?/.test(line)) {
      flushPara()
      flushList()
      out.push({ kind: 'quote', text: line.replace(/^>\s?/, '') })
      continue
    }
    if (list) {
      const l = list as { kind: 'ul' | 'ol'; items: string[] }
      l.items[l.items.length - 1] += ` ${line.trim()}`
      continue
    }
    para.push(line.trim())
  }
  if (code) out.push({ kind: 'code', text: (code as string[]).join('\n') })
  flushPara()
  flushList()
  return out
}

const INLINE = /(\*\*[^*]+\*\*|__[^_]+__|\*[^*\s][^*]*\*|_[^_\s][^_]*_|`[^`]+`|\[[^\]]+\]\([^)\s]+\)|(?<![\d.:])(?:\d+:)?\d{1,2}:\d{2}(?:\.\d+)?(?![\d]))/g

function inline(text: string, opts: Options, key: string): ReactNode[] {
  const out: ReactNode[] = []
  let last = 0
  let i = 0
  for (const m of text.matchAll(INLINE)) {
    const tok = m[0]
    const at = m.index ?? 0
    if (at > last) out.push(text.slice(last, at))
    const k = `${key}-${i++}`
    if (tok.startsWith('**') || tok.startsWith('__')) {
      out.push(<strong key={k} className="font-semibold text-fg">{inline(tok.slice(2, -2), opts, k)}</strong>)
    } else if (tok.startsWith('`')) {
      out.push(<code key={k} className="rounded bg-white/8 px-1 py-0.5 font-mono text-[0.85em]">{tok.slice(1, -1)}</code>)
    } else if (tok.startsWith('[')) {
      const lm = /^\[([^\]]+)\]\(([^)\s]+)\)$/.exec(tok)
      const href = lm?.[2] ?? ''
      if (lm && /^https?:\/\//i.test(href)) {
        out.push(
          <a key={k} href={href} target="_blank" rel="noopener noreferrer nofollow" className="text-flame-300 underline decoration-flame-500/40 underline-offset-2 hover:decoration-flame-300">
            {lm[1]}
          </a>,
        )
      } else out.push(lm?.[1] ?? tok)
    } else if (tok.startsWith('*') || tok.startsWith('_')) {
      out.push(<em key={k}>{inline(tok.slice(1, -1), opts, k)}</em>)
    } else {
      const t = parseClock(tok)
      if (t != null && opts.onTime) {
        const onTime = opts.onTime
        out.push(
          <button
            key={k}
            type="button"
            onClick={() => onTime(t)}
            className="mx-0.5 inline-flex items-center rounded-md bg-flame-500/12 px-1.5 py-px align-baseline text-[0.92em] font-semibold text-flame-300 tnum transition hover:bg-flame-500/22 hover:text-flame-200"
            aria-label={`Jump to ${tok}`}
          >
            {tok}
          </button>,
        )
      } else out.push(tok)
    }
    last = at + tok.length
  }
  if (last < text.length) out.push(text.slice(last))
  return out
}

/** Split text into normal / flagged spans by searching for each flagged claim. */
function withFlags(text: string, opts: Options, key: string): ReactNode[] {
  const flags = (opts.flags ?? []).map((f) => f.trim().replace(/[.!?]+$/, '')).filter((f) => f.length > 8)
  if (!flags.length) return inline(text, opts, key)
  const ranges: [number, number][] = []
  for (const f of flags) {
    const at = text.indexOf(f)
    if (at >= 0) ranges.push([at, at + f.length])
  }
  if (!ranges.length) return inline(text, opts, key)
  ranges.sort((a, b) => a[0] - b[0])
  const out: ReactNode[] = []
  let last = 0
  ranges.forEach(([s, e], i) => {
    if (s < last) return
    if (s > last) out.push(...inline(text.slice(last, s), opts, `${key}n${i}`))
    out.push(
      <Tip key={`${key}f${i}`} content="System One's claim check couldn't match this to the event log. Treat it with caution.">
        <mark tabIndex={0} className="rounded-sm bg-attempt/12 px-0.5 text-fg underline decoration-attempt decoration-wavy decoration-1 underline-offset-4 outline-none focus-visible:ring-2 focus-visible:ring-attempt">
          {inline(text.slice(s, e), opts, `${key}m${i}`)}
        </mark>
      </Tip>,
    )
    last = e
  })
  if (last < text.length) out.push(...inline(text.slice(last), opts, `${key}t`))
  return out
}

export function Markdown({ text, onTime, flags, className }: { text: string; onTime?: (t: number) => void; flags?: string[]; className?: string }) {
  const opts: Options = { onTime, flags }
  const blocks = parseBlocks(text)
  return (
    <div className={cn('space-y-3 text-[0.95rem] leading-relaxed text-fg-2', className)}>
      {blocks.map((b, i) => {
        const k = `b${i}`
        switch (b.kind) {
          case 'h':
            return (
              <p key={k} className={cn('font-display font-semibold uppercase tracking-wide text-fg', b.level <= 2 ? 'text-lg' : 'text-base')}>
                {withFlags(b.text, opts, k)}
              </p>
            )
          case 'p':
            return <p key={k}>{withFlags(b.text, opts, k)}</p>
          case 'ul':
          case 'ol': {
            const Tag = b.kind
            return (
              <Tag key={k} className={cn('space-y-1.5 pl-5', b.kind === 'ul' ? 'list-disc marker:text-flame-500' : 'list-decimal marker:text-fg-3')}>
                {b.items.map((it, j) => (
                  <li key={j}>{withFlags(it, opts, `${k}-${j}`)}</li>
                ))}
              </Tag>
            )
          }
          case 'quote':
            return (
              <blockquote key={k} className="border-l-2 border-flame-500/60 pl-3 italic">
                {withFlags(b.text, opts, k)}
              </blockquote>
            )
          case 'code':
            return (
              <pre key={k} className="overflow-x-auto rounded-lg bg-black/40 p-3 font-mono text-xs text-fg-2 scrollbar-thin">
                {b.text}
              </pre>
            )
        }
      })}
    </div>
  )
}
