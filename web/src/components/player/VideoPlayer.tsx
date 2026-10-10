import { AlertTriangle, Captions, CaptionsOff, Maximize2, Pause, Play, RotateCcw } from 'lucide-react'
import { useCallback, useEffect, useRef, useState, type KeyboardEvent } from 'react'

import { isMockMedia } from '../../api'
import type { HoopsEvent, TeamRef } from '../../api/types'
import { cn, fmtClock } from '../../lib/format'
import { Segmented } from '../ui/segmented'
import { Tip } from '../ui/tip'
import { EventTimeline } from './EventTimeline'
import { usePlayer } from './PlayerContext'
import { SimulatedVideo } from './SimulatedVideo'

export interface Source {
  key: string
  label: string
  url: string
}

const RATES = [0.5, 1, 1.5, 2]

/**
 * Video + custom controls + event timeline. Real media plays in a <video> (Range-served MP4);
 * mock media ("mock://") renders the canvas simulation with the same clock and controls.
 */
export function VideoPlayer({
  sources,
  captionsUrl,
  poster,
  fallbackDuration,
  events,
  teams,
  className,
}: {
  sources: Source[]
  captionsUrl: string | null
  poster: string | null
  fallbackDuration: number
  events: HoopsEvent[]
  teams: TeamRef[]
  className?: string
}) {
  const player = usePlayer()
  const root = useRef<HTMLDivElement>(null)
  const video = useRef<HTMLVideoElement>(null)
  const [srcKey, setSrcKey] = useState(sources[0]?.key ?? '')
  const source = sources.find((s) => s.key === srcKey) ?? sources[0]
  const sim = isMockMedia(source?.url)
  const [playing, setPlaying] = useState(false)
  const [time, setTime] = useState(0)
  const [duration, setDuration] = useState(fallbackDuration)
  const [rate, setRate] = useState(1)
  const [cc, setCc] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [buffering, setBuffering] = useState(false)
  const simClock = useRef({ t: 0, last: 0 })
  const pending = useRef<{ t: number; play: boolean } | null>(null)
  const timeRef = useRef(0)

  const report = player.report
  const update = useCallback(
    (t: number) => {
      timeRef.current = t
      setTime(t)
      report(t)
    },
    [report],
  )

  // Animation loop while playing: read the video clock, or advance the simulated one.
  useEffect(() => {
    if (!playing) return
    let raf = 0
    const tick = (now: number) => {
      if (sim) {
        const c = simClock.current
        const dt = c.last ? (now - c.last) / 1000 : 0
        c.last = now
        c.t = Math.min(duration, c.t + dt * rate)
        update(c.t)
        if (c.t >= duration) {
          setPlaying(false)
          return
        }
      } else if (video.current) update(video.current.currentTime)
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => {
      cancelAnimationFrame(raf)
      simClock.current.last = 0
    }
  }, [playing, sim, duration, rate, update])

  const play = useCallback(() => {
    if (sim) {
      if (simClock.current.t >= duration - 0.05) simClock.current.t = 0
      setPlaying(true)
    } else video.current?.play().catch(() => setPlaying(false))
  }, [sim, duration])
  const pause = useCallback(() => {
    if (sim) setPlaying(false)
    else video.current?.pause()
  }, [sim])
  const toggle = () => (playing ? pause() : play())
  const seek = useCallback(
    (t: number) => {
      const c = Math.max(0, Math.min(duration || t, t))
      if (sim) simClock.current.t = c
      else if (video.current) video.current.currentTime = c
      update(c)
    },
    [sim, duration, update],
  )

  useEffect(() => {
    player.register({ seek, play, element: () => root.current })
    return () => player.register(null)
  }, [player, seek, play])

  // Keep playback position when switching annotated <-> original.
  const switchSource = (key: string) => {
    if (key === srcKey) return
    pending.current = { t: timeRef.current, play: playing }
    setError(null)
    setSrcKey(key)
    const next = sources.find((s) => s.key === key)
    if (next && isMockMedia(next.url)) {
      simClock.current.t = timeRef.current
      pending.current = null
    }
  }

  useEffect(() => {
    if (video.current) video.current.playbackRate = rate
  }, [rate, srcKey])

  useEffect(() => {
    const tracks = video.current?.textTracks
    if (tracks) for (const t of Array.from(tracks)) t.mode = cc ? 'showing' : 'hidden'
  }, [cc, srcKey, captionsUrl])

  const fullscreen = () => {
    const el = root.current as (HTMLDivElement & { webkitRequestFullscreen?: () => void }) | null
    const v = video.current as (HTMLVideoElement & { webkitEnterFullscreen?: () => void }) | null
    if (document.fullscreenElement) void document.exitFullscreen()
    else if (el?.requestFullscreen) void el.requestFullscreen()
    else if (el?.webkitRequestFullscreen) el.webkitRequestFullscreen()
    else v?.webkitEnterFullscreen?.()
  }

  const onKey = (e: KeyboardEvent) => {
    if (e.target !== e.currentTarget && (e.target as HTMLElement).getAttribute('role') === 'slider') return
    const k = e.key.toLowerCase()
    if (k === ' ' || k === 'k') toggle()
    else if (k === 'arrowleft' || k === 'j') seek(timeRef.current - (k === 'j' ? 10 : 5))
    else if (k === 'arrowright' || k === 'l') seek(timeRef.current + (k === 'l' ? 10 : 5))
    else if (k === ',') seek(timeRef.current - 1 / 30)
    else if (k === '.') seek(timeRef.current + 1 / 30)
    else if (k === 'f') fullscreen()
    else if (k === 'c') setCc((v) => !v)
    else return
    e.preventDefault()
  }

  const hasCaptions = sim || !!captionsUrl

  return (
    <div ref={root} className={cn('glass overflow-hidden rounded-[var(--radius-card)] [&:fullscreen]:flex [&:fullscreen]:flex-col [&:fullscreen]:justify-center [&:fullscreen]:bg-black', className)}>
      <div
        className="group relative aspect-video w-full bg-black outline-none focus-visible:ring-2 focus-visible:ring-flame-400 focus-visible:ring-inset"
        tabIndex={0}
        role="region"
        aria-label="Video player. Space plays or pauses, arrows seek, F for fullscreen, C for captions."
        onKeyDown={onKey}
      >
        {source && sim ? (
          <SimulatedVideo time={time} annotated={source.key === 'annotated'} cc={cc} events={events} teams={teams} poster={poster} />
        ) : source ? (
          <video
            key={source.url}
            ref={video}
            src={source.url}
            poster={poster ?? undefined}
            playsInline
            preload="metadata"
            className="absolute inset-0 h-full w-full"
            onClick={toggle}
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
            onEnded={() => setPlaying(false)}
            onWaiting={() => setBuffering(true)}
            onPlaying={() => setBuffering(false)}
            onCanPlay={() => setBuffering(false)}
            onTimeUpdate={(e) => !playing && update(e.currentTarget.currentTime)}
            onSeeked={(e) => update(e.currentTarget.currentTime)}
            onLoadedMetadata={(e) => {
              const v = e.currentTarget
              if (Number.isFinite(v.duration) && v.duration > 0) setDuration(v.duration)
              v.playbackRate = rate
              const p = pending.current
              if (p) {
                v.currentTime = p.t
                if (p.play) void v.play().catch(() => {})
                pending.current = null
              }
            }}
            onError={() => setError(`Couldn't load the ${source.label.toLowerCase()} video.`)}
          >
            {captionsUrl && <track kind="captions" srcLang="en" label="Events" src={captionsUrl} default />}
          </video>
        ) : null}

        {sim && (
          <button type="button" className="absolute inset-0" onClick={toggle} aria-label={playing ? 'Pause' : 'Play'} tabIndex={-1} />
        )}

        {!playing && !error && (
          <button
            type="button"
            onClick={toggle}
            tabIndex={-1}
            aria-hidden
            className="absolute top-1/2 left-1/2 grid size-16 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full bg-flame-500/90 text-ink-950 shadow-[0_12px_40px_-8px_rgb(255_122_26/0.8)] transition hover:scale-105 sm:size-20"
          >
            <Play className="ml-1 size-7 sm:size-9" fill="currentColor" />
          </button>
        )}
        {buffering && playing && <div className="absolute top-3 right-3 size-6 animate-spin rounded-full border-2 border-white/30 border-t-white" aria-hidden />}
        {error && (
          <div role="alert" className="absolute inset-0 grid place-items-center bg-ink-950/85 p-6 text-center">
            <div>
              <AlertTriangle className="mx-auto size-7 text-attempt" aria-hidden />
              <p className="mt-2 font-semibold text-fg">{error}</p>
              <p className="mt-1 text-sm text-fg-2">The file may still be rendering, or the connection dropped.</p>
              <button
                type="button"
                className="mt-3 inline-flex items-center gap-1.5 text-sm font-semibold text-flame-300 hover:text-flame-200"
                onClick={() => {
                  setError(null)
                  video.current?.load()
                }}
              >
                <RotateCcw className="size-4" /> Try again
              </button>
            </div>
          </div>
        )}
      </div>

      {/* controls */}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2 px-3 pt-3 sm:px-4">
        <button type="button" onClick={toggle} className="grid size-10 place-items-center rounded-xl bg-white/[0.08] text-fg hover:bg-white/[0.14]" aria-label={playing ? 'Pause' : 'Play'}>
          {playing ? <Pause className="size-5" fill="currentColor" /> : <Play className="ml-0.5 size-5" fill="currentColor" />}
        </button>
        <div className="font-mono text-sm text-fg tnum" aria-live="off">
          {fmtClock(time)} <span className="text-fg-3">/ {fmtClock(duration)}</span>
        </div>
        {sources.length > 1 && <Segmented size="sm" label="Video version" value={srcKey} onChange={switchSource} options={sources.map((s) => ({ key: s.key, label: s.label }))} />}
        <div className="ml-auto flex items-center gap-1">
          <Tip content="Playback speed">
            <button
              type="button"
              onClick={() => setRate((r) => RATES[(RATES.indexOf(r) + 1) % RATES.length])}
              className="h-9 min-w-12 rounded-xl px-2 text-sm font-semibold text-fg-2 tnum hover:bg-white/[0.08] hover:text-fg"
              aria-label={`Playback speed ${rate}x`}
            >
              {rate}×
            </button>
          </Tip>
          {hasCaptions && (
            <Tip content={cc ? 'Hide event captions (C)' : 'Show event captions (C)'}>
              <button type="button" onClick={() => setCc((v) => !v)} className={cn('grid size-9 place-items-center rounded-xl hover:bg-white/[0.08]', cc ? 'text-flame-300' : 'text-fg-3')} aria-pressed={cc} aria-label="Event captions">
                {cc ? <Captions className="size-5" /> : <CaptionsOff className="size-5" />}
              </button>
            </Tip>
          )}
          <Tip content="Fullscreen (F)">
            <button type="button" onClick={fullscreen} className="grid size-9 place-items-center rounded-xl text-fg-2 hover:bg-white/[0.08] hover:text-fg" aria-label="Fullscreen">
              <Maximize2 className="size-4.5" />
            </button>
          </Tip>
        </div>
      </div>

      <div className="px-1 pt-2 pb-3 sm:px-2">
        <EventTimeline events={events} teams={teams} duration={duration} time={time} onSeek={seek} />
      </div>
    </div>
  )
}
