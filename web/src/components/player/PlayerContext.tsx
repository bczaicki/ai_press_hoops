import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from 'react'

export interface PlayerEngine {
  seek: (t: number) => void
  play: () => void
  element: () => HTMLElement | null
}

interface PlayerCtx {
  /** Seek the video; scrolls it into view when it's off screen (e.g. clicked from the event log). */
  seek: (t: number, opts?: { play?: boolean; reveal?: boolean }) => void
  getTime: () => number
  /** Coarse playback time (~4 Hz) for components that only need "roughly now". */
  time: number
  register: (engine: PlayerEngine | null) => void
  report: (t: number) => void
}

const Ctx = createContext<PlayerCtx | null>(null)

export function PlayerProvider({ children }: { children: ReactNode }) {
  const engine = useRef<PlayerEngine | null>(null)
  const timeRef = useRef(0)
  const lastPush = useRef(0)
  const [time, setTime] = useState(0)

  const report = useCallback((t: number) => {
    timeRef.current = t
    const now = performance.now()
    if (now - lastPush.current > 240) {
      lastPush.current = now
      setTime(t)
    }
  }, [])

  const seek = useCallback((t: number, opts?: { play?: boolean; reveal?: boolean }) => {
    const e = engine.current
    if (!e) return
    e.seek(Math.max(0, t))
    timeRef.current = Math.max(0, t)
    setTime(Math.max(0, t))
    if (opts?.play) e.play()
    if (opts?.reveal !== false) {
      const el = e.element()
      if (el) {
        const r = el.getBoundingClientRect()
        const visible = r.top >= 0 && r.top < window.innerHeight * 0.6 && r.bottom > 80
        if (!visible) el.scrollIntoView({ behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' })
      }
    }
  }, [])

  const value = useMemo<PlayerCtx>(
    () => ({
      seek,
      getTime: () => timeRef.current,
      time,
      register: (e) => {
        engine.current = e
      },
      report,
    }),
    [seek, time, report],
  )
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

const noop: PlayerCtx = { seek: () => {}, getTime: () => 0, time: 0, register: () => {}, report: () => {} }

export function usePlayer(): PlayerCtx {
  return useContext(Ctx) ?? noop
}
