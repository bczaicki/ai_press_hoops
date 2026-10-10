import { useCallback, useEffect, useRef, useState } from 'react'

import { errorMessage } from '../api/client'

export function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value)
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms)
    return () => clearTimeout(t)
  }, [value, ms])
  return v
}

export interface AsyncState<T> {
  data: T | undefined
  error: string | null
  status: number | null
  loading: boolean
  reload: () => void
  setData: (updater: T | ((prev: T | undefined) => T)) => void
}

/** Minimal data-fetching hook: refetches when `deps` change; keeps the previous data while reloading. */
export function useAsync<T>(fn: () => Promise<T>, deps: unknown[]): AsyncState<T> {
  const [data, setDataState] = useState<T | undefined>(undefined)
  const [error, setError] = useState<string | null>(null)
  const [status, setStatus] = useState<number | null>(null)
  const [loading, setLoading] = useState(true)
  const [nonce, setNonce] = useState(0)
  const fnRef = useRef(fn)
  fnRef.current = fn

  useEffect(() => {
    let alive = true
    setLoading(true)
    fnRef.current().then(
      (d) => {
        if (!alive) return
        setDataState(d)
        setError(null)
        setStatus(null)
        setLoading(false)
      },
      (err: unknown) => {
        if (!alive) return
        setError(errorMessage(err))
        setStatus(typeof err === 'object' && err && 'status' in err ? Number((err as { status: number }).status) : null)
        setLoading(false)
      },
    )
    return () => {
      alive = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce])

  const reload = useCallback(() => setNonce((n) => n + 1), [])
  const setData = useCallback((u: T | ((prev: T | undefined) => T)) => {
    setDataState((prev) => (typeof u === 'function' ? (u as (p: T | undefined) => T)(prev) : u))
  }, [])
  return { data, error, status, loading, reload, setData }
}

export function useMediaQuery(query: string): boolean {
  const [match, setMatch] = useState(() => typeof window !== 'undefined' && window.matchMedia(query).matches)
  useEffect(() => {
    const mq = window.matchMedia(query)
    const on = () => setMatch(mq.matches)
    on()
    mq.addEventListener('change', on)
    return () => mq.removeEventListener('change', on)
  }, [query])
  return match
}

export function useNow(intervalMs: number, active = true): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (!active) return
    const t = setInterval(() => setNow(Date.now()), intervalMs)
    return () => clearInterval(t)
  }, [intervalMs, active])
  return now
}
