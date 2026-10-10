import { useEffect, useState } from 'react'

import { ApiError, errorMessage, type Api } from './client'
import { api as defaultApi } from './index'
import { TERMINAL_JOB, type Job } from './types'

export type Transport = 'sse' | 'poll'

const POLL_MS = 1000
/**
 * Some tunnels buffer SSE silently, and the server only emits on change (AI calls can be quiet for a
 * while). If nothing arrives for this long, start polling too; the stream stays open alongside.
 */
const SSE_WATCHDOG_MS = 6000

/**
 * Follow a job until it reaches a terminal status. Uses SSE (`/api/jobs/{id}/stream`) and falls
 * back to polling `GET /api/jobs/{id}` every second when the stream errors or goes quiet.
 */
export function followJob(
  api: Api,
  jobId: string,
  onJob: (job: Job, via: Transport) => void,
  signal: AbortSignal,
): Promise<Job> {
  return new Promise<Job>((resolve, reject) => {
    let done = false
    let polling = false
    let failures = 0
    let pollTimer: ReturnType<typeof setTimeout> | undefined
    let watchdog: ReturnType<typeof setTimeout> | undefined
    let closeSse: (() => void) | null = null

    const cleanup = () => {
      closeSse?.()
      closeSse = null
      clearTimeout(pollTimer)
      clearTimeout(watchdog)
    }
    const fail = (err: unknown) => {
      if (done) return
      done = true
      cleanup()
      reject(err)
    }
    const handle = (job: Job, via: Transport) => {
      if (done) return
      onJob(job, via)
      if (TERMINAL_JOB.includes(job.status)) {
        done = true
        cleanup()
        resolve(job)
      }
    }
    const startPolling = (sseDead = false) => {
      if (sseDead) {
        closeSse?.()
        closeSse = null
      }
      if (done || polling) return
      polling = true
      clearTimeout(watchdog)
      const tick = async () => {
        if (done) return
        try {
          handle(await api.job(jobId), 'poll')
          failures = 0
        } catch (err) {
          if (err instanceof ApiError && (err.status === 401 || err.status === 404)) return fail(err)
          failures += 1
          if (failures >= 30) return fail(err)
        }
        if (!done) pollTimer = setTimeout(tick, POLL_MS)
      }
      void tick()
    }
    const armWatchdog = () => {
      clearTimeout(watchdog)
      if (!polling) watchdog = setTimeout(() => startPolling(false), SSE_WATCHDOG_MS)
    }

    signal.addEventListener('abort', () => fail(new DOMException('Aborted', 'AbortError')), { once: true })

    // Paint immediately from a plain GET, then stream.
    api.job(jobId).then(
      (job) => handle(job, polling ? 'poll' : 'sse'),
      (err) => {
        if (err instanceof ApiError && (err.status === 401 || err.status === 404)) fail(err)
      },
    )
    closeSse = api.streamJob(
      jobId,
      (job) => {
        armWatchdog()
        handle(job, 'sse')
      },
      () => startPolling(true),
    )
    armWatchdog()
  })
}

export interface JobState {
  job: Job | null
  transport: Transport | null
  error: string | null
}

/** React hook around followJob. Pass null to idle. */
export function useJob(jobId: string | null, api: Api = defaultApi): JobState {
  const [state, setState] = useState<JobState>({ job: null, transport: null, error: null })
  useEffect(() => {
    if (!jobId) return
    setState({ job: null, transport: null, error: null })
    const ac = new AbortController()
    followJob(api, jobId, (job, transport) => setState({ job, transport, error: null }), ac.signal).catch((err) => {
      if (!ac.signal.aborted) setState((s) => ({ ...s, error: errorMessage(err) }))
    })
    return () => ac.abort()
  }, [jobId, api])
  return state
}
