import type {
  AddEventRequest,
  Answer,
  AppConfig,
  AskRequest,
  ClipCard,
  ClipView,
  CompareResponse,
  CreateRunResponse,
  Estimate,
  FootageType,
  Health,
  HoopsEvent,
  Job,
  RecomputeResponse,
  ReviewRequest,
  RunRequest,
  RunSummary,
  RunView,
  Session,
  UploadInit,
} from './types'

/** An error from the API: `status` is the HTTP status, `detail` the server's human message. */
export class ApiError extends Error {
  readonly status: number
  readonly detail: string
  constructor(status: number, detail: string) {
    super(detail)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

export function errorMessage(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 0) return "Can't reach the server. Is it running?"
    return err.detail
  }
  if (err instanceof DOMException && err.name === 'AbortError') return 'Cancelled'
  if (err instanceof Error) return err.message
  return 'Something went wrong'
}

export interface ChunkOptions {
  signal?: AbortSignal
  onProgress?: (loadedBytes: number) => void
}

/** Everything the UI needs from the backend. Implemented by the HTTP client and by the mock. */
export interface Api {
  readonly mock: boolean
  health(): Promise<Health>
  session(): Promise<Session>
  login(password: string): Promise<{ ok: boolean }>
  logout(): Promise<{ ok: boolean }>
  config(): Promise<AppConfig>

  createUpload(body: { filename: string; size_bytes: number; content_type: string }): Promise<UploadInit>
  putChunk(uploadId: string, index: number, chunk: Blob, opts?: ChunkOptions): Promise<{ received_bytes: number }>
  completeUpload(uploadId: string): Promise<{ job_id: string }>

  job(jobId: string): Promise<Job>
  cancelJob(jobId: string): Promise<Job>
  /** Open the SSE stream. Returns a closer. `onError` fires once if the stream drops before a terminal status. */
  streamJob(jobId: string, onJob: (job: Job) => void, onError: () => void): () => void

  clips(): Promise<ClipCard[]>
  clip(clipId: string): Promise<ClipView>
  patchClip(clipId: string, body: { footage_type: FootageType }): Promise<ClipView>

  estimate(req: RunRequest, signal?: AbortSignal): Promise<Estimate>
  createRun(req: RunRequest): Promise<CreateRunResponse>
  runs(): Promise<RunSummary[]>
  run(runId: string): Promise<RunView>
  recompute(runId: string, thresholdPreset: string): Promise<RecomputeResponse>
  review(runId: string, eventId: string, body: ReviewRequest): Promise<HoopsEvent>
  addEvent(runId: string, body: AddEventRequest): Promise<HoopsEvent>
  ask(runId: string, body: AskRequest): Promise<Answer>
  compare(a: string, b: string): Promise<CompareResponse>
}

// ------------------------------------------------------------------ 401 broadcast

type Listener = () => void
const unauthorizedListeners = new Set<Listener>()

/** Subscribe to "the server said 401" (session expired). Returns an unsubscribe function. */
export function onUnauthorized(fn: Listener): () => void {
  unauthorizedListeners.add(fn)
  return () => unauthorizedListeners.delete(fn)
}

function emitUnauthorized() {
  unauthorizedListeners.forEach((fn) => fn())
}

// ------------------------------------------------------------------ HTTP implementation

async function parseError(res: Response): Promise<ApiError> {
  let detail = res.statusText || `HTTP ${res.status}`
  try {
    const body = (await res.json()) as { detail?: unknown }
    if (typeof body.detail === 'string') detail = body.detail
    else if (Array.isArray(body.detail)) {
      // FastAPI validation errors: [{loc, msg, type}]
      detail = body.detail
        .map((d: { msg?: string; loc?: unknown[] }) => [d.loc?.slice(1).join('.'), d.msg].filter(Boolean).join(': '))
        .join('; ')
    }
  } catch {
    /* non-JSON error body */
  }
  return new ApiError(res.status, detail)
}

async function request<T>(method: string, path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  let res: Response
  try {
    res = await fetch(path, {
      method,
      credentials: 'same-origin',
      headers: body === undefined ? { Accept: 'application/json' } : { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
    })
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') throw err
    throw new ApiError(0, "Can't reach the server")
  }
  if (!res.ok) {
    const err = await parseError(res)
    const isAuthRoute = path.startsWith('/api/login') || path.startsWith('/api/session')
    if (res.status === 401 && !isAuthRoute) emitUnauthorized()
    throw err
  }
  return (await res.json()) as T
}

const enc = encodeURIComponent

function putChunkXhr(url: string, chunk: Blob, opts: ChunkOptions = {}): Promise<{ received_bytes: number }> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('PUT', url)
    xhr.withCredentials = true
    xhr.setRequestHeader('Content-Type', 'application/octet-stream')
    xhr.upload.onprogress = (e) => opts.onProgress?.(e.loaded)
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as { received_bytes: number })
        } catch {
          resolve({ received_bytes: chunk.size })
        }
        return
      }
      let detail = xhr.statusText || `HTTP ${xhr.status}`
      try {
        const body = JSON.parse(xhr.responseText) as { detail?: unknown }
        if (typeof body.detail === 'string') detail = body.detail
      } catch {
        /* ignore */
      }
      if (xhr.status === 401) emitUnauthorized()
      reject(new ApiError(xhr.status, detail))
    }
    xhr.onerror = () => reject(new ApiError(0, 'Network error while uploading'))
    xhr.ontimeout = () => reject(new ApiError(0, 'Upload timed out'))
    xhr.onabort = () => reject(new DOMException('Aborted', 'AbortError'))
    if (opts.signal) {
      if (opts.signal.aborted) {
        reject(new DOMException('Aborted', 'AbortError'))
        return
      }
      opts.signal.addEventListener('abort', () => xhr.abort(), { once: true })
    }
    xhr.send(chunk)
  })
}

export function createHttpApi(): Api {
  return {
    mock: false,
    health: () => request('GET', '/api/health'),
    session: () => request('GET', '/api/session'),
    login: (password) => request('POST', '/api/login', { password }),
    logout: () => request('POST', '/api/logout'),
    config: () => request('GET', '/api/config'),

    createUpload: (body) => request('POST', '/api/uploads', body),
    putChunk: (uploadId, index, chunk, opts) =>
      putChunkXhr(`/api/uploads/${enc(uploadId)}/chunks/${index}`, chunk, opts),
    completeUpload: (uploadId) => request('POST', `/api/uploads/${enc(uploadId)}/complete`),

    job: (jobId) => request('GET', `/api/jobs/${enc(jobId)}`),
    cancelJob: (jobId) => request('POST', `/api/jobs/${enc(jobId)}/cancel`),
    streamJob(jobId, onJob, onError) {
      if (typeof EventSource === 'undefined') {
        queueMicrotask(onError)
        return () => {}
      }
      const es = new EventSource(`/api/jobs/${enc(jobId)}/stream`, { withCredentials: true })
      let closed = false
      const close = () => {
        closed = true
        es.close()
      }
      es.addEventListener('job', (e) => {
        try {
          const job = JSON.parse((e as MessageEvent<string>).data) as Job
          onJob(job)
          if (job.status === 'done' || job.status === 'failed' || job.status === 'cancelled') close()
        } catch {
          /* malformed frame; polling fallback will catch up */
        }
      })
      es.onerror = () => {
        if (closed) return
        close()
        onError()
      }
      return close
    },

    clips: () => request('GET', '/api/clips'),
    clip: (clipId) => request('GET', `/api/clips/${enc(clipId)}`),
    patchClip: (clipId, body) => request('PATCH', `/api/clips/${enc(clipId)}`, body),

    estimate: (req, signal) => request('POST', '/api/estimate', req, signal),
    createRun: (req) => request('POST', '/api/runs', req),
    runs: () => request('GET', '/api/runs'),
    run: (runId) => request('GET', `/api/runs/${enc(runId)}`),
    recompute: (runId, threshold_preset) =>
      request('POST', `/api/runs/${enc(runId)}/recompute`, { threshold_preset }),
    review: (runId, eventId, body) =>
      request('POST', `/api/runs/${enc(runId)}/events/${enc(eventId)}/review`, body),
    addEvent: (runId, body) => request('POST', `/api/runs/${enc(runId)}/events`, body),
    ask: (runId, body) => request('POST', `/api/runs/${enc(runId)}/ask`, body),
    compare: (a, b) => request('GET', `/api/compare?a=${enc(a)}&b=${enc(b)}`),
  }
}
