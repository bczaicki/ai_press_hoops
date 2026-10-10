import { ApiError, type Api } from './client'

export interface UploadProgress {
  phase: 'starting' | 'uploading' | 'retrying' | 'finishing'
  sentBytes: number
  totalBytes: number
  chunkIndex: number
  chunkCount: number
  bytesPerSec: number
  retry: { attempt: number; maxAttempts: number; inSeconds: number; reason: string } | null
}

const MAX_ATTEMPTS = 5
const NOT_RETRYABLE = new Set([400, 401, 402, 403, 404, 409, 413, 415, 422])
const sleep = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const t = setTimeout(resolve, ms)
    signal?.addEventListener(
      'abort',
      () => {
        clearTimeout(t)
        reject(new DOMException('Aborted', 'AbortError'))
      },
      { once: true },
    )
  })

/**
 * Chunked upload per docs/api.md: POST /api/uploads → PUT each chunk (retried with backoff) →
 * POST /complete. Resolves with the ingest job id.
 */
export async function uploadFile(
  api: Api,
  file: File,
  onProgress: (p: UploadProgress) => void,
  signal: AbortSignal,
): Promise<string> {
  const total = file.size
  const started = performance.now()
  let confirmed = 0
  let chunkCount = 1
  const emit = (phase: UploadProgress['phase'], sent: number, chunkIndex: number, retry: UploadProgress['retry'] = null) => {
    const secs = Math.max(0.25, (performance.now() - started) / 1000)
    onProgress({ phase, sentBytes: Math.min(total, sent), totalBytes: total, chunkIndex, chunkCount, bytesPerSec: sent / secs, retry })
  }

  emit('starting', 0, 0)
  const init = await api.createUpload({
    filename: file.name,
    size_bytes: file.size,
    content_type: file.type || 'application/octet-stream',
  })
  const size = Math.max(1, init.chunk_size_bytes)
  chunkCount = Math.max(1, Math.ceil(total / size))

  for (let i = 0; i < chunkCount; i++) {
    const blob = file.slice(i * size, Math.min(total, (i + 1) * size))
    for (let attempt = 1; ; attempt++) {
      try {
        emit('uploading', confirmed, i)
        await api.putChunk(init.upload_id, i, blob, {
          signal,
          onProgress: (loaded) => emit('uploading', confirmed + loaded, i),
        })
        confirmed += blob.size
        break
      } catch (err) {
        if (signal.aborted || (err instanceof DOMException && err.name === 'AbortError')) throw err
        if (err instanceof ApiError && NOT_RETRYABLE.has(err.status)) throw err
        if (attempt >= MAX_ATTEMPTS) throw err
        const wait = Math.min(8000, 700 * 2 ** (attempt - 1))
        emit('retrying', confirmed, i, {
          attempt: attempt + 1,
          maxAttempts: MAX_ATTEMPTS,
          inSeconds: Math.round(wait / 100) / 10,
          reason: err instanceof Error ? err.message : 'Network error',
        })
        await sleep(wait, signal)
      }
    }
  }

  emit('finishing', total, chunkCount - 1)
  const { job_id } = await api.completeUpload(init.upload_id)
  return job_id
}
