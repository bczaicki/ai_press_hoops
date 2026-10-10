import { AnimatePresence, motion } from 'framer-motion'
import { CloudUpload, Film, RotateCcw, Wifi, X } from 'lucide-react'
import { useCallback, useEffect, useRef, useState, type DragEvent } from 'react'
import { useNavigate } from 'react-router'

import { api, errorMessage } from '../api'
import { useJob } from '../api/jobs'
import { uploadFile, type UploadProgress } from '../api/upload'
import { cn, fmtBytes, fmtClock } from '../lib/format'
import { useConfig } from './ConfigContext'
import { Button } from './ui/button'
import { useToast } from './ui/toast'

type Phase =
  | { kind: 'idle' }
  | { kind: 'uploading'; file: File; progress: UploadProgress | null }
  | { kind: 'processing'; file: File; jobId: string }
  | { kind: 'error'; file: File | null; message: string }

const VIDEO_EXT = /\.(mov|mp4|m4v|webm|mkv|avi|mts|m2ts|3gp|hevc)$/i

function probeDuration(file: File): Promise<number | null> {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file)
    const v = document.createElement('video')
    const done = (d: number | null) => {
      URL.revokeObjectURL(url)
      resolve(d)
    }
    const t = setTimeout(() => done(null), 4000)
    v.preload = 'metadata'
    v.muted = true
    v.onloadedmetadata = () => {
      clearTimeout(t)
      done(Number.isFinite(v.duration) ? v.duration : null)
    }
    v.onerror = () => {
      clearTimeout(t)
      done(null) // e.g. HEVC in a browser that can't decode it; the server will check
    }
    v.src = url
  })
}

export function UploadZone() {
  const { config } = useConfig()
  const navigate = useNavigate()
  const toast = useToast()
  const [phase, setPhase] = useState<Phase>({ kind: 'idle' })
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const abortRef = useRef<AbortController | null>(null)
  const dragDepth = useRef(0)

  const jobId = phase.kind === 'processing' ? phase.jobId : null
  const { job, error: jobError } = useJob(jobId)

  useEffect(() => {
    if (phase.kind !== 'processing' || !job) return
    if (job.status === 'done') {
      if (job.clip_id) {
        toast({ title: 'Clip ready', body: `${phase.file.name} is ready for analysis.`, tone: 'success' })
        navigate(`/clips/${job.clip_id}`)
      } else setPhase({ kind: 'error', file: phase.file, message: 'Processing finished but no clip came back.' })
    } else if (job.status === 'failed') setPhase({ kind: 'error', file: phase.file, message: job.error ?? 'Processing failed.' })
    else if (job.status === 'cancelled') setPhase({ kind: 'idle' })
  }, [job, phase, navigate, toast])

  useEffect(() => {
    if (jobError && phase.kind === 'processing') setPhase({ kind: 'error', file: phase.file, message: jobError })
  }, [jobError, phase])

  useEffect(() => () => abortRef.current?.abort(), [])

  const start = useCallback(
    async (file: File) => {
      const maxMb = config?.caps.max_upload_mb ?? 500
      if (!file.type.startsWith('video/') && !VIDEO_EXT.test(file.name)) {
        setPhase({ kind: 'error', file, message: `“${file.name}” doesn't look like a video. Try an .mov or .mp4 from your phone.` })
        return
      }
      if (file.size > maxMb * 1024 * 1024) {
        setPhase({ kind: 'error', file, message: `That file is ${fmtBytes(file.size)}. The limit is ${maxMb} MB. Trim it or export at a lower resolution.` })
        return
      }
      const maxS = config?.caps.max_clip_seconds
      if (maxS) {
        const d = await probeDuration(file)
        if (d && d > maxS) {
          setPhase({ kind: 'error', file, message: `That clip is ${fmtClock(d)} long. The limit is ${fmtClock(maxS)}. Trim it to the part you care about.` })
          return
        }
      }
      const ac = new AbortController()
      abortRef.current = ac
      setPhase({ kind: 'uploading', file, progress: null })
      try {
        const id = await uploadFile(api, file, (progress) => setPhase({ kind: 'uploading', file, progress }), ac.signal)
        setPhase({ kind: 'processing', file, jobId: id })
      } catch (err) {
        if (ac.signal.aborted) {
          setPhase({ kind: 'idle' })
          return
        }
        setPhase({ kind: 'error', file, message: errorMessage(err) })
      }
    },
    [config],
  )

  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    dragDepth.current = 0
    setDragging(false)
    const f = e.dataTransfer.files?.[0]
    if (f && (phase.kind === 'idle' || phase.kind === 'error')) void start(f)
  }

  const busy = phase.kind === 'uploading' || phase.kind === 'processing'
  const pct =
    phase.kind === 'uploading'
      ? phase.progress
        ? phase.progress.sentBytes / Math.max(1, phase.progress.totalBytes)
        : 0
      : phase.kind === 'processing'
        ? (job?.progress ?? 0)
        : 0

  return (
    <div
      onDragEnter={(e) => {
        e.preventDefault()
        dragDepth.current++
        if (!busy) setDragging(true)
      }}
      onDragOver={(e) => e.preventDefault()}
      onDragLeave={() => {
        dragDepth.current = Math.max(0, dragDepth.current - 1)
        if (dragDepth.current === 0) setDragging(false)
      }}
      onDrop={onDrop}
      className={cn(
        'group relative overflow-hidden rounded-[1.75rem] border-2 border-dashed transition-all duration-200',
        dragging ? 'scale-[1.01] border-flame-400 bg-flame-500/[0.08]' : 'border-white/12 bg-white/[0.025] hover:border-flame-500/50',
      )}
    >
      {/* court arc motif */}
      <svg aria-hidden viewBox="0 0 800 300" className="pointer-events-none absolute inset-x-0 bottom-0 h-full w-full opacity-[0.07]" preserveAspectRatio="xMidYMax slice">
        <g fill="none" stroke="#e8c08f" strokeWidth="3">
          <path d="M120 300 A 280 280 0 0 1 680 300" />
          <rect x="300" y="160" width="200" height="140" />
          <path d="M340 160 A 60 60 0 0 1 460 160" />
        </g>
      </svg>
      <input
        ref={inputRef}
        type="file"
        accept="video/*,.mov,.mp4,.m4v,.webm,.mkv,.avi"
        className="sr-only"
        tabIndex={-1}
        onChange={(e) => {
          const f = e.target.files?.[0]
          e.target.value = ''
          if (f) void start(f)
        }}
      />
      <AnimatePresence mode="wait" initial={false}>
        {phase.kind === 'idle' || phase.kind === 'error' ? (
          <motion.div key="idle" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="relative flex min-w-0 flex-col items-center px-4 py-10 text-center sm:px-6 sm:py-14">
            <motion.div
              animate={dragging ? { y: -6, scale: 1.08 } : { y: 0, scale: 1 }}
              className="grid size-16 place-items-center rounded-2xl bg-flame-500 text-ink-950 shadow-[0_18px_40px_-12px_rgb(255_122_26/0.7)]"
            >
              <CloudUpload className="size-8" aria-hidden />
            </motion.div>
            <h2 className="mt-5 font-display text-[1.7rem] leading-tight font-bold tracking-wide text-fg uppercase sm:text-4xl">{dragging ? 'Drop it like a bounce pass' : 'Drop a game clip here'}</h2>
            <p className="mt-2 max-w-md text-fg-2">
              Phone video is perfect: .mov, .mp4, portrait or landscape, HDR is fine. Up to {config?.caps.max_upload_mb ?? 500} MB and {fmtClock(config?.caps.max_clip_seconds ?? 600)} long.
            </p>
            <Button variant="primary" size="lg" className="mt-6" onClick={() => inputRef.current?.click()} icon={<Film className="size-5" aria-hidden />}>
              Choose a video
            </Button>
            {phase.kind === 'error' && (
              <div role="alert" className="mt-6 flex max-w-lg items-start gap-3 rounded-2xl border border-miss/40 bg-miss/[0.08] p-3.5 text-left">
                <X className="mt-0.5 size-4.5 shrink-0 text-miss-text" aria-hidden />
                <div className="min-w-0 flex-1 text-sm">
                  <div className="font-semibold text-fg">Upload didn't finish</div>
                  <div className="text-fg-2">{phase.message}</div>
                </div>
                {phase.file && (
                  <Button size="sm" variant="secondary" icon={<RotateCcw className="size-3.5" />} onClick={() => void start(phase.file!)}>
                    Retry
                  </Button>
                )}
              </div>
            )}
          </motion.div>
        ) : (
          <motion.div key="busy" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="relative min-w-0 px-4 py-10 sm:px-10 sm:py-12">
            <div className="flex items-center gap-4">
              <div className="grid size-12 shrink-0 place-items-center rounded-xl border border-line bg-white/[0.05]">
                <Film className="size-6 text-flame-400" aria-hidden />
              </div>
              <div className="min-w-0 flex-1">
                <div className="truncate font-semibold text-fg">{phase.file.name}</div>
                <div className="text-sm text-fg-3">{fmtBytes(phase.file.size)}</div>
              </div>
              <div className="numeral text-5xl text-fg tnum">
                {Math.floor(pct * 100)}
                <span className="text-2xl text-fg-3">%</span>
              </div>
            </div>
            <div className="mt-6 h-2.5 overflow-hidden rounded-full bg-white/[0.07]" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(pct * 100)} aria-label={phase.kind === 'uploading' ? 'Upload progress' : 'Processing progress'}>
              <motion.div
                className={cn('h-full rounded-full', phase.kind === 'processing' ? 'bg-[linear-gradient(90deg,var(--color-flame-600),var(--color-flame-400))]' : 'bg-flame-500')}
                animate={{ width: `${Math.max(2, pct * 100)}%` }}
                transition={{ ease: 'easeOut', duration: 0.3 }}
              />
            </div>
            <div className="mt-3 flex flex-wrap items-center justify-between gap-x-4 gap-y-1 text-sm">
              <div className="text-fg-2" aria-live="polite">
                {phase.kind === 'uploading' ? (
                  phase.progress?.phase === 'retrying' && phase.progress.retry ? (
                    <span className="inline-flex items-center gap-1.5 text-attempt">
                      <Wifi className="size-4" aria-hidden /> Connection hiccup on piece {phase.progress.chunkIndex + 1}. Retrying (try {phase.progress.retry.attempt} of {phase.progress.retry.maxAttempts})…
                    </span>
                  ) : phase.progress?.phase === 'finishing' ? (
                    'Upload complete. Handing off to the server…'
                  ) : phase.progress ? (
                    <>
                      Uploading piece {phase.progress.chunkIndex + 1} of {phase.progress.chunkCount} · {fmtBytes(phase.progress.sentBytes)} of {fmtBytes(phase.progress.totalBytes)}
                      {phase.progress.bytesPerSec > 0 && <span className="text-fg-3"> · {fmtBytes(phase.progress.bytesPerSec)}/s</span>}
                    </>
                  ) : (
                    'Starting upload…'
                  )
                ) : (
                  <>
                    <span className="font-semibold text-fg">Preparing video.</span> {job?.message ?? 'Waiting for the server…'}
                  </>
                )}
              </div>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => {
                  if (phase.kind === 'uploading') abortRef.current?.abort()
                  else if (phase.kind === 'processing') void api.cancelJob(phase.jobId).catch(() => setPhase({ kind: 'idle' }))
                }}
              >
                Cancel
              </Button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
