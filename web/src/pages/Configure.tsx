import { AnimatePresence, motion } from 'framer-motion'
import {
  AlertTriangle,
  ArrowLeft,
  ChevronDown,
  CircleDollarSign,
  Clock3,
  Crosshair,
  Grid2x2,
  HardDrive,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Wand2,
} from 'lucide-react'
import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { Link, useNavigate, useParams } from 'react-router'

import { api, ApiError, errorMessage } from '../api'
import type { AppConfig, ClipView, CourtPoint, Effort, Estimate, FootageType, OutputKind, RunRequest } from '../api/types'
import { useConfig } from '../components/ConfigContext'
import { DeciderList, OutputCards, ProviderCards, ThresholdControl } from '../components/Pickers'
import { CourtPointsPicker, RimPicker, type Box } from '../components/PosterPickers'
import { Poster } from '../components/Poster'
import { Button, ButtonLink } from '../components/ui/button'
import { Badge, Card, ErrorPanel, Field, inputClass, Meter, Select, Skeleton } from '../components/ui/misc'
import { Switch } from '../components/ui/segmented'
import { Tip } from '../components/ui/tip'
import { useToast } from '../components/ui/toast'
import { cn, fmtBytes, fmtClock, fmtUsd, footageLabel } from '../lib/format'
import { useAsync, useDebounced } from '../lib/hooks'

interface FormState {
  outputs: OutputKind[]
  footageType: FootageType | null
  provider: string | null
  model: string
  effort: Effort
  nativeVideo: boolean
  decider: string
  threshold: string
  detector: string
  tracker: string
  promptPreset: string
  customPrompt: string
  structured: boolean
  rimBox: Box | null
  courtPoints: CourtPoint[]
  maxCost: number
}

const EFFORTS: { key: Effort; label: string; note: string }[] = [
  { key: 'low', label: 'Low', note: 'Fastest and cheapest' },
  { key: 'medium', label: 'Medium', note: 'Recommended' },
  { key: 'high', label: 'High', note: 'More careful, slower' },
  { key: 'xhigh', label: 'Extra high', note: 'Slow; for tricky footage' },
  { key: 'max', label: 'Max', note: 'Slowest and priciest' },
]

const MOVING_CAMERA = new Set<FootageType>(['handheld', 'broadcast', 'highlights'])

function defaults(config: AppConfig): FormState {
  const prov = config.providers.find((p) => p.available && !p.local) ?? config.providers.find((p) => p.available) ?? null
  const decider =
    ['jev', 'rules'].map((k) => config.deciders.find((d) => d.key === k && d.available)).find(Boolean)?.key ??
    config.deciders.find((d) => d.available)?.key ??
    'rules'
  return {
    outputs: config.outputs.map((o) => o.key),
    footageType: null,
    provider: prov?.key ?? null,
    model: prov?.default_model ?? '',
    effort: 'medium',
    nativeVideo: false,
    decider,
    threshold: config.threshold_presets.find((p) => p.key === 'balanced')?.key ?? config.threshold_presets[0]?.key ?? 'balanced',
    detector: config.detectors.find((d) => d.available)?.key ?? config.detectors[0]?.key ?? '',
    tracker: config.trackers.find((t) => t.available)?.key ?? config.trackers[0]?.key ?? '',
    promptPreset: config.prompt_presets[0]?.key ?? 'recap',
    customPrompt: '',
    structured: true,
    rimBox: null,
    courtPoints: [],
    maxCost: config.caps.run_cost_cap_usd,
  }
}

/** Who receives what. Choosing a provider is the consent that populates egress_ok. */
function egress(form: FormState, config: AppConfig) {
  const keys: string[] = []
  const lines: { who: string; what: string; local?: boolean }[] = []
  const p = config.providers.find((x) => x.key === form.provider)
  if (p) {
    keys.push(p.key)
    lines.push(
      p.local
        ? { who: 'This computer', what: `${p.label} runs locally. Nothing leaves.`, local: true }
        : { who: p.sends_data_to, what: form.nativeVideo && p.native_video ? 'short video windows and sampled frames' : 'sampled frames from close calls (and the recap)' },
    )
  }
  if (form.decider === 'openai-decisions') {
    if (!keys.includes('openai')) keys.push('openai')
    lines.push({ who: 'OpenAI', what: 'small rim crops for each candidate shot' })
  } else if (form.decider === 'jev') {
    lines.push({ who: 'TypeSafe', what: 'short text descriptions only (no images)' })
  } else if (form.decider.startsWith('adapter:')) {
    const vendor = form.decider.split(':')[1] ?? 'the adapter'
    lines.push({ who: vendor[0].toUpperCase() + vendor.slice(1), what: 'short text descriptions only (no images)' })
  }
  return { keys, lines }
}

function buildRequest(form: FormState, config: AppConfig, clip: ClipView): RunRequest {
  const p = config.providers.find((x) => x.key === form.provider) ?? null
  const footage = form.footageType ?? clip.footage_type
  const outputs = form.outputs.filter((o) => o !== 'summary' || !!p)
  return {
    clip_id: clip.clip_id,
    outputs,
    detector: form.detector,
    tracker: form.tracker,
    decider: form.decider,
    vlm: p ? { provider: p.key, model: form.model || p.default_model, effort: form.effort, native_video: form.nativeVideo && p.native_video } : null,
    prompt_preset: form.promptPreset,
    custom_prompt: form.promptPreset === 'custom' ? form.customPrompt.trim() || null : null,
    structured: form.structured,
    threshold_preset: form.threshold,
    egress_ok: egress(form, config).keys,
    footage_type: form.footageType,
    rim_box: form.rimBox,
    court_points: footage === 'fixed_camera' && form.courtPoints.length >= 4 ? form.courtPoints : null,
    max_cost_usd: form.maxCost,
  }
}

export default function ConfigurePage() {
  const { clipId = '' } = useParams()
  const { config, error: cfgError, reload: reloadConfig } = useConfig()
  const clip = useAsync(() => api.clip(clipId), [clipId])

  if (clip.error || cfgError) {
    const notFound = clip.status === 404
    return (
      <div className="mx-auto max-w-xl py-10">
        <ErrorPanel
          title={notFound ? 'Clip not found' : clip.status === 409 ? 'Still preparing this clip' : "Couldn't load this clip"}
          message={notFound ? 'It may have been removed. Upload it again from the home page.' : (clip.error ?? cfgError ?? '')}
          action={
            <div className="flex gap-2">
              <Button size="sm" onClick={() => (cfgError ? reloadConfig() : clip.reload())}>
                Try again
              </Button>
              <ButtonLink size="sm" variant="ghost" to="/">
                Home
              </ButtonLink>
            </div>
          }
        />
      </div>
    )
  }
  if (!config || !clip.data) return <ConfigureSkeleton />
  return <ConfigureForm key={clip.data.clip_id} config={config} clip={clip.data} />
}

function ConfigureSkeleton() {
  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_22rem]" aria-busy>
      <div className="space-y-5">
        <Skeleton className="h-28" />
        <Skeleton className="h-64" />
        <Skeleton className="h-56" />
      </div>
      <Skeleton className="h-96" />
    </div>
  )
}

function Step({ n, title, subtitle, badge, children, className }: { n: number; title: string; subtitle?: ReactNode; badge?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <Card className={cn('p-4 sm:p-6', className)}>
      <div className="mb-4 flex items-start gap-3">
        <span className="numeral grid size-9 shrink-0 place-items-center rounded-xl bg-white/[0.06] text-xl text-flame-400">{n}</span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="font-display text-2xl leading-none font-bold tracking-wide text-fg uppercase">{title}</h2>
            {badge}
          </div>
          {subtitle && <p className="mt-1 text-sm text-fg-2">{subtitle}</p>}
        </div>
      </div>
      {children}
    </Card>
  )
}

function ConfigureForm({ config, clip }: { config: AppConfig; clip: ClipView }) {
  const navigate = useNavigate()
  const toast = useToast()
  const [form, setForm] = useState<FormState>(() => defaults(config))
  const set = (patch: Partial<FormState>) => setForm((f) => ({ ...f, ...patch }))
  const [advanced, setAdvanced] = useState(false)
  const [showRim, setShowRim] = useState(false)
  const [showCourt, setShowCourt] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState<string | null>(null)

  const provider = config.providers.find((p) => p.key === form.provider) ?? null
  const footage = form.footageType ?? clip.footage_type
  const detector = config.detectors.find((d) => d.key === form.detector)
  const needsRim = !!detector && /coco|generic/i.test(detector.key)
  const req = useMemo(() => buildRequest(form, config, clip), [form, config, clip])
  const reqKey = JSON.stringify(req)
  const debouncedKey = useDebounced(reqKey, 400)
  const eg = egress(form, config)

  const [estimate, setEstimate] = useState<{ data: Estimate | null; loading: boolean; error: string | null }>({ data: null, loading: true, error: null })
  useEffect(() => {
    const ac = new AbortController()
    setEstimate((e) => ({ ...e, loading: true }))
    api.estimate(JSON.parse(debouncedKey) as RunRequest, ac.signal).then(
      (data) => setEstimate({ data, loading: false, error: null }),
      (err) => {
        if (ac.signal.aborted) return
        setEstimate((e) => ({ data: e.data, loading: false, error: errorMessage(err) }))
      },
    )
    return () => ac.abort()
  }, [debouncedKey])

  const est = estimate.data?.estimate_usd ?? null
  const overRunCap = est != null && est > form.maxCost
  const overSession = est != null && config.spend.session_usd + est > config.caps.session_cost_cap_usd
  const customMissing = form.promptPreset === 'custom' && !form.customPrompt.trim()
  const blockReason = !req.outputs.length
    ? 'Pick at least one output.'
    : overRunCap
      ? `The estimate is over your ${fmtUsd(form.maxCost)} limit for one run.`
      : overSession
        ? 'This would go past the spending cap for this session.'
        : customMissing
          ? 'Write your custom prompt, or pick a preset.'
          : null

  async function analyze() {
    if (blockReason) return
    setSubmitting(true)
    setSubmitError(null)
    try {
      const res = await api.createRun(req)
      navigate(`/runs/${encodeURIComponent(res.run_id)}?job=${encodeURIComponent(res.job_id)}`)
    } catch (err) {
      const msg = errorMessage(err)
      setSubmitError(msg)
      toast({ title: err instanceof ApiError && err.status === 402 ? 'Over the spending cap' : "Couldn't start the analysis", body: msg, tone: 'error' })
      setSubmitting(false)
    }
  }

  const setFootage = (v: FootageType | null) => {
    const recommended = v && MOVING_CAMERA.has(v) ? 'botsort' : 'bytetrack'
    const tracker = config.trackers.find((t) => t.key === recommended && t.available)?.key ?? form.tracker
    set({ footageType: v, tracker })
  }

  const cheaper = (() => {
    if (!overRunCap || !provider) return null
    const models = provider.models.length ? provider.models : [provider.default_model]
    const i = models.indexOf(form.model || provider.default_model)
    return i >= 0 && i < models.length - 1 ? models[i + 1] : null
  })()

  const rail = (
    <EstimateRail
      config={config}
      form={form}
      cheaperModel={cheaper}
      onCheaper={() => cheaper && set({ model: cheaper })}
      estimate={estimate}
      blockReason={blockReason}
      submitting={submitting}
      submitError={submitError}
      egressLines={eg.lines}
      onAnalyze={analyze}
    />
  )

  return (
    <div className="pb-28 lg:pb-0">
      <Link to="/" className="mb-4 inline-flex items-center gap-1.5 text-sm font-semibold text-fg-2 hover:text-fg">
        <ArrowLeft className="size-4" aria-hidden /> All clips
      </Link>
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_22rem] xl:grid-cols-[minmax(0,1fr)_24rem]">
        <div className="min-w-0 space-y-5">
          {/* clip header */}
          <Card className="flex flex-col gap-4 p-4 sm:flex-row sm:items-center sm:p-5">
            <Poster src={clip.urls.poster} alt="" className="w-full shrink-0 rounded-xl sm:w-56" />
            <div className="min-w-0 flex-1">
              <div className="eyebrow">Analyze</div>
              <h1 className="truncate font-display text-3xl font-bold tracking-wide text-fg uppercase">{clip.original_filename ?? clip.clip_id}</h1>
              <div className="mt-1.5 flex flex-wrap gap-1.5">
                <Badge icon={<Clock3 className="size-3" aria-hidden />}>{fmtClock(clip.master.duration_s)}</Badge>
                <Badge>
                  {clip.master.width}×{clip.master.height} · {Math.round(clip.master.fps_float)} fps
                </Badge>
                <Badge>{fmtBytes(clip.source_size_bytes)}</Badge>
              </div>
              {clip.normalization.summary && (
                <p className="mt-2 flex items-start gap-1.5 text-xs text-fg-3">
                  <Wand2 className="mt-px size-3.5 shrink-0 text-flame-400" aria-hidden /> Prepared for the browser: {clip.normalization.summary}
                </p>
              )}
            </div>
          </Card>

          <Step n={1} title="What kind of footage?" subtitle="This picks the right tracker and decides whether a real court map is possible.">
            <div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-end">
              <Field label="Footage type" htmlFor="footage">
                <Select id="footage" value={form.footageType ?? ''} onChange={(e) => setFootage((e.target.value || null) as FootageType | null)}>
                  <option value="">Auto: let System One decide{clip.footage_type !== 'unknown' ? ` (looks like ${footageLabel(clip.footage_type).toLowerCase()})` : ''}</option>
                  {config.footage_types
                    .filter((f) => f !== 'unknown')
                    .map((f) => (
                      <option key={f} value={f}>
                        {footageLabel(f)}
                      </option>
                    ))}
                </Select>
              </Field>
              {clip.footage_type_source && clip.footage_type !== 'unknown' && (
                <Badge tone="info" className="mb-2 justify-self-start sm:self-end">
                  Detected by {clip.footage_type_source === 'system_one' ? 'System One' : clip.footage_type_source}
                </Badge>
              )}
            </div>
          </Step>

          <Step n={2} title="What do you want?" subtitle="Turn on what you need. Fewer outputs finish faster and cost less.">
            <OutputCards
              options={config.outputs}
              value={form.outputs}
              onChange={(outputs) => set({ outputs })}
              disabledReason={(k) => (k === 'summary' && !provider ? 'Needs an AI provider. Pick one in step 3.' : null)}
            />
          </Step>

          <Step n={3} title="Who explains it?" subtitle="The AI reviews close calls, writes the recap, and answers your questions. It never does the counting.">
            <ProviderCards
              providers={config.providers}
              value={form.provider}
              onChange={(k) => {
                const p = config.providers.find((x) => x.key === k)
                set({ provider: k, model: p?.default_model ?? '', nativeVideo: false })
              }}
            />
            <AnimatePresence initial={false}>
              {provider && (
                <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                  <div className="mt-4 grid gap-3 sm:grid-cols-2">
                    <Field label="Model" htmlFor="model">
                      <Select id="model" value={form.model} onChange={(e) => set({ model: e.target.value })}>
                        {(provider.models.length ? provider.models : [provider.default_model]).map((m) => (
                          <option key={m} value={m}>
                            {m}
                            {m === provider.default_model ? ' (default)' : ''}
                          </option>
                        ))}
                      </Select>
                    </Field>
                    <div
                      className={cn(
                        'flex items-start gap-2.5 rounded-xl border p-3 text-sm',
                        provider.local ? 'border-made/30 bg-made/[0.06]' : 'border-line-strong bg-white/[0.03]',
                      )}
                    >
                      {provider.local ? <HardDrive className="mt-0.5 size-4.5 shrink-0 text-made-text" aria-hidden /> : <ShieldCheck className="mt-0.5 size-4.5 shrink-0 text-flame-400" aria-hidden />}
                      <span className="text-fg-2">
                        {provider.local ? (
                          <>
                            <b className="text-fg">Runs on this computer.</b> No frames leave.
                          </>
                        ) : (
                          <>
                            Picking this <b className="text-fg">sends frames to {provider.sends_data_to}</b>. Only short windows around close calls, plus a few frames for the recap.
                          </>
                        )}
                      </span>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </Step>

          <Step n={4} title="Who makes the calls?" subtitle="A fast System One model gives every candidate shot a typed, calibrated verdict. Unsure ones go to the AI or to you.">
            <DeciderList deciders={config.deciders} value={form.decider} onChange={(decider) => set({ decider })} />
            <div className="mt-5 border-t border-line pt-4">
              <div className="mb-2 text-[0.8rem] font-semibold text-fg-2">How sure before a shot counts?</div>
              <ThresholdControl presets={config.threshold_presets} value={form.threshold} onChange={(threshold) => set({ threshold })} />
              <p className="mt-2 text-xs text-fg-3">You can change this after the run too. Results update instantly with no re-analysis.</p>
            </div>
          </Step>

          <Step
            n={5}
            title="Mark the rim"
            badge={<Badge tone={needsRim ? 'flame' : 'muted'}>{needsRim ? 'Recommended' : 'Optional'}</Badge>}
            subtitle={needsRim ? 'The selected detector has no rim class, so showing it where the rim is makes shot calls much better.' : 'The detector can find rims itself, but marking it helps on tricky footage.'}
          >
            {showRim || form.rimBox ? (
              <RimPicker poster={clip.urls.poster} width={clip.proxy.width} height={clip.proxy.height} value={form.rimBox} onChange={(rimBox) => set({ rimBox })} />
            ) : (
              <Button icon={<Crosshair className="size-4" />} onClick={() => setShowRim(true)}>
                Mark the rim on the poster
              </Button>
            )}
          </Step>

          {footage === 'fixed_camera' && (
            <Step
              n={6}
              title="Calibrate the court"
              badge={<Badge tone="muted">Optional</Badge>}
              subtitle="Click 4 or more court landmarks so shots land on a real court map instead of an image-space approximation."
            >
              {showCourt || form.courtPoints.length ? (
                <CourtPointsPicker poster={clip.urls.poster} width={clip.proxy.width} height={clip.proxy.height} value={form.courtPoints} onChange={(courtPoints) => set({ courtPoints })} />
              ) : (
                <Button icon={<Grid2x2 className="size-4" />} onClick={() => setShowCourt(true)}>
                  Place court landmarks
                </Button>
              )}
            </Step>
          )}

          {/* advanced */}
          <Card className="overflow-hidden">
            <button type="button" aria-expanded={advanced} aria-controls="adv" onClick={() => setAdvanced((a) => !a)} className="flex w-full items-center gap-3 px-4 py-4 text-left sm:px-6">
              <SlidersHorizontal className="size-5 text-fg-3" aria-hidden />
              <span className="flex-1">
                <span className="block font-display text-xl font-bold tracking-wide text-fg uppercase">Advanced</span>
                <span className="block text-sm text-fg-3">Detector, tracker, prompt, effort, cost limit</span>
              </span>
              <ChevronDown className={cn('size-5 text-fg-3 transition-transform', advanced && 'rotate-180')} aria-hidden />
            </button>
            <AnimatePresence initial={false}>
              {advanced && (
                <motion.div id="adv" initial={{ height: 0 }} animate={{ height: 'auto' }} exit={{ height: 0 }} className="overflow-hidden">
                  <div className="grid gap-4 border-t border-line px-4 py-5 sm:grid-cols-2 sm:px-6">
                    <Field label="Detector" htmlFor="det" hint={detector ? `${detector.license || 'license n/a'}${detector.notes ? ` · ${detector.notes}` : ''}` : undefined}>
                      <Select id="det" value={form.detector} onChange={(e) => set({ detector: e.target.value })}>
                        {config.detectors.map((d) => (
                          <option key={d.key} value={d.key} disabled={!d.available}>
                            {d.label}
                            {d.available ? '' : ` (unavailable: ${d.reason})`}
                          </option>
                        ))}
                      </Select>
                    </Field>
                    <Field label="Tracker" htmlFor="trk" hint={MOVING_CAMERA.has(footage) ? 'Moving camera: BoT-SORT handles pans best.' : 'Steady camera: ByteTrack is fast and solid.'}>
                      <Select id="trk" value={form.tracker} onChange={(e) => set({ tracker: e.target.value })}>
                        {config.trackers.map((t) => (
                          <option key={t.key} value={t.key} disabled={!t.available}>
                            {t.label}
                            {t.available ? '' : ` (unavailable: ${t.reason})`}
                          </option>
                        ))}
                      </Select>
                    </Field>
                    <Field label="Recap style" htmlFor="preset">
                      <Select id="preset" value={form.promptPreset} onChange={(e) => set({ promptPreset: e.target.value })} disabled={!provider}>
                        {config.prompt_presets.map((p) => (
                          <option key={p.key} value={p.key}>
                            {p.label}
                          </option>
                        ))}
                      </Select>
                    </Field>
                    <Field label="AI effort" htmlFor="effort" hint={EFFORTS.find((e) => e.key === form.effort)?.note}>
                      <Select id="effort" value={form.effort} onChange={(e) => set({ effort: e.target.value as Effort })} disabled={!provider}>
                        {EFFORTS.map((e) => (
                          <option key={e.key} value={e.key}>
                            {e.label}
                          </option>
                        ))}
                      </Select>
                    </Field>
                    {form.promptPreset === 'custom' && (
                      <div className="sm:col-span-2">
                        <Field label="Your prompt" htmlFor="custom" hint="The AI also gets the event log and the shot clips. Ask for what you want; it won't count shots itself.">
                          <textarea
                            id="custom"
                            rows={5}
                            value={form.customPrompt}
                            onChange={(e) => set({ customPrompt: e.target.value })}
                            placeholder="e.g. Focus on our defense: where did they get open looks, and who closed out late?"
                            className={cn(inputClass, 'h-auto py-2.5 leading-relaxed')}
                          />
                        </Field>
                      </div>
                    )}
                    <ToggleRow label="Structured output" hint="Ask the AI for checked JSON (recommended). Turn off for free-form text." checked={form.structured} onChange={(structured) => set({ structured })} disabled={!provider} />
                    {provider?.native_video && (
                      <ToggleRow
                        label="Send real video"
                        hint={`${provider.label.replace(/\s*\(.*\)$/, '')} can watch short video windows instead of still frames.`}
                        checked={form.nativeVideo}
                        onChange={(nativeVideo) => set({ nativeVideo })}
                      />
                    )}
                    <Field label="Cost limit for this run" htmlFor="maxcost" hint={`Up to the server cap of ${fmtUsd(config.caps.run_cost_cap_usd)}.`}>
                      <div className="relative">
                        <span className="pointer-events-none absolute top-1/2 left-3 -translate-y-1/2 text-fg-3">$</span>
                        <input
                          id="maxcost"
                          type="number"
                          inputMode="decimal"
                          min={0.01}
                          step={0.05}
                          max={config.caps.run_cost_cap_usd}
                          value={form.maxCost}
                          onChange={(e) => {
                            const v = Number(e.target.value)
                            if (Number.isFinite(v)) set({ maxCost: Math.max(0, Math.min(config.caps.run_cost_cap_usd, v)) })
                          }}
                          className={cn(inputClass, 'pl-7 tnum')}
                        />
                      </div>
                    </Field>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </Card>

          <div className="lg:hidden">{rail}</div>
        </div>

        <aside className="hidden lg:block">
          <div className="sticky top-24">{rail}</div>
        </aside>
      </div>

      {/* phone: sticky analyze bar */}
      <div className="fixed inset-x-0 bottom-0 z-30 border-t border-line bg-ink-950/90 px-4 pt-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] backdrop-blur-xl lg:hidden">
        <div className="mx-auto flex max-w-xl items-center gap-3">
          <div className="min-w-0 flex-1">
            <div className="eyebrow text-[0.65rem]">Estimated</div>
            <div className={cn('numeral text-3xl tnum', overRunCap ? 'text-miss-text' : 'text-fg')}>{est == null ? '…' : est === 0 ? 'Free' : fmtUsd(est)}</div>
          </div>
          <Button variant="primary" size="lg" onClick={analyze} loading={submitting} disabled={!!blockReason} icon={!submitting && <Sparkles className="size-5" aria-hidden />}>
            Analyze
          </Button>
        </div>
      </div>
    </div>
  )
}

function ToggleRow({ label, hint, checked, onChange, disabled }: { label: string; hint: string; checked: boolean; onChange: (v: boolean) => void; disabled?: boolean }) {
  const id = `t-${label.replace(/\W+/g, '-').toLowerCase()}`
  return (
    <div className={cn('flex items-start justify-between gap-3 rounded-xl border border-line bg-white/[0.02] p-3', disabled && 'opacity-50')}>
      <div>
        <label htmlFor={id} className="text-sm font-semibold text-fg">
          {label}
        </label>
        <p className="text-xs text-fg-3">{hint}</p>
      </div>
      <Switch id={id} label={label} checked={checked} onChange={onChange} disabled={disabled} />
    </div>
  )
}

function EstimateRail({
  config,
  form,
  cheaperModel,
  onCheaper,
  estimate,
  blockReason,
  submitting,
  submitError,
  egressLines,
  onAnalyze,
}: {
  config: AppConfig
  form: FormState
  cheaperModel: string | null
  onCheaper: () => void
  estimate: { data: Estimate | null; loading: boolean; error: string | null }
  blockReason: string | null
  submitting: boolean
  submitError: string | null
  egressLines: { who: string; what: string; local?: boolean }[]
  onAnalyze: () => void
}) {
  const est = estimate.data?.estimate_usd ?? null
  const over = est != null && est > form.maxCost
  const sessionLeft = Math.max(0, config.caps.session_cost_cap_usd - config.spend.session_usd)
  return (
    <Card className="overflow-hidden">
      <div className="relative px-5 pt-5 pb-4">
        <div aria-hidden className="pointer-events-none absolute inset-x-0 top-0 h-24 bg-[radial-gradient(60%_100%_at_50%_0%,rgb(255_122_26/0.18),transparent)]" />
        <div className="relative flex items-center justify-between">
          <div className="eyebrow flex items-center gap-1.5">
            <CircleDollarSign className="size-3.5" aria-hidden /> Estimated cost
          </div>
          {estimate.loading && <span className="text-xs text-fg-3">updating…</span>}
        </div>
        <div className={cn('relative mt-1 numeral text-6xl transition-opacity', estimate.loading && 'opacity-60', over ? 'text-miss-text' : 'text-fg')} aria-live="polite">
          {est == null ? <Skeleton className="h-14 w-32" /> : est === 0 ? 'Free' : fmtUsd(est)}
        </div>
        <Meter value={est == null ? 0 : est / Math.max(0.0001, form.maxCost)} tone={over ? 'miss' : 'flame'} className="relative mt-3" label="Estimate against this run's limit" />
        <div className="relative mt-1.5 flex justify-between text-xs text-fg-3 tnum">
          <span>Limit {fmtUsd(form.maxCost)} per run</span>
          <span>{fmtUsd(sessionLeft)} left this session</span>
        </div>
        {over && cheaperModel && (
          <Button size="sm" variant="outline" className="relative mt-3 w-full" onClick={onCheaper}>
            Try {cheaperModel} instead
          </Button>
        )}
      </div>

      {estimate.data && estimate.data.breakdown.length > 0 && (
        <ul className="space-y-1.5 border-t border-line px-5 py-3 text-sm">
          {estimate.data.breakdown.map((b, i) => (
            <li key={i} className="flex items-baseline justify-between gap-3">
              <span className="min-w-0">
                <span className="font-semibold text-fg capitalize">{b.stage}</span>
                <span className="block truncate text-xs text-fg-3">{[b.model ?? b.provider, b.note].filter(Boolean).join(' · ')}</span>
              </span>
              <span className="shrink-0 text-fg-2 tnum">{b.usd === 0 ? 'free' : fmtUsd(b.usd)}</span>
            </li>
          ))}
        </ul>
      )}

      {(estimate.data?.warnings.length || estimate.error) && (
        <ul className="space-y-2 border-t border-line px-5 py-3 text-[0.8rem]">
          {estimate.error && (
            <li className="flex gap-2 text-fg-2">
              <AlertTriangle className="mt-0.5 size-3.5 shrink-0 text-miss-text" aria-hidden />
              Couldn't estimate: {estimate.error}
            </li>
          )}
          {estimate.data?.warnings.map((w, i) => (
            <li key={i} className="flex gap-2 text-fg-2">
              <AlertTriangle className="mt-0.5 size-3.5 shrink-0 text-attempt" aria-hidden />
              {w}
            </li>
          ))}
        </ul>
      )}

      <div className="border-t border-line px-5 py-3">
        <div className="mb-1.5 text-xs font-semibold text-fg-2">Where your footage goes</div>
        {egressLines.length ? (
          <ul className="space-y-1.5 text-[0.8rem]">
            {egressLines.map((l, i) => (
              <li key={i} className="flex gap-2">
                {l.local ? <HardDrive className="mt-0.5 size-3.5 shrink-0 text-made-text" aria-hidden /> : <ShieldCheck className="mt-0.5 size-3.5 shrink-0 text-flame-400" aria-hidden />}
                <span className="text-fg-2">
                  <b className="text-fg">{l.who}</b>: {l.what}
                </span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="flex gap-2 text-[0.8rem] text-fg-2">
            <HardDrive className="mt-0.5 size-3.5 shrink-0 text-made-text" aria-hidden /> Nothing leaves this computer.
          </p>
        )}
      </div>

      <div className="hidden border-t border-line p-5 lg:block">
        <Tip content={blockReason} disabled={!blockReason}>
          <span className="block" tabIndex={blockReason ? 0 : -1}>
            <Button variant="primary" size="lg" className="w-full" onClick={onAnalyze} loading={submitting} disabled={!!blockReason} icon={!submitting && <Sparkles className="size-5" aria-hidden />}>
              Analyze
            </Button>
          </span>
        </Tip>
        {blockReason && <p className="mt-2 text-center text-xs text-attempt">{blockReason}</p>}
        {submitError && <p className="mt-2 text-center text-xs text-miss-text">{submitError}</p>}
        {!blockReason && !submitError && <p className="mt-2 text-center text-xs text-fg-3">Usually takes 1–3 minutes for a short clip.</p>}
      </div>
    </Card>
  )
}
