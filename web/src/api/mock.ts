/**
 * In-memory implementation of the Api interface, so the whole UI is demoable without the backend.
 * Enable with VITE_MOCK=1 or ?mock=1 (sticky for the browser tab; ?mock=0 turns it off).
 */
import { ApiError, type Api } from './client'
import {
  MOCK_CLIP_ID,
  MOCK_RUN_B_ID,
  MOCK_RUN_ID,
  buildEvents,
  buildRun,
  makeClip,
  mockClips,
  mockConfig,
  mockPosterSvg,
  teamSummaries,
} from './fixtures'
import type {
  Answer,
  AppConfig,
  ClipCard,
  ClipMeta,
  ClipView,
  Estimate,
  EstimateLine,
  HoopsEvent,
  Job,
  JobStage,
  RunRequest,
  RunSummary,
  RunView,
} from './types'
import { STAGES } from './types'

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))
const jitter = (ms: number) => ms * (0.7 + Math.random() * 0.6)
const hex = (n: number) => Array.from({ length: n }, () => Math.floor(Math.random() * 16).toString(16)).join('')
const clone = <T>(v: T): T => structuredClone(v)

interface PlanStage {
  key: string
  label: string
  ms: number
  skipped: boolean
  messages: ((p: number) => string)[]
}

interface MockJob {
  job_id: string
  kind: Job['kind']
  clip_id: string | null
  run_id: string | null
  plan: PlanStage[]
  startedAt: number
  cancelledAt: number | null
  failAt: string | null
  finished: boolean
  onDone: () => void
}

const VLM_PRICE_PER_MIN: Record<string, number> = {
  'claude-opus-5-5': 0.33,
  'claude-sonnet-5-5': 0.17,
  'claude-haiku-5-5': 0.008,
  'gpt-6.1-sol': 0.15,
  'gpt-6-luna': 0.008,
  'gpt-6-astra': 0.75,
  'gemini-3.8-flash': 0.013,
  'gemini-3.5-flash-lite': 0.006,
  'gemini-3.1-pro-preview': 0.09,
}
const EFFORT_MULT: Record<string, number> = { low: 0.7, medium: 1, high: 1.4, xhigh: 1.8, max: 2.4 }

function frames(p: number, total: number) {
  return `${Math.round(p * total).toLocaleString()}/${total.toLocaleString()} frames`
}

export function createMockApi(): Api {
  const params = new URLSearchParams(window.location.search)
  const config: AppConfig = clone(mockConfig)
  const clips = new Map<string, ClipMeta>(mockClips.map((c) => [c.clip_id, clone(c)]))
  const posters = new Map<string, string>([
    [MOCK_CLIP_ID, mockPosterSvg('gym')],
    ['7d02be91c4aa3f10', mockPosterSvg('driveway')],
  ])
  const runs = new Map<string, RunView>()
  const jobs = new Map<string, MockJob>()
  const uploads = new Map<string, { filename: string; size: number; received: number; failedOnce: Set<number> }>()
  let authRequired = params.has('login') || sessionStorage.getItem('hoops_mock_login') === '1'
  if (params.has('login')) sessionStorage.setItem('hoops_mock_login', '1')
  let authenticated = !authRequired

  // Seed two finished runs over the same clip so Compare has something to show.
  const clipA = clips.get(MOCK_CLIP_ID)!
  runs.set(
    MOCK_RUN_ID,
    buildRun({
      run_id: MOCK_RUN_ID,
      clip: clipA,
      minutesAgo: 48,
      decider: 'jev',
      provider: 'claude',
      model: 'claude-opus-5-5',
      threshold: { key: 'balanced', accept: 0.85, reject: 0.15 },
      outputs: ['annotated_video', 'events', 'summary', 'court_map'],
      egress: ['claude'],
      wall_s: 142.6,
    }),
  )
  const runB = buildRun({
    run_id: MOCK_RUN_B_ID,
    clip: clipA,
    minutesAgo: 24,
    decider: 'rules',
    provider: 'openai',
    model: 'gpt-6.1-sol',
    threshold: { key: 'eager', accept: 0.7, reject: 0.05 },
    outputs: ['annotated_video', 'events', 'summary', 'court_map'],
    egress: ['openai'],
    wall_s: 118.2,
  })
  runB.events = runB.events.filter((e) => e.event_id !== 'e9' && e.event_id !== 'e12')
  runB.team_summaries = teamSummaries(runB.events, runB.teams)
  runs.set(MOCK_RUN_B_ID, runB)

  function clipView(c: ClipMeta): ClipView {
    return {
      ...clone(c),
      urls: {
        master: `mock:///media/clips/${c.clip_id}/master.mp4`,
        proxy: `mock:///media/clips/${c.clip_id}/proxy_720p.mp4`,
        poster: posters.get(c.clip_id) ?? mockPosterSvg('gym'),
      },
    }
  }

  function requireAuth() {
    if (authRequired && !authenticated) throw new ApiError(401, 'Not signed in')
  }

  // ---------------------------------------------------------------- jobs (state derived from elapsed time)

  function snapshot(mj: MockJob): Job {
    const now = mj.cancelledAt ?? Date.now()
    let elapsed = now - mj.startedAt
    const active = mj.plan.filter((s) => !s.skipped)
    const total = active.reduce((s, x) => s + x.ms, 0)
    let status: Job['status'] = 'running'
    let current: PlanStage | null = null
    let currentP = 0
    const stages: JobStage[] = mj.plan.map((s) => {
      if (s.skipped) return { key: s.key, label: s.label, status: 'skipped', progress: 0 }
      if (elapsed >= s.ms) {
        elapsed -= s.ms
        return { key: s.key, label: s.label, status: 'done', progress: 1 }
      }
      if (current === null) {
        current = s
        currentP = Math.max(0, elapsed / s.ms)
        elapsed = -Infinity
        return { key: s.key, label: s.label, status: 'running', progress: currentP }
      }
      return { key: s.key, label: s.label, status: 'pending', progress: 0 }
    })
    const doneMs = active.reduce((acc, s) => {
      const st = stages.find((x) => x.key === s.key)!
      return acc + (st.status === 'done' ? s.ms : st.status === 'running' ? s.ms * st.progress : 0)
    }, 0)
    let error: string | null = null
    if (current === null) status = 'done'
    if (mj.cancelledAt) {
      status = 'cancelled'
      for (const st of stages) if (st.status === 'running') st.status = 'cancelled'
    }
    if (mj.failAt && current && (current as PlanStage).key === mj.failAt && currentP > 0.5) {
      status = 'failed'
      error = 'The provider returned an error (simulated).'
      for (const st of stages) if (st.status === 'running') st.status = 'failed'
    }
    if (status === 'done' && !mj.finished) {
      mj.finished = true
      mj.onDone()
    }
    const cur = current as PlanStage | null
    const message =
      status === 'done' ? 'All done' : status === 'cancelled' ? 'Cancelled' : cur ? cur.messages[Math.min(cur.messages.length - 1, Math.floor(currentP * cur.messages.length))](currentP) : null
    return {
      job_id: mj.job_id,
      kind: mj.kind,
      status,
      progress: status === 'done' ? 1 : Math.min(0.999, doneMs / total),
      stage: cur ? cur.key : null,
      message,
      clip_id: mj.clip_id,
      run_id: mj.run_id,
      error,
      started_at: new Date(mj.startedAt).toISOString(),
      finished_at: status === 'running' ? null : new Date(now).toISOString(),
      stages,
    }
  }

  function getJob(jobId: string): MockJob {
    const mj = jobs.get(jobId)
    if (!mj) throw new ApiError(404, 'No such job')
    return mj
  }

  function analyzePlan(req: RunRequest, nFrames: number): PlanStage[] {
    const needs: Record<string, boolean> = {
      ingest: true,
      detect: true,
      track: true,
      shots: true,
      decide: true,
      adjudicate: !!req.vlm,
      summarize: !!req.vlm && req.outputs.includes('summary'),
      court: req.outputs.includes('court_map'),
      render: req.outputs.includes('annotated_video'),
    }
    const ms: Record<string, number> = { ingest: 500, detect: 2600, track: 900, shots: 700, decide: 1100, adjudicate: 1500, summarize: 1700, court: 700, render: 1500 }
    const msgs: Record<string, ((p: number) => string)[]> = {
      ingest: [() => 'Using the cached 720p proxy'],
      detect: [(p) => `Finding players & ball (${frames(p, nFrames)})`],
      track: [() => 'Linking detections into tracks', () => 'Splitting teams by jersey color'],
      shots: [() => 'Watching the rim zone for ball flights', () => 'Merging duplicate candidates'],
      decide: [() => `Asking ${req.decider} about each candidate shot`, () => 'Routing confident calls; escalating close ones'],
      adjudicate: [() => `Sending 2 close calls to ${req.vlm?.provider ?? 'the AI'}`],
      summarize: [() => 'Writing the recap', () => 'Checking every claim against the event log'],
      court: [() => 'Mapping shot spots onto the court'],
      render: [(p) => `Drawing overlays (${frames(p, nFrames)})`, () => 'Encoding H.264 for the browser'],
    }
    return STAGES.map((s) => ({ key: s.key, label: s.label, ms: jitter(ms[s.key]), skipped: !needs[s.key], messages: msgs[s.key] }))
  }

  // ---------------------------------------------------------------- estimate

  function estimate(req: RunRequest): Estimate {
    const clip = clips.get(req.clip_id)
    const minutes = (clip?.master.duration_s ?? 60) / 60
    const lines: EstimateLine[] = [{ stage: 'detect + track', provider: null, model: 'local (MPS)', usd: 0, note: 'Runs on this Mac' }]
    const decCost = req.decider === 'rules' ? 0 : req.decider === 'jev' ? 0.0012 : req.decider === 'openai-decisions' ? 0.009 : 0.018
    lines.push({ stage: 'decide', provider: req.decider, model: null, usd: decCost, note: req.decider === 'rules' ? 'Free' : '~10 candidate shots' })
    if (req.vlm) {
      const per = req.vlm.provider === 'ollama' ? 0 : (VLM_PRICE_PER_MIN[req.vlm.model] ?? 0.1)
      const effort = EFFORT_MULT[req.vlm.effort] ?? 1
      const nv = req.vlm.native_video ? 0.6 : 1
      const adj = per * minutes * 0.2 * effort * nv
      lines.push({ stage: 'adjudicate', provider: req.vlm.provider, model: req.vlm.model, usd: adj, note: 'Close calls only (~2 windows)' })
      if (req.outputs.includes('summary')) {
        lines.push({ stage: 'summarize', provider: req.vlm.provider, model: req.vlm.model, usd: per * minutes * 0.4 * effort * nv, note: 'Recap + claim check' })
      }
    }
    const total = lines.reduce((s, l) => s + l.usd, 0)
    const warnings: string[] = []
    if (total > req.max_cost_usd) warnings.push(`About $${total.toFixed(2)}, above the $${req.max_cost_usd.toFixed(2)} per-run cap. Pick a cheaper model or fewer outputs.`)
    if (config.spend.session_usd + total > config.caps.session_cost_cap_usd) warnings.push('This run would push the session past its spending cap.')
    if (req.detector === 'rfdetr:coco' && !req.rim_box) warnings.push('COCO weights have no rim class. Marking the rim makes shot detection much better.')
    if (req.outputs.includes('court_map') && !req.court_points && (req.footage_type ?? clip?.footage_type) !== 'broadcast')
      warnings.push('No court calibration, so the court map will be image-space (approximate).')
    return { estimate_usd: Math.round(total * 10000) / 10000, breakdown: lines, warnings }
  }

  function presetBounds(key: string) {
    const p = config.threshold_presets.find((x) => x.key === key) ?? config.threshold_presets[1]
    return { accept: p.accept, reject: p.reject }
  }

  function recomputeRun(run: RunView, preset: string) {
    const { accept, reject } = presetBounds(preset)
    const fresh = buildEvents(accept, reject, {
      decider: run.pipeline.find((s) => s.stage === 'decide')?.impl ?? 'rules',
      vlmProducer: run.vlm_calls[0] ? `${run.vlm_calls[0].provider}:${run.vlm_calls[0].model}` : null,
    })
    const old = new Map(run.events.map((e) => [e.event_id, e]))
    const merged: HoopsEvent[] = []
    for (const e of fresh) {
      const prev = old.get(e.event_id)
      if (!prev) continue // not in this run (e.g. dropped in run B)
      if (prev.review !== 'unreviewed' || prev.provenance.source === 'human') merged.push(prev)
      else merged.push(e)
      old.delete(e.event_id)
    }
    for (const e of old.values()) merged.push(e) // human-added events survive recompute
    merged.sort((a, b) => a.t_start_s - b.t_start_s)
    run.events = merged
    run.team_summaries = teamSummaries(merged, run.teams)
    const decide = run.pipeline.find((s) => s.stage === 'decide')
    if (decide) decide.params = { ...decide.params, threshold_preset: preset, accept, reject }
  }

  function getRun(runId: string): RunView {
    const r = runs.get(runId)
    if (!r) throw new ApiError(404, 'Run not found')
    return r
  }

  function fmt(t: number) {
    const m = Math.floor(t / 60)
    const s = Math.floor(t % 60)
    return `${m}:${s.toString().padStart(2, '0')}`
  }

  function answer(run: RunView, q: string, provider: string | null, model: string | null): Answer {
    const text = q.toLowerCase()
    const shots = run.events.filter((e) => e.shot && e.status !== 'rejected')
    const name = (id: string | null) => run.teams.find((t) => t.team_id === id)?.name ?? id ?? 'Unknown team'
    const vlmProvider = provider ?? run.vlm_calls[0]?.provider ?? null
    const vlmModel = model ?? run.vlm_calls[0]?.model ?? null
    const decision = { decider: run.pipeline.find((s) => s.stage === 'decide')?.impl ?? 'rules', route_confidence: 0.9 }
    if (/three|3s|3-point|deep|threes/.test(text)) {
      const threes = shots.filter((e) => e.shot?.points === 3)
      const made = threes.filter((e) => e.type === 'made_shot')
      return {
        answer: made.length
          ? `There ${made.length === 1 ? 'is' : 'are'} **${made.length} made three${made.length === 1 ? '' : 's'}** out of ${threes.length} attempts from deep:\n\n${made.map((e) => `- ${fmt(e.t_start_s)}: ${name(e.team_id)}, ${e.description?.toLowerCase() ?? 'three'}`).join('\n')}`
          : `No made threes yet. There ${threes.length === 1 ? 'was' : 'were'} ${threes.length} attempt${threes.length === 1 ? '' : 's'} from deep.`,
        route: 'find_moments',
        citations: (made.length ? made : threes).map((e) => ({ t_s: e.t_key_s ?? e.t_start_s, event_id: e.event_id, label: `${e.type === 'made_shot' ? 'Made' : 'Missed'} three, ${fmt(e.t_start_s)}` })),
        cost_usd: 0,
        provider: null,
        model: null,
        latency_ms: 140,
        decision: { ...decision, route_confidence: 0.94 },
      }
    }
    if (/who|better|percent|%|stats|score|points|shoot|shot|fg/.test(text) && !/why|explain/.test(text)) {
      const lines = run.team_summaries.map((ts) => {
        const pct = ts.stats.fga ? Math.round((ts.stats.fgm / ts.stats.fga) * 100) : 0
        return `- **${ts.team.name ?? ts.team.team_id}**: ${ts.stats.fgm}/${ts.stats.fga} (${pct}%), ${ts.stats.tpm}/${ts.stats.tpa} from three, ${ts.stats.pts} pts`
      })
      const best = [...run.team_summaries].sort((a, b) => b.stats.fgm / Math.max(1, b.stats.fga) - a.stats.fgm / Math.max(1, a.stats.fga))[0]
      return {
        answer: `**${best?.team.name ?? 'Neither team'}** shot better, counting confirmed and provisional calls:\n\n${lines.join('\n')}\n\nThis comes straight from the event log, so it changes if you review close calls.`,
        route: 'stats',
        citations: shots.filter((e) => e.type === 'made_shot').slice(0, 4).map((e) => ({ t_s: e.t_key_s ?? e.t_start_s, event_id: e.event_id, label: `${name(e.team_id)} make, ${fmt(e.t_start_s)}` })),
        cost_usd: 0,
        provider: null,
        model: null,
        latency_ms: 120,
        decision: { ...decision, route_confidence: 0.96 },
      }
    }
    if (/why|explain|how did|what happened|flag/.test(text)) {
      const target = shots.find((e) => e.status === 'needs_review') ?? shots[0]
      return {
        answer: `The shot at **${fmt(target.t_start_s)}** was flagged because the ball disappears behind a defender right at the rim. The tracker lost it for about 0.4 s, and no frame shows it below the rim. System One put the make at ${Math.round(Number(target.features.p_make ?? 0.5) * 100)}%, inside the uncertain band, so it went to ${vlmProvider ?? 'the AI'}. The AI also called it **unclear**.\n\nIf you can see it on the video, accept or reject it in the event log.`,
        route: 'explain_play',
        citations: [{ t_s: target.t_key_s ?? target.t_start_s, event_id: target.event_id, label: `Flagged shot, ${fmt(target.t_start_s)}` }],
        cost_usd: vlmProvider === 'ollama' ? 0 : 0.0124,
        provider: vlmProvider,
        model: vlmModel,
        latency_ms: 3420,
        decision: { ...decision, route_confidence: 0.88 },
      }
    }
    if (/summar|recap|overview|tl;?dr|story/.test(text)) {
      const n = run.narratives[0]
      const headline = (n?.provenance.params?.summary as { headline?: string } | undefined)?.headline
      return {
        answer: headline
          ? `${headline}. White tees led most of the way: a wing three at 0:38 swung it, and a fast-break layup at 1:12 sealed it.`
          : 'This run has no AI summary. Run it again with "AI summary & Q&A" on.',
        route: 'summary',
        citations: headline ? [{ t_s: 39.3, event_id: 'e5', label: 'Wing three, 0:38' }, { t_s: 72.5, event_id: 'e9', label: 'Layup, 1:12' }] : [],
        cost_usd: 0,
        provider: null,
        model: null,
        latency_ms: 90,
        decision: { ...decision, route_confidence: 0.91 },
      }
    }
    return {
      answer: "I can only answer questions about this clip: shots, makes and misses, when things happened, and what happened on a play. Try **\"Show me every made three\"** or **\"Why was the 0:31 shot flagged?\"**",
      route: 'unsupported',
      citations: [],
      cost_usd: 0,
      provider: null,
      model: null,
      latency_ms: 80,
      decision: { ...decision, route_confidence: 0.83 },
    }
  }

  const api: Api = {
    mock: true,
    async health() {
      return { ok: true, version: config.version }
    },
    async session() {
      await sleep(120)
      return { authenticated, auth_required: authRequired, user: authenticated ? 'friend' : null }
    },
    async login(password) {
      await sleep(500)
      if (password !== 'hoops') throw new ApiError(401, 'Wrong password')
      authenticated = true
      return { ok: true }
    },
    async logout() {
      authenticated = false
      authRequired = true
      sessionStorage.setItem('hoops_mock_login', '1')
      return { ok: true }
    },
    async config() {
      requireAuth()
      await sleep(250)
      return clone(config)
    },

    async createUpload({ filename, size_bytes }) {
      requireAuth()
      await sleep(200)
      if (size_bytes > config.caps.max_upload_mb * 1024 * 1024) throw new ApiError(400, `File is larger than ${config.caps.max_upload_mb} MB`)
      const id = `u_${hex(8)}`
      uploads.set(id, { filename, size: size_bytes, received: 0, failedOnce: new Set() })
      // Small chunks in the demo so multi-chunk progress is visible.
      return { upload_id: id, chunk_size_bytes: Math.max(256 * 1024, Math.ceil(size_bytes / 6)) }
    },
    async putChunk(uploadId, index, chunk, opts) {
      const up = uploads.get(uploadId)
      if (!up) throw new ApiError(404, 'Upload not found')
      const steps = 8
      for (let i = 1; i <= steps; i++) {
        if (opts?.signal?.aborted) throw new DOMException('Aborted', 'AbortError')
        await sleep(55 + Math.random() * 60)
        opts?.onProgress?.((chunk.size * i) / steps)
        // Simulate one flaky tunnel hiccup so the retry path is visible.
        if (index === 1 && i === 5 && !up.failedOnce.has(index)) {
          up.failedOnce.add(index)
          throw new ApiError(0, 'Network error while uploading')
        }
      }
      up.received += chunk.size
      return { received_bytes: up.received }
    },
    async completeUpload(uploadId) {
      const up = uploads.get(uploadId)
      if (!up) throw new ApiError(404, 'Upload not found')
      await sleep(250)
      const clipId = hex(16)
      const jobId = `j_${hex(6)}`
      const duration = 30 + Math.round(Math.random() * 80)
      const mj: MockJob = {
        job_id: jobId,
        kind: 'ingest',
        clip_id: null,
        run_id: null,
        startedAt: Date.now(),
        cancelledAt: null,
        failAt: null,
        finished: false,
        plan: [
          {
            key: 'ingest',
            label: 'Preparing video',
            ms: 4200,
            skipped: false,
            messages: [() => 'Checking the file', () => 'Converting to a browser-friendly format', (p) => `Making a 720p copy (${Math.round(p * 100)}%)`, () => 'Looking for camera cuts'],
          },
        ],
        onDone: () => {
          const c = makeClip(clipId, up.filename, duration, 'unknown', 0)
          clips.set(clipId, c)
          posters.set(clipId, mockPosterSvg('gym'))
          mj.clip_id = clipId
        },
      }
      jobs.set(jobId, mj)
      return { job_id: jobId }
    },

    async job(jobId) {
      requireAuth()
      await sleep(60)
      return snapshot(getJob(jobId))
    },
    async cancelJob(jobId) {
      const mj = getJob(jobId)
      const s = snapshot(mj)
      if (s.status === 'running' || s.status === 'queued') mj.cancelledAt = Date.now()
      return snapshot(mj)
    },
    streamJob(jobId, onJob, onError) {
      // Pretend the tunnel drops SSE when ?nosse=1, to exercise the polling fallback.
      if (params.has('nosse')) {
        const t = setTimeout(onError, 300)
        return () => clearTimeout(t)
      }
      const timer = setInterval(() => {
        const mj = jobs.get(jobId)
        if (!mj) {
          clearInterval(timer)
          onError()
          return
        }
        const s = snapshot(mj)
        onJob(s)
        if (s.status !== 'running' && s.status !== 'queued') clearInterval(timer)
      }, 250)
      return () => clearInterval(timer)
    },

    async clips() {
      requireAuth()
      await sleep(300)
      return [...clips.values()]
        .sort((a, b) => b.created_at.localeCompare(a.created_at))
        .map<ClipCard>((c) => ({
          clip_id: c.clip_id,
          original_filename: c.original_filename,
          duration_s: c.master.duration_s,
          footage_type: c.footage_type,
          poster_url: posters.get(c.clip_id) ?? mockPosterSvg('gym'),
          created_at: c.created_at,
        }))
    },
    async clip(clipId) {
      requireAuth()
      await sleep(220)
      const c = clips.get(clipId)
      if (!c) throw new ApiError(404, 'Clip not found')
      return clipView(c)
    },
    async patchClip(clipId, { footage_type }) {
      const c = clips.get(clipId)
      if (!c) throw new ApiError(404, 'Clip not found')
      c.footage_type = footage_type
      c.footage_type_source = 'human'
      return clipView(c)
    },

    async estimate(req, signal) {
      await sleep(260)
      if (signal?.aborted) throw new DOMException('Aborted', 'AbortError')
      return estimate(req)
    },
    async createRun(req) {
      requireAuth()
      await sleep(400)
      const clip = clips.get(req.clip_id)
      if (!clip) throw new ApiError(404, 'Clip not found')
      if (req.vlm) {
        const p = config.providers.find((x) => x.key === req.vlm!.provider)
        if (!p || !p.available) throw new ApiError(503, `${p?.label ?? req.vlm.provider} is unavailable: ${p?.reason ?? 'unknown provider'}`)
        if (!req.egress_ok.includes(req.vlm.provider)) throw new ApiError(400, `Allow frames to be sent to ${p.sends_data_to} first.`)
      }
      const est = estimate(req)
      if (est.estimate_usd > req.max_cost_usd)
        throw new ApiError(402, `This run would cost about $${est.estimate_usd.toFixed(2)}, above your $${req.max_cost_usd.toFixed(2)} per-run cap.`)
      if (config.spend.session_usd + est.estimate_usd > config.caps.session_cost_cap_usd)
        throw new ApiError(402, `This session has spent $${config.spend.session_usd.toFixed(2)} of its $${config.caps.session_cost_cap_usd.toFixed(2)} cap.`)
      const runId = `${new Date().toISOString().slice(0, 19).replace(/[-:]/g, '').replace('T', '-')}-${hex(4)}`
      const jobId = `j_${hex(6)}`
      const preset = config.threshold_presets.find((p) => p.key === req.threshold_preset) ?? config.threshold_presets[1]
      const mj: MockJob = {
        job_id: jobId,
        kind: 'analyze',
        clip_id: clip.clip_id,
        run_id: runId,
        plan: analyzePlan(req, clip.proxy.n_frames),
        startedAt: Date.now(),
        cancelledAt: null,
        failAt: params.has('fail') ? 'adjudicate' : null,
        finished: false,
        onDone: () => {
          const run = buildRun({
            run_id: runId,
            clip: { ...clip, footage_type: req.footage_type ?? (clip.footage_type === 'unknown' ? 'fixed_camera' : clip.footage_type) },
            minutesAgo: 0,
            decider: req.decider,
            provider: req.vlm?.provider ?? null,
            model: req.vlm?.model ?? null,
            threshold: { key: preset.key, accept: preset.accept, reject: preset.reject },
            outputs: req.outputs,
            egress: req.vlm && !config.providers.find((p) => p.key === req.vlm!.provider)?.local ? [req.vlm.provider] : [],
            wall_s: Math.round((Date.now() - mj.startedAt) / 100) / 10,
          })
          runs.set(runId, run)
          config.spend.session_usd = Math.round((config.spend.session_usd + run.metrics.cost_usd) * 10000) / 10000
        },
      }
      jobs.set(jobId, mj)
      return { run_id: runId, job_id: jobId, estimate_usd: est.estimate_usd }
    },
    async runs() {
      requireAuth()
      await sleep(320)
      return [...runs.values()]
        .sort((a, b) => b.created_at.localeCompare(a.created_at))
        .map<RunSummary>((r) => ({
          run_id: r.run_id,
          created_at: r.created_at,
          status: r.status,
          clip_id: r.clip.clip_id,
          filename: r.clip.original_filename,
          footage_type: r.clip.footage_type,
          outputs: r.outputs_requested,
          n_events: r.events.length,
          cost_usd: r.metrics.cost_usd,
          providers: [...new Set(r.vlm_calls.map((c) => c.provider))].sort(),
        }))
    },
    async run(runId) {
      requireAuth()
      await sleep(300)
      const r = runs.get(runId)
      if (r) return clone(r)
      const pending = [...jobs.values()].find((j) => j.run_id === runId)
      if (pending) throw new ApiError(409, 'This run is still in progress')
      throw new ApiError(404, 'Run not found')
    },
    async recompute(runId, preset) {
      await sleep(450)
      const run = getRun(runId)
      recomputeRun(run, preset)
      return { events: clone(run.events), team_summaries: clone(run.team_summaries) }
    },
    async review(runId, eventId, { action, note }) {
      await sleep(250)
      const run = getRun(runId)
      const e = run.events.find((x) => x.event_id === eventId)
      if (!e) throw new ApiError(404, 'Event not found')
      e.review = action === 'accept' ? 'accepted' : 'rejected'
      e.status = action === 'accept' ? 'confirmed' : 'rejected'
      if (note) e.features = { ...e.features, review_note: note }
      run.team_summaries = teamSummaries(run.events, run.teams)
      return clone(e)
    },
    async addEvent(runId, { t_s, type, team_id }) {
      await sleep(300)
      const run = getRun(runId)
      const e: HoopsEvent = {
        event_id: `h${hex(4)}`,
        type,
        t_start_s: t_s,
        t_end_s: t_s + 1,
        t_key_s: t_s,
        team_id,
        player_ids: [],
        related_event_ids: [],
        shot: { points: 2, result: type === 'made_shot' ? 'made' : type === 'missed_shot' ? 'missed' : 'unknown', kind: 'unknown', court_x_ft: null, court_y_ft: null, image_x: null, image_y: null },
        is_replay: false,
        confidence: 1,
        confidence_kind: 'human',
        status: 'confirmed',
        review: 'edited',
        description: 'Added by you',
        features: {},
        evidence: [],
        provenance: { source: 'human', producer: 'web-ui', stage: 'review', prompt_id: null, params: {}, created_at: new Date().toISOString() },
      }
      run.events = [...run.events, e].sort((a, b) => a.t_start_s - b.t_start_s)
      run.team_summaries = teamSummaries(run.events, run.teams)
      return clone(e)
    },
    async ask(runId, { question, provider, model }) {
      const run = getRun(runId)
      const a = answer(run, question, provider, model)
      await sleep(Math.min(2200, a.latency_ms / 2 + 350))
      return a
    },
    async compare(a, b) {
      requireAuth()
      await sleep(350)
      const ra = getRun(a)
      const rb = getRun(b)
      const family = (e: HoopsEvent) => (e.shot || ['made_shot', 'missed_shot', 'shot_attempt', 'three_pointer', 'free_throw'].includes(e.type) ? 'shot' : e.type)
      const ea = ra.events.filter((e) => e.status !== 'rejected')
      const eb = rb.events.filter((e) => e.status !== 'rejected')
      const used = new Set<string>()
      let matched = 0
      let sameOutcome = 0
      let comparable = 0
      for (const x of ea) {
        const y = eb.find((c) => !used.has(c.event_id) && family(c) === family(x) && Math.abs(c.t_start_s - x.t_start_s) <= 1)
        if (!y) continue
        used.add(y.event_id)
        matched++
        if (family(x) === 'shot') {
          comparable++
          if (x.type === y.type) sameOutcome++
        }
      }
      return {
        a: clone(ra),
        b: clone(rb),
        agreement: { matched, only_a: ea.length - matched, only_b: eb.length - matched, outcome_agreement: comparable ? sameOutcome / comparable : null },
      }
    },
  }
  return api
}
