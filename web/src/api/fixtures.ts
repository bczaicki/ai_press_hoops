/**
 * Demo fixtures for MOCK mode. Everything here is synthetic: no real footage, people, or teams.
 * The "video" is drawn live on a canvas (see components/SimulatedVideo.tsx) on top of the
 * generated poster below, so rim/court clicks and shot positions line up with the picture.
 */
import type {
  AppConfig,
  ClipMeta,
  CourtSpec,
  DecisionRecord,
  EventStatus,
  EventType,
  HoopsEvent,
  Narrative,
  RunView,
  ShotKind,
  Source,
  StageConfig,
  TeamRef,
  VLMCallRecord,
} from './types'

export const MOCK_PROXY_W = 1280
export const MOCK_PROXY_H = 720
/** Rim center in proxy pixels for the generated scene. */
export const MOCK_RIM = { x: 1004, y: 262, w: 46, h: 10 }

const iso = (minutesAgo: number) => new Date(Date.now() - minutesAgo * 60_000).toISOString()

// ------------------------------------------------------------------ poster (generated SVG)

export function mockPosterSvg(variant: 'gym' | 'driveway' = 'gym'): string {
  const W = MOCK_PROXY_W
  const H = MOCK_PROXY_H
  const sky = variant === 'gym' ? ['#1a2236', '#0b1020'] : ['#2b3b5c', '#141d31']
  const floorTop = variant === 'gym' ? '#b77a43' : '#4a4f5a'
  const floorBottom = variant === 'gym' ? '#7a4a22' : '#2d3038'
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">
  <defs>
    <linearGradient id="wall" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${sky[0]}"/><stop offset="1" stop-color="${sky[1]}"/></linearGradient>
    <linearGradient id="floor" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${floorTop}"/><stop offset="1" stop-color="${floorBottom}"/></linearGradient>
    <radialGradient id="light" cx="0.5" cy="0" r="0.8"><stop offset="0" stop-color="#fff" stop-opacity="0.18"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></radialGradient>
    <pattern id="planks" width="56" height="720" patternUnits="userSpaceOnUse"><rect width="56" height="720" fill="none"/><line x1="0" y1="0" x2="0" y2="720" stroke="#000" stroke-opacity="0.08" stroke-width="2"/></pattern>
  </defs>
  <rect width="${W}" height="${H}" fill="url(#wall)"/>
  <rect width="${W}" height="${H}" fill="url(#light)"/>
  ${variant === 'gym' ? `<g opacity="0.35">${Array.from({ length: 9 }, (_, i) => `<rect x="${60 + i * 140}" y="40" width="70" height="26" rx="4" fill="#dfe8ff" opacity="${0.25 + (i % 3) * 0.1}"/>`).join('')}</g>` : ''}
  <path d="M0 330 L1280 300 L1280 720 L0 720 Z" fill="url(#floor)"/>
  <path d="M0 330 L1280 300 L1280 720 L0 720 Z" fill="url(#planks)"/>
  <g fill="none" stroke="#f4efe6" stroke-opacity="0.75" stroke-width="4">
    <path d="M40 690 L240 340"/>
    <path d="M1240 690 L1110 318"/>
    <path d="M300 520 Q 640 600 1000 470"/>
    <path d="M770 330 L 1180 318 L 1220 470 L 860 492 Z"/>
    <ellipse cx="900" cy="410" rx="120" ry="34"/>
  </g>
  <g>
    <rect x="1118" y="150" width="10" height="230" fill="#9aa3b5"/>
    <rect x="960" y="170" width="92" height="66" rx="3" fill="#e9eef7" fill-opacity="0.9" stroke="#2b3449" stroke-width="3"/>
    <rect x="988" y="196" width="36" height="28" fill="none" stroke="#d0442f" stroke-width="3"/>
    <line x1="1052" y1="214" x2="1118" y2="214" stroke="#9aa3b5" stroke-width="6"/>
    <ellipse cx="${MOCK_RIM.x}" cy="${MOCK_RIM.y}" rx="${MOCK_RIM.w / 2}" ry="${MOCK_RIM.h / 2}" fill="none" stroke="#ff5a1f" stroke-width="5"/>
    <path d="M${MOCK_RIM.x - 21} ${MOCK_RIM.y + 3} L${MOCK_RIM.x - 13} ${MOCK_RIM.y + 34} L${MOCK_RIM.x + 13} ${MOCK_RIM.y + 34} L${MOCK_RIM.x + 21} ${MOCK_RIM.y + 3}" fill="none" stroke="#f1f1f1" stroke-opacity="0.7" stroke-width="2"/>
  </g>
  <g font-family="Arial, sans-serif" font-weight="700" fill="#ffffff" fill-opacity="0.55">
    <text x="36" y="${H - 34}" font-size="22" letter-spacing="4">SIMULATED DEMO FOOTAGE</text>
  </g>
</svg>`
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`
}

// ------------------------------------------------------------------ config

/** Landmarks in the /api/config convention: feet from center court, x toward the filmed basket, +y camera-left. */
function court(key: string, label: string, L: number, Wd: number, lane: number, ft: number): CourtSpec {
  const b = L / 2
  const w = Wd / 2
  const l = lane / 2
  const lm = (name: string, lbl: string, x: number, y: number, filmed = true) => ({ name, label: lbl, x_ft: x, y_ft: y, filmed_half: filmed })
  return {
    key,
    label,
    length_ft: L,
    width_ft: Wd,
    landmarks: [
      lm('baseline_left_corner', 'Baseline left corner', b, w),
      lm('baseline_right_corner', 'Baseline right corner', b, -w),
      lm('lane_left_baseline', 'Paint, left edge at the baseline', b, l),
      lm('lane_right_baseline', 'Paint, right edge at the baseline', b, -l),
      lm('free_throw_left', 'Free-throw line, left end', b - ft, l),
      lm('free_throw_right', 'Free-throw line, right end', b - ft, -l),
      lm('halfcourt_left', 'Half-court line, left sideline', 0, w),
      lm('halfcourt_right', 'Half-court line, right sideline', 0, -w),
      lm('center_court', 'Center of the center circle', 0, 0),
      lm('other_baseline_left_corner', 'Far baseline, left corner', -b, w, false),
      lm('other_baseline_right_corner', 'Far baseline, right corner', -b, -w, false),
    ],
  }
}

export const mockConfig: AppConfig = {
  version: '0.1.0-mock',
  providers: [
    {
      key: 'claude',
      label: 'Claude (Anthropic)',
      available: true,
      reason: null,
      native_video: false,
      audio: false,
      local: false,
      sends_data_to: 'Anthropic',
      default_model: 'claude-opus-5-5',
      models: ['claude-opus-5-5', 'claude-sonnet-5-5', 'claude-haiku-5-5'],
    },
    {
      key: 'gemini',
      label: 'Gemini (Google, native video)',
      available: false,
      reason: 'missing GEMINI_API_KEY in .env',
      native_video: true,
      audio: true,
      local: false,
      sends_data_to: 'Google (paid tier only)',
      default_model: 'gemini-3.8-flash',
      models: ['gemini-3.8-flash', 'gemini-3.5-flash-lite', 'gemini-3.1-pro-preview'],
    },
    {
      key: 'openai',
      label: 'OpenAI',
      available: false,
      reason: 'OpenAI account is out of credits (add billing at platform.openai.com)',
      native_video: false,
      audio: false,
      local: false,
      sends_data_to: 'OpenAI',
      default_model: 'gpt-6.1-sol',
      models: ['gpt-6.1-sol', 'gpt-6-luna', 'gpt-6-astra'],
    },
    {
      key: 'openrouter',
      label: 'OpenRouter (any model)',
      available: false,
      reason: 'missing OPENROUTER_API_KEY in .env',
      native_video: false,
      audio: false,
      local: false,
      sends_data_to: 'OpenRouter + the routed model provider',
      default_model: 'google/gemini-3.8-flash',
      models: ['google/gemini-3.8-flash', 'qwen/qwen3.5-vl-72b'],
    },
    {
      key: 'ollama',
      label: 'Local (Ollama Qwen3.5)',
      available: true,
      reason: null,
      native_video: false,
      audio: false,
      local: true,
      sends_data_to: 'nobody (runs on this Mac)',
      default_model: 'gemma4:latest',
      models: ['gemma4:latest', 'qwen3.5:4b'],
    },
  ],
  detectors: [
    { key: 'rfdetr:coco', label: 'RF-DETR (COCO)', available: true, reason: null, license: 'Apache-2.0', notes: 'General people + ball. No rim class, so mark the rim.' },
    { key: 'rfdetr:basketball', label: 'RF-DETR (basketball weights)', available: false, reason: 'missing ROBOFLOW_API_KEY in .env', license: 'Apache-2.0', notes: 'Knows rims and made baskets. NBA-trained.' },
    { key: 'yoloe:26', label: 'YOLOE-26 (type what to find)', available: false, reason: "missing Python package 'ultralytics' (install the matching uv extra)", license: 'AGPL-3.0', notes: 'Experimental.' },
  ],
  trackers: [
    { key: 'bytetrack', label: 'ByteTrack (fixed camera)', available: true, reason: null, license: 'Apache-2.0', notes: '' },
    { key: 'botsort', label: 'BoT-SORT (moving camera)', available: true, reason: null, license: 'Apache-2.0', notes: 'Compensates for camera pans.' },
    { key: 'ocsort', label: 'OC-SORT', available: true, reason: null, license: 'Apache-2.0', notes: '' },
  ],
  deciders: [
    { key: 'rules', label: 'Rules (baseline)', available: true, reason: null, license: '', notes: 'Deterministic thresholds. Free and instant.', supports_images: false },
    { key: 'jev', label: 'TypeSafe Jev', available: true, reason: null, license: 'TypeSafe API', notes: 'Calibrated System One model.', supports_images: false },
    { key: 'adapter:anthropic:claude-opus-5-5', label: 'Adapter: Claude', available: true, reason: null, license: 'MIT', notes: 'LLM-backed stand-in with the Jev API.', supports_images: false },
    { key: 'adapter:gemini:gemini-3.8-flash', label: 'Adapter: Gemini', available: false, reason: 'missing GEMINI_API_KEY in .env', license: 'MIT', notes: '', supports_images: false },
    { key: 'openai-decisions', label: 'OpenAI Decisions (gpt-6-luna)', available: true, reason: null, license: 'OpenAI API', notes: 'Looks at rim crops (images).', supports_images: true },
  ],
  threshold_presets: [
    { key: 'strict', label: 'Strict', accept: 0.92, reject: 0.3, note: 'fewer phantom makes' },
    { key: 'balanced', label: 'Balanced', accept: 0.85, reject: 0.15, note: '' },
    { key: 'eager', label: 'Eager', accept: 0.7, reject: 0.05, note: 'miss fewer shots' },
  ],
  prompt_presets: [
    { key: 'recap', label: 'Recap' },
    { key: 'play_by_play', label: 'Play-by-play' },
    { key: 'coach', label: "Coach's notes" },
    { key: 'custom', label: 'Custom' },
  ],
  outputs: [
    { key: 'annotated_video', label: 'Annotated video' },
    { key: 'events', label: 'Event log + stats' },
    { key: 'summary', label: 'AI summary & Q&A' },
    { key: 'court_map', label: 'Court map / heatmap' },
  ],
  footage_types: ['unknown', 'broadcast', 'fixed_camera', 'handheld', 'highlights', 'practice'],
  courts: [court('nba', 'NBA', 94, 50, 16, 19), court('ncaa', 'NCAA', 94, 50, 12, 19), court('fiba', 'FIBA', 91.86, 49.21, 16.08, 19.03), court('hs', 'High school', 84, 50, 12, 19)],
  caps: { run_cost_cap_usd: 0.5, session_cost_cap_usd: 5, max_upload_mb: 500, max_clip_seconds: 600 },
  spend: { session_usd: 0.42 },
}

// ------------------------------------------------------------------ clips

function videoAsset(path: string, w: number, h: number, fps: string, fpsFloat: number, dur: number, audio: boolean) {
  return {
    path,
    width: w,
    height: h,
    fps,
    fps_float: fpsFloat,
    n_frames: Math.round(dur * fpsFloat),
    duration_s: dur,
    codec: 'h264',
    pix_fmt: 'yuv420p',
    has_audio: audio,
  }
}

export function makeClip(id: string, filename: string, duration: number, footage: ClipMeta['footage_type'], minutesAgo: number): ClipMeta {
  return {
    clip_id: id,
    original_filename: filename,
    source_size_bytes: Math.round(duration * 4_350_000),
    master: videoAsset(`clips/${id}/master.mp4`, 1920, 1080, '30000/1001', 29.97, duration, true),
    proxy: videoAsset(`clips/${id}/proxy_720p.mp4`, MOCK_PROXY_W, MOCK_PROXY_H, '30/1', 30, duration, false),
    normalization: {
      rotation_deg: 0,
      hdr: 'hlg',
      tonemapped: true,
      deinterlaced: false,
      vfr_source: true,
      sar_fixed: false,
      fps_snapped_from: '29.83',
      source_resolution: [3840, 2160],
      ffmpeg_version: '7.1.1',
      warnings: [],
      summary: 'HDR→SDR (HLG), 4K→1080p, 29.97 fps CFR, location metadata removed',
    },
    footage_type: footage,
    footage_type_source: footage === 'unknown' ? null : 'system_one',
    scene_cuts_s: [],
    created_at: iso(minutesAgo),
  }
}

export const MOCK_CLIP_ID = 'c3f9a1b2d4e5f607'
export const mockClips: ClipMeta[] = [
  makeClip(MOCK_CLIP_ID, 'saturday-run-court2.mov', 94.6, 'fixed_camera', 52),
  makeClip('7d02be91c4aa3f10', 'driveway-1v1.mp4', 41.2, 'handheld', 60 * 26),
]

// ------------------------------------------------------------------ events

export const mockTeams: TeamRef[] = [
  { team_id: 'A', name: 'White tees', color_hex: '#e9edf3' },
  { team_id: 'B', name: 'Dark tees', color_hex: '#26324d' },
]

interface ShotSeed {
  id: string
  t: number
  team: 'A' | 'B'
  p_make: number
  p_attempt: number
  points: 2 | 3
  kind: ShotKind
  court: [number, number]
  img: [number, number]
  source?: Source
  desc: string
  vlm?: 'made' | 'missed' | 'unclear'
}

/** Court coordinates: feet from center court (x along the 94 ft length, y across the 50 ft width). */
export const SHOT_SEEDS: ShotSeed[] = [
  { id: 'e1', t: 6.2, team: 'A', p_make: 0.95, p_attempt: 0.98, points: 2, kind: 'jumper', court: [29.0, 8.5], img: [742, 452], desc: 'Elbow jumper, swish', source: 'system_one' },
  { id: 'e2', t: 14.8, team: 'B', p_make: 0.06, p_attempt: 0.97, points: 3, kind: 'jumper', court: [19.5, -6.0], img: [612, 520], desc: 'Top-of-key three, back rim', source: 'system_one' },
  { id: 'e3', t: 22.1, team: 'B', p_make: 0.74, p_attempt: 0.96, points: 2, kind: 'layup', court: [39.5, -2.5], img: [948, 380], desc: 'Backdoor cut, left-hand layup', source: 'vlm', vlm: 'made' },
  { id: 'e4', t: 31.4, team: 'A', p_make: 0.47, p_attempt: 0.71, points: 2, kind: 'jumper', court: [31.0, 13.5], img: [812, 488], desc: 'Baseline fadeaway, ball hidden behind defender', source: 'system_one', vlm: 'unclear' },
  { id: 'e5', t: 38.9, team: 'A', p_make: 0.81, p_attempt: 0.95, points: 3, kind: 'jumper', court: [24.0, 17.5], img: [520, 470], desc: 'Left-wing three off a kick-out', source: 'system_one' },
  { id: 'e6', t: 47.3, team: 'A', p_make: 0.07, p_attempt: 0.94, points: 2, kind: 'jumper', court: [33.0, -9.5], img: [868, 520], desc: 'Pull-up from the right block, short', source: 'system_one' },
  { id: 'e7', t: 55.0, team: 'B', p_make: 0.96, p_attempt: 0.99, points: 2, kind: 'jumper', court: [30.5, 0.5], img: [760, 430], desc: 'Free-throw-line jumper', source: 'system_one' },
  { id: 'e8', t: 63.6, team: 'B', p_make: 0.22, p_attempt: 0.9, points: 2, kind: 'layup', court: [40.0, 3.0], img: [980, 392], desc: 'Contested layup rolls off', source: 'system_one' },
  { id: 'e10', t: 81.5, team: 'B', p_make: 0.62, p_attempt: 0.88, points: 3, kind: 'jumper', court: [22.0, -16.0], img: [452, 548], desc: 'Deep right-wing three, ball leaves frame', source: 'system_one', vlm: 'unclear' },
  { id: 'e11', t: 88.4, team: 'A', p_make: 0.88, p_attempt: 0.22, points: 2, kind: 'tip', court: [41.0, -1.0], img: [992, 360], desc: 'Possible tip-in in traffic', source: 'cv' },
]

function provenance(source: Source, stage: string, producer: string, params: Record<string, unknown> = {}) {
  return { source, producer, stage, prompt_id: null, params, created_at: iso(48) }
}

/** Derive type/status/confidence from stored scores, exactly what /recompute does server-side (no re-inference). */
export function classifyShot(seed: ShotSeed, accept: number, reject: number): { type: EventType; status: EventStatus; confidence: number; result: 'made' | 'missed' | 'unknown' } {
  if (seed.p_attempt <= reject) return { type: 'shot_attempt', status: 'rejected', confidence: 1 - seed.p_attempt, result: 'unknown' }
  const p = seed.p_make
  if (p >= accept) return { type: 'made_shot', status: 'confirmed', confidence: p, result: 'made' }
  if (p <= 1 - accept) return { type: 'missed_shot', status: 'confirmed', confidence: 1 - p, result: 'missed' }
  if (seed.vlm === 'made') return { type: 'made_shot', status: 'confirmed', confidence: 0.88, result: 'made' }
  if (seed.vlm === 'missed') return { type: 'missed_shot', status: 'confirmed', confidence: 0.86, result: 'missed' }
  if (Math.abs(p - 0.5) >= 0.2) {
    return p > 0.5
      ? { type: 'made_shot', status: 'provisional', confidence: p, result: 'made' }
      : { type: 'missed_shot', status: 'provisional', confidence: 1 - p, result: 'missed' }
  }
  return { type: 'shot_attempt', status: 'needs_review', confidence: Math.max(p, 1 - p), result: 'unknown' }
}

export function buildEvents(accept: number, reject: number, opts: { decider: string; vlmProducer: string | null } = { decider: 'jev', vlmProducer: 'anthropic:claude-opus-5-5' }): HoopsEvent[] {
  const shots: HoopsEvent[] = SHOT_SEEDS.map((s) => {
    const c = classifyShot(s, accept, reject)
    const usedVlm = s.vlm && opts.vlmProducer && s.vlm !== 'unclear' && c.status === 'confirmed' && s.p_make < accept && s.p_make > 1 - accept
    const source: Source = usedVlm ? 'vlm' : s.source === 'vlm' ? 'system_one' : (s.source ?? 'system_one')
    const producer =
      source === 'vlm' ? (opts.vlmProducer ?? 'none') : source === 'cv' ? 'rfdetr==1.11.2:coco-small' : `typesafe:${opts.decider}`
    return {
      event_id: s.id,
      type: c.type,
      t_start_s: s.t,
      t_end_s: s.t + 1.6,
      t_key_s: s.t + 0.4,
      team_id: s.team,
      player_ids: [],
      related_event_ids: [],
      shot: { points: s.points, result: c.result, kind: s.kind, court_x_ft: s.court[0], court_y_ft: s.court[1], image_x: s.img[0], image_y: s.img[1] },
      is_replay: false,
      confidence: Math.round(c.confidence * 100) / 100,
      confidence_kind: source === 'vlm' ? 'self_reported' : 'calibrated',
      status: c.status,
      review: 'unreviewed',
      description: s.desc,
      features: {
        p_attempt: s.p_attempt,
        p_make: s.p_make,
        ball_path: s.p_make > 0.5 ? 'rose above rim, then dropped through rim region' : 'rose above rim, deflected sideways off the rim',
        rim_visible: s.vlm === 'unclear' ? 'partly hidden' : 'throughout',
        scene_cut_nearby: 'no',
        ...(s.vlm ? { vlm_verdict: s.vlm } : {}),
      },
      evidence: [{ t_s: s.t + 0.4, frame: Math.round((s.t + 0.4) * 30), image_path: null, tile_id: `S${Math.floor(s.t / 10) + 1}#${(Math.round(s.t) % 9) + 1}`, note: null }],
      provenance: provenance(source, source === 'vlm' ? 'adjudicate' : 'decide', producer),
    }
  })
  const human: HoopsEvent = {
    event_id: 'e9',
    type: 'made_shot',
    t_start_s: 72.2,
    t_end_s: 73.4,
    t_key_s: 72.5,
    team_id: 'A',
    player_ids: [],
    related_event_ids: [],
    shot: { points: 2, result: 'made', kind: 'layup', court_x_ft: 38.5, court_y_ft: 4.0, image_x: 930, image_y: 400 },
    is_replay: false,
    confidence: 1,
    confidence_kind: 'human',
    status: 'confirmed',
    review: 'edited',
    description: 'Fast-break layup (added by you)',
    features: {},
    evidence: [],
    provenance: provenance('human', 'review', 'web-ui'),
  }
  const rebound: HoopsEvent = {
    event_id: 'e12',
    type: 'rebound',
    t_start_s: 48.4,
    t_end_s: 49.0,
    t_key_s: null,
    team_id: 'B',
    player_ids: ['B#12'],
    related_event_ids: ['e6'],
    shot: null,
    is_replay: false,
    confidence: 0.77,
    confidence_kind: 'heuristic',
    status: 'provisional',
    review: 'unreviewed',
    description: 'Defensive rebound',
    features: {},
    evidence: [],
    provenance: provenance('rule', 'shots', 'hoops.events.stats'),
  }
  return [...shots, human, rebound].sort((a, b) => a.t_start_s - b.t_start_s)
}

export function teamSummaries(events: HoopsEvent[], teams: TeamRef[]) {
  return teams.map((team) => {
    const stats = { pts: 0, fga: 0, fgm: 0, tpa: 0, tpm: 0, fta: 0, ftm: 0, reb: 0, ast: 0, stl: 0, blk: 0, tov: 0, pf: 0 }
    const ids: string[] = []
    for (const e of events) {
      if (e.team_id !== team.team_id || e.status === 'rejected') continue
      if (e.type === 'rebound') {
        stats.reb += 1
        ids.push(e.event_id)
        continue
      }
      if (!['made_shot', 'missed_shot', 'shot_attempt', 'three_pointer'].includes(e.type)) continue
      ids.push(e.event_id)
      const three = e.shot?.points === 3
      stats.fga += 1
      if (three) stats.tpa += 1
      if (e.type === 'made_shot') {
        stats.fgm += 1
        if (three) stats.tpm += 1
        stats.pts += three ? 3 : 2
      }
    }
    return { team, stats, derived_from_event_ids: ids }
  })
}

// ------------------------------------------------------------------ narrative, decisions, VLM calls

export function mockNarrative(producer: string): Narrative {
  return {
    kind: 'recap',
    text: `# White tees win the shooting battle in a scrappy 1:35 run

**White tees** opened with a smooth elbow jumper at 0:06 and never trailed for long. **Dark tees** answered at the rim — their layup at 0:22 came off a quick backdoor cut — and stayed close through the middle stretch.

The swing came at 0:38, when White tees buried a three from the left wing. Dark tees' #12 hit three straight jumpers to keep it close. A late White tees layup at 1:12 restored the cushion.

- Most attempts came from inside the arc; only a few were from deep.
- Two close calls (0:31 and 1:21) were too hard to see and are waiting for your review.`,
    question: null,
    cited_event_ids: ['e1', 'e3', 'e5', 'e9'],
    cited_times_s: [6.6, 22.5, 39.3, 72.5],
    unsupported_claims: ["Dark tees' #12 hit three straight jumpers to keep it close."],
    provenance: provenance('vlm', 'summarize', producer, {
      summary: {
        headline: 'White tees win the shooting battle in a scrappy 1:35 run',
        key_moments: [
          { ref: '0:06', description: 'Elbow jumper opens the scoring' },
          { ref: '0:22', description: 'Backdoor cut ends in a layup' },
          { ref: '0:38', description: 'Left-wing three swings it' },
          { ref: '1:12', description: 'Fast-break layup restores the lead' },
        ],
        caveats: [
          'The ball was hard to see against the dark backdrop between 0:28 and 0:33.',
          'Jersey numbers come from the footage only. No names are inferred.',
        ],
      },
    }),
  }
}

function decision(
  id: string,
  decider: string,
  purpose: DecisionRecord['purpose'],
  subject: string | null,
  state: DecisionRecord['state'],
  questions: DecisionRecord['questions'],
  answers: DecisionRecord['answers'],
  routed: DecisionRecord['routed'],
  latency: number,
  cost: number | null,
): DecisionRecord {
  return { decision_id: id, decider, purpose, subject_id: subject, state, questions, answers, routed, latency_ms: latency, cost_usd: cost, error: null, created_at: iso(47) }
}

export function mockDecisions(decider: string): DecisionRecord[] {
  const cost = decider === 'rules' ? 0 : decider === 'jev' ? 0.00004 : 0.0011
  const lat = (ms: number) => (decider === 'rules' ? Math.round(ms / 60) : ms)
  return [
    decision(
      'd1', decider, 'triage', null,
      { probe: '1080p, 29.97 fps, 95 s, no audio track issues', scene_cuts: 'none', rim: 'one, steady', court_lines: 'visible near the basket', caption: 'A pickup game in a gym filmed from a tripod at half court.' },
      {
        footage_type: { kind: 'choice', instructions: 'What kind of basketball footage is this?', criteria: { broadcast: null, fixed_camera: null, handheld: null, highlights: null, practice: null, not_basketball: null } },
        court_lines_visible: { kind: 'noul', instructions: 'Court lines near the basket are clearly visible.' },
      },
      {
        footage_type: { kind: 'choice', value: 'fixed_camera', probabilities: { fixed_camera: 0.86, handheld: 0.07, practice: 0.04, broadcast: 0.02, highlights: 0.01, not_basketball: 0 }, confidence: 0.86 },
        court_lines_visible: { kind: 'noul', value: 0.91, probabilities: null, confidence: null },
      },
      'accept', lat(212), cost,
    ),
    decision(
      'd2', decider, 'event_typing', 'e1',
      { ball_path: 'rose above rim, then dropped through rim region', rim_visible: 'throughout', ball_in_basket_class: '3 consecutive frames', scene_cut_nearby: 'no' },
      {
        is_shot_attempt: { kind: 'noul', instructions: 'A player attempted a shot at the basket.' },
        outcome: { kind: 'choice', instructions: 'Did the shot go in?', criteria: { made: null, missed: null, unclear: null } },
        shot_kind: { kind: 'choice', instructions: 'What kind of shot was it?', criteria: { jumper: null, layup: null, dunk: null, free_throw: null, tip: null, unknown: null } },
      },
      {
        is_shot_attempt: { kind: 'noul', value: 0.98, probabilities: null, confidence: null },
        outcome: { kind: 'choice', value: 'made', probabilities: { made: 0.95, missed: 0.03, unclear: 0.02 }, confidence: 0.95 },
        shot_kind: { kind: 'choice', value: 'jumper', probabilities: { jumper: 0.83, layup: 0.09, tip: 0.04, unknown: 0.04, dunk: 0, free_throw: 0 }, confidence: 0.83 },
      },
      'accept', lat(188), cost,
    ),
    decision(
      'd3', decider, 'event_typing', 'e4',
      { ball_path: 'rose toward rim, then lost behind a defender', rim_visible: 'partly hidden', ball_in_basket_class: 'never seen', scene_cut_nearby: 'no' },
      {
        is_shot_attempt: { kind: 'noul', instructions: 'A player attempted a shot at the basket.' },
        outcome: { kind: 'choice', instructions: 'Did the shot go in?', criteria: { made: null, missed: null, unclear: null } },
      },
      {
        is_shot_attempt: { kind: 'noul', value: 0.71, probabilities: null, confidence: null },
        outcome: { kind: 'choice', value: 'made', probabilities: { made: 0.47, missed: 0.36, unclear: 0.17 }, confidence: 0.47 },
      },
      'escalate', lat(231), cost,
    ),
    decision(
      'd4', decider, 'event_typing', 'e8',
      { ball_path: 'rose above rim, deflected sideways off the rim', rim_visible: 'throughout', ball_in_basket_class: 'never seen', scene_cut_nearby: 'no' },
      {
        outcome: { kind: 'choice', instructions: 'Did the shot go in?', criteria: { made: null, missed: null, unclear: null } },
      },
      {
        outcome: { kind: 'choice', value: 'missed', probabilities: { missed: 0.78, made: 0.14, unclear: 0.08 }, confidence: 0.78 },
      },
      'escalate', lat(176), cost,
    ),
    decision(
      'd5', decider, 'claim_check', 'narrative:recap',
      "Claim: Dark tees' #12 hit three straight jumpers to keep it close. Event log: Dark tees made 1 jumper in the middle stretch.",
      { supported: { kind: 'noul', instructions: 'The event log supports this claim.' } },
      { supported: { kind: 'noul', value: 0.09, probabilities: null, confidence: null } },
      'reject', lat(164), cost,
    ),
  ]
}

export function mockVlmCalls(provider: string, model: string): VLMCallRecord[] {
  const local = provider === 'ollama'
  const price = (i: number, o: number) => (local ? 0 : Math.round(((i * 4 + o * 20) / 1e6) * 10000) / 10000)
  const rec = (id: string, purpose: VLMCallRecord['purpose'], n: number, i: number, o: number, ms: number): VLMCallRecord => ({
    call_id: id, provider, model, purpose, n_images: n, video_seconds: 0, input_tokens: i, output_tokens: o, cost_usd: price(i, o), latency_ms: local ? ms * 2.4 : ms, stop_reason: 'end_turn', raw_path: null, error: null, created_at: iso(46),
  })
  return [
    rec('v1', 'adjudicate', 14, 9812, 412, 6210),
    rec('v2', 'adjudicate', 12, 8904, 388, 5790),
    rec('v3', 'summary', 18, 14220, 1104, 11420),
    rec('v4', 'claim_check', 0, 1880, 96, 1340),
  ]
}

export function mockPipeline(req: { detector: string; tracker: string; decider: string; provider: string | null; model: string | null; court: boolean; render: boolean; preset: { key: string; accept: number; reject: number } }): StageConfig[] {
  const out: StageConfig[] = [
    { stage: 'ingest', impl: 'ffmpeg', version: '7.1.1', license: 'LGPL/GPL (Homebrew build)', params: { crf: 20, preset: 'veryfast' } },
    { stage: 'detect', impl: req.detector, version: 'rfdetr 1.11.2', license: 'Apache-2.0', params: { size: 'small', imgsz: 640, stride: 2 } },
    { stage: 'track', impl: req.tracker, version: 'trackers 2.6.0', license: 'Apache-2.0', params: {} },
    { stage: 'shots', impl: 'hoops.events.shots', version: '0.1.0', license: 'project', params: { cooldown_s: 1.5 } },
    { stage: 'decide', impl: req.decider, version: req.decider === 'jev' ? 'jev-latest' : null, license: req.decider === 'rules' ? 'project' : 'provider API terms', params: { threshold_preset: req.preset.key, accept: req.preset.accept, reject: req.preset.reject } },
  ]
  if (req.provider) {
    out.push({ stage: 'adjudicate', impl: req.provider, version: req.model, license: 'provider API terms', params: { effort: 'low' } })
    out.push({ stage: 'summarize', impl: req.provider, version: req.model, license: 'provider API terms', params: { effort: 'medium' } })
  }
  if (req.court) out.push({ stage: 'court', impl: 'homography:manual', version: 'supervision 0.30', license: 'MIT', params: { points: 4 } })
  if (req.render) out.push({ stage: 'render', impl: 'pyav:libx264', version: 'av 19.0', license: 'BSD-3 / GPL (x264)', params: { pix_fmt: 'yuv420p', faststart: true } })
  return out
}

// ------------------------------------------------------------------ runs

export interface MockRunOptions {
  run_id: string
  clip: ClipMeta
  minutesAgo: number
  decider: string
  provider: string | null
  model: string | null
  threshold: { key: string; accept: number; reject: number }
  outputs: RunView['outputs_requested']
  egress: string[]
  wall_s: number
}

export function buildRun(o: MockRunOptions): RunView {
  const vlmProducer = o.provider ? `${o.provider}:${o.model}` : null
  const events = buildEvents(o.threshold.accept, o.threshold.reject, { decider: o.decider, vlmProducer })
  const wantsSummary = o.outputs.includes('summary') && !!o.provider
  const wantsCourt = o.outputs.includes('court_map')
  const wantsVideo = o.outputs.includes('annotated_video')
  const vlmCalls = o.provider && o.model ? mockVlmCalls(o.provider, o.model).filter((c) => wantsSummary || c.purpose === 'adjudicate') : []
  const decisions = mockDecisions(o.decider)
  const decisionsCost = decisions.reduce((s, d) => s + (d.cost_usd ?? 0), 0)
  const cost = vlmCalls.reduce((s, c) => s + (c.cost_usd ?? 0), 0) + decisionsCost
  const base = `/media/runs/${o.run_id}`
  const needsReview = events.filter((e) => e.status === 'needs_review').length
  return {
    schema_version: '0.1.0',
    run_id: o.run_id,
    created_at: iso(o.minutesAgo),
    clip: o.clip,
    outputs_requested: o.outputs,
    pipeline: mockPipeline({ detector: 'rfdetr:coco', tracker: 'bytetrack', decider: o.decider, provider: o.provider, model: o.model, court: wantsCourt, render: wantsVideo, preset: o.threshold }),
    data_egress: o.egress,
    teams: mockTeams,
    players: [],
    detections_path: `runs/${o.run_id}/detections.parquet`,
    annotated_video_path: wantsVideo ? `runs/${o.run_id}/annotated.mp4` : null,
    captions_path: `runs/${o.run_id}/events.vtt`,
    court_map_paths: wantsCourt ? [`runs/${o.run_id}/charts/shot_chart.png`] : [],
    tracks: [],
    events,
    team_summaries: teamSummaries(events, mockTeams),
    player_summaries: [],
    narratives: wantsSummary ? [mockNarrative(vlmProducer ?? 'none')] : [],
    decisions,
    vlm_calls: vlmCalls,
    metrics: { wall_s: o.wall_s, detect_fps: 41.8, tokens_in: vlmCalls.reduce((s, c) => s + (c.input_tokens ?? 0), 0), tokens_out: vlmCalls.reduce((s, c) => s + (c.output_tokens ?? 0), 0), cost_usd: Math.round(cost * 10000) / 10000, cost_estimate_usd: Math.round(cost * 1.15 * 10000) / 10000 },
    warnings: [
      'COCO weights have no rim class, so the rim you marked was used for every frame.',
      ...(needsReview ? [`${needsReview} close call${needsReview === 1 ? '' : 's'} need${needsReview === 1 ? 's' : ''} your review.`] : []),
    ],
    status: 'done',
    error: null,
    urls: {
      annotated: wantsVideo ? `mock://${base}/annotated.mp4` : null,
      master: `mock:///media/clips/${o.clip.clip_id}/master.mp4`,
      proxy: `mock:///media/clips/${o.clip.clip_id}/proxy_720p.mp4`,
      captions: null,
      court_maps: [],
      download: `/api/runs/${o.run_id}/download`,
    },
  }
}

export const MOCK_RUN_ID = '20261010-141522-9f1c'
export const MOCK_RUN_B_ID = '20261010-143907-2be4'
