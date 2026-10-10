/**
 * TypeScript mirror of hoops/schemas.py (serialized with model_dump(mode="json")) and the HTTP
 * shapes in docs/api.md. Datetimes arrive as ISO strings. Keep these in sync with the Python side.
 */

// ------------------------------------------------------------------ enums / literals (schemas.py)

export type EventType =
  | 'shot_attempt'
  | 'made_shot'
  | 'missed_shot'
  | 'three_pointer'
  | 'free_throw'
  | 'rebound'
  | 'assist'
  | 'turnover'
  | 'steal'
  | 'block'
  | 'foul'
  | 'possession_change'
  | 'timeout'
  | 'other'

export type FootageType =
  | 'broadcast'
  | 'fixed_camera'
  | 'handheld'
  | 'highlights'
  | 'practice'
  | 'not_basketball'
  | 'unknown'

export type EventStatus = 'provisional' | 'confirmed' | 'rejected' | 'needs_review'
export type Source = 'cv' | 'rule' | 'system_one' | 'vlm' | 'human' | 'import'
export type ShotKind = 'jumper' | 'layup' | 'dunk' | 'hook' | 'tip' | 'free_throw' | 'unknown'
export type OutputKind = 'annotated_video' | 'events' | 'summary' | 'court_map'
export type ConfidenceKind = 'model_score' | 'calibrated' | 'self_reported' | 'heuristic' | 'consensus' | 'human'
export type ReviewState = 'unreviewed' | 'accepted' | 'rejected' | 'edited'
export type Effort = 'low' | 'medium' | 'high' | 'xhigh' | 'max'
export type RunStatus = 'running' | 'done' | 'failed' | 'cancelled'

// ------------------------------------------------------------------ canonical models

export interface Provenance {
  source: Source
  producer: string
  stage: string
  prompt_id: string | null
  params: Record<string, unknown>
  created_at: string
}

export interface Evidence {
  t_s: number
  frame: number | null
  image_path: string | null
  tile_id: string | null
  note: string | null
}

export interface VideoAsset {
  path: string
  width: number
  height: number
  fps: string
  fps_float: number
  n_frames: number
  duration_s: number
  codec: string
  pix_fmt: string
  has_audio: boolean
}

export interface NormalizationReport {
  rotation_deg: number
  hdr: 'none' | 'hlg' | 'pq' | 'dolby_vision'
  tonemapped: boolean
  deinterlaced: boolean
  vfr_source: boolean
  sar_fixed: boolean
  fps_snapped_from: string | null
  source_resolution: [number, number] | null
  ffmpeg_version: string | null
  warnings: string[]
  summary: string
}

export interface ClipMeta {
  clip_id: string
  original_filename: string | null
  source_size_bytes: number
  master: VideoAsset
  proxy: VideoAsset
  normalization: NormalizationReport
  footage_type: FootageType
  footage_type_source: Source | null
  scene_cuts_s: number[]
  created_at: string
}

export interface TeamRef {
  team_id: string
  name: string | null
  color_hex: string | null
}

export interface PlayerRef {
  player_id: string
  team_id: string | null
  jersey_number: string | null
  name: string | null
  track_ids: number[]
}

export interface Track {
  track_id: number
  cls: string
  first_t_s: number
  last_t_s: number
  n_frames: number
  mean_score: number
  team_id: string | null
  player_id: string | null
}

export interface ShotInfo {
  points: 1 | 2 | 3 | null
  result: 'made' | 'missed' | 'unknown'
  kind: ShotKind
  court_x_ft: number | null
  court_y_ft: number | null
  image_x: number | null
  image_y: number | null
}

/** schemas.Event (named HoopsEvent to avoid clashing with the DOM Event type). */
export interface HoopsEvent {
  event_id: string
  type: EventType
  t_start_s: number
  t_end_s: number | null
  t_key_s: number | null
  team_id: string | null
  player_ids: string[]
  related_event_ids: string[]
  shot: ShotInfo | null
  is_replay: boolean
  confidence: number
  confidence_kind: ConfidenceKind
  status: EventStatus
  review: ReviewState
  description: string | null
  features: Record<string, unknown>
  evidence: Evidence[]
  provenance: Provenance
}

export interface StatLine {
  pts: number
  fga: number
  fgm: number
  tpa: number
  tpm: number
  fta: number
  ftm: number
  reb: number
  ast: number
  stl: number
  blk: number
  tov: number
  pf: number
}

export interface TeamSummary {
  team: TeamRef
  stats: StatLine
  derived_from_event_ids: string[]
}

export interface PlayerSummary {
  player: PlayerRef
  stats: StatLine
  on_screen_s: number | null
  derived_from_event_ids: string[]
}

export interface Narrative {
  kind: 'recap' | 'play_by_play' | 'coaching' | 'qa' | 'custom'
  text: string
  question: string | null
  cited_event_ids: string[]
  cited_times_s: number[]
  unsupported_claims: string[]
  provenance: Provenance
}

/** One answer inside DecisionRecord.answers (hoops.decide.base.Answer.spec()). */
export interface DecisionAnswer {
  kind: 'choice' | 'score' | 'noul'
  value: string | number
  probabilities: Record<string, number> | null
  confidence: number | null
}

/** One question inside DecisionRecord.questions (hoops.decide.base.question_spec()). */
export interface QuestionSpec {
  kind: 'choice' | 'score' | 'noul'
  instructions: string
  criteria?: Record<string, string | null> | string[]
}

export interface DecisionRecord {
  decision_id: string
  decider: string
  purpose: 'triage' | 'event_typing' | 'escalation' | 'qa_route' | 'claim_check'
  subject_id: string | null
  state: Record<string, unknown> | string
  questions: Record<string, QuestionSpec>
  answers: Record<string, DecisionAnswer>
  routed: 'accept' | 'reject' | 'escalate' | null
  latency_ms: number | null
  cost_usd: number | null
  error: string | null
  created_at: string
}

export interface VLMCallRecord {
  call_id: string
  provider: string
  model: string
  purpose: 'summary' | 'adjudicate' | 'qa' | 'caption' | 'triage' | 'claim_check' | 'custom'
  n_images: number
  video_seconds: number
  input_tokens: number | null
  output_tokens: number | null
  cost_usd: number | null
  latency_ms: number | null
  stop_reason: string | null
  raw_path: string | null
  error: string | null
  created_at: string
}

export interface StageConfig {
  stage: string
  impl: string
  version: string | null
  license: string | null
  params: Record<string, unknown>
}

export interface RunMetrics {
  wall_s: number
  detect_fps: number | null
  tokens_in: number
  tokens_out: number
  cost_usd: number
  cost_estimate_usd: number | null
}

export interface AnalysisRun {
  schema_version: string
  run_id: string
  created_at: string
  clip: ClipMeta
  outputs_requested: OutputKind[]
  pipeline: StageConfig[]
  data_egress: string[]
  teams: TeamRef[]
  players: PlayerRef[]
  detections_path: string | null
  annotated_video_path: string | null
  captions_path: string | null
  court_map_paths: string[]
  tracks: Track[]
  events: HoopsEvent[]
  team_summaries: TeamSummary[]
  player_summaries: PlayerSummary[]
  narratives: Narrative[]
  decisions: DecisionRecord[]
  vlm_calls: VLMCallRecord[]
  metrics: RunMetrics
  warnings: string[]
  status: RunStatus
  error: string | null
}

// ------------------------------------------------------------------ HTTP shapes (docs/api.md)

export interface Health {
  ok: boolean
  version: string
}

export interface Session {
  authenticated: boolean
  auth_required: boolean
  user: string | null
}

export interface ProviderInfo {
  key: string
  label: string
  available: boolean
  reason: string | null
  native_video: boolean
  audio: boolean
  local: boolean
  sends_data_to: string
  default_model: string
  models: string[]
}

export interface RegistryItem {
  key: string
  label: string
  available: boolean
  reason: string | null
  license: string
  notes: string
  supports_images?: boolean
}

export interface ThresholdPreset {
  key: string
  label: string
  accept: number
  reject: number
  note: string
}

export interface KeyLabel<K extends string = string> {
  key: K
  label: string
  description?: string
}

export interface Caps {
  run_cost_cap_usd: number
  session_cost_cap_usd: number
  max_upload_mb: number
  max_clip_seconds: number
}

/** /api/config courts[]: landmark names are the only valid court_points[].name values. */
export interface CourtLandmark {
  name: string
  label: string
  /** feet from center court, toward the filmed basket */
  x_ft: number
  /** feet from center court, across; +y is camera-left (camera at half court facing the filmed basket) */
  y_ft: number
  filmed_half: boolean
}

export interface CourtSpec {
  key: string
  label: string
  length_ft: number
  width_ft: number
  landmarks: CourtLandmark[]
}

export interface AppConfig {
  version: string
  providers: ProviderInfo[]
  detectors: RegistryItem[]
  trackers: RegistryItem[]
  deciders: RegistryItem[]
  threshold_presets: ThresholdPreset[]
  prompt_presets: KeyLabel[]
  outputs: KeyLabel<OutputKind>[]
  footage_types: FootageType[]
  /** Added after v0; optional so older servers still work. */
  courts?: CourtSpec[]
  caps: Caps
  spend: { session_usd: number }
}

export interface UploadInit {
  upload_id: string
  chunk_size_bytes: number
}

export type JobStatus = 'queued' | 'running' | 'done' | 'failed' | 'cancelled'
/** Server sends pending / running / done / skipped / failed; 'cancelled' is UI-only (job cancelled mid-stage). */
export type StageStatus = 'pending' | 'running' | 'done' | 'skipped' | 'failed' | 'cancelled'

export interface JobStage {
  key: string
  label: string
  status: StageStatus
  progress: number
}

export interface Job {
  job_id: string
  kind: 'ingest' | 'analyze'
  status: JobStatus
  progress: number
  stage: string | null
  message: string | null
  clip_id: string | null
  run_id: string | null
  error: string | null
  started_at: string | null
  finished_at: string | null
  stages: JobStage[]
}

export interface ClipCard {
  clip_id: string
  original_filename: string | null
  duration_s: number
  footage_type: FootageType
  poster_url: string
  created_at: string
}

/** GET /api/clips/{id}/rim-suggestions: likely rims (proxy pixels), best first. */
export interface RimSuggestion {
  box: [number, number, number, number]
  score: number
  t_s: number
  /** The frame the box was found on (the poster is a different moment). Older servers omit it. */
  frame_url?: string | null
}

export interface ClipUrls {
  master: string
  proxy: string
  poster: string
}

export interface ClipView extends ClipMeta {
  urls: ClipUrls
}

export interface VlmChoice {
  provider: string
  model: string
  effort: Effort
  native_video: boolean
}

export interface CourtPoint {
  name: string
  x: number
  y: number
}

export interface RunRequest {
  clip_id: string
  outputs: OutputKind[]
  detector: string
  tracker: string
  decider: string
  vlm: VlmChoice | null
  prompt_preset: string
  custom_prompt: string | null
  structured: boolean
  threshold_preset: string
  egress_ok: string[]
  footage_type: FootageType | null
  rim_box: [number, number, number, number] | null
  court_points: CourtPoint[] | null
  max_cost_usd: number
}

export interface EstimateLine {
  stage: string
  provider: string | null
  model: string | null
  usd: number
  note: string | null
}

export interface Estimate {
  estimate_usd: number
  breakdown: EstimateLine[]
  warnings: string[]
}

export interface CreateRunResponse {
  run_id: string
  job_id: string
  estimate_usd: number
}

/** hoops/store.py RunSummary */
export interface RunSummary {
  run_id: string
  created_at: string
  status: string
  clip_id: string
  filename: string | null
  footage_type: string
  outputs: string[]
  n_events: number
  cost_usd: number
  providers: string[]
}

export interface RunUrls {
  /** Present on the real server (always /media/clips/<id>/poster.jpg); not in docs/api.md v0. */
  poster?: string
  annotated: string | null
  master: string
  proxy: string
  captions: string | null
  court_maps: string[]
  download: string
}

export interface RunView extends AnalysisRun {
  urls: RunUrls
}

export interface RecomputeResponse {
  events: HoopsEvent[]
  team_summaries: TeamSummary[]
}

export interface ReviewRequest {
  action: 'accept' | 'reject'
  note: string | null
}

export type AddableEventType = 'made_shot' | 'missed_shot' | 'shot_attempt'

export interface AddEventRequest {
  t_s: number
  type: AddableEventType
  team_id: string | null
}

export interface ChatTurn {
  role: 'user' | 'assistant'
  text: string
}

export interface AskRequest {
  question: string
  provider: string | null
  model: string | null
  history: ChatTurn[]
}

export interface Citation {
  t_s: number
  event_id: string | null
  label: string
}

export type AnswerRoute = 'stats' | 'find_moments' | 'explain_play' | 'summary' | 'unsupported'

export interface Answer {
  answer: string
  route: AnswerRoute
  citations: Citation[]
  cost_usd: number
  provider: string | null
  model: string | null
  latency_ms: number
  decision: {
    decider: string
    route_confidence: number
    model?: string | null
    probabilities?: Record<string, number> | null
    cost_usd?: number | null
    latency_ms?: number | null
    error?: string
    fallback?: string
  } | null
}

export interface Agreement {
  matched: number
  only_a: number
  only_b: number
  outcome_agreement: number | null
}

export interface CompareResponse {
  a: RunView
  b: RunView
  agreement: Agreement
}

// ------------------------------------------------------------------ constants from the contract

export const STAGES: { key: string; label: string }[] = [
  { key: 'ingest', label: 'Preparing video' },
  { key: 'detect', label: 'Finding players & ball' },
  { key: 'track', label: 'Tracking' },
  { key: 'shots', label: 'Spotting shots' },
  { key: 'decide', label: 'System One decisions' },
  { key: 'adjudicate', label: 'AI review of close calls' },
  { key: 'summarize', label: 'AI summary' },
  { key: 'court', label: 'Court map' },
  { key: 'render', label: 'Annotated video' },
]

export const TERMINAL_JOB: JobStatus[] = ['done', 'failed', 'cancelled']
