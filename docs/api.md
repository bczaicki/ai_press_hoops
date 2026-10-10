# HTTP API contract (v0)

This is the contract between the FastAPI backend (`hoops/server.py`) and the web UI (`web/`).
- All bodies are JSON unless noted. Times are seconds (`t_s`) on the clip's master timeline.
- Pixel coordinates are in **proxy** pixels (720p).
- Shapes in `snake_case` mirror the Pydantic models in `hoops/schemas.py`. `Event`, `ClipMeta`, `AnalysisRun`, and so on are those models serialized with `model_dump(mode="json")`.

## Auth
If `HOOPS_APP_PASSWORD` is set, every `/api/*` and `/media/*` route requires the `hoops_session` cookie. It is a first-party HttpOnly cookie, so it works in Safari. The exceptions are `/api/health`, `/api/session`, and `/api/login`.

| Method | Path | Body | Response |
|---|---|---|---|
| GET | `/api/health` | – | `{"ok": true, "version": "0.1.0"}` |
| GET | `/api/session` | – | `{"authenticated": bool, "auth_required": bool, "user": str \| null}` |
| POST | `/api/login` | `{"password": str}` | `200 {"ok": true}` + cookie, or `401 {"detail": "Wrong password"}` |
| POST | `/api/logout` | – | `{"ok": true}` |

## Config (drives every picker in the UI)
`GET /api/config`
```json
{
  "version": "0.1.0",
  "providers": [
    {"key": "claude", "label": "Claude (Anthropic)", "available": true, "reason": null,
     "native_video": false, "audio": false, "local": false, "sends_data_to": "Anthropic",
     "default_model": "claude-opus-5-5", "models": ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]}
  ],
  "detectors": [{"key": "rfdetr:coco", "label": "RF-DETR (COCO)", "available": true, "reason": null, "license": "Apache-2.0", "notes": ""}],
  "trackers":  [{"key": "bytetrack", "label": "ByteTrack (fixed camera)", "available": true, "reason": null, "license": "Apache-2.0", "notes": ""}],
  "deciders":  [{"key": "rules", "label": "Rules (baseline)", "available": true, "reason": null, "license": "", "notes": "", "supports_images": false}],
  "threshold_presets": [{"key": "strict", "label": "Strict", "accept": 0.92, "reject": 0.30, "note": "fewer phantom makes"},
                        {"key": "balanced", "label": "Balanced", "accept": 0.85, "reject": 0.15, "note": ""},
                        {"key": "eager", "label": "Eager", "accept": 0.70, "reject": 0.05, "note": "miss fewer shots"}],
  "prompt_presets": [{"key": "recap", "label": "Recap"}, {"key": "play_by_play", "label": "Play-by-play"},
                     {"key": "coach", "label": "Coach's notes"}, {"key": "custom", "label": "Custom"}],
  "outputs": [{"key": "annotated_video", "label": "Annotated video"}, {"key": "events", "label": "Event log + stats"},
              {"key": "summary", "label": "AI summary & Q&A"}, {"key": "court_map", "label": "Court map / heatmap"}],
  "footage_types": ["unknown", "broadcast", "fixed_camera", "handheld", "highlights", "practice"],
  "courts": [{"key": "nba", "label": "NBA", "length_ft": 94, "width_ft": 50,
              "landmarks": [{"name": "baseline_left_corner", "label": "Baseline left corner", "x_ft": 47.0, "y_ft": 25.0,
                             "filmed_half": true}]}],
  "caps": {"run_cost_cap_usd": 0.5, "session_cost_cap_usd": 5.0, "max_upload_mb": 500, "max_clip_seconds": 600},
  "spend": {"session_usd": 0.0}
}
```
Unavailable items still appear, with `available: false` and a human `reason` such as "missing GEMINI_API_KEY in .env". Show them disabled, with the reason as a tooltip.

`courts[].landmarks` are the only valid `court_points[].name` values. Coordinates are in feet with the origin at center court; `x` runs along the length toward the filmed basket and `y` across the width. "Left" means camera-left for a camera at half court facing the filmed basket. Landmarks on the other half are prefixed `other_`.

## Uploads (chunked, so tunnels with 100 MB body caps still work)
| Method | Path | Body | Response |
|---|---|---|---|
| POST | `/api/uploads` | `{"filename": str, "size_bytes": int, "content_type": str}` | `{"upload_id": str, "chunk_size_bytes": 8388608}` |
| PUT | `/api/uploads/{upload_id}/chunks/{index}` | raw bytes (`application/octet-stream`) | `{"received_bytes": int}` |
| POST | `/api/uploads/{upload_id}/complete` | – | `{"job_id": str}` (an ingest job; its result carries `clip_id`) |

## Jobs (long-running work)
`GET /api/jobs/{job_id}` returns a `Job`:
```json
{
  "job_id": "j_ab12", "kind": "ingest" | "analyze", "status": "queued" | "running" | "done" | "failed" | "cancelled",
  "progress": 0.42, "stage": "detect", "message": "Finding players & ball (312/900 frames)",
  "clip_id": "c3f9...", "run_id": "20261010-142233-9f1c" | null, "error": null,
  "started_at": "2026-10-10T14:22:33Z", "finished_at": null,
  "stages": [{"key": "ingest", "label": "Preparing video", "status": "done", "progress": 1.0},
             {"key": "detect", "label": "Finding players & ball", "status": "running", "progress": 0.35}]
}
```
- **Stage keys, in order:**

  | Key | Label |
  |---|---|
  | `ingest` | "Preparing video" |
  | `detect` | "Finding players & ball" |
  | `track` | "Tracking" |
  | `shots` | "Spotting shots" |
  | `decide` | "System One decisions" |
  | `adjudicate` | "AI review of close calls" |
  | `summarize` | "AI summary" |
  | `court` | "Court map" |
  | `render` | "Annotated video" |

  Stages a run doesn't need report `skipped`.
- `GET /api/jobs/{job_id}/stream` is Server-Sent Events. Each message is `event: job`, with `data` being the `Job` JSON; the stream ends at a terminal status. **Fall back to polling every 1 s**, because some tunnels drop SSE.
- `POST /api/jobs/{job_id}/cancel` → `Job`.

## Clips
| Method | Path | Body | Response |
|---|---|---|---|
| GET | `/api/clips` | – | `[ClipCard]`, newest first |
| GET | `/api/clips/{clip_id}` | – | `ClipView` |
| PATCH | `/api/clips/{clip_id}` | `{"footage_type": str}` | `ClipView` |

- `ClipCard = {"clip_id", "original_filename", "duration_s", "footage_type", "poster_url", "created_at"}`
- `ClipView = ClipMeta & {"urls": {"master": "/media/clips/<id>/master.mp4", "proxy": "/media/clips/<id>/proxy_720p.mp4", "poster": "/media/clips/<id>/poster.jpg"}}`

`GET /api/clips/{clip_id}/rim-suggestions` → `[{"box": [x1, y1, x2, y2], "score": float, "t_s": float, "frame_url": str}]`
- Up to 4 likely rims in proxy pixels, best first, found zero-shot (OWLv2) on frames at 10% and 50% of the clip.
- Show each box on its own `frame_url` (the proxy frame at `t_s`), not the poster: the poster is a different moment, so on a moving camera the boxes would not line up. The tapped box becomes `rim_box` ("Tap the hoop being played at").
- An empty list means no hoop is visible.
- The first call takes a few seconds (model load); results are then cached per clip.
- When a run has no `rim_box` and the detector finds no rim, the pipeline picks the suggested hoop the ball approaches and adds a run warning.

## Runs
`RunRequest`:
```json
{
  "clip_id": "c3f9...",
  "outputs": ["annotated_video", "events", "summary", "court_map"],
  "detector": "rfdetr:coco", "tracker": "bytetrack", "decider": "rules",
  "vlm": {"provider": "claude", "model": "claude-opus-5-5", "effort": "medium", "native_video": false},
  "prompt_preset": "recap", "custom_prompt": null, "structured": true,
  "threshold_preset": "balanced",
  "egress_ok": ["claude"],
  "footage_type": null,
  "rim_box": null,
  "court_points": null,
  "max_cost_usd": 0.5
}
```
Field notes:
- `vlm` may be `null` (no AI).
- `egress_ok` lists the providers the user allowed to receive frames. A run that needs a provider that isn't listed fails fast.
- `footage_type`: `null` means auto (System One triage).
- `rim_box`: `null` or `[x1, y1, x2, y2]` in proxy pixels (the "click the rim" step).
- `court_points`: `null` or `[{"name": "baseline_left_corner", "x": 102, "y": 540}, ...]`, at least 4 points (fixed-camera calibration).

| Method | Path | Body | Response |
|---|---|---|---|
| POST | `/api/estimate` | `RunRequest` | `{"estimate_usd": float, "breakdown": [{"stage", "provider", "model", "usd", "note"}], "warnings": [str]}` |
| POST | `/api/runs` | `RunRequest` | `{"run_id": str, "job_id": str, "estimate_usd": float}` |
| GET | `/api/runs` | – | `[RunSummary]` (see `hoops/store.py`) |
| GET | `/api/runs/{run_id}` | – | `RunView` |
| GET | `/api/runs/{run_id}/download` | – | `application/zip` |
| POST | `/api/runs/{run_id}/recompute` | `{"threshold_preset": str}` | `{"events": [Event], "team_summaries": [TeamSummary]}` (no re-inference) |
| POST | `/api/runs/{run_id}/events/{event_id}/review` | `{"action": "accept" \| "reject", "note": str \| null}` | `Event` |
| POST | `/api/runs/{run_id}/events` | `{"t_s": float, "type": "made_shot" \| "missed_shot" \| "shot_attempt", "team_id": str \| null}` | `Event` (`source: "human"`) |
| POST | `/api/runs/{run_id}/ask` | `{"question": str, "provider": str \| null, "model": str \| null, "history": [{"role": "user" \| "assistant", "text": str}]}` | `Answer` |

`RunView` is the `AnalysisRun` plus:
```json
{"urls": {"annotated": "/media/runs/<id>/annotated.mp4" | null, "master": "...", "proxy": "...",
          "captions": "/media/runs/<id>/events.vtt" | null, "court_maps": ["/media/runs/<id>/charts/shot_chart.png"],
          "poster": "/media/clips/<clip_id>/poster.jpg", "download": "/api/runs/<id>/download"},
 "job_id": "j_ab12" | null}
```
- `job_id` is the live job for this run while it is still queued or running.
- `status` becomes `failed` or `cancelled` (with `error`) when a run doesn't finish. Runs interrupted by a server restart are marked failed at startup.
- Captions are regenerated after review, add-event and recompute.
- AI summaries keep their structured form in `narratives[].provenance.params.summary`, and `params.key_moments` lists `{ref, description, t_s}`.
- `GET /api/config` also returns `"defaults": {"detector", "tracker", "decider", "provider"}`, where provider is the first working cloud provider, else local.

`Answer`:
```json
{"answer": "markdown text", "route": "stats" | "find_moments" | "explain_play" | "summary" | "unsupported",
 "citations": [{"t_s": 12.4, "event_id": "e3" | null, "label": "Made jumper, 0:12"}],
 "cost_usd": 0.004, "provider": "claude" | null, "model": "claude-opus-5-5" | null, "latency_ms": 2300,
 "decision": {"decider": "jev", "route_confidence": 0.93} | null}
```

## Compare
`GET /api/compare?a=<run_id>&b=<run_id>` → `{"a": RunView, "b": RunView, "agreement": {"matched": int, "only_a": int, "only_b": int, "outcome_agreement": float | null}}`. Events match when they are within ±1 s and have the same type family.

## Media
`GET /media/{path}` serves files under the data dir, with HTTP Range support (needed for seeking and for Safari). Only paths under `clips/` and `runs/` are allowed.

## Errors
Errors are JSON: `{"detail": str}`. Statuses:
- 400: bad input
- 401: auth
- 402: cost cap would be exceeded (`detail` explains the cap)
- 404: not found
- 409: wrong state (e.g. the clip is not ingested yet)
- 503: the provider is unavailable
