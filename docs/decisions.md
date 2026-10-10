# Decisions

We sort decisions into two kinds:
- **One-way doors** are hard or impossible to undo. Each gets a short record below (Context → Decision → Consequences).
- **Two-way doors** are everything else. They live in config: decide fast, change freely.

Status for all five records: **accepted** (2026-10-10). Revisit them before the app moves to a public host or gets new users.

## ADR-1: Data egress to third-party AI providers
**Context.** Frames sent to an API can't be recalled, and each provider has different rules:
- **Anthropic:** API data is deleted within 30 days, with some exceptions.
- **OpenAI:** abuse-monitoring logs are kept for up to 30 days, and API data isn't used for training by default.
- **Gemini free tier:** inputs may be used to improve Google's products and may be read by human reviewers.

**Decision.**
- Nothing leaves the Mac unless a run names each receiving provider and the user ticks it (`egress_ok`). The run records that list (`AnalysisRun.data_egress`).
- Send the fewest frames that do the job: candidate windows and contact sheets, not whole games, unless the user picks native video.
- Use Gemini on a paid key only.
- A local-only mode (CV + Ollama) is always available.
- Keys stay on the server. Users pick from fixed provider profiles and can never type a base URL.

**Consequences.** Running a clip takes an extra click. Some providers' features, such as Gemini's whole-video input, are only used when the user opts in. In return, we always know what went where.

## ADR-2: Footage rights and minors
**Context.**
- YouTube's terms forbid downloading without permission, and NBA content can't be redistributed.
- Youth games show minors. Under the 2025 COPPA rule, biometric identifiers count as personal information, and sharing children's data with third parties needs parental consent.
- Gemini's API terms forbid apps likely to be used by under-18s.

**Decision.**
- Only footage the user has rights to.
- No URL ingestion in the POC (no yt-dlp).
- No face recognition, ever. Player identity comes only from jersey numbers plus a roster the user types in.
- Youth footage goes to cloud providers only with a guardian's OK; otherwise use local-only mode.
- Exported or annotated broadcast footage is never published.

**Consequences.** No "paste a YouTube link" feature. Player names need a manual roster step.

## ADR-3: Detector and weights licensing
**Context.**
- RF-DETR Nano–Large, supervision and trackers are Apache/MIT.
- RF-DETR XL/2XL are under PML 1.0, which needs a Roboflow plan.
- Ultralytics code and every model trained with it are AGPL-3.0, which requires releasing the source of the whole networked application.
- Roboflow Universe weights carry per-project licenses that we couldn't verify.

**Decision.**
- The default stack is Apache/MIT only.
- Ultralytics (YOLOE/YOLO26-pose) ships only as the opt-in `[ultra]` extra, for local experiments. It is never part of a hosted or commercial build without an Ultralytics license.
- No RF-DETR XL/2XL.
- Check each Universe model's license before hosting.
- Every run records each stage's implementation and license (`StageConfig.license`).

**Consequences.** We give up some prompt-based detection features by default. Hosting stays license-clean.

## ADR-4: Public hosting and visibility
**Context.**
- A public link to this app spends the owner's API credit and exposes uploaded footage.
- Gradio share links last about a week, and anyone holding one can reach the app.
- Hugging Face Docker/Gradio Spaces need a paid plan; ZeroGPU is the free exception.

**Decision.**
- For now, the app runs on the owner's Mac behind a password (`HOOPS_APP_PASSWORD`, a first-party cookie that works in Safari), shared by a share link or Tailscale Funnel.
- No unauthenticated public deployment.
- A later move to a private or protected HF Space or Modal keeps the password, the cost caps and server-side keys.

**Consequences.** The friend's access depends on the Mac being awake (`caffeinate`). See [sharing.md](sharing.md).

## ADR-5: Freezing output schema v1
**Context.**
- `run.json` (`AnalysisRun`, `SCHEMA_VERSION = "0.1.0"`) is what the UI, Compare, eval and any exports read.
- While only we read it, changing it is cheap.
- Once the friend's spreadsheets or a hosted API depend on it, changing it breaks them.

**Decision.**
- Breaking changes are allowed while the version is 0.x.
- Freeze it as **1.0.0** the first time anyone outside this repo depends on it. After that, changes are additive only, and old runs get a migration.
- Runs stay append-only and keep raw detections and scores, so thresholds can be re-applied without re-running models.

**Consequences.** A small amount of version bookkeeping. Old runs stay readable and comparable.

## Two-way doors (in config, change any time)
- Models and providers, plus their default model IDs and prices
- Prompts
- Thresholds and Strict/Balanced/Eager presets
- Detector preset
- Tracker: ByteTrack or BoT-SORT
- Decider: rules, adapter, Jev, or OpenAI Decisions
- Sampling fps and contact-sheet layout
- Proxy resolution
- Chart styling
- UI framework: we already switched from Gradio to a React SPA + FastAPI
- Tunnel or host choice
- File layout under `data/`
