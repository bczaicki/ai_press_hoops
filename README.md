# ai_press_hoops

Upload a basketball clip and get:
- an annotated video;
- a log of shots (made or missed) with simple stats;
- an AI-written summary you can ask questions about;
- a shot chart or heatmap.

It is a **proof of concept** that runs on a Mac and can be shared with a friend through a password-protected link.

**How it works.** Computer-vision models find the players, ball and rim in every frame. Plain code turns their tracks into shot events. A fast "System One" model makes quick typed calls on those events. Slower AI models (Claude, Gemini, OpenAI, OpenRouter, or a local Qwen model) only explain, double-check close calls, and answer questions. **Counts, times and positions always come from code, never from an AI's guess.** Vision-language models are bad at counting.

## Quickstart (macOS, Apple Silicon)
You need [uv](https://docs.astral.sh/uv/), Node 22 (for the web UI), and Homebrew `ffmpeg` (`brew install ffmpeg`, then `brew pin ffmpeg`).

```bash
make setup                    # = uv sync --extra cv --extra vlm --extra decide --extra court + npm install (web/)
cp .env.example .env          # add the API keys you have; every key is optional; set HOOPS_APP_PASSWORD
uv run hoops doctor           # checks ffmpeg, GPU, keys, and Ollama, and tells you how to fix each one
make demo                     # optional: analyze 3 CC BY sample clips so the app opens with real breakdowns
uv run hoops serve --share    # builds the UI on first run, prints a link; log in with HOOPS_APP_PASSWORD
```

- **Local development:** `make dev` runs the API (:8000) and the UI with hot reload (:5173).

- **Missing keys.** A missing key just greys out that provider in the app.
- **No cloud at all.** Install [Ollama](https://ollama.com) and run `ollama pull qwen3.5:4b`, the recommended model. Any installed vision model also works: the app picks one automatically, for example `gemma4`, or pin one with `HOOPS_OLLAMA_MODEL`. That gives you a local-only mode where no footage leaves the Mac.
- **Out of credits?** Providers whose account has no credit show as unavailable, with a link to fix billing. The local mode keeps working.
- **Basketball-trained detector (optional).** Run `uv sync --extra roboflow` and set `ROBOFLOW_API_KEY` (free) to unlock a detector trained on NBA footage. It knows rims, referees and shot classes. The default COCO detector finds people and the ball, and you mark the rim once.
- **Before sharing.** Read [docs/sharing.md](docs/sharing.md).
- **Before filming.** Read [docs/filming-guide.md](docs/filming-guide.md). How a game is filmed matters more than which AI model you pick.

## What you get
| Output | What it is | Where it comes from |
|---|---|---|
| **Prepared video** | Your upload, converted so every browser plays it upright, with normal colors and a steady frame rate. A one-line note says what changed, e.g. "Rotated 90°, HDR→SDR, 29.97 fps constant frame rate, 1080p". Location data from phones is removed. | ffmpeg |
| **Annotated video** | Players, ball and rim marked on the video. | Detector + tracker |
| **Event log + stats** | Each shot with its time, made/missed, and a status. *Provisional* means the code's call. *Confirmed* means a second check agreed. *Needs review* means it's a close call: tap 👍 or 👎. Click a row to jump to that moment. Stats (FGA/FGM/FG%) are computed from the log. | Code, plus System One and AI review of close calls |
| **AI summary & Q&A** | A short recap and a chat about the clip. "How many shots did white make?" is answered from the event log, not by the AI counting. | Your chosen AI provider |
| **Court map** | A shot chart in court feet, after you click 4+ court landmarks once per camera position. Without that calibration you get an **image-space** heatmap, which is labeled as such and is not to scale. | Code |

**Strict ↔ Eager.** This slider re-decides which shots count, without re-running anything.
- **Strict:** fewer phantom makes, but more go to review.
- **Eager:** misses fewer shots, but raises more false alarms.

## How accurate is it? (honest version)
Nothing has been measured on your footage yet. The notes below are expectations from published results and from similar tools. To get real numbers, label 3–5 clips and run `uv run hoops eval` (see [eval/README.md](eval/README.md)).

| Footage | Expect | Why |
|---|---|---|
| **Tripod, raised, half court, rim visible** | Best case. Shot attempts and makes should be mostly right. The shot chart works after calibration. | Fixed view, rim always in frame, no cuts. |
| **TV broadcast** | Players and ball are found well, especially with the optional basketball detector, which was trained on NBA broadcasts. Expect some missed shots around camera cuts and zooms. Replays could double-count, so they are flagged. | Cuts, replays, zooms. The court map works only when the court lines are clearly visible. |
| **Phone from the stands (handheld)** | Rough. Expect missed shots and some false alarms. The heatmap is image-space only. | The camera moves and zooms, and the rim leaves the frame. |
| **Highlight reels** | The summary is fine, but don't trust the counts. | Constant cuts and replays. |
| **Shooting practice (one hoop)** | Good for made/missed if the rim stays in frame. | Simple scene. |

Small or blurry balls, dark gyms and people standing in front of the rim all cost accuracy. Detection models trained on NBA games do worse on high-school gyms and outdoor courts.

## Privacy and cost
- **What gets sent.** Each run lists the AI providers that will receive frames, and nothing is sent until you tick them. The run's record keeps that list.
- **Local mode.** CV + Ollama never leaves this Mac.
- **Gemini.** Use it with a paid key only. The free tier may use your uploads for training.
- **Minors.** Don't upload footage of minors to cloud providers without a guardian's OK. Use local mode instead.
- **Cost.** Each run shows a cost estimate first. Default caps are $0.50 per run and $5 per session. Also set spend limits in each provider's console.

## For developers
- `hoops/` is a UI-agnostic package. FastAPI + the React app in `web/` sit on top. See [docs/api.md](docs/api.md).
- Run the tests with `uv run pytest`. Paid-API tests need `HOOPS_LIVE=1`.
- The big, hard-to-reverse choices are written up in [docs/decisions.md](docs/decisions.md).
