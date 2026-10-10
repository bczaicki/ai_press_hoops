"""`hoops` command line: serve the app, check the setup, run analyses headless."""

from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from hoops.config import PROFILES, ROOT, get_settings

app = typer.Typer(
    add_completion=False, help="ai_press_hoops: basketball video analysis POC", no_args_is_help=True
)
console = Console()


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    share: Annotated[
        bool, typer.Option(help="Open a public *.gradio.live link (1 week). Needs a password.")
    ] = False,
    build_web: Annotated[bool, typer.Option(help="Build the web UI first if web/dist is missing.")] = True,
    reload: bool = False,
) -> None:
    """Run the web app (API + UI)."""
    import uvicorn

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    settings = get_settings()
    if build_web and not (ROOT / "web" / "dist" / "index.html").exists():
        _build_web()
    if share:
        if not settings.key("app_password"):
            console.print(
                "[red]Refusing to share without a password.[/] Set HOOPS_APP_PASSWORD in .env "
                f"(e.g. HOOPS_APP_PASSWORD={secrets.token_urlsafe(9)})."
            )
            raise typer.Exit(1)
        url = _open_share_tunnel(host, port)
        console.print(f"\n[bold green]Share link:[/] {url}  (valid ~1 week, while this Mac is awake)")
        console.print(
            f"Keep the Mac awake: [cyan]caffeinate -dimsu &[/]   Login user: [cyan]{settings.app_user}[/]\n"
        )
    console.print(f"Local: http://{host}:{port}")
    uvicorn.run(
        "hoops.server:create_app", factory=True, host=host, port=port, reload=reload, log_level="info"
    )


@app.command()
def doctor() -> None:
    """Check ffmpeg, MPS, keys, and optional pieces."""
    from hoops.doctor import run_doctor

    table = Table(title="hoops doctor")
    for col in ("check", "status", "detail", "fix"):
        table.add_column(col)
    worst = 0
    for c in run_doctor():
        status = "[green]ok[/]" if c.ok else ("[yellow]warn[/]" if c.severity != "error" else "[red]fail[/]")
        if not c.ok and c.severity == "error":
            worst = 1
        table.add_row(c.name, status, c.detail, c.fix or "")
    console.print(table)
    raise typer.Exit(worst)


@app.command()
def ingest(path: Path) -> None:
    """Normalize a clip (master + 720p proxy) and print its id."""
    from hoops.ingest import ingest as do_ingest

    clip = do_ingest(path, original_filename=path.name, progress=_progress_printer("ingest"))
    console.print(f"\nclip_id [bold]{clip.clip_id}[/]  ·  {clip.normalization.summary}")


@app.command()
def analyze(
    path: Path,
    outputs: Annotated[
        str, typer.Option(help="Comma list: annotated_video,events,summary,court_map")
    ] = "annotated_video,events,summary",
    detector: str = "rfdetr:coco",
    tracker: str = "bytetrack",
    decider: str = "rules",
    vlm: Annotated[
        str, typer.Option(help="Provider[:model], e.g. claude or claude:claude-sonnet-5-5; 'none'")
    ] = "none",
    rim: Annotated[str | None, typer.Option(help="Rim box in proxy px: x1,y1,x2,y2")] = None,
    preset: str = "balanced",
    footage: str | None = None,
) -> None:
    """Ingest + analyze a clip without the web UI. Sends frames only to the provider you name."""
    from hoops.ingest import ingest as do_ingest
    from hoops.pipeline import RunRequest, VLMChoice, run_analysis
    from hoops.store import Store

    clip = do_ingest(path, original_filename=path.name, progress=_progress_printer("ingest"))
    choice = None
    if vlm != "none":
        provider, _, model = vlm.partition(":")
        if provider not in PROFILES:
            raise typer.BadParameter(f"unknown provider {provider}; known: {', '.join(PROFILES)}")
        choice = VLMChoice(provider=provider, model=model or None)
    req = RunRequest(
        clip_id=clip.clip_id,
        outputs=[o.strip() for o in outputs.split(",") if o.strip()],  # type: ignore[misc]
        detector=detector,
        tracker=tracker,
        decider=decider,
        vlm=choice,
        threshold_preset=preset,
        egress_ok=[choice.provider] if choice else [],
        rim_box=[float(v) for v in rim.split(",")] if rim else None,
        footage_type=footage,  # type: ignore[arg-type]
    )
    run = run_analysis(req, None, Store())
    console.print(
        f"\nrun [bold]{run.run_id}[/]  ·  {len(run.events)} events  ·  ${run.metrics.cost_usd:.4f}  ·  "
        f"{run.metrics.wall_s:.1f}s"
    )
    for w in run.warnings:
        console.print(f"[yellow]warning:[/] {w}")
    for n in run.narratives:
        console.print(f"\n{n.text}")
    console.print(f"\nfiles: {Store().run_dir(run.run_id)}")


@app.command()
def bench(clip: Path | None = None, detectors: str = "rfdetr:coco") -> None:
    """Measure detector speed on this machine (640 and 1280 px)."""
    from hoops.bench import bench as do_bench

    rows = do_bench([d.strip() for d in detectors.split(",")], clip=clip)
    table = Table(title="detector speed")
    for col in rows[0].keys() if rows else ["result"]:
        table.add_column(str(col))
    for r in rows:
        table.add_row(*[f"{v:.1f}" if isinstance(v, float) else str(v) for v in r.values()])
    console.print(table)


@app.command()
def smoke(providers: str = "all", deciders: str = "rules") -> None:
    """Tiny end-to-end check of each configured AI provider and decider (costs a few cents)."""
    import tempfile

    from hoops import registry
    from hoops.reason.base import AnalyzeRequest, ImagePart

    registry.load_plugins()
    with tempfile.TemporaryDirectory() as d:
        frame = Path(d) / "f.jpg"
        subprocess.run(
            [
                get_settings().ffmpeg_bin,
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=640x360:duration=1",
                "-frames:v",
                "1",
                str(frame),
            ],
            check=True,
        )
        jpeg = frame.read_bytes()
    table = Table(title="smoke")
    for col in ("kind", "key", "ok", "detail"):
        table.add_column(col)
    keys = list(PROFILES) if providers == "all" else providers.split(",")
    for key in keys:
        try:
            vlm = registry.vlms.create(key)
        except Exception as exc:  # noqa: BLE001
            table.add_row("vlm", key, "skip", str(exc)[:80])
            continue
        res = vlm.analyze(
            AnalyzeRequest(
                prompt="Describe this test pattern in five words.",
                images=[ImagePart(jpeg=jpeg, label="frame 1")],
                max_tokens=2000,
                effort="low",
                purpose="custom",
            )
        )
        ok = res.error is None and bool(res.text)
        table.add_row(
            "vlm",
            key,
            "[green]ok[/]" if ok else "[red]fail[/]",
            f"{res.model_id} · ${res.cost_usd or 0:.4f} · {res.latency_ms:.0f} ms · "
            f"{(res.error or res.text)[:60]}",
        )
    from hoops.decide.base import Noul

    for key in deciders.split(","):
        try:
            dec = registry.deciders.create(key)
        except Exception as exc:  # noqa: BLE001
            table.add_row("decider", key, "skip", str(exc)[:80])
            continue
        r = dec.decide(
            {"ball_path": "rose above rim, then dropped through rim region"},
            {"made": Noul(instructions="The shot went in.")},
        )
        ok = r.error is None and "made" in r.answers
        detail = r.error or f"p={r.answers['made'].value} · {r.latency_ms:.0f} ms"
        table.add_row("decider", key, "[green]ok[/]" if ok else "[red]fail[/]", detail[:80])
    console.print(table)


DEMOS = [
    # (file, footage type, rim box in proxy px or None, outputs)
    ("plos_shot_01.ogv", "fixed_camera", "768,240,824,266", "annotated_video,events,summary,court_map"),
    ("plos_shot_04.ogv", "fixed_camera", "340,58,470,104", "annotated_video,events,summary,court_map"),
    ("bandits_practice_01.webm", "handheld", None, "annotated_video,events,summary"),
]


@app.command()
def demo(
    vlm: Annotated[str, typer.Option(help="AI provider for summaries ('none' for CV only)")] = "none",
) -> None:
    """Download CC BY sample clips and analyze them so the UI has finished breakdowns to show."""
    subprocess.run([sys.executable, str(ROOT / "scripts" / "fetch_samples.py")], check=True)
    samples = ROOT / "data" / "samples"
    for name, footage, rim, outs in DEMOS:
        path = samples / name
        if not path.exists():
            console.print(f"[yellow]missing {name}; skipping[/]")
            continue
        console.rule(name)
        analyze(
            path,
            outputs=outs,
            vlm=vlm,
            rim=rim,
            footage=footage,
            detector="rfdetr:coco",
            tracker="botsort" if footage == "handheld" else "bytetrack",
            decider="rules",
            preset="balanced",
        )


@app.command("eval")
def eval_cmd(run_id: str, labels: Path) -> None:
    """Score a run against hand labels (precision/recall with confidence intervals)."""
    from hoops.eval import report_markdown, score_run
    from hoops.store import Store

    report = score_run(Store().load_run(run_id), json.loads(labels.read_text()))
    console.print(report_markdown(report))


# ---------------------------------------------------------------------------------------- helpers


def _progress_printer(stage: str):
    last = {"pct": -1}

    def cb(fraction: float, message: str = "") -> None:
        pct = int(fraction * 100)
        if pct != last["pct"] and pct % 10 == 0:
            last["pct"] = pct
            console.print(f"  {stage}: {pct:3d}%  {message}")

    return cb


def _build_web() -> None:
    web = ROOT / "web"
    if not (web / "package.json").exists():
        console.print("[yellow]web/ not found; serving the API only[/]")
        return
    npm = shutil.which("npm")
    if not npm:
        console.print("[yellow]npm not found; install Node 22 to build the web UI[/]")
        return
    console.print("Building the web UI (first run only)…")
    if not (web / "node_modules").exists():
        subprocess.run([npm, "--prefix", str(web), "install", "--no-audit", "--no-fund"], check=True)
    subprocess.run([npm, "--prefix", str(web), "run", "build"], check=True)


def _open_share_tunnel(host: str, port: int) -> str:
    """Reuse Gradio's share-link tunnel (FRP) for our FastAPI app."""
    from gradio.networking import setup_tunnel

    local = "127.0.0.1" if host in ("0.0.0.0", "localhost") else host
    return setup_tunnel(local, port, secrets.token_urlsafe(32), None, None)


def main() -> None:  # pragma: no cover
    sys.exit(app())
