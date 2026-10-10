"""Download a few openly licensed basketball clips into data/samples/ (gitignored) for testing and demos.

Usage: uv run python scripts/fetch_samples.py [--all]

Every clip is Creative Commons Attribution licensed; ATTRIBUTION.md is written next to the files.
"""

from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data" / "samples"
UA = "ai_press_hoops/0.1 (basketball analysis POC; sample fetch)"

PLOS = "https://upload.wikimedia.org/wikipedia/commons"
PLOS_CREDIT = (
    "Supplementary video from Erčulj F, Štrumbelj E (2015) 'Basketball Shot Types and Shot Success in Different "
    "Levels of Competitive Basketball', PLOS ONE 10(6): e0128885, via Wikimedia Commons. CC BY 4.0."
)

PLOS_PAPER = (
    "Basketball-Shot-Types-and-Shot-Success-in-Different-Levels-of-Competitive-Basketball-pone.0128885"
)
BANDITS = "ShotBug_Press_LIVE%21_at_the_Langley_Event_Centre_for_the_Vancouver_Bandits_first_practice_of_the_year_01"
BANDITS_CREDIT = (
    "'ShotBug Press LIVE! at the Langley Event Centre for the Vancouver Bandits first practice of the year 01' "
    "by ShotBug Press, via Wikimedia Commons. CC BY 3.0."
)

SAMPLES = [
    ("plos_shot_01.ogv", f"{PLOS}/2/2e/{PLOS_PAPER}.s001.ogv", PLOS_CREDIT, True),
    ("plos_shot_04.ogv", f"{PLOS}/d/df/{PLOS_PAPER}.s004.ogv", PLOS_CREDIT, True),
    ("bandits_practice_01.webm", f"{PLOS}/c/ce/{BANDITS}.webm", BANDITS_CREDIT, True),
]


def main(all_clips: bool) -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    lines = ["# Sample clip attribution", ""]
    for name, url, credit, default in SAMPLES:
        if not (default or all_clips):
            continue
        out = DEST / name
        if not out.exists():
            print(f"downloading {name} …")
            _download(url, out)
            time.sleep(2)  # be polite to upload.wikimedia.org
        lines.append(f"- `{name}`: {credit} Source: {url}")
        print(f"ok  {name}  ({out.stat().st_size / 1e6:.1f} MB)")
    (DEST / "ATTRIBUTION.md").write_text("\n".join(lines) + "\n")


def _download(url: str, out: Path, attempts: int = 5) -> None:
    for i in range(attempts):
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            tmp = out.with_suffix(out.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(out)
            return
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or i == attempts - 1:
                raise
            wait = int(exc.headers.get("Retry-After") or 10 * (i + 1))
            print(f"  rate limited; retrying in {wait}s")
            time.sleep(min(wait, 60))


if __name__ == "__main__":
    main(all_clips="--all" in sys.argv)
