"""Phone-accurate full-page screenshots of the web UI via the Chrome DevTools Protocol.

usage: uv run python scripts/screenshot.py <url> <out.png> [width=390] [height=844] [wait_s=6]

Headless Chrome's --window-size can't go below 500 px wide, so this emulates the device instead.
"""

from __future__ import annotations

import asyncio
import base64
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PORT = 9333


def start_chrome() -> tuple[subprocess.Popen, str]:
    proc = subprocess.Popen(
        [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--remote-debugging-port={PORT}",
            f"--user-data-dir={tempfile.mkdtemp(prefix='hoops-cdp-')}",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(50):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json") as resp:
                targets = json.load(resp)
            page = next(t for t in targets if t["type"] == "page")
            return proc, page["webSocketDebuggerUrl"]
        except Exception:  # noqa: BLE001 - Chrome still starting
            time.sleep(0.2)
    proc.terminate()
    raise SystemExit("Chrome didn't start")


async def capture(ws_url: str, url: str, width: int, height: int, wait_s: float) -> tuple[bytes, int]:
    async with websockets.connect(ws_url, max_size=50_000_000) as ws:
        n = 0

        async def call(method: str, **params: object) -> dict:
            nonlocal n
            n += 1
            await ws.send(json.dumps({"id": n, "method": method, "params": params}))
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("id") == n:
                    return msg.get("result", {})

        mobile = width < 800
        await call(
            "Emulation.setDeviceMetricsOverride",
            width=width,
            height=height,
            deviceScaleFactor=2,
            mobile=mobile,
        )
        await call("Emulation.setTouchEmulationEnabled", enabled=mobile)
        await call("Page.enable")
        await call("Page.navigate", url=url)
        await asyncio.sleep(wait_s)
        full_h = min(int((await call("Page.getLayoutMetrics"))["cssContentSize"]["height"]), 6000)
        await call(
            "Emulation.setDeviceMetricsOverride",
            width=width,
            height=full_h,
            deviceScaleFactor=1,
            mobile=mobile,
        )
        await asyncio.sleep(1.0)
        shot = await call("Page.captureScreenshot", format="png")
        return base64.b64decode(shot["data"]), full_h


def main(url: str, out: str, width: int = 390, height: int = 844, wait_s: float = 6) -> None:
    proc, ws_url = start_chrome()
    try:
        png, full_h = asyncio.run(capture(ws_url, url, width, height, wait_s))
    finally:
        proc.terminate()
    Path(out).write_bytes(png)
    print(f"{out}: {width}x{full_h} css px")


if __name__ == "__main__":
    a = sys.argv
    main(
        a[1],
        a[2],
        int(a[3]) if len(a) > 3 else 390,
        int(a[4]) if len(a) > 4 else 844,
        float(a[5]) if len(a) > 5 else 6,
    )
