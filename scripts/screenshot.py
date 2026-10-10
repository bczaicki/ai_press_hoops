"""Phone-accurate full-page screenshots of the web UI via the Chrome DevTools Protocol.

usage: uv run python scripts/screenshot.py <url> <out.png> [width=390] [height=844] [wait_s=6]
"""

import asyncio
import base64
import json
import subprocess
import sys
import tempfile
import time
import urllib.request

import websockets

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PORT = 9333


async def main(url: str, out: str, width: int, height: int, wait_s: float) -> None:
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
    page = None
    try:
        for _ in range(50):
            try:
                targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json"))
                page = next(t for t in targets if t["type"] == "page")
                break
            except Exception:  # noqa: BLE001 - Chrome still starting
                time.sleep(0.2)
        if page is None:
            raise SystemExit("Chrome didn't start")
        async with websockets.connect(page["webSocketDebuggerUrl"], max_size=50_000_000) as ws:
            n = 0

            async def call(method, **params):
                nonlocal n
                n += 1
                await ws.send(json.dumps({"id": n, "method": method, "params": params}))
                while True:
                    msg = json.loads(await ws.recv())
                    if msg.get("id") == n:
                        return msg.get("result", {})

            await call(
                "Emulation.setDeviceMetricsOverride",
                width=width,
                height=height,
                deviceScaleFactor=2,
                mobile=width < 800,
            )
            await call("Emulation.setTouchEmulationEnabled", enabled=width < 800)
            await call("Page.enable")
            await call("Page.navigate", url=url)
            await asyncio.sleep(wait_s)
            metrics = await call("Page.getLayoutMetrics")
            full_h = int(metrics["cssContentSize"]["height"])
            await call(
                "Emulation.setDeviceMetricsOverride",
                width=width,
                height=min(full_h, 6000),
                deviceScaleFactor=1,
                mobile=width < 800,
            )
            await asyncio.sleep(1.0)
            shot = await call("Page.captureScreenshot", format="png")
            with open(out, "wb") as f:
                f.write(base64.b64decode(shot["data"]))
            print(f"{out}: {width}x{min(full_h, 6000)} css px")
    finally:
        proc.terminate()


if __name__ == "__main__":
    a = sys.argv
    asyncio.run(
        main(
            a[1],
            a[2],
            int(a[3]) if len(a) > 3 else 390,
            int(a[4]) if len(a) > 4 else 844,
            float(a[5]) if len(a) > 5 else 6,
        )
    )
