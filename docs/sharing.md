# Sharing the app with a friend

The app runs on your Mac. A tunnel gives your friend a link to it.

## Before sharing (every option)
1. **Set a password.** Put `HOOPS_APP_PASSWORD=...` in `.env`, and send the password separately from the link.
   - The login is a first-party cookie, so it works in Safari and private windows.
   - Still, test the link once from your friend's browser.
2. **Cap spending.**
   - The app's caps are `HOOPS_RUN_COST_CAP_USD` (default $0.50) and `HOOPS_SESSION_COST_CAP_USD` (default $5).
   - Also set a monthly limit in each provider's console: Anthropic, OpenAI, Google, OpenRouter. The app's caps are estimates; the console limit is the hard stop.
3. **Keep the Mac awake.**
   - Run `caffeinate -dimsu` in a second terminal while sharing.
   - Plug in power and turn off "sleep when the lid is closed" if you close the lid.
4. **Run `uv run hoops doctor`.** It should show no errors. It warns if the password is missing.

## Option A: share link (quickest)
```bash
uv run hoops serve --share
```
- This prints a public `https://….gradio.live` link, using Gradio's share tunnel.
- Links expire after about a week; run the command again for a new one.
- Uploads are sent in 8 MB chunks, so tunnel size limits don't matter.
- If live progress stalls, the page falls back to checking every second.

## Option B: Tailscale Funnel (stable URL)
- Install Tailscale and log in.
- Start `uv run hoops serve` (without `--share`) and note its port.
- Then run:
  ```bash
  tailscale funnel --bg <port>
  ```
- You get a fixed `https://<your-mac>.<tailnet>.ts.net` address that doesn't change between sessions.
- Stop sharing with `tailscale funnel --bg off` (or `tailscale funnel reset`).

## Later: a permanent host
The `hoops/` package doesn't depend on the tunnel, so moving it is a deployment change, not a rewrite. We'll keep the password, the caps and server-side keys either way.

| Host | Notes |
|---|---|
| **Hugging Face Space** | Make it *private* or *protected*. A FastAPI + React app is a Docker Space, which needs a paid plan. Store keys as Space secrets. Footage needs a storage bucket. |
| **Modal** | Pay per second, with GPUs available. Keep `data/` on a volume. |

On Linux hosts there is no Apple GPU (MPS) and no VideoToolbox, so CV runs on CUDA or the CPU.

## Don't use
| Option | Why not |
|---|---|
| **Cloudflare quick tunnels** (`trycloudflare.com`) | No server-sent events, so live progress breaks, and they cap upload size. |
| **ngrok free tier** | About 1 GB/month of bandwidth, which a few game videos use up. It also adds an interstitial warning page. |
| **Any link without a password** | Anyone who finds it can use your API credit and see the footage. |
