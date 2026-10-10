import { useEffect, useRef } from 'react'

import { MOCK_PROXY_H, MOCK_PROXY_W, MOCK_RIM } from '../../api/fixtures'
import type { HoopsEvent, TeamRef } from '../../api/types'
import { eventTitle, keyTime, outcomeOf, teamName } from '../../lib/events'
import { fmtClock } from '../../lib/format'

/**
 * Mock-mode "footage": a synthetic scene drawn on a canvas over the generated poster, driven by the
 * player clock. Shots fly to the rim at each event's key time, so the timeline, captions, and
 * annotated/original toggle all behave like the real thing. No real footage is involved.
 */
const W = MOCK_PROXY_W
const H = MOCK_PROXY_H
const TEAM_DATA = ['#3987e5', '#d55181']
const STATUS = { made: '#0ca30c', missed: '#d03b3b', attempt: '#fab219', other: '#8a94a9' }

interface P {
  team: 0 | 1
  id: number
  cx: number
  cy: number
  ax: number
  ay: number
  fx: number
  fy: number
  ph: number
}

const PLAYERS: P[] = Array.from({ length: 10 }, (_, i) => ({
  team: (i < 5 ? 0 : 1) as 0 | 1,
  id: [3, 7, 11, 23, 30, 2, 5, 12, 14, 21][i],
  cx: 520 + (i % 5) * 120 + (i < 5 ? 0 : 40),
  cy: 400 + ((i * 37) % 5) * 52,
  ax: 70 + ((i * 13) % 60),
  ay: 22 + ((i * 7) % 28),
  fx: 0.29 + (i % 4) * 0.07,
  fy: 0.21 + (i % 3) * 0.09,
  ph: i * 1.7,
}))

const pos = (p: P, t: number) => ({ x: p.cx + p.ax * Math.sin(t * p.fx + p.ph), y: p.cy + p.ay * Math.sin(t * p.fy + p.ph * 0.7) })
const scaleAt = (y: number) => 0.55 + ((y - 300) / 420) * 0.75
const clamp01 = (v: number) => Math.max(0, Math.min(1, v))

function activeShot(events: HoopsEvent[], t: number): HoopsEvent | null {
  let best: HoopsEvent | null = null
  for (const e of events) {
    if (!e.shot || e.shot.image_x == null || e.status === 'rejected') continue
    const k = keyTime(e)
    if (t >= k - 1.05 && t <= k + 0.9 && (!best || Math.abs(k - t) < Math.abs(keyTime(best) - t))) best = e
  }
  return best
}

function ballAt(events: HoopsEvent[], t: number): { x: number; y: number } {
  const e = activeShot(events, t)
  if (e && e.shot?.image_x != null && e.shot.image_y != null) {
    const k = keyTime(e)
    const sx = e.shot.image_x
    const sy = e.shot.image_y - 78 * scaleAt(e.shot.image_y)
    const o = outcomeOf(e)
    if (t <= k) {
      const p = clamp01((t - (k - 1.0)) / 1.0)
      return { x: sx + (MOCK_RIM.x - sx) * p, y: sy + (MOCK_RIM.y - 10 - sy) * p - 190 * 4 * p * (1 - p) }
    }
    const dt = t - k
    if (o === 'made') return { x: MOCK_RIM.x + Math.sin(dt * 30) * 2, y: MOCK_RIM.y - 6 + dt * 300 }
    const dir = sx < MOCK_RIM.x ? 1 : -1
    return { x: MOCK_RIM.x + dir * dt * 240, y: MOCK_RIM.y - 12 - 220 * dt + 420 * dt * dt }
  }
  const handler = PLAYERS[Math.floor(t / 5.5) % PLAYERS.length]
  const hp = pos(handler, t)
  const s = scaleAt(hp.y)
  return { x: hp.x + 15 * s, y: hp.y - 6 * s - Math.abs(Math.sin(t * 8.5)) * 26 * s }
}

function draw(ctx: CanvasRenderingContext2D, img: HTMLImageElement | null, t: number, annotated: boolean, cc: boolean, events: HoopsEvent[], teams: TeamRef[]) {
  ctx.clearRect(0, 0, W, H)
  if (img?.complete && img.naturalWidth) ctx.drawImage(img, 0, 0, W, H)
  else {
    ctx.fillStyle = '#141c30'
    ctx.fillRect(0, 0, W, H)
  }
  const jerseys = [teams[0]?.color_hex ?? '#e9edf3', teams[1]?.color_hex ?? '#26324d']

  // players, back to front
  const ps = PLAYERS.map((p) => ({ p, ...pos(p, t) })).sort((a, b) => a.y - b.y)
  for (const { p, x, y } of ps) {
    const s = scaleAt(y)
    ctx.fillStyle = 'rgba(0,0,0,0.35)'
    ctx.beginPath()
    ctx.ellipse(x, y, 17 * s, 5 * s, 0, 0, Math.PI * 2)
    ctx.fill()
    if (annotated) {
      ctx.strokeStyle = TEAM_DATA[p.team]
      ctx.lineWidth = 3
      ctx.beginPath()
      ctx.ellipse(x, y, 24 * s, 7.5 * s, 0, 0, Math.PI * 2)
      ctx.stroke()
    }
    ctx.fillStyle = '#1b1f2a'
    ctx.fillRect(x - 7 * s, y - 26 * s, 6 * s, 26 * s)
    ctx.fillRect(x + 1 * s, y - 26 * s, 6 * s, 26 * s)
    ctx.fillStyle = jerseys[p.team]
    ctx.beginPath()
    ctx.roundRect(x - 11 * s, y - 60 * s, 22 * s, 36 * s, 6 * s)
    ctx.fill()
    ctx.strokeStyle = 'rgba(0,0,0,0.35)'
    ctx.lineWidth = 1.5
    ctx.stroke()
    ctx.fillStyle = '#7a5a45'
    ctx.beginPath()
    ctx.arc(x, y - 69 * s, 8 * s, 0, Math.PI * 2)
    ctx.fill()
    if (annotated) {
      const label = `#${p.id}`
      ctx.font = `700 ${Math.round(13 * s + 4)}px Inter Variable, sans-serif`
      const w = ctx.measureText(label).width + 8
      ctx.fillStyle = TEAM_DATA[p.team]
      ctx.beginPath()
      ctx.roundRect(x - w / 2, y - 98 * s, w, 17 * s + 4, 4)
      ctx.fill()
      ctx.fillStyle = '#fff'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillText(label, x, y - 98 * s + (17 * s + 4) / 2)
    }
  }

  // rim box
  if (annotated) {
    ctx.strokeStyle = '#ff9447'
    ctx.lineWidth = 2.5
    ctx.strokeRect(MOCK_RIM.x - MOCK_RIM.w / 2 - 8, MOCK_RIM.y - MOCK_RIM.h / 2 - 8, MOCK_RIM.w + 16, MOCK_RIM.h + 16)
    ctx.font = '700 14px Inter Variable, sans-serif'
    ctx.fillStyle = '#ffb27a'
    ctx.textAlign = 'left'
    ctx.textBaseline = 'bottom'
    ctx.fillText('rim 0.94', MOCK_RIM.x - MOCK_RIM.w / 2 - 8, MOCK_RIM.y - MOCK_RIM.h / 2 - 12)
  }

  // ball (+ trail when annotated)
  if (annotated) {
    for (let k = 8; k >= 1; k--) {
      const b = ballAt(events, t - k * 0.035)
      ctx.fillStyle = `rgba(255,148,71,${0.05 + (8 - k) * 0.035})`
      ctx.beginPath()
      ctx.arc(b.x, b.y, 8 - k * 0.5, 0, Math.PI * 2)
      ctx.fill()
    }
  }
  const b = ballAt(events, t)
  const g = ctx.createRadialGradient(b.x - 3, b.y - 3, 1, b.x, b.y, 10)
  g.addColorStop(0, '#ffb27a')
  g.addColorStop(1, '#c4520a')
  ctx.fillStyle = g
  ctx.beginPath()
  ctx.arc(b.x, b.y, 9.5, 0, Math.PI * 2)
  ctx.fill()
  if (annotated) {
    ctx.fillStyle = '#ff7a1a'
    ctx.beginPath()
    ctx.moveTo(b.x, b.y - 16)
    ctx.lineTo(b.x - 7, b.y - 28)
    ctx.lineTo(b.x + 7, b.y - 28)
    ctx.closePath()
    ctx.fill()
  }

  const e = activeShot(events, t)
  // event banner (annotated)
  if (annotated && e) {
    const o = outcomeOf(e)
    const text = `${o === 'made' ? 'MADE' : o === 'missed' ? 'MISSED' : 'SHOT?'} · ${teamName(e.team_id, teams).toUpperCase()}`
    ctx.font = '700 20px Barlow Condensed, Inter Variable, sans-serif'
    const w = ctx.measureText(text).width + 40
    ctx.fillStyle = 'rgba(7,11,20,0.82)'
    ctx.beginPath()
    ctx.roundRect(28, 28, w, 38, 8)
    ctx.fill()
    ctx.fillStyle = STATUS[o]
    ctx.beginPath()
    ctx.arc(46, 47, 6, 0, Math.PI * 2)
    ctx.fill()
    ctx.fillStyle = '#fff'
    ctx.textAlign = 'left'
    ctx.textBaseline = 'middle'
    ctx.fillText(text, 60, 48)
  }
  if (annotated) {
    ctx.font = '600 13px Inter Variable, sans-serif'
    ctx.fillStyle = 'rgba(255,255,255,0.7)'
    ctx.textAlign = 'left'
    ctx.textBaseline = 'alphabetic'
    ctx.fillText('rfdetr:coco · bytetrack · 10 tracks', 30, H - 70)
  }

  // clock
  ctx.font = '600 18px ui-monospace, SF Mono, Menlo, monospace'
  ctx.fillStyle = 'rgba(7,11,20,0.7)'
  ctx.beginPath()
  ctx.roundRect(W - 112, 24, 86, 32, 6)
  ctx.fill()
  ctx.fillStyle = '#fff'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillText(fmtClock(t, true), W - 69, 41)

  // captions
  if (cc && e) {
    const text = `${eventTitle(e)} · ${teamName(e.team_id, teams)}`
    ctx.font = '600 24px Inter Variable, sans-serif'
    const w = ctx.measureText(text).width + 36
    ctx.fillStyle = 'rgba(7,11,20,0.82)'
    ctx.beginPath()
    ctx.roundRect(W / 2 - w / 2, H - 118, w, 44, 8)
    ctx.fill()
    ctx.fillStyle = '#fff'
    ctx.fillText(text, W / 2, H - 95)
  }
}

export function SimulatedVideo({ time, annotated, cc, events, teams, poster }: { time: number; annotated: boolean; cc: boolean; events: HoopsEvent[]; teams: TeamRef[]; poster: string | null }) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const img = useRef<HTMLImageElement | null>(null)
  const latest = useRef({ time, annotated, cc, events, teams })
  latest.current = { time, annotated, cc, events, teams }

  useEffect(() => {
    if (!poster) return
    const im = new Image()
    im.onload = () => {
      const c = canvas.current?.getContext('2d')
      const l = latest.current
      if (c) draw(c, im, l.time, l.annotated, l.cc, l.events, l.teams)
    }
    im.src = poster
    img.current = im
  }, [poster])

  useEffect(() => {
    const c = canvas.current?.getContext('2d')
    if (c) draw(c, img.current, time, annotated, cc, events, teams)
  }, [time, annotated, cc, events, teams])

  return <canvas ref={canvas} width={W} height={H} className="absolute inset-0 h-full w-full" role="img" aria-label="Simulated demo footage" />
}
