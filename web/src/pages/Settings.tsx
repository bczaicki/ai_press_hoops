import { Ban, Camera, Check, FlaskConical, HardDrive, LogOut, ShieldCheck, Video } from 'lucide-react'
import { useState, type ReactNode } from 'react'

import { api, errorMessage, isMock } from '../api'
import type { RegistryItem } from '../api/types'
import { useConfig } from '../components/ConfigContext'
import { Button } from '../components/ui/button'
import { Badge, Card, CardHeader, ErrorPanel, Meter, Skeleton } from '../components/ui/misc'
import { useToast } from '../components/ui/toast'
import { fmtClock, fmtUsd } from '../lib/format'
import { useAsync } from '../lib/hooks'

function Row({ ok, title, sub, right }: { ok: boolean; title: ReactNode; sub?: ReactNode; right?: ReactNode }) {
  return (
    <li className="flex items-start gap-3 px-5 py-3">
      <span className={ok ? 'mt-0.5 grid size-5 shrink-0 place-items-center rounded-full bg-made text-ink-950' : 'mt-0.5 grid size-5 shrink-0 place-items-center rounded-full border border-fg-3/50 text-fg-3'}>
        {ok ? <Check className="size-3.5" strokeWidth={3.5} aria-hidden /> : <Ban className="size-3" aria-hidden />}
      </span>
      <div className="min-w-0 flex-1">
        <div className="font-semibold text-fg">{title}</div>
        {sub && <div className="text-sm break-words text-fg-3">{sub}</div>}
      </div>
      {right}
      <span className="sr-only">{ok ? 'available' : 'unavailable'}</span>
    </li>
  )
}

function RegistryList({ items }: { items: RegistryItem[] }) {
  return (
    <ul className="divide-y divide-line">
      {items.map((d) => (
        <Row key={d.key} ok={d.available} title={d.label} sub={d.available ? [d.license, d.notes].filter(Boolean).join(' · ') || d.key : d.reason} right={d.supports_images ? <Badge tone="info">Sees images</Badge> : undefined} />
      ))}
    </ul>
  )
}

const TIPS = [
  { icon: Camera, title: 'Steady beats fancy', body: 'A phone on a tripod or a bleacher rail, filming one half of the court, gives the best shot calls.' },
  { icon: Video, title: 'Keep the hoop in frame', body: 'The rim should be visible the whole time. Shots are judged by the ball crossing it.' },
  { icon: Camera, title: 'Landscape, normal speed', body: 'Landscape at 30 or 60 fps. Skip slow-motion and filters. HDR and portrait clips work, they just convert first.' },
  { icon: Video, title: 'Short clips first', body: "Try a 1-3 minute clip before a full game. You'll get results in a couple of minutes." },
]

export default function SettingsPage() {
  const { config, error, reload } = useConfig()
  const toast = useToast()
  const session = useAsync(() => api.session(), [])
  const [busy, setBusy] = useState(false)

  if (error) return <ErrorPanel message={error} action={<Button size="sm" onClick={reload}>Try again</Button>} />
  if (!config) return <Skeleton className="h-96" />
  const spend = config.spend.session_usd
  const cap = config.caps.session_cost_cap_usd

  return (
    <div className="space-y-6">
      <div>
        <div className="eyebrow mb-1 text-flame-300">Behind the bench</div>
        <h1 className="font-display text-5xl leading-none font-bold tracking-wide text-fg uppercase">Settings & about</h1>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader eyebrow="Who can explain your clips" title="AI providers" />
          <ul className="divide-y divide-line border-t border-line">
            {config.providers.map((p) => (
              <Row
                key={p.key}
                ok={p.available}
                title={
                  <span className="flex flex-wrap items-center gap-2">
                    {p.label}
                    {p.local && (
                      <Badge tone="made" icon={<HardDrive className="size-3" aria-hidden />}>
                        Local
                      </Badge>
                    )}
                    {p.native_video && <Badge tone="info">Native video</Badge>}
                  </span>
                }
                sub={p.available ? `${p.default_model} · sends frames to ${p.sends_data_to}` : p.reason}
              />
            ))}
          </ul>
          <p className="border-t border-line px-5 py-3 text-xs text-fg-3">Keys stay on the server. This page only shows whether each one works, never the key itself.</p>
        </Card>

        <div className="space-y-6">
          <Card className="p-5">
            <div className="eyebrow">Spent this session</div>
            <div className="numeral mt-1 text-6xl text-fg">{fmtUsd(spend)}</div>
            <Meter value={spend / cap} tone={spend / cap > 0.8 ? 'miss' : 'flame'} className="mt-3" label="Session spend against the cap" />
            <div className="mt-1.5 flex justify-between text-xs text-fg-3 tnum">
              <span>Cap {fmtUsd(cap)}</span>
              <span>{fmtUsd(Math.max(0, cap - spend))} left</span>
            </div>
            <dl className="mt-4 grid grid-cols-2 gap-3 text-sm">
              <div>
                <dt className="text-xs text-fg-3">Per-run cap</dt>
                <dd className="font-semibold text-fg tnum">{fmtUsd(config.caps.run_cost_cap_usd)}</dd>
              </div>
              <div>
                <dt className="text-xs text-fg-3">Biggest upload</dt>
                <dd className="font-semibold text-fg tnum">{config.caps.max_upload_mb} MB</dd>
              </div>
              <div>
                <dt className="text-xs text-fg-3">Longest clip</dt>
                <dd className="font-semibold text-fg tnum">{fmtClock(config.caps.max_clip_seconds)}</dd>
              </div>
              <div>
                <dt className="text-xs text-fg-3">Version</dt>
                <dd className="font-semibold text-fg tnum">{config.version}</dd>
              </div>
            </dl>
          </Card>

          <Card className="space-y-3 p-5">
            <div className="eyebrow">This browser</div>
            {isMock ? (
              <a href="/?mock=0" className="flex items-center gap-2 text-sm font-semibold text-flame-300 hover:text-flame-200">
                <FlaskConical className="size-4" aria-hidden /> Leave demo mode
              </a>
            ) : (
              <a href="/?mock=1" className="flex items-center gap-2 text-sm font-semibold text-flame-300 hover:text-flame-200">
                <FlaskConical className="size-4" aria-hidden /> Try demo mode (sample data, no server)
              </a>
            )}
            {session.data?.auth_required && (
              <Button
                variant="secondary"
                className="w-full"
                loading={busy}
                icon={<LogOut className="size-4" />}
                onClick={async () => {
                  setBusy(true)
                  try {
                    await api.logout()
                    window.location.assign('/')
                  } catch (err) {
                    toast({ title: "Couldn't sign out", body: errorMessage(err), tone: 'error' })
                    setBusy(false)
                  }
                }}
              >
                Sign out
              </Button>
            )}
          </Card>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <Card>
          <CardHeader eyebrow="System One" title="Shot callers" />
          <div className="border-t border-line">
            <RegistryList items={config.deciders} />
          </div>
        </Card>
        <Card>
          <CardHeader eyebrow="Computer vision" title="Detectors" />
          <div className="border-t border-line">
            <RegistryList items={config.detectors} />
          </div>
        </Card>
        <Card>
          <CardHeader eyebrow="Computer vision" title="Trackers" />
          <div className="border-t border-line">
            <RegistryList items={config.trackers} />
          </div>
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card className="p-5">
          <div className="mb-3 flex items-center gap-2">
            <ShieldCheck className="size-5 text-flame-400" aria-hidden />
            <h2 className="font-display text-2xl font-bold tracking-wide text-fg uppercase">Privacy</h2>
          </div>
          <ul className="space-y-2 text-sm text-fg-2">
            <li>• Your video stays on this computer. Detection and tracking run locally.</li>
            <li>• Frames go to an AI company only when you pick that provider, and only short windows around close calls plus a few frames for the recap.</li>
            <li>• Pick “Local” or “No AI” and nothing leaves this machine.</li>
            <li>• Every run lists exactly which services received data (Under the hood → Privacy).</li>
            <li>• No face recognition. Players are only ever “#23 in white”, never a name.</li>
            <li>• Only film people who are OK with it, and get a guardian's OK for minors.</li>
          </ul>
        </Card>
        <Card className="p-5">
          <div className="mb-3 flex items-center gap-2">
            <Camera className="size-5 text-flame-400" aria-hidden />
            <h2 className="font-display text-2xl font-bold tracking-wide text-fg uppercase">Filming tips</h2>
          </div>
          <ul className="grid gap-3 sm:grid-cols-2">
            {TIPS.map((t) => (
              <li key={t.title} className="rounded-xl border border-line bg-white/[0.02] p-3">
                <div className="flex items-center gap-2 font-semibold text-fg">
                  <t.icon className="size-4 text-flame-400" aria-hidden />
                  {t.title}
                </div>
                <p className="mt-1 text-sm text-fg-2">{t.body}</p>
              </li>
            ))}
          </ul>
        </Card>
      </div>
    </div>
  )
}
