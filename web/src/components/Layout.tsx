import { FlaskConical, GitCompareArrows, House, Settings2 } from 'lucide-react'
import { NavLink, Outlet, useLocation } from 'react-router'

import { isMock } from '../api'
import { cn, fmtUsd } from '../lib/format'
import { CourtBackdrop, Wordmark } from './Brand'
import { useConfig } from './ConfigContext'
import { Tip } from './ui/tip'

const NAV = [
  { to: '/', label: 'Home', icon: House, end: true },
  { to: '/compare', label: 'Compare', icon: GitCompareArrows, end: false },
  { to: '/settings', label: 'Settings', icon: Settings2, end: false },
]

export function Layout() {
  const { config } = useConfig()
  const loc = useLocation()
  return (
    <div className="flex min-h-dvh flex-col">
      <CourtBackdrop />
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:top-3 focus:left-3 focus:z-[100] focus:rounded-lg focus:bg-flame-500 focus:px-3 focus:py-2 focus:text-ink-950">
        Skip to content
      </a>
      <header className="sticky top-0 z-40 border-b border-line bg-ink-950/75 pt-[env(safe-area-inset-top)] backdrop-blur-xl">
        <div className="mx-auto flex h-16 max-w-7xl items-center gap-3 px-4 sm:px-6">
          <NavLink to="/" className="mr-auto rounded-lg" aria-label="Press Hoops home">
            <Wordmark />
          </NavLink>
          {isMock && (
            <Tip content="Demo mode: fake data, no backend, nothing leaves your browser. Add ?mock=0 to the URL to switch it off.">
              <span tabIndex={0} className="hidden items-center gap-1.5 rounded-full border border-flame-500/40 bg-flame-500/10 px-2.5 py-1 font-display text-xs font-semibold tracking-[0.14em] text-flame-200 uppercase sm:inline-flex">
                <FlaskConical className="size-3.5" aria-hidden /> Demo
              </span>
            </Tip>
          )}
          {config && (
            <Tip content={`AI spend this session, out of a ${fmtUsd(config.caps.session_cost_cap_usd)} cap`}>
              <NavLink to="/settings" className="hidden rounded-full border border-line px-2.5 py-1 text-xs font-semibold text-fg-2 tnum hover:text-fg md:inline-flex">
                {fmtUsd(config.spend.session_usd)} spent
              </NavLink>
            </Tip>
          )}
          <nav aria-label="Main" className="flex items-center gap-1">
            {NAV.map(({ to, label, icon: Icon, end }) => (
              <NavLink
                key={to}
                to={to}
                end={end}
                className={({ isActive }) =>
                  cn(
                    'flex h-9 items-center gap-2 rounded-xl px-2.5 text-sm font-semibold transition-colors sm:px-3',
                    isActive ? 'bg-white/[0.08] text-fg' : 'text-fg-2 hover:bg-white/[0.05] hover:text-fg',
                  )
                }
              >
                <Icon className="size-4.5" aria-hidden />
                <span className="sr-only sm:not-sr-only">{label}</span>
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <main id="main" key={loc.pathname} className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 sm:px-6 sm:py-8">
        <Outlet />
      </main>
      <footer className="mx-auto w-full max-w-7xl px-4 pt-2 pb-[max(1.5rem,env(safe-area-inset-bottom))] text-xs text-fg-3 sm:px-6">
        Footage stays on this computer unless you pick a cloud AI provider. Every run lists exactly what was sent where.
      </footer>
    </div>
  )
}
