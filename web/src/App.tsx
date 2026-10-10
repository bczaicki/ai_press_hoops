import { MotionConfig } from 'framer-motion'
import { RefreshCw, ServerCrash } from 'lucide-react'
import { lazy, Suspense, useCallback, useEffect, useState, type ReactNode } from 'react'
import { BrowserRouter, Route, Routes } from 'react-router'

import { api, errorMessage, isMock, onUnauthorized } from './api'
import type { Session } from './api/types'
import { CourtBackdrop, Wordmark } from './components/Brand'
import { ConfigProvider } from './components/ConfigContext'
import { Layout } from './components/Layout'
import { Button } from './components/ui/button'
import { Skeleton, Spinner } from './components/ui/misc'
import { TipProvider } from './components/ui/tip'
import { ToastProvider } from './components/ui/toast'
import { Home } from './pages/Home'
import { Login } from './pages/Login'
import { NotFound } from './pages/NotFound'

const Configure = lazy(() => import('./pages/Configure'))
const RunPage = lazy(() => import('./pages/Run'))
const Compare = lazy(() => import('./pages/Compare'))
const Settings = lazy(() => import('./pages/Settings'))

function PageFallback() {
  return (
    <div className="space-y-4" aria-busy>
      <Skeleton className="h-10 w-64" />
      <Skeleton className="h-72 w-full" />
      <div className="grid gap-4 md:grid-cols-3">
        <Skeleton className="h-32" />
        <Skeleton className="h-32" />
        <Skeleton className="h-32" />
      </div>
    </div>
  )
}

function Splash({ children }: { children: ReactNode }) {
  return (
    <div className="grid min-h-dvh place-items-center px-4">
      <CourtBackdrop />
      <div className="flex flex-col items-center gap-6 text-center">
        <Wordmark />
        {children}
      </div>
    </div>
  )
}

/** Checks GET /api/session; shows Login when the server wants a password. Re-checks on any 401. */
function SessionGate({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [nonce, setNonce] = useState(0)
  const refresh = useCallback(() => setNonce((n) => n + 1), [])

  useEffect(() => {
    let alive = true
    setError(null)
    api.session().then(
      (s) => alive && setSession(s),
      (err) => alive && setError(errorMessage(err)),
    )
    return () => {
      alive = false
    }
  }, [nonce])

  useEffect(() => onUnauthorized(() => setSession((s) => (s ? { ...s, authenticated: false, auth_required: true } : s))), [])

  if (error && !session) {
    return (
      <Splash>
        <div className="glass max-w-sm rounded-3xl p-6">
          <ServerCrash className="mx-auto size-8 text-flame-400" aria-hidden />
          <h1 className="mt-3 font-display text-2xl font-bold tracking-wide text-fg uppercase">Can't reach the court</h1>
          <p className="mt-1.5 text-sm text-fg-2">The analysis server isn't answering ({error}). It may be starting up or the computer may be asleep.</p>
          <div className="mt-5 flex flex-col gap-2">
            <Button variant="primary" icon={<RefreshCw className="size-4" />} onClick={refresh}>
              Try again
            </Button>
            {!isMock && (
              <a href="/?mock=1" className="text-sm font-semibold text-flame-300 hover:text-flame-200">
                Or explore the demo with sample data
              </a>
            )}
          </div>
        </div>
      </Splash>
    )
  }
  if (!session) {
    return (
      <Splash>
        <Spinner label="Checking your session" />
      </Splash>
    )
  }
  if (session.auth_required && !session.authenticated) return <Login onSuccess={refresh} />
  return <>{children}</>
}

export default function App() {
  return (
    <MotionConfig reducedMotion="user">
      <TipProvider>
        <ToastProvider>
          <BrowserRouter>
            <SessionGate>
              <ConfigProvider>
                <Suspense fallback={<PageFallback />}>
                  <Routes>
                    <Route element={<Layout />}>
                      <Route index element={<Home />} />
                      <Route path="clips/:clipId" element={<Configure />} />
                      <Route path="runs/:runId" element={<RunPage />} />
                      <Route path="compare" element={<Compare />} />
                      <Route path="settings" element={<Settings />} />
                      <Route path="*" element={<NotFound />} />
                    </Route>
                  </Routes>
                </Suspense>
              </ConfigProvider>
            </SessionGate>
          </BrowserRouter>
        </ToastProvider>
      </TipProvider>
    </MotionConfig>
  )
}
