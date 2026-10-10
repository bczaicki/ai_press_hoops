import { ArrowRight, Eye, EyeOff, Lock } from 'lucide-react'
import { useState, type FormEvent } from 'react'

import { api, errorMessage, isMock } from '../api'
import { CourtBackdrop, Wordmark } from '../components/Brand'
import { Button } from '../components/ui/button'
import { inputClass } from '../components/ui/misc'
import { cn } from '../lib/format'

export function Login({ onSuccess }: { onSuccess: () => void }) {
  const [password, setPassword] = useState('')
  const [show, setShow] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  async function submit(e: FormEvent) {
    e.preventDefault()
    if (!password) return
    setBusy(true)
    setError(null)
    try {
      await api.login(password)
      onSuccess()
    } catch (err) {
      setError(errorMessage(err))
      setBusy(false)
    }
  }

  return (
    <div className="grid min-h-dvh place-items-center px-4 py-10">
      <CourtBackdrop />
      <div className="w-full max-w-sm">
        <div className="mb-8 flex justify-center">
          <Wordmark />
        </div>
        <form onSubmit={submit} className="glass rounded-3xl p-6 sm:p-7">
          <div className="mb-1 flex items-center gap-2 text-flame-300">
            <Lock className="size-4" aria-hidden />
            <span className="eyebrow text-flame-300">Private gym</span>
          </div>
          <h1 className="font-display text-3xl font-bold tracking-wide text-fg uppercase">Welcome in</h1>
          <p className="mt-1.5 text-sm text-fg-2">Enter the password you were sent and we'll get you to the film room.</p>
          <label htmlFor="pw" className="mt-6 mb-1.5 block text-[0.8rem] font-semibold text-fg-2">
            Password
          </label>
          <div className="relative">
            <input
              id="pw"
              type={show ? 'text' : 'password'}
              autoComplete="current-password"
              autoFocus
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className={cn(inputClass, 'h-12 pr-11 text-base', error && 'border-miss/70')}
              aria-invalid={!!error}
              aria-describedby={error ? 'pw-err' : undefined}
            />
            <button type="button" onClick={() => setShow((s) => !s)} className="absolute top-1/2 right-2 -translate-y-1/2 rounded-lg p-1.5 text-fg-3 hover:text-fg" aria-label={show ? 'Hide password' : 'Show password'}>
              {show ? <EyeOff className="size-4.5" /> : <Eye className="size-4.5" />}
            </button>
          </div>
          {error && (
            <p id="pw-err" role="alert" className="mt-2 text-sm text-miss-text">
              {error === 'Wrong password' ? "That's not it. Check for typos and try again." : error}
            </p>
          )}
          <Button type="submit" variant="primary" size="lg" loading={busy} disabled={!password} className="mt-6 w-full">
            Let's go <ArrowRight className="size-5" aria-hidden />
          </Button>
          {isMock && <p className="mt-4 text-center text-xs text-fg-3">Demo mode: the password is “hoops”.</p>}
        </form>
      </div>
    </div>
  )
}
