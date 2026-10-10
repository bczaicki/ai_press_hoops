import { AnimatePresence, motion } from 'framer-motion'
import { AlertTriangle, CheckCircle2, Info, X } from 'lucide-react'
import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from 'react'

import { cn } from '../../lib/format'

type Tone = 'success' | 'error' | 'info'
interface Toast {
  id: number
  title: string
  body?: ReactNode
  tone: Tone
}

type ToastFn = (t: { title: string; body?: ReactNode; tone?: Tone; durationMs?: number }) => void
const Ctx = createContext<ToastFn>(() => {})

export function useToast(): ToastFn {
  return useContext(Ctx)
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([])
  const seq = useRef(0)
  const dismiss = useCallback((id: number) => setToasts((ts) => ts.filter((t) => t.id !== id)), [])
  const push = useCallback<ToastFn>(
    ({ title, body, tone = 'info', durationMs }) => {
      const id = ++seq.current
      setToasts((ts) => [...ts.slice(-3), { id, title, body, tone }])
      setTimeout(() => dismiss(id), durationMs ?? (tone === 'error' ? 7000 : 4200))
    },
    [dismiss],
  )
  const value = useMemo(() => push, [push])
  return (
    <Ctx.Provider value={value}>
      {children}
      <div aria-live="polite" aria-atomic="false" className="pointer-events-none fixed inset-x-0 bottom-0 z-[90] flex flex-col items-center gap-2 p-4 sm:items-end sm:p-6">
        <AnimatePresence initial={false}>
          {toasts.map((t) => (
            <motion.div
              key={t.id}
              layout
              initial={{ opacity: 0, y: 16, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 8, scale: 0.98, transition: { duration: 0.15 } }}
              transition={{ type: 'spring', stiffness: 420, damping: 34 }}
              role={t.tone === 'error' ? 'alert' : 'status'}
              className={cn(
                'glass pointer-events-auto flex w-full max-w-sm items-start gap-3 rounded-2xl p-3.5 pr-2.5',
                t.tone === 'error' && 'border-miss/50',
                t.tone === 'success' && 'border-made/40',
              )}
            >
              {t.tone === 'success' ? (
                <CheckCircle2 className="mt-0.5 size-5 shrink-0 text-made-text" aria-hidden />
              ) : t.tone === 'error' ? (
                <AlertTriangle className="mt-0.5 size-5 shrink-0 text-miss-text" aria-hidden />
              ) : (
                <Info className="mt-0.5 size-5 shrink-0 text-flame-400" aria-hidden />
              )}
              <div className="min-w-0 flex-1">
                <div className="text-sm font-semibold text-fg">{t.title}</div>
                {t.body && <div className="mt-0.5 text-[0.8rem] text-fg-2">{t.body}</div>}
              </div>
              <button type="button" onClick={() => dismiss(t.id)} className="rounded-lg p-1 text-fg-3 hover:bg-white/5 hover:text-fg" aria-label="Dismiss">
                <X className="size-4" />
              </button>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </Ctx.Provider>
  )
}
