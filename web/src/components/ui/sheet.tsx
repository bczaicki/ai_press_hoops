import * as Dialog from '@radix-ui/react-dialog'
import { X } from 'lucide-react'
import type { ReactNode } from 'react'

import { cn } from '../../lib/format'
import { useMediaQuery } from '../../lib/hooks'

/** Side drawer on desktop, bottom sheet on phones. Focus-trapped, Esc to close (Radix Dialog). */
export function Sheet({
  open,
  onOpenChange,
  title,
  description,
  children,
  footer,
  width = 'max-w-md',
}: {
  open: boolean
  onOpenChange: (v: boolean) => void
  title: ReactNode
  description?: ReactNode
  children: ReactNode
  footer?: ReactNode
  width?: string
}) {
  const desktop = useMediaQuery('(min-width: 768px)')
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-[60] bg-ink-950/70 backdrop-blur-[2px] data-[state=open]:animate-float-in" />
        <Dialog.Content
          className={cn(
            'fixed z-[70] flex flex-col border-line-strong bg-ink-900/95 shadow-2xl shadow-black/60 backdrop-blur-xl outline-none',
            desktop
              ? cn('inset-y-0 right-0 w-full border-l', width, 'data-[state=open]:animate-[sheet-in-right_0.28s_cubic-bezier(0.2,0.8,0.2,1)]')
              : 'inset-x-0 bottom-0 max-h-[88dvh] rounded-t-3xl border-t data-[state=open]:animate-[sheet-in-up_0.3s_cubic-bezier(0.2,0.8,0.2,1)]',
          )}
        >
          {!desktop && <div className="mx-auto mt-2.5 h-1.5 w-10 rounded-full bg-white/15" aria-hidden />}
          <div className="flex items-start justify-between gap-3 border-b border-line px-5 pt-4 pb-3">
            <div className="min-w-0">
              <Dialog.Title className="font-display text-2xl font-semibold uppercase tracking-wide text-fg">{title}</Dialog.Title>
              {description ? (
                <Dialog.Description className="mt-0.5 text-sm text-fg-2">{description}</Dialog.Description>
              ) : (
                <Dialog.Description className="sr-only">Panel</Dialog.Description>
              )}
            </div>
            <Dialog.Close className="rounded-xl p-2 text-fg-3 hover:bg-white/5 hover:text-fg" aria-label="Close">
              <X className="size-5" />
            </Dialog.Close>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain scrollbar-thin">{children}</div>
          {footer && <div className="border-t border-line p-3 pb-[max(0.75rem,env(safe-area-inset-bottom))]">{footer}</div>}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}
