import * as Tooltip from '@radix-ui/react-tooltip'
import type { ReactNode } from 'react'

export const TipProvider = ({ children }: { children: ReactNode }) => (
  <Tooltip.Provider delayDuration={180} skipDelayDuration={300}>
    {children}
  </Tooltip.Provider>
)

/** Accessible tooltip (hover + keyboard focus). The child must be focusable or wrap one. */
export function Tip({
  content,
  children,
  side = 'top',
  disabled,
}: {
  content: ReactNode
  children: ReactNode
  side?: 'top' | 'bottom' | 'left' | 'right'
  disabled?: boolean
}) {
  if (disabled || !content) return <>{children}</>
  return (
    <Tooltip.Root>
      <Tooltip.Trigger asChild>{children}</Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Content
          side={side}
          sideOffset={6}
          collisionPadding={12}
          className="z-[80] max-w-72 rounded-lg border border-line-strong bg-ink-800/95 px-3 py-2 text-[0.8rem] leading-snug text-fg shadow-xl shadow-black/50 backdrop-blur-md data-[state=delayed-open]:animate-float-in"
        >
          {content}
          <Tooltip.Arrow className="fill-ink-800" />
        </Tooltip.Content>
      </Tooltip.Portal>
    </Tooltip.Root>
  )
}
