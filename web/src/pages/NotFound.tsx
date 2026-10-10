import { House } from 'lucide-react'

import { ButtonLink } from '../components/ui/button'

export function NotFound() {
  return (
    <div className="grid place-items-center py-20 text-center">
      <div className="numeral text-[7rem] text-flame-500/90">404</div>
      <h1 className="mt-2 font-display text-3xl font-bold tracking-wide text-fg uppercase">Airball</h1>
      <p className="mt-2 max-w-sm text-fg-2">That page isn't on the court. It may have moved, or the link was cut short.</p>
      <ButtonLink to="/" variant="primary" className="mt-6" icon={<House className="size-4" />}>
        Back home
      </ButtonLink>
    </div>
  )
}
