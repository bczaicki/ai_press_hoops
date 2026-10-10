import { RefreshCw } from 'lucide-react'
import { Component, type ErrorInfo, type ReactNode } from 'react'

import { ErrorPanel } from './ui/misc'

/** Keeps one broken screen from blanking the whole app; resets when the route changes. */
export class ErrorBoundary extends Component<{ children: ReactNode; resetKey?: string }, { error: Error | null }> {
  state: { error: Error | null } = { error: null }

  static getDerivedStateFromError(error: Error) {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('UI error', error, info.componentStack)
  }

  componentDidUpdate(prev: { resetKey?: string }) {
    if (prev.resetKey !== this.props.resetKey && this.state.error) this.setState({ error: null })
  }

  render() {
    if (!this.state.error) return this.props.children
    return (
      <div className="mx-auto max-w-xl py-10">
        <ErrorPanel
          title="This screen hit a snag"
          message={`${this.state.error.message || 'Unexpected error'}. Your data is safe on the server.`}
          action={
            <button type="button" onClick={() => window.location.reload()} className="inline-flex items-center gap-1.5 text-sm font-semibold text-flame-300 hover:text-flame-200">
              <RefreshCw className="size-4" aria-hidden /> Reload the page
            </button>
          }
        />
      </div>
    )
  }
}
