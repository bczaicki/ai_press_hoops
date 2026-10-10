import { createContext, useContext, type ReactNode } from 'react'

import { api } from '../api'
import type { AppConfig } from '../api/types'
import { useAsync } from '../lib/hooks'

interface ConfigState {
  config: AppConfig | undefined
  error: string | null
  loading: boolean
  reload: () => void
}

const Ctx = createContext<ConfigState>({ config: undefined, error: null, loading: true, reload: () => {} })

export function ConfigProvider({ children }: { children: ReactNode }) {
  const { data, error, loading, reload } = useAsync(() => api.config(), [])
  return <Ctx.Provider value={{ config: data, error, loading, reload }}>{children}</Ctx.Provider>
}

export function useConfig(): ConfigState {
  return useContext(Ctx)
}
