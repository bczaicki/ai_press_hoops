import { createHttpApi, type Api } from './client'
import { createMockApi } from './mock'

export * from './client'
export type * from './types'

/**
 * Mock mode: VITE_MOCK=1 at build/dev time, or ?mock=1 in the URL (remembered for this tab).
 * ?mock=0 switches back to the real backend.
 */
function detectMock(): boolean {
  if (import.meta.env.VITE_MOCK === '1') return true
  try {
    const p = new URLSearchParams(window.location.search)
    if (p.get('mock') === '0') {
      sessionStorage.removeItem('hoops_mock')
      return false
    }
    if (p.has('mock')) {
      sessionStorage.setItem('hoops_mock', '1')
      return true
    }
    return sessionStorage.getItem('hoops_mock') === '1'
  } catch {
    return false
  }
}

export const isMock = detectMock()
export const api: Api = isMock ? createMockApi() : createHttpApi()

/** Media URLs from the API are same-origin paths ("/media/..."); mock ones use the "mock://" scheme. */
export function isMockMedia(url: string | null | undefined): boolean {
  return !!url && url.startsWith('mock://')
}

/** RunSummary has no poster URL; posters live at a fixed path per clip (docs/api.md ClipView.urls). */
export function posterForClip(clipId: string): string {
  return `/media/clips/${encodeURIComponent(clipId)}/poster.jpg`
}
