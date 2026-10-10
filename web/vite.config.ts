import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The FastAPI backend (hoops/server.py) listens on 127.0.0.1:8000 and serves web/dist in production.
// In dev, Vite proxies the API and media routes so cookies stay first-party.
const backend = process.env.HOOPS_API_URL ?? 'http://127.0.0.1:8000'
const proxy = {
  '/api': { target: backend, changeOrigin: false },
  '/media': { target: backend, changeOrigin: false },
}

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { proxy },
  preview: { proxy },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    chunkSizeWarningLimit: 900,
  },
})
