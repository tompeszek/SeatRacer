import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: { port: 8088 },
  test: {
    include: ['src/**/*.test.ts'],
    environment: 'node',
  },
} as Parameters<typeof defineConfig>[0])
