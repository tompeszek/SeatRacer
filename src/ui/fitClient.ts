// Main-thread client for the fit worker. Later requests supersede earlier
// ones: a stale response is dropped, so the UI always shows the newest fit.
import type { FitPayload, FitRequest, FitResponse, SerializableSpec } from '../workers/fit.worker'
import type { WeightSettings } from '../engine/types'

type Pending = {
  resolve: (payload: FitPayload) => void
  reject: (err: Error) => void
}

export class FitClient {
  private worker: Worker
  private nextId = 1
  private latestId = 0
  private pending = new Map<number, Pending>()

  constructor() {
    this.worker = new Worker(new URL('../workers/fit.worker.ts', import.meta.url), {
      type: 'module',
    })
    this.worker.onmessage = (event: MessageEvent<FitResponse>) => {
      const resp = event.data
      const entry = this.pending.get(resp.id)
      this.pending.delete(resp.id)
      if (!entry) return
      if (resp.id !== this.latestId) return // superseded
      if (resp.ok) entry.resolve(resp.result)
      else entry.reject(new Error(resp.error))
    }
  }

  fit(csvText: string, settings: WeightSettings, spec: SerializableSpec): Promise<FitPayload> {
    const id = this.nextId++
    this.latestId = id
    const req: FitRequest = { id, csvText, settings, spec }
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject })
      this.worker.postMessage(req)
    })
  }
}
