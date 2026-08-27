// Main-thread client for the model worker. Requests of each kind supersede
// earlier requests of the same kind; a stale response is dropped.
import type { FitPayload, FitRequest, FitResponse, SerializableSpec } from '../workers/fit.worker'
import type { AthleteInfluence, TimeSeriesResult } from '../engine/influence'
import type { WeightSettings } from '../engine/types'

type Pending = {
  kind: FitRequest['kind']
  resolve: (payload: unknown) => void
  reject: (err: Error) => void
  onProgress?: (done: number, total: number) => void
}

export class FitClient {
  private worker: Worker
  private nextId = 1
  private latestByKind = new Map<string, number>()
  private pending = new Map<number, Pending>()

  constructor() {
    this.worker = new Worker(new URL('../workers/fit.worker.ts', import.meta.url), {
      type: 'module',
    })
    this.worker.onmessage = (event: MessageEvent<FitResponse>) => {
      const resp = event.data
      const entry = this.pending.get(resp.id)
      if (!entry) return
      if (resp.ok && resp.kind === 'progress') {
        entry.onProgress?.(resp.done, resp.total)
        return
      }
      this.pending.delete(resp.id)
      if (resp.id !== this.latestByKind.get(entry.kind)) return // superseded
      if (resp.ok) entry.resolve(resp.result)
      else entry.reject(new Error(resp.error))
    }
  }

  private request<T>(
    kind: FitRequest['kind'],
    csvText: string,
    settings: WeightSettings,
    spec: SerializableSpec,
    onProgress?: (done: number, total: number) => void,
  ): Promise<T> {
    const id = this.nextId++
    this.latestByKind.set(kind, id)
    const req: FitRequest = { id, kind, csvText, settings, spec }
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, {
        kind,
        resolve: resolve as (p: unknown) => void,
        reject,
        onProgress,
      })
      this.worker.postMessage(req)
    })
  }

  fit(csvText: string, settings: WeightSettings, spec: SerializableSpec): Promise<FitPayload> {
    return this.request<FitPayload>('fit', csvText, settings, spec)
  }

  leaveOneOut(
    csvText: string,
    settings: WeightSettings,
    spec: SerializableSpec,
    onProgress?: (done: number, total: number) => void,
  ): Promise<AthleteInfluence[]> {
    return this.request<AthleteInfluence[]>('loo', csvText, settings, spec, onProgress)
  }

  overTime(
    csvText: string,
    settings: WeightSettings,
    spec: SerializableSpec,
    onProgress?: (done: number, total: number) => void,
  ): Promise<TimeSeriesResult> {
    return this.request<TimeSeriesResult>('time', csvText, settings, spec, onProgress)
  }
}
