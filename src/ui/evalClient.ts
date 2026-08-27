// Main-thread client for the evaluation worker, with progress callbacks.
import type { EvalRequest, EvalResponse, SerializableCandidate } from '../workers/eval.worker'
import type { EvalResult } from '../engine/evaluate'
import type { WeightSettings } from '../engine/types'

export class EvalClient {
  private worker: Worker
  private nextId = 1
  private latestId = 0
  private handlers = new Map<
    number,
    { resolve: (r: EvalResult) => void; reject: (e: Error) => void; onProgress?: (d: number, t: number) => void }
  >()

  constructor() {
    this.worker = new Worker(new URL('../workers/eval.worker.ts', import.meta.url), {
      type: 'module',
    })
    this.worker.onmessage = (event: MessageEvent<EvalResponse>) => {
      const resp = event.data
      const entry = this.handlers.get(resp.id)
      if (!entry || resp.id !== this.latestId) return
      if (resp.type === 'progress') entry.onProgress?.(resp.done, resp.total)
      else if (resp.type === 'done') {
        this.handlers.delete(resp.id)
        entry.resolve(resp.result)
      } else {
        this.handlers.delete(resp.id)
        entry.reject(new Error(resp.error))
      }
    }
  }

  evaluate(
    csvText: string,
    settings: WeightSettings,
    candidates: SerializableCandidate[],
    onProgress?: (done: number, total: number) => void,
  ): Promise<EvalResult> {
    const id = this.nextId++
    this.latestId = id
    const req: EvalRequest = { id, csvText, settings, candidates }
    return new Promise((resolve, reject) => {
      this.handlers.set(id, { resolve, reject, onProgress })
      this.worker.postMessage(req)
    })
  }
}
