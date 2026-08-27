// Walk-forward evaluation worker (Model Lab).
import { parseRaceCsv } from '../engine/parse'
import { walkForward, type Candidate, type EvalResult } from '../engine/evaluate'
import type { Loss, Shrinkage, WeightSettings } from '../engine/types'

export interface SerializableCandidate {
  key: string
  label: string
  loss: Loss
  shrinkage: Shrinkage
  ergCenters?: Array<[string, number]>
}

export interface EvalRequest {
  id: number
  csvText: string
  settings: WeightSettings
  candidates: SerializableCandidate[]
}

export type EvalResponse =
  | { id: number; type: 'progress'; done: number; total: number }
  | { id: number; type: 'done'; result: EvalResult }
  | { id: number; type: 'error'; error: string }

self.onmessage = (event: MessageEvent<EvalRequest>) => {
  const req = event.data
  try {
    const raw = parseRaceCsv(req.csvText)
    const candidates: Candidate[] = req.candidates.map((c) => ({
      ...c,
      ergCenters: c.ergCenters ? new Map(c.ergCenters) : undefined,
    }))
    let lastReport = 0
    const result = walkForward(raw, req.settings, candidates, (done, total) => {
      const now = Date.now()
      if (now - lastReport > 150 || done === total) {
        lastReport = now
        const resp: EvalResponse = { id: req.id, type: 'progress', done, total }
        self.postMessage(resp)
      }
    })
    const resp: EvalResponse = { id: req.id, type: 'done', result }
    self.postMessage(resp)
  } catch (err) {
    const resp: EvalResponse = { id: req.id, type: 'error', error: String(err) }
    self.postMessage(resp)
  }
}
