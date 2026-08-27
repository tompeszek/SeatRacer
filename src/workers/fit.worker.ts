// Model-fitting worker: prep, design, fit, and derived stats off the main
// thread. One request kind for now; the evaluation worker arrives with the
// Model Lab.
import { parseRaceCsv } from '../engine/parse'
import { prepRows, collectShellClasses, shellClassFromRigging } from '../engine/prep'
import { buildDesign } from '../engine/design'
import { fitModel } from '../engine/model'
import { athleteStats, shellStats, fittedRows } from '../engine/derived'
import type { ModelSpec, WeightSettings } from '../engine/types'

export interface SerializableSpec {
  loss: ModelSpec['loss']
  shrinkage: ModelSpec['shrinkage']
  ergCenters?: Array<[string, number]>
}

export interface FitRequest {
  id: number
  csvText: string
  settings: WeightSettings
  spec: SerializableSpec
}

export interface FitPayload {
  athletes: ReturnType<typeof athleteStats>
  shells: ReturnType<typeof shellStats>
  fitted: ReturnType<typeof fittedRows>
  params: Array<[string, number]>
  dfResid: number
  rowCount: number
  pieceCount: number
  athleteNames: string[]
  shellClasses: string[]
  /** Every shell class present in the file (before filtering). */
  allShellClasses: string[]
}

export type FitResponse =
  | { id: number; ok: true; result: FitPayload }
  | { id: number; ok: false; error: string }

export function runFit(req: FitRequest): FitPayload {
  const raw = parseRaceCsv(req.csvText)
  const allShellClasses = [...new Set(raw.map((r) => shellClassFromRigging(r.rigging)))].sort()
  const rows = prepRows(raw, req.settings)
  if (rows.length === 0) {
    return {
      athletes: [],
      shells: [],
      fitted: [],
      params: [],
      dfResid: NaN,
      rowCount: 0,
      pieceCount: 0,
      athleteNames: [],
      shellClasses: [],
      allShellClasses,
    }
  }
  const design = buildDesign(rows, req.settings)
  const spec: ModelSpec = {
    loss: req.spec.loss,
    shrinkage: req.spec.shrinkage,
    weights: req.settings,
    ergCenters: req.spec.ergCenters ? new Map(req.spec.ergCenters) : undefined,
  }
  const fit = fitModel(design, spec)
  return {
    athletes: athleteStats(design, fit),
    shells: shellStats(design, fit),
    fitted: fittedRows(design, fit),
    params: design.columns.map((c, i) => [c, fit.params[i]]),
    dfResid: fit.dfResid,
    rowCount: rows.length,
    pieceCount: collectShellClasses(rows).length === 0 ? 0 : design.pieces.length,
    athleteNames: design.athletes,
    shellClasses: design.shellClasses,
    allShellClasses,
  }
}

self.onmessage = (event: MessageEvent<FitRequest>) => {
  const req = event.data
  try {
    const result = runFit(req)
    const resp: FitResponse = { id: req.id, ok: true, result }
    self.postMessage(resp)
  } catch (err) {
    const resp: FitResponse = { id: req.id, ok: false, error: String(err) }
    self.postMessage(resp)
  }
}
