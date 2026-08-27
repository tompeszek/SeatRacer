// Model worker: fitting, derived stats, leave-one-out influence, and
// over-time series, all off the main thread.
import tCdf from '@stdlib/stats-base-dists-t-cdf'
import { parseRaceCsv } from '../engine/parse'
import { prepRows, shellClassFromRigging } from '../engine/prep'
import { buildDesign } from '../engine/design'
import { fitModel } from '../engine/model'
import {
  athleteStats,
  shellStats,
  fittedRows,
  athletePairs,
  biasStats,
  correlationPairs,
  duplicateAthletes,
} from '../engine/derived'
import { leaveOneOut, overTime, type AthleteInfluence, type TimeSeriesResult } from '../engine/influence'
import type { Candidate } from '../engine/evaluate'
import type { Loss, ModelSpec, Shrinkage, WeightSettings } from '../engine/types'

export interface SerializableSpec {
  loss: Loss
  shrinkage: Shrinkage
  ergCenters?: Array<[string, number]>
}

export interface FitRequest {
  id: number
  kind: 'fit' | 'loo' | 'time'
  csvText: string
  settings: WeightSettings
  spec: SerializableSpec
}

export interface FitPayload {
  athletes: ReturnType<typeof athleteStats>
  shells: ReturnType<typeof shellStats>
  fitted: ReturnType<typeof fittedRows>
  pairs: ReturnType<typeof athletePairs>
  bias: ReturnType<typeof biasStats>
  correlations: ReturnType<typeof correlationPairs>
  duplicates: ReturnType<typeof duplicateAthletes>
  params: Array<[string, number]>
  dfResid: number
  rowCount: number
  pieceCount: number
  athleteNames: string[]
  shellClasses: string[]
  allShellClasses: string[]
}

export type FitResponse =
  | { id: number; ok: true; kind: 'fit'; result: FitPayload }
  | { id: number; ok: true; kind: 'loo'; result: AthleteInfluence[] }
  | { id: number; ok: true; kind: 'time'; result: TimeSeriesResult }
  | { id: number; ok: false; error: string }
  | { id: number; ok: true; kind: 'progress'; done: number; total: number }

function toCandidate(spec: SerializableSpec): Candidate {
  return {
    key: 'current',
    label: 'Current',
    loss: spec.loss,
    shrinkage: spec.shrinkage,
    ergCenters: spec.ergCenters ? new Map(spec.ergCenters) : undefined,
  }
}

export function runFit(req: FitRequest): FitPayload {
  const raw = parseRaceCsv(req.csvText)
  const allShellClasses = [...new Set(raw.map((r) => shellClassFromRigging(r.rigging)))].sort()
  const rows = prepRows(raw, req.settings)
  if (rows.length === 0) {
    return {
      athletes: [],
      shells: [],
      fitted: [],
      pairs: [],
      bias: [],
      correlations: [],
      duplicates: [],
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
    pairs: athletePairs(design, fit, tCdf),
    bias: biasStats(design, fit, tCdf),
    correlations: correlationPairs(design),
    duplicates: duplicateAthletes(design),
    params: design.columns.map((c, i) => [c, fit.params[i]]),
    dfResid: fit.dfResid,
    rowCount: rows.length,
    pieceCount: design.pieces.length,
    athleteNames: design.athletes,
    shellClasses: design.shellClasses,
    allShellClasses,
  }
}

self.onmessage = (event: MessageEvent<FitRequest>) => {
  const req = event.data
  const progress = (done: number, total: number) => {
    const resp: FitResponse = { id: req.id, ok: true, kind: 'progress', done, total }
    self.postMessage(resp)
  }
  try {
    if (req.kind === 'fit') {
      const resp: FitResponse = { id: req.id, ok: true, kind: 'fit', result: runFit(req) }
      self.postMessage(resp)
    } else {
      const raw = parseRaceCsv(req.csvText)
      const rows = prepRows(raw, req.settings)
      const candidate = toCandidate(req.spec)
      if (req.kind === 'loo') {
        const result = leaveOneOut(rows, req.settings, candidate, progress)
        const resp: FitResponse = { id: req.id, ok: true, kind: 'loo', result }
        self.postMessage(resp)
      } else {
        const result = overTime(rows, req.settings, candidate, progress)
        const resp: FitResponse = { id: req.id, ok: true, kind: 'time', result }
        self.postMessage(resp)
      }
    }
  } catch (err) {
    const resp: FitResponse = { id: req.id, ok: false, error: String(err) }
    self.postMessage(resp)
  }
}
