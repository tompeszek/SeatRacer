// Walk-forward evaluation ("Model Lab"): for each piece past a warm-up, fit
// each candidate model on all strictly earlier pieces and score predicted
// within-piece margins against the actual ones. Errors are reported split by
// horizon (same-day vs future-day). See REWRITE_PLAN.md section 5.
//
// Fits here use normal equations with a Cholesky solve plus a tiny diagonal
// jitter (the fast equivalent of the SVD minimum-norm solution; the jitter
// plays the role of the pseudoinverse cutoff). Hundreds of refits make the
// SVD path too slow for this loop.
import type { Loss, PreppedRow, Shrinkage, WeightSettings } from './types'
import { prepRows } from './prep'
import type { RaceRow } from './types'
import { boatFractions } from './weights'
import { madAboutZero } from './robust'

export interface Candidate {
  key: string
  label: string
  loss: Loss
  shrinkage: Shrinkage
  ergCenters?: Map<string, number>
}

export interface HorizonStats {
  pairs: number
  medianAbsError: number
  correctPickRate: number
}

export interface PieceMiss {
  piece: string
  meanAbsError: number
  pairs: number
}

export interface CandidateResult {
  key: string
  label: string
  sameDay: HorizonStats
  futureDay: HorizonStats
  excludedPairs: number
  worstPieces: PieceMiss[]
}

export interface EvalResult {
  candidates: CandidateResult[]
  testedPieces: number
  totalPieces: number
  /** Set when the test window was capped (train data still uses everything). */
  cappedAt: number | null
}

const WARMUP_PIECES = 6
const MAX_TEST_PIECES = 80

interface Solved {
  coef: Float64Array
  columns: Map<string, number>
}

/** Cholesky solve of (A + jitter I) x = b; A must be symmetric PSD. */
function choleskySolve(a: Float64Array[], b: Float64Array, jitter: number): Float64Array {
  const n = b.length
  const L = a.map((row) => Float64Array.from(row))
  for (let i = 0; i < n; i++) L[i][i] += jitter
  for (let j = 0; j < n; j++) {
    let d = L[j][j]
    for (let k = 0; k < j; k++) d -= L[j][k] * L[j][k]
    const diag = Math.sqrt(Math.max(d, jitter * 1e-3))
    L[j][j] = diag
    for (let i = j + 1; i < n; i++) {
      let v = L[i][j]
      for (let k = 0; k < j; k++) v -= L[i][k] * L[j][k]
      L[i][j] = v / diag
    }
  }
  // forward then backward substitution
  const y = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let v = b[i]
    for (let k = 0; k < i; k++) v -= L[i][k] * y[k]
    y[i] = v / L[i][i]
  }
  const x = new Float64Array(n)
  for (let i = n - 1; i >= 0; i--) {
    let v = y[i]
    for (let k = i + 1; k < n; k++) v -= L[k][i] * x[k]
    x[i] = v / L[i][i]
  }
  return x
}

/**
 * Fit one candidate on the given rows via weighted normal equations, with
 * IRLS on top for Huber and Lp losses. Returns coefficients keyed by column.
 */
export function fitCandidate(
  rows: PreppedRow[],
  settings: WeightSettings,
  candidate: Candidate,
): Solved {
  // Active columns: athletes, shells, pieces present in the training rows.
  const columns = new Map<string, number>()
  const athletes: string[] = []
  for (const row of rows) {
    for (const name of row.athleteFractions.keys()) {
      if (!settings.includeCoxswains && name.endsWith('ᶜ')) continue
      if (!columns.has(name)) {
        columns.set(name, columns.size)
        athletes.push(name)
      }
    }
  }
  const athleteCount = columns.size
  for (const row of rows) {
    if (!columns.has(row.shellClass)) columns.set(row.shellClass, columns.size)
  }
  for (const row of rows) {
    const key = `Piece_${row.piece}`
    if (!columns.has(key)) columns.set(key, columns.size)
  }
  const k = columns.size
  const n = rows.length

  const xRow = (row: PreppedRow): Float64Array => {
    const line = new Float64Array(k)
    for (const [name, frac] of row.athleteFractions) {
      const c = columns.get(name)
      if (c !== undefined) line[c] = frac
    }
    line[columns.get(row.shellClass)!] = 1
    line[columns.get(`Piece_${row.piece}`)!] = 1
    return line
  }
  const X = rows.map(xRow)
  const y = Float64Array.from(rows, (r) => r.timePer500m)
  const baseW = Float64Array.from(rows, (r) => r.totalWeight)

  const lambda = candidate.shrinkage.kind === 'ridge' ? candidate.shrinkage.lambda : 0
  const centers = new Float64Array(k)
  if (candidate.shrinkage.kind === 'ridge' && candidate.shrinkage.center === 'erg') {
    const provided = candidate.ergCenters ? [...candidate.ergCenters.values()] : []
    const fallback = provided.length ? provided.reduce((a, b) => a + b, 0) / provided.length : 0
    for (let c = 0; c < athleteCount; c++) {
      centers[c] = candidate.ergCenters?.get(athletes[c]) ?? fallback
    }
  }

  const solveWeighted = (w: Float64Array): Float64Array => {
    const A: Float64Array[] = Array.from({ length: k }, () => new Float64Array(k))
    const b = new Float64Array(k)
    for (let i = 0; i < n; i++) {
      const row = X[i]
      const wi = w[i]
      for (let c1 = 0; c1 < k; c1++) {
        const v1 = row[c1]
        if (v1 === 0) continue
        b[c1] += wi * v1 * y[i]
        const scaled = wi * v1
        for (let c2 = c1; c2 < k; c2++) {
          if (row[c2] !== 0) A[c1][c2] += scaled * row[c2]
        }
      }
    }
    for (let c1 = 0; c1 < k; c1++) for (let c2 = 0; c2 < c1; c2++) A[c1][c2] = A[c2][c1]
    let trace = 0
    for (let c = 0; c < k; c++) trace += A[c][c]
    const jitter = Math.max(trace / k, 1) * 1e-10
    if (lambda > 0) {
      for (let c = 0; c < athleteCount; c++) {
        A[c][c] += lambda
        b[c] += lambda * centers[c]
      }
    }
    return choleskySolve(A, b, jitter)
  }

  let coef = solveWeighted(baseW)
  if (candidate.loss.kind !== 'squared') {
    const resid = new Float64Array(n)
    for (let iter = 0; iter < 30; iter++) {
      for (let i = 0; i < n; i++) {
        let acc = 0
        const row = X[i]
        for (let c = 0; c < k; c++) if (row[c] !== 0) acc += row[c] * coef[c]
        resid[i] = y[i] - acc
      }
      const scale = Math.max(madAboutZero(resid), 1e-8)
      const w = new Float64Array(n)
      if (candidate.loss.kind === 'huber') {
        const t = candidate.loss.c
        for (let i = 0; i < n; i++) {
          const z = Math.abs(resid[i]) / scale
          w[i] = baseW[i] * (z <= t ? 1 : t / z)
        }
      } else {
        const p = candidate.loss.p
        const floor = 1e-6 * scale
        for (let i = 0; i < n; i++) {
          w[i] = baseW[i] * Math.pow(Math.max(Math.abs(resid[i]), floor), p - 2)
        }
      }
      const next = solveWeighted(w)
      let delta = 0
      for (let c = 0; c < k; c++) delta = Math.max(delta, Math.abs(next[c] - coef[c]))
      coef = next
      if (delta < 1e-7) break
    }
  }
  return { coef, columns }
}

function median(values: number[]): number {
  if (values.length === 0) return NaN
  const s = [...values].sort((a, b) => a - b)
  const m = Math.floor(s.length / 2)
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2
}

export function walkForward(
  raw: RaceRow[],
  settings: WeightSettings,
  candidates: Candidate[],
  onProgress?: (done: number, total: number) => void,
): EvalResult {
  // Prep once with neutral recency (recency is re-derived per training window
  // below, because "most recent" changes as the window grows).
  const allRows = prepRows(raw, { ...settings, halflife: null })
  const pieceOrder: string[] = []
  const rowsByPiece = new Map<string, PreppedRow[]>()
  for (const row of allRows) {
    if (!rowsByPiece.has(row.piece)) {
      rowsByPiece.set(row.piece, [])
      pieceOrder.push(row.piece)
    }
    rowsByPiece.get(row.piece)!.push(row)
  }
  const totalPieces = pieceOrder.length
  const testStart = Math.max(WARMUP_PIECES, totalPieces - MAX_TEST_PIECES)
  const cappedAt = totalPieces - WARMUP_PIECES > MAX_TEST_PIECES ? MAX_TEST_PIECES : null

  type Acc = {
    same: { errors: number[]; correct: number; total: number }
    future: { errors: number[]; correct: number; total: number }
    excluded: number
    pieceMisses: Map<string, { sum: number; pairs: number }>
  }
  const acc = new Map<string, Acc>(
    candidates.map((c) => [
      c.key,
      {
        same: { errors: [], correct: 0, total: 0 },
        future: { errors: [], correct: 0, total: 0 },
        excluded: 0,
        pieceMisses: new Map(),
      },
    ]),
  )

  const totalWork = (totalPieces - testStart) * candidates.length
  let done = 0

  for (let kPiece = testStart; kPiece < totalPieces; kPiece++) {
    const testPiece = pieceOrder[kPiece]
    const testRows = rowsByPiece.get(testPiece)!
    if (testRows.length < 2) {
      done += candidates.length
      continue
    }
    const trainRows = pieceOrder.slice(0, kPiece).flatMap((p) => rowsByPiece.get(p)!)
    // Recompute recency weights relative to the training window's latest date.
    let windowRows = trainRows
    if (settings.halflife != null && settings.halflife > 0) {
      let latest = trainRows[0].date
      for (const r of trainRows) if (r.date > latest) latest = r.date
      let minFactor = Infinity
      const factors = trainRows.map((r) => {
        const days = Math.round((latest.getTime() - r.date.getTime()) / 86_400_000)
        const f = Math.max(Math.exp(-days / settings.halflife!), 0.1)
        if (f < minFactor) minFactor = f
        return f
      })
      windowRows = trainRows.map((r, i) => ({
        ...r,
        recencyFactor: Math.min(factors[i] / minFactor, 10),
        totalWeight: r.closenessFactor * Math.min(factors[i] / minFactor, 10),
      }))
    }
    const sameDay = trainRows.some((r) => r.dateRaw === testRows[0].dateRaw)

    for (const candidate of candidates) {
      const solved = fitCandidate(windowRows, settings, candidate)
      const a = acc.get(candidate.key)!
      const ridge = candidate.shrinkage.kind === 'ridge'

      // Predict each boat's pace without the piece effect.
      const preds: Array<number | null> = testRows.map((row) => {
        const fractions = boatFractions(row.personnel, settings.weightStern)
        let pace = 0
        const shellIdx = solved.columns.get(row.shellClass)
        if (shellIdx === undefined) return null // shell class never seen
        pace += solved.coef[shellIdx]
        for (const [name, frac] of fractions) {
          if (!settings.includeCoxswains && name.endsWith('ᶜ')) continue
          const idx = solved.columns.get(name)
          if (idx === undefined) {
            // Unseen athlete: ridge models predict at the prior center
            // (their own prior mean); unpenalized models cannot predict.
            if (!ridge) return null
            if (candidate.shrinkage.kind === 'ridge' && candidate.shrinkage.center === 'erg') {
              const provided = candidate.ergCenters ? [...candidate.ergCenters.values()] : []
              const fallback = provided.length
                ? provided.reduce((s, v) => s + v, 0) / provided.length
                : 0
              pace += (candidate.ergCenters?.get(name) ?? fallback) * frac
            }
            continue
          }
          pace += solved.coef[idx] * frac
        }
        return pace
      })

      const bucket = sameDay ? a.same : a.future
      for (let i = 0; i < testRows.length; i++) {
        for (let j = i + 1; j < testRows.length; j++) {
          if (preds[i] == null || preds[j] == null) {
            a.excluded++
            continue
          }
          const predDiff = preds[i]! - preds[j]!
          const actualDiff = testRows[i].timePer500m - testRows[j].timePer500m
          const err = Math.abs(predDiff - actualDiff)
          bucket.errors.push(err)
          bucket.total++
          if (actualDiff !== 0 && Math.sign(predDiff) === Math.sign(actualDiff)) bucket.correct++
          const pm = a.pieceMisses.get(testPiece) ?? { sum: 0, pairs: 0 }
          pm.sum += err
          pm.pairs++
          a.pieceMisses.set(testPiece, pm)
        }
      }
      done++
      onProgress?.(done, totalWork)
    }
  }

  const results: CandidateResult[] = candidates.map((c) => {
    const a = acc.get(c.key)!
    const stats = (b: Acc['same']): HorizonStats => ({
      pairs: b.total,
      medianAbsError: median(b.errors),
      correctPickRate: b.total > 0 ? b.correct / b.total : NaN,
    })
    const worst = [...a.pieceMisses.entries()]
      .map(([piece, { sum, pairs }]) => ({ piece, meanAbsError: sum / pairs, pairs }))
      .sort((x, y) => y.meanAbsError - x.meanAbsError)
      .slice(0, 5)
    return {
      key: c.key,
      label: c.label,
      sameDay: stats(a.same),
      futureDay: stats(a.future),
      excludedPairs: a.excluded,
      worstPieces: worst,
    }
  })

  return {
    candidates: results,
    testedPieces: totalPieces - testStart,
    totalPieces,
    cappedAt,
  }
}
