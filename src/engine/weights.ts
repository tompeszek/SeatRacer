// Observation weights and per-seat athlete fractions. Ports the formulas in
// analysis_base.py (_apply_weights, _compute_athlete_weights,
// calculate_closest_margin) 1:1, including every clip value.
import type { PreppedRow, WeightSettings } from './types'
import { daysBetween } from './prep'

/** Closest absolute margin to another boat in the same piece (null if alone). */
export function closestMargins(rows: PreppedRow[]): void {
  const byPiece = new Map<string, PreppedRow[]>()
  for (const row of rows) {
    const list = byPiece.get(row.piece)
    if (list) list.push(row)
    else byPiece.set(row.piece, [row])
  }
  for (const list of byPiece.values()) {
    for (const row of list) {
      if (list.length < 2) {
        row.closestMargin = null
        continue
      }
      let best = Infinity
      for (const other of list) {
        if (other === row) continue
        const d = Math.abs(row.timeSeconds - other.timeSeconds)
        if (d < best) best = d
      }
      row.closestMargin = best
    }
  }
}

const MAX_MARGIN = 12

export function applyWeights(rows: PreppedRow[], settings: WeightSettings): void {
  // Close-race weighting: races decided by 1s weighted 2x vs weightClose margin.
  for (const row of rows) {
    if (settings.weightClose != null) {
      const margin = Math.min(row.closestMargin ?? Infinity, MAX_MARGIN)
      const factor = Math.exp((-Math.LN2 * margin) / settings.weightClose)
      row.closenessFactor = Math.max(factor, 0.1)
    } else {
      row.closenessFactor = 1
    }
  }
  // Recency: exponential decay by days before the most recent session,
  // floored at 0.1, rescaled so the oldest kept weight is 1, capped at 10.
  const halflife = settings.halflife
  if (halflife != null && halflife > 0 && rows.length > 0) {
    let latest = rows[0].date
    for (const row of rows) if (row.date > latest) latest = row.date
    let minFactor = Infinity
    const factors = rows.map((row) => {
      const days = daysBetween(latest, row.date)
      const f = Math.max(Math.exp(-days / halflife), 0.1)
      if (f < minFactor) minFactor = f
      return f
    })
    rows.forEach((row, i) => {
      row.recencyFactor = Math.min(factors[i] / minFactor, 10)
    })
  } else {
    for (const row of rows) row.recencyFactor = 1
  }
  for (const row of rows) row.totalWeight = row.closenessFactor * row.recencyFactor
}

/**
 * Per-boat athlete fractions. Even 1/n split, or a stern-biased linear split
 * where the first-listed seat gets the most weight, normalized to sum to 1.
 * Fractions are stored for every seat (including coxswains); the design
 * matrix later includes only the selected athletes' columns.
 */
/** Seat fractions for one boat: even 1/n, or stern-biased linear split. */
export function boatFractions(boat: string[], stern: number | null): Map<string, number> {
  const n = boat.length
  const out = new Map<string, number>()
  if (stern != null && stern !== 0 && n > 1) {
    const base = 1 / n
    const adjustment = base * stern
    const weights = boat.map((_, pos) => {
      const relative = (n - 1 - pos) / Math.max(1, n - 1)
      return base + relative * adjustment - adjustment / 2
    })
    const total = weights.reduce((a, b) => a + b, 0)
    boat.forEach((name, i) => out.set(name, weights[i] / total))
  } else {
    const w = n > 0 ? 1 / n : 0
    for (const name of boat) out.set(name, w)
  }
  return out
}

export function athleteFractions(rows: PreppedRow[], settings: WeightSettings): void {
  for (const row of rows) {
    row.athleteFractions = boatFractions(row.personnel, settings.weightStern)
  }
}
