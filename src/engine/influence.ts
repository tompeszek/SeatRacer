// Leave-one-out influence and coefficients-over-time, both built on the fast
// normal-equations fit used by the walk-forward evaluator.
import type { PreppedRow, WeightSettings } from './types'
import { fitCandidate, type Candidate } from './evaluate'
import { parseDate } from './prep'

export interface InfluenceEntry {
  piece: string
  /** Change in the athlete's coefficient when the piece is removed (seconds
   * per 500m; negative means the piece was making the athlete look slower). */
  delta: number
}

export interface AthleteInfluence {
  name: string
  entries: InfluenceEntry[]
}

/** Refit without each piece; report how every athlete's estimate moves. */
export function leaveOneOut(
  rows: PreppedRow[],
  settings: WeightSettings,
  candidate: Candidate,
  onProgress?: (done: number, total: number) => void,
): AthleteInfluence[] {
  const full = fitCandidate(rows, settings, candidate)
  const pieces = [...new Set(rows.map((r) => r.piece))]
  const athletes = [...full.columns.keys()].filter(
    (c) => !c.startsWith('Piece_') && rows.some((r) => r.athleteFractions.has(c)),
  )
  const byAthlete = new Map<string, InfluenceEntry[]>(athletes.map((a) => [a, []]))
  pieces.forEach((piece, i) => {
    const subset = rows.filter((r) => r.piece !== piece)
    const fit = fitCandidate(subset, settings, candidate)
    for (const athlete of athletes) {
      const fullIdx = full.columns.get(athlete)
      const subIdx = fit.columns.get(athlete)
      if (fullIdx === undefined || subIdx === undefined) continue
      byAthlete.get(athlete)!.push({ piece, delta: fit.coef[subIdx] - full.coef[fullIdx] })
    }
    onProgress?.(i + 1, pieces.length)
  })
  return athletes.map((name) => ({
    name,
    entries: byAthlete.get(name)!.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta)),
  }))
}

export interface TimeSeriesResult {
  dates: string[]
  series: Array<{ name: string; suffix: string; values: Array<number | null> }>
}

/**
 * Fit on all data up to each session date and report each athlete's standing
 * relative to the mean of their side at that date (absolute coefficients are
 * not comparable across fits, relative standing is).
 */
export function overTime(
  rows: PreppedRow[],
  settings: WeightSettings,
  candidate: Candidate,
  onProgress?: (done: number, total: number) => void,
): TimeSeriesResult {
  const dates = [...new Set(rows.map((r) => r.dateRaw))].sort(
    (a, b) => parseDate(a).getTime() - parseDate(b).getTime(),
  )
  const athletes = new Map<string, string>()
  for (const row of rows) {
    for (const name of row.athleteFractions.keys()) {
      if (!settings.includeCoxswains && name.endsWith('ᶜ')) continue
      if (!athletes.has(name)) athletes.set(name, name[name.length - 1])
    }
  }
  const values = new Map<string, Array<number | null>>(
    [...athletes.keys()].map((a) => [a, dates.map(() => null)]),
  )

  dates.forEach((date, di) => {
    const cutoff = parseDate(date).getTime()
    let window = rows.filter((r) => r.date.getTime() <= cutoff)
    if (window.length === 0) return
    if (settings.halflife != null && settings.halflife > 0) {
      let latest = window[0].date
      for (const r of window) if (r.date > latest) latest = r.date
      let minFactor = Infinity
      const factors = window.map((r) => {
        const days = Math.round((latest.getTime() - r.date.getTime()) / 86_400_000)
        const f = Math.max(Math.exp(-days / settings.halflife!), 0.1)
        if (f < minFactor) minFactor = f
        return f
      })
      window = window.map((r, i) => ({
        ...r,
        totalWeight: r.closenessFactor * Math.min(factors[i] / minFactor, 10),
      }))
    }
    const fit = fitCandidate(window, settings, candidate)
    // Side means at this date, over athletes present in the window.
    const bySide = new Map<string, number[]>()
    for (const [name, suffix] of athletes) {
      const idx = fit.columns.get(name)
      if (idx === undefined) continue
      if (!bySide.has(suffix)) bySide.set(suffix, [])
      bySide.get(suffix)!.push(fit.coef[idx])
    }
    const sideMean = new Map<string, number>()
    for (const [suffix, list] of bySide) {
      sideMean.set(suffix, list.reduce((s, v) => s + v, 0) / list.length)
    }
    for (const [name, suffix] of athletes) {
      const idx = fit.columns.get(name)
      if (idx === undefined) continue
      values.get(name)![di] = fit.coef[idx] - (sideMean.get(suffix) ?? 0)
    }
    onProgress?.(di + 1, dates.length)
  })

  return {
    dates,
    series: [...athletes.entries()].map(([name, suffix]) => ({
      name,
      suffix,
      values: values.get(name)!,
    })),
  }
}
