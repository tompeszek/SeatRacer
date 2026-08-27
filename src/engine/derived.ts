// Display-level statistics derived from a fit: per-athlete rows for the
// Performance tab, shell class effects, correlation diagnostics, fitted vs
// actual rows, and lineup prediction. Ports the display logic of
// analysis_base.py (_add_side_aware_speed, _add_correlations,
// predict_lineup) without the correlation-based dropping.
import type { Design, FitResult } from './types'
import { boatFractions } from './weights'

export const SUFFIX_POSITIONS: Record<string, string> = {
  'ˢ': 'Starboard',
  'ᵖ': 'Port',
  'ˣ': 'Scull',
  'ᶜ': 'Coxswain',
}

export interface AthleteStat {
  name: string
  suffix: string
  coefficient: number
  lower: number
  upper: number
  /** Seconds per 500m behind the fastest athlete with the same suffix. */
  speedBehind: number
  rank: number
  totalInPosition: number
  /** Appearances in the data. */
  races: number
  maxCorrelation: number
  maxCorrelatedWith: string
  minCorrelation: number
  minCorrelatedWith: string
}

export interface ShellStat {
  shellClass: string
  coefficient: number
  lower: number
  upper: number
}

export interface FittedRow {
  piece: string
  crew: string
  shellClass: string
  km: number
  actual: number
  fitted: number
  delta: number
  weight: number
}

function pearson(a: Float64Array, b: Float64Array): number {
  const n = a.length
  let ma = 0
  let mb = 0
  for (let i = 0; i < n; i++) {
    ma += a[i]
    mb += b[i]
  }
  ma /= n
  mb /= n
  let cov = 0
  let va = 0
  let vb = 0
  for (let i = 0; i < n; i++) {
    const da = a[i] - ma
    const db = b[i] - mb
    cov += da * db
    va += da * da
    vb += db * db
  }
  const denom = Math.sqrt(va * vb)
  return denom === 0 ? 0 : cov / denom
}

export function athleteStats(design: Design, fit: FitResult): AthleteStat[] {
  const { athletes } = design
  const cols = athletes.map((_, c) => {
    const v = new Float64Array(design.x.length)
    for (let r = 0; r < design.x.length; r++) v[r] = design.x[r][c]
    return v
  })
  const races = athletes.map((_, c) => {
    let count = 0
    for (let r = 0; r < design.x.length; r++) if (design.x[r][c] !== 0) count++
    return count
  })

  const corr: number[][] = athletes.map(() => [])
  for (let i = 0; i < athletes.length; i++) {
    for (let j = 0; j < athletes.length; j++) {
      corr[i][j] = i === j ? 1 : pearson(cols[i], cols[j])
    }
  }

  const stats: AthleteStat[] = athletes.map((name, i) => {
    const suffix = SUFFIX_POSITIONS[name[name.length - 1]] ? name[name.length - 1] : ''
    let maxC = -Infinity
    let minC = Infinity
    let maxWith = ''
    let minWith = ''
    for (let j = 0; j < athletes.length; j++) {
      if (j === i) continue
      if (corr[i][j] > maxC) {
        maxC = corr[i][j]
        maxWith = athletes[j]
      }
      if (corr[i][j] < minC) {
        minC = corr[i][j]
        minWith = athletes[j]
      }
    }
    return {
      name,
      suffix,
      coefficient: fit.params[i],
      lower: fit.ciLower[i],
      upper: fit.ciUpper[i],
      speedBehind: 0,
      rank: 0,
      totalInPosition: 0,
      races: races[i],
      maxCorrelation: athletes.length > 1 ? maxC : 0,
      maxCorrelatedWith: maxWith,
      minCorrelation: athletes.length > 1 ? minC : 0,
      minCorrelatedWith: minWith,
    }
  })

  // Speed behind the fastest and rank, within each suffix group.
  const groups = new Map<string, AthleteStat[]>()
  for (const s of stats) {
    const list = groups.get(s.suffix)
    if (list) list.push(s)
    else groups.set(s.suffix, [s])
  }
  for (const list of groups.values()) {
    const fastest = Math.min(...list.map((s) => s.coefficient))
    const sorted = [...list].sort((a, b) => a.coefficient - b.coefficient)
    for (const s of list) {
      s.speedBehind = s.coefficient - fastest
      s.rank = sorted.findIndex((x) => x.coefficient === s.coefficient) + 1
      s.totalInPosition = list.length
    }
  }
  return stats
}

export function shellStats(design: Design, fit: FitResult): ShellStat[] {
  const offset = design.athletes.length
  return design.shellClasses.map((shellClass, i) => ({
    shellClass,
    coefficient: fit.params[offset + i],
    lower: fit.ciLower[offset + i],
    upper: fit.ciUpper[offset + i],
  }))
}

export function fittedRows(design: Design, fit: FitResult): FittedRow[] {
  return design.rows.map((row, r) => ({
    piece: row.piece,
    crew: row.personnel.join('/'),
    shellClass: row.shellClass,
    km: row.km,
    actual: row.timePer500m,
    fitted: fit.fitted[r],
    delta: row.timePer500m - fit.fitted[r],
    weight: row.totalWeight,
  }))
}

/**
 * Predicted pace per 500m for an arbitrary lineup: athlete fractions plus the
 * shell class effect. Piece effects are unknown for a future race, so the
 * prediction is comparative (differences between lineups are meaningful).
 */
export function predictLineup(
  paramMap: Map<string, number>,
  personnel: string[],
  shellClass: string,
  weightStern: number | null,
): number {
  const fractions = boatFractions(personnel, weightStern)
  let pace = paramMap.get(shellClass) ?? 0
  for (const [name, frac] of fractions) {
    pace += (paramMap.get(name) ?? 0) * frac
  }
  return pace
}
