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
  /** 80% range of plausible ranks within the side, from joint simulation. */
  rankLow: number | null
  rankHigh: number | null
  /** Appearances in the data. */
  races: number
  maxCorrelation: number
  maxCorrelatedWith: string
  minCorrelation: number
  minCorrelatedWith: string
}

/** Deterministic PRNG (mulberry32) for reproducible simulations. */
function makeRng(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/**
 * 80% rank ranges by simulation from the joint coefficient distribution.
 * Correlations between athletes' estimates are respected: each draw perturbs
 * every coefficient together via the covariance square root, then ranks
 * athletes within their side. Deterministic (fixed seed).
 */
export function simulateRankRanges(
  design: Design,
  fit: FitResult,
  draws = 1000,
  alpha = 0.2,
  seed = 20260827,
): Map<string, [number, number]> | null {
  const covHalf = fit.covHalf
  if (!covHalf || covHalf.length === 0) return null
  const nAthletes = design.athletes.length
  const m = covHalf[0].length
  const rng = makeRng(seed)
  // Box-Muller with a spare.
  let spare: number | null = null
  const normal = () => {
    if (spare != null) {
      const v = spare
      spare = null
      return v
    }
    let u = 0
    let v = 0
    while (u === 0) u = rng()
    v = rng()
    const r = Math.sqrt(-2 * Math.log(u))
    spare = r * Math.sin(2 * Math.PI * v)
    return r * Math.cos(2 * Math.PI * v)
  }

  const groups = new Map<string, number[]>()
  design.athletes.forEach((name, i) => {
    const suffix = name[name.length - 1]
    if (!groups.has(suffix)) groups.set(suffix, [])
    groups.get(suffix)!.push(i)
  })

  const ranks: Int16Array[] = design.athletes.map(() => new Int16Array(draws))
  const z = new Float64Array(m)
  const drawCoef = new Float64Array(nAthletes)
  for (let d = 0; d < draws; d++) {
    for (let j = 0; j < m; j++) z[j] = normal()
    for (let c = 0; c < nAthletes; c++) {
      let delta = 0
      const row = covHalf[c]
      for (let j = 0; j < m; j++) delta += row[j] * z[j]
      drawCoef[c] = fit.params[c] + delta
    }
    for (const members of groups.values()) {
      const order = [...members].sort((a, b) => drawCoef[a] - drawCoef[b])
      order.forEach((athleteIdx, pos) => {
        ranks[athleteIdx][d] = pos + 1
      })
    }
  }

  const out = new Map<string, [number, number]>()
  const loIdx = Math.floor((alpha / 2) * (draws - 1))
  const hiIdx = Math.ceil((1 - alpha / 2) * (draws - 1))
  design.athletes.forEach((name, i) => {
    const sorted = [...ranks[i]].sort((a, b) => a - b)
    out.set(name, [sorted[loIdx], sorted[hiIdx]])
  })
  return out
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
      rankLow: null,
      rankHigh: null,
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

  const rankRanges = simulateRankRanges(design, fit)
  if (rankRanges) {
    for (const s of stats) {
      const range = rankRanges.get(s.name)
      if (range) {
        s.rankLow = range[0]
        s.rankHigh = range[1]
      }
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

export interface PairStat {
  a: string
  b: string
  avgDelta: number
  races: number
  tStat: number
  pValue: number
}

/**
 * Athlete pairs' joint performance vs the model: mean residual of boats
 * containing both, with a one-sample t-test. Negative means faster together
 * than the model expects (synergy). Ports _create_athlete_pairs_df.
 */
export function athletePairs(
  design: Design,
  fit: FitResult,
  tCdf: (x: number, df: number) => number,
): PairStat[] {
  const resid = design.rows.map((row, r) => row.timePer500m - fit.fitted[r])
  const byPair = new Map<string, { deltas: number[]; a: string; b: string }>()
  design.rows.forEach((row, r) => {
    const names = row.personnel.filter((n) => design.athletes.includes(n))
    for (let i = 0; i < names.length; i++) {
      for (let j = i + 1; j < names.length; j++) {
        const [a, b] = [names[i], names[j]].sort()
        const key = `${a}|${b}`
        if (!byPair.has(key)) byPair.set(key, { deltas: [], a, b })
        byPair.get(key)!.deltas.push(resid[r])
      }
    }
  })
  const out: PairStat[] = []
  for (const { deltas, a, b } of byPair.values()) {
    if (deltas.length < 2) continue
    const n = deltas.length
    const mean = deltas.reduce((s, v) => s + v, 0) / n
    const sd = Math.sqrt(deltas.reduce((s, v) => s + (v - mean) ** 2, 0) / n)
    const tStat = sd > 0 ? mean / (sd / Math.sqrt(n)) : Infinity
    const pValue = sd > 0 ? 2 * (1 - tCdf(Math.abs(tStat), n - 1)) : 0
    out.push({ a, b, avgDelta: mean, races: n, tStat, pValue })
  }
  return out.sort((x, y) => x.avgDelta - y.avgDelta)
}

export interface BiasStat {
  name: string
  suffix: string
  avgDelta: number
  sd: number
  races: number
  pValue: number
  significant: boolean
}

/**
 * Per-athlete prediction bias: mean residual (actual minus model) across the
 * athlete's boats. Negative means their boats go faster than the model
 * predicts. Ports the Fairness tab's analysis with the sign stated correctly.
 */
export function biasStats(
  design: Design,
  fit: FitResult,
  tCdf: (x: number, df: number) => number,
): BiasStat[] {
  const byAthlete = new Map<string, number[]>()
  design.rows.forEach((row, r) => {
    const delta = row.timePer500m - fit.fitted[r]
    for (const name of row.personnel) {
      if (!design.athletes.includes(name)) continue
      if (!byAthlete.has(name)) byAthlete.set(name, [])
      byAthlete.get(name)!.push(delta)
    }
  })
  const out: BiasStat[] = []
  for (const [name, deltas] of byAthlete) {
    if (deltas.length < 2) continue
    const n = deltas.length
    const mean = deltas.reduce((s, v) => s + v, 0) / n
    const sd = Math.sqrt(deltas.reduce((s, v) => s + (v - mean) ** 2, 0) / n)
    const tStat = sd > 0 ? mean / (sd / Math.sqrt(n)) : Infinity
    const pValue = sd > 0 ? 2 * (1 - tCdf(Math.abs(tStat), n - 1)) : 0
    const suffix = SUFFIX_POSITIONS[name[name.length - 1]] ? name[name.length - 1] : ''
    out.push({ name, suffix, avgDelta: mean, sd, races: n, pValue, significant: pValue < 0.05 })
  }
  return out.sort((a, b) => a.avgDelta - b.avgDelta)
}

export interface CorrelationPair {
  a: string
  b: string
  correlation: number
  racesTogether: number
}

/**
 * Design-column correlations between athletes. High positive correlation
 * means the data cannot separate the two athletes' contributions.
 */
export function correlationPairs(design: Design): CorrelationPair[] {
  const cols = design.athletes.map((_, c) => {
    const v = new Float64Array(design.x.length)
    for (let r = 0; r < design.x.length; r++) v[r] = design.x[r][c]
    return v
  })
  const out: CorrelationPair[] = []
  for (let i = 0; i < design.athletes.length; i++) {
    for (let j = i + 1; j < design.athletes.length; j++) {
      let together = 0
      for (let r = 0; r < design.x.length; r++) {
        if (cols[i][r] !== 0 && cols[j][r] !== 0) together++
      }
      out.push({
        a: design.athletes[i],
        b: design.athletes[j],
        correlation: pearson(cols[i], cols[j]),
        racesTogether: together,
      })
    }
  }
  return out.sort((x, y) => y.correlation - x.correlation)
}

export interface DuplicateEntry {
  piece: string
  athlete: string
  boats: number
}

/** Athletes appearing in more than one boat within the same piece. */
export function duplicateAthletes(design: Design): DuplicateEntry[] {
  const byPiece = new Map<string, Map<string, number>>()
  for (const row of design.rows) {
    if (!byPiece.has(row.piece)) byPiece.set(row.piece, new Map())
    const seen = byPiece.get(row.piece)!
    for (const name of row.personnel) {
      if (name === 'Coxᶜ') continue
      seen.set(name, (seen.get(name) ?? 0) + 1)
    }
  }
  const out: DuplicateEntry[] = []
  for (const [piece, seen] of byPiece) {
    for (const [athlete, boats] of seen) {
      if (boats > 1) out.push({ piece, athlete, boats })
    }
  }
  return out
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
