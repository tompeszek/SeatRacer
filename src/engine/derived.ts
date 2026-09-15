// Display-level statistics derived from a fit: per-athlete rows for the
// Performance tab, shell class effects, correlation diagnostics, fitted vs
// actual rows, and lineup prediction. Ports the display logic of
// analysis_base.py (_add_side_aware_speed, _add_correlations,
// predict_lineup) without the correlation-based dropping.
import { Matrix, SingularValueDecomposition } from 'ml-matrix'
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
  /**
   * Seconds per 500m behind the fastest athlete in the same comparison
   * group; NaN when the athlete is not comparable to anyone.
   */
  speedBehind: number
  /** Rank within the comparison group; 0 when not comparable. */
  rank: number
  totalInPosition: number
  /**
   * Athletes on the same side whose gaps the data pins down share a group.
   * An athlete alone in their group (never raced against anyone on their
   * side in a way the model can separate) is not comparable: their gap and
   * uncertainty are unknowable, not small.
   */
  group: number
  comparable: boolean
  /** Index into the fit's lumps when this athlete is inseparable from others. */
  lump: number | null
  /** 80% range of plausible ranks within the side, from joint simulation. */
  rankLow: number | null
  rankHigh: number | null
  /** Appearances in the data. */
  races: number
  maxCorrelation: number
  maxCorrelatedWith: string
  /** Other athletes tied (within noise) with maxCorrelatedWith. */
  maxCorrelatedOthers: number
  minCorrelation: number
  minCorrelatedWith: string
}

/**
 * Comparison groups from the design matrix alone. A gap b_i - b_j is
 * estimable iff e_i - e_j lies in the row space of X; with V_k the kept
 * right singular vectors, that is iff the rows of V_k for i and j differ by
 * a vector of squared length 2 (the full length of e_i - e_j). Comparability
 * is transitive, so union-find within each side yields the groups.
 */
export function comparisonGroups(design: Design): number[] {
  return columnGroups(
    design,
    design.athletes.map((_, i) => i),
    (a, b) => design.athletes[a].slice(-1) === design.athletes[b].slice(-1),
  )
}

/** Comparison groups over any set of design columns (see comparisonGroups). */
export function columnGroups(
  design: Design,
  cols: number[],
  eligible: (a: number, b: number) => boolean = () => true,
): number[] {
  const nAthletes = cols.length
  const parent = cols.map((_, i) => i)
  const find = (i: number): number => (parent[i] === i ? i : (parent[i] = find(parent[i])))
  if (nAthletes > 0 && design.x.length > 0) {
    const X = new Matrix(design.x.map((r) => Array.from(r)))
    const svd = new SingularValueDecomposition(X, { autoTranspose: true })
    const s = svd.diagonal
    const V = svd.rightSingularVectors
    const cut = 1e-8 * (s.length ? Math.max(...s) : 0)
    const kept: number[] = []
    s.forEach((v, j) => {
      if (v > cut) kept.push(j)
    })
    const p = cols.map((c) => kept.map((j) => V.get(c, j)))
    for (let i = 0; i < nAthletes; i++) {
      for (let j = i + 1; j < nAthletes; j++) {
        if (!eligible(i, j)) continue
        let d2 = 0
        for (let m = 0; m < kept.length; m++) d2 += (p[i][m] - p[j][m]) ** 2
        if (Math.abs(d2 - 2) < 1e-6) parent[find(i)] = find(j)
      }
    }
  }
  const ids = new Map<number, number>()
  return cols.map((_, i) => {
    const root = find(i)
    if (!ids.has(root)) ids.set(root, ids.size)
    return ids.get(root)!
  })
}

/**
 * Test whether a contrast c'b is estimable: c must lie in the row space of
 * X, i.e. keep its full length when projected onto the kept right singular
 * vectors.
 */
function estimableTest(design: Design): (c: Float64Array) => boolean {
  if (design.x.length === 0) return () => false
  const X = new Matrix(design.x.map((r) => Array.from(r)))
  const svd = new SingularValueDecomposition(X, { autoTranspose: true })
  const s = svd.diagonal
  const V = svd.rightSingularVectors
  const cut = 1e-8 * (s.length ? Math.max(...s) : 0)
  const kept: number[] = []
  s.forEach((v, j) => {
    if (v > cut) kept.push(j)
  })
  const k = design.columns.length
  return (c) => {
    let len2 = 0
    for (let i = 0; i < k; i++) len2 += c[i] * c[i]
    if (len2 === 0) return true
    let proj2 = 0
    for (const j of kept) {
      let acc = 0
      for (let i = 0; i < k; i++) if (c[i] !== 0) acc += c[i] * V.get(i, j)
      proj2 += acc * acc
    }
    return Math.abs(proj2 - len2) < 1e-6 * len2
  }
}

/** The t multiplier the solver used for its intervals (NaN if none). */
function intervalMultiplier(fit: FitResult): number {
  for (let c = 0; c < fit.bse.length; c++) {
    if (fit.bse[c] > 0 && Number.isFinite(fit.ciUpper[c])) {
      return (fit.ciUpper[c] - fit.ciLower[c]) / (2 * fit.bse[c])
    }
  }
  return NaN
}

/**
 * A lump: athletes, coxswains and named shells that never appeared apart,
 * so the data cannot tell them from each other. What it can tell is how
 * the lump as a whole compares, per seat, with the fastest free rower on
 * each side (and the fastest free shell); failing that, with other lumps.
 */
export interface Lump {
  id: number
  /** Column names: athletes with suffix, shells as the bare boat name. */
  members: string[]
  /** Per-seat gap behind the fastest, in coefficient units; NaN if unknown. */
  behind: number
  lower: number
  upper: number
  known: boolean
}

export function lumpStats(design: Design, fit: FitResult): Lump[] {
  const { athletes, shells, x } = design
  const nA = athletes.length
  const shellOffset = nA + design.shellClasses.length
  const k = design.columns.length
  const n = x.length
  const cols = [...athletes.map((_, i) => i), ...shells.map((_, i) => shellOffset + i)]
  const column = (c: number) => Float64Array.from({ length: n }, (_, r) => x[r][c])
  const vec = new Map(cols.map((c) => [c, column(c)]))
  const together = (a: number, b: number) => {
    const va = vec.get(a)!
    const vb = vec.get(b)!
    let ab = 0
    let aa = 0
    let bb = 0
    for (let r = 0; r < n; r++) {
      ab += va[r] * vb[r]
      aa += va[r] ** 2
      bb += vb[r] ** 2
    }
    return aa > 0 && bb > 0 && ab * ab > (1 - 1e-9) * aa * bb
  }
  const parent = new Map(cols.map((c) => [c, c]))
  const find = (c: number): number => {
    const p = parent.get(c)!
    if (p === c) return c
    const root = find(p)
    parent.set(c, root)
    return root
  }
  for (let i = 0; i < cols.length; i++) {
    for (let j = i + 1; j < cols.length; j++) {
      if (together(cols[i], cols[j])) parent.set(find(cols[i]), find(cols[j]))
    }
  }
  const byRoot = new Map<number, number[]>()
  for (const c of cols) {
    const r = find(c)
    if (!byRoot.has(r)) byRoot.set(r, [])
    byRoot.get(r)!.push(c)
  }
  const lumpCols = [...byRoot.values()].filter((m) => m.length > 1)
  if (lumpCols.length === 0) return []
  const inLump = new Set(lumpCols.flat())

  // Free entities: separable athletes and shells outside every lump.
  const groupOf = comparisonGroups(design)
  const freeAthletes = athletes
    .map((_, i) => i)
    .filter((i) => !inLump.has(i) && groupOf.filter((g) => g === groupOf[i]).length > 1)
  const fastestFree = new Map<string, number>()
  for (const i of freeAthletes) {
    const side = athletes[i].slice(-1)
    const best = fastestFree.get(side)
    if (best === undefined || fit.params[i] < fit.params[best]) fastestFree.set(side, i)
  }
  const freeShells = shells.map((_, i) => shellOffset + i).filter((c) => !inLump.has(c))
  const fastestShell = freeShells.length
    ? freeShells.reduce((best, c) => (fit.params[c] < fit.params[best] ? c : best))
    : undefined

  const estimable = estimableTest(design)
  const tMult = intervalMultiplier(fit)
  const halfWidth = (c: Float64Array): number => {
    if (!fit.covHalf || !Number.isFinite(tMult)) return NaN
    const m = fit.covHalf[0].length
    let variance = 0
    for (let j = 0; j < m; j++) {
      let acc = 0
      for (let col = 0; col < k; col++) if (c[col] !== 0) acc += c[col] * fit.covHalf[col][j]
      variance += acc * acc
    }
    return tMult * Math.sqrt(variance)
  }
  const dot = (c: Float64Array) => c.reduce((acc, v, i) => acc + v * fit.params[i], 0)

  // Per-seat value of a lump: athletes weigh 1/n; a shell's whole-boat
  // effect is converted to one seat's share via the crew's mean fraction.
  const valueContrast = (members: number[]) => {
    const ath = members.filter((c) => c < nA)
    const nSeats = ath.length
    const c = new Float64Array(k)
    let fracSum = 0
    let fracCount = 0
    for (const a of ath) {
      c[a] += 1 / nSeats
      for (let r = 0; r < n; r++) {
        if (x[r][a] !== 0) {
          fracSum += x[r][a]
          fracCount++
        }
      }
    }
    const frac = fracCount ? fracSum / fracCount : 1
    for (const s of members) if (s >= nA) c[s] += 1 / (nSeats * frac)
    return { c, nSeats, frac }
  }
  // Baseline: the same seats filled with the fastest free rower per side
  // (and the fastest free shell); null when some seat has no free peer.
  const baselineContrast = (members: number[]): Float64Array | null => {
    const { c, nSeats, frac } = valueContrast(members)
    const out = Float64Array.from(c)
    for (const m of members) {
      if (m < nA) {
        const best = fastestFree.get(athletes[m].slice(-1))
        if (best === undefined) return null
        out[best] -= 1 / nSeats
      } else {
        if (fastestShell === undefined) return null
        out[fastestShell] -= 1 / (nSeats * frac)
      }
    }
    return out
  }

  const lumps: Lump[] = lumpCols.map((members, id) => ({
    id,
    members: members.map((c) => (c < nA ? athletes[c] : shells[c - shellOffset])),
    behind: NaN,
    lower: -Infinity,
    upper: Infinity,
    known: false,
  }))
  const values = lumpCols.map((m) => valueContrast(m).c)
  const baselines = lumpCols.map((m) => {
    const b = baselineContrast(m)
    return b && estimable(b) ? b : null
  })

  // Lumps that can be told from each other form components.
  const lp = lumps.map((_, i) => i)
  const lfind = (i: number): number => (lp[i] === i ? i : (lp[i] = lfind(lp[i])))
  for (let a = 0; a < lumps.length; a++) {
    for (let b = a + 1; b < lumps.length; b++) {
      if (estimable(values[a].map((v, i) => v - values[b][i]))) lp[lfind(a)] = lfind(b)
    }
  }
  const components = new Set(lumps.map((_, i) => lfind(i)))
  for (const root of components) {
    const idx = lumps.map((_, i) => i).filter((i) => lfind(i) === root)
    if (idx.every((i) => baselines[i] != null)) {
      // Everyone measures against the fastest free rowers; the lump that
      // beats them shows as Fastest.
      const raw = idx.map((i) => dot(baselines[i]!))
      const floor = Math.min(0, ...raw)
      idx.forEach((i, j) => {
        const half = halfWidth(baselines[i]!)
        lumps[i].behind = raw[j] - floor
        lumps[i].lower = lumps[i].behind - half
        lumps[i].upper = lumps[i].behind + half
        lumps[i].known = true
      })
    } else if (idx.length > 1) {
      const vals = idx.map((i) => dot(values[i]))
      const fastest = Math.min(...vals)
      idx.forEach((i, j) => {
        const mean = values[i].map((v, col) => v - idx.reduce((acc, o) => acc + values[o][col], 0) / idx.length)
        const half = halfWidth(mean)
        lumps[i].behind = vals[j] - fastest
        lumps[i].lower = lumps[i].behind - half
        lumps[i].upper = lumps[i].behind + half
        lumps[i].known = true
      })
    }
  }
  return lumps
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
  groupOf: number[] = comparisonGroups(design),
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

  const groups = new Map<number, number[]>()
  design.athletes.forEach((_, i) => {
    if (!groups.has(groupOf[i])) groups.set(groupOf[i], [])
    groups.get(groupOf[i])!.push(i)
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
  /**
   * Predicted pace for this class with an average-coefficient rower in every
   * seat (per side; coxswain included only when modeled). Comparable across
   * classes, unlike the bare shell effect, which absorbs a class-dependent
   * share of the crew level.
   */
  averageCrewPace: number
  /**
   * Pieces where this class raced at least one other class. Zero means the
   * class's gap to other classes is not identified by the data at all.
   */
  crossClassPieces: number
}

/** One named boat's effect, when named shells are modeled. */
export interface NamedShellStat {
  shell: string
  coefficient: number
  lower: number
  upper: number
  /** Seconds per 500m behind the fastest comparable shell; NaN if none. */
  behind: number
  races: number
  /** False when the data cannot separate this boat from its crews. */
  comparable: boolean
  /** Index into the fit's lumps when this boat is inseparable from its crew. */
  lump: number | null
}

export function namedShellStats(design: Design, fit: FitResult): NamedShellStat[] {
  const offset = design.athletes.length + design.shellClasses.length
  const cols = design.shells.map((_, i) => offset + i)
  const groupOf = columnGroups(design, cols)
  const stats: NamedShellStat[] = design.shells.map((shell, i) => ({
    shell,
    coefficient: fit.params[offset + i],
    lower: -Infinity,
    upper: Infinity,
    behind: NaN,
    races: design.rows.filter((r) => r.shell === shell).length,
    comparable: groupOf.filter((g) => g === groupOf[i]).length > 1,
    lump: null,
  }))
  const groups = new Set(groupOf)
  for (const g of groups) {
    const members = stats.filter((_, i) => groupOf[i] === g)
    if (members.length < 2) continue
    const fastest = Math.min(...members.map((s) => s.coefficient))
    members.forEach((s) => {
      const i = stats.indexOf(s)
      s.behind = s.coefficient - fastest
      s.lower = fit.ciLower[offset + i]
      s.upper = fit.ciUpper[offset + i]
    })
  }
  return stats
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
    let tied = 0
    for (let j = 0; j < athletes.length; j++) if (j !== i && corr[i][j] > maxC - 1e-9) tied++
    return {
      name,
      suffix,
      coefficient: fit.params[i],
      lower: fit.ciLower[i],
      upper: fit.ciUpper[i],
      speedBehind: NaN,
      rank: 0,
      totalInPosition: 1,
      group: 0,
      comparable: false,
      lump: null,
      rankLow: null,
      rankHigh: null,
      races: races[i],
      maxCorrelation: athletes.length > 1 ? maxC : 0,
      maxCorrelatedWith: maxWith,
      maxCorrelatedOthers: Math.max(0, tied - 1),
      minCorrelation: athletes.length > 1 ? minC : 0,
      minCorrelatedWith: minWith,
    }
  })

  // Speed behind the fastest and rank, within each comparison group. An
  // athlete alone in their group has no estimable gap to anyone: their
  // interval is unbounded and they carry no rank.
  const groupOf = comparisonGroups(design)
  const groups = new Map<number, AthleteStat[]>()
  stats.forEach((s, i) => {
    s.group = groupOf[i]
    const list = groups.get(s.group)
    if (list) list.push(s)
    else groups.set(s.group, [s])
  })
  // Solver noise (1e-13 or so) separates coefficients the data ties; treat
  // differences below this as exact ties so they share a rank.
  const TIE = 1e-9
  // Same t multiplier the solver used for its intervals.
  let tMult = NaN
  for (let c = 0; c < fit.bse.length; c++) {
    if (fit.bse[c] > 0 && Number.isFinite(fit.ciUpper[c])) {
      tMult = (fit.ciUpper[c] - fit.ciLower[c]) / (2 * fit.bse[c])
      break
    }
  }
  for (const list of groups.values()) {
    if (list.length < 2) {
      list[0].lower = -Infinity
      list[0].upper = Infinity
      continue
    }
    const fastest = Math.min(...list.map((s) => s.coefficient))
    const members = list.map((s) => athletes.indexOf(s.name))
    for (const s of list) {
      const behind = s.coefficient - fastest
      s.comparable = true
      s.speedBehind = behind < TIE ? 0 : behind
      s.rank = 1 + list.filter((x) => x.coefficient < s.coefficient - TIE).length
      s.totalInPosition = list.length
      // Interval on the athlete's gap from the group average, which is
      // estimable even though the athlete's own coefficient is not.
      if (fit.covHalf && Number.isFinite(tMult)) {
        const own = athletes.indexOf(s.name)
        const m = fit.covHalf[0].length
        let variance = 0
        for (let j = 0; j < m; j++) {
          let acc = fit.covHalf[own][j]
          for (const idx of members) acc -= fit.covHalf[idx][j] / members.length
          variance += acc * acc
        }
        const half = tMult * Math.sqrt(variance)
        s.lower = s.coefficient - half
        s.upper = s.coefficient + half
      }
    }
  }

  const rankRanges = simulateRankRanges(design, fit, undefined, undefined, undefined, groupOf)
  if (rankRanges) {
    for (const s of stats) {
      const range = rankRanges.get(s.name)
      if (range && s.comparable) {
        s.rankLow = range[0]
        s.rankHigh = range[1]
      }
    }
  }
  return stats
}

export function shellStats(design: Design, fit: FitResult): ShellStat[] {
  const offset = design.athletes.length
  // Mean coefficient per side.
  const sums = new Map<string, { sum: number; n: number }>()
  design.athletes.forEach((name, i) => {
    const suffix = name[name.length - 1]
    if (!sums.has(suffix)) sums.set(suffix, { sum: 0, n: 0 })
    const s = sums.get(suffix)!
    s.sum += fit.params[i]
    s.n++
  })
  const avg = (suffix: string): number => {
    const s = sums.get(suffix)
    if (s && s.n > 0) return s.sum / s.n
    // Fall back to the overall mean if this side has no athletes.
    let total = 0
    let n = 0
    for (const v of sums.values()) {
      total += v.sum
      n += v.n
    }
    return n > 0 ? total / n : 0
  }

  // Pieces where a class races at least one other class: the only source of
  // cross-class identification.
  const pieceClasses = new Map<string, Set<string>>()
  for (const row of design.rows) {
    if (!pieceClasses.has(row.piece)) pieceClasses.set(row.piece, new Set())
    pieceClasses.get(row.piece)!.add(row.shellClass)
  }
  const crossCount = new Map<string, number>()
  for (const classes of pieceClasses.values()) {
    if (classes.size < 2) continue
    for (const cl of classes) crossCount.set(cl, (crossCount.get(cl) ?? 0) + 1)
  }

  return design.shellClasses.map((shellClass, i) => {
    const rowers = parseInt(shellClass, 10) || 0
    const hasCox = shellClass.includes('+')
    const scull = shellClass.includes('x')
    const seats = rowers + (hasCox ? 1 : 0)
    let crew = 0
    if (seats > 0) {
      if (scull) crew += (rowers / seats) * avg('ˣ')
      else crew += ((rowers / 2) / seats) * (avg('ᵖ') + avg('ˢ'))
      // The cox seat contributes only when coxswains are modeled; otherwise
      // its share drops, exactly as in training rows.
      if (hasCox && sums.has('ᶜ')) crew += (1 / seats) * (sums.get('ᶜ')!.sum / sums.get('ᶜ')!.n)
    }
    return {
      shellClass,
      coefficient: fit.params[offset + i],
      lower: fit.ciLower[offset + i],
      upper: fit.ciUpper[offset + i],
      averageCrewPace: fit.params[offset + i] + crew,
      crossClassPieces: crossCount.get(shellClass) ?? 0,
    }
  })
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
