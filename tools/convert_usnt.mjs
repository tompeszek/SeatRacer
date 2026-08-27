// Converts the USNT 2018-2021 pairwise-margin data into the app's schema
// (Race Session (date), Piece, KM, Rigging, Personnel, Result).
//
// The source records winner/loser margins per piece, not boat times, and
// carries no side information. This script:
//  1. Normalizes names (C./E Carlson variants, Ezra = E Carlson, M/B = MB).
//  2. Assigns each rower one sweep side: seeded from the actual rigging in
//     the 2012/2021 datasets (ground truth), the rest chosen to minimize
//     boats that violate the half-port half-starboard quota. Flagg and Best
//     are known side-switchers and take whichever side balances each boat.
//  3. Reconstructs per-boat times per piece by least squares over the margin
//     constraints (margins can be over-determined and mildly inconsistent),
//     anchored at a class-typical base time over an assumed 2 km piece.
//
// Run: node tools/convert_usnt.mjs
import { readFileSync, writeFileSync } from 'node:fs'

const SOURCE = 'tools/USNT 2018-2021 - Sheet1.csv'
const OUT = 'public/data/USNT 2018-2021.csv'
const KM = 2
const BASE_TIME = { Pair: 420, Four: 380, Eight: 330 } // seconds for the fastest boat
const SWITCHERS = new Set(['Flagg', 'Best'])
// Ground truth from datasets with real rigging, plus user-confirmed sides.
const FIXED = {
  Richards: 's', Weiss: 'p', Wallis: 'p', DiSanto: 'p', Peszek: 's', Reed: 's',
  Karwoski: 's', Harrity: 's', Mead: 'p', Dethlefs: 'p', Ochal: 'p', Hack: 'p',
  Corrigan: 'p', Dean: 'p', Carlson: 's', Rummel: 's', Davison: 's', Grady: 's',
  Miklasevich: 's', 'C Carlson': 's', 'E Carlson': 's',
}
const RENAME = { 'C. Carlson': 'C Carlson', Ezra: 'E Carlson', 'M/B': 'MB' }

const parseLine = (line) => {
  const out = []
  let field = ''
  let quoted = false
  for (const ch of line) {
    if (quoted) {
      if (ch === '"') quoted = false
      else field += ch
    } else if (ch === '"') quoted = true
    else if (ch === ',') {
      out.push(field)
      field = ''
    } else field += ch
  }
  out.push(field)
  return out
}

const rows = readFileSync(SOURCE, 'utf8').trim().split(/\r?\n/).slice(1).map(parseLine)
const crewOf = (s) => s.split(',').map((n) => RENAME[n.trim()] ?? n.trim())

// ---- Side assignment ----------------------------------------------------
const boats = []
const seenBoats = new Set()
const names = new Set()
for (const r of rows) {
  for (const crew of [crewOf(r[5]), crewOf(r[6])]) {
    crew.forEach((n) => names.add(n))
    const key = crew.slice().sort().join('|')
    if (!seenBoats.has(key)) {
      seenBoats.add(key)
      boats.push(crew)
    }
  }
}
const free = [...names].filter((n) => !FIXED[n] && !SWITCHERS.has(n)).sort()

function cost(asg) {
  let total = 0
  for (const boat of boats) {
    const half = boat.length / 2
    let ps = 0
    let ss = 0
    let wild = 0
    for (const n of boat) {
      if (SWITCHERS.has(n)) wild++
      else if (asg[n] === 'p') ps++
      else ss++
    }
    const needP = half - ps
    const needS = half - ss
    if (needP < 0) total += -needP
    if (needS < 0) total += -needS
    if (needP >= 0 && needS >= 0 && needP + needS !== wild) total += Math.abs(needP + needS - wild)
  }
  return total
}

let seed = 20260827
const rand = () => {
  seed = (seed * 1103515245 + 12345) >>> 0
  return seed / 4294967296
}
let best = null
let bestCost = Infinity
for (let restart = 0; restart < 80; restart++) {
  const asg = { ...FIXED }
  for (const n of free) asg[n] = rand() < 0.5 ? 'p' : 's'
  let improved = true
  while (improved) {
    improved = false
    for (const n of free) {
      const before = cost(asg)
      asg[n] = asg[n] === 'p' ? 's' : 'p'
      if (cost(asg) < before) improved = true
      else asg[n] = asg[n] === 'p' ? 's' : 'p'
    }
  }
  const c = cost(asg)
  if (c < bestCost) {
    bestCost = c
    best = { ...asg }
  }
}
console.log(`side assignment: residual imbalance ${bestCost} over ${boats.length} boats`)

/** Sides for one boat: fixed rowers keep their side, switchers balance it. */
function boatSides(crew) {
  const half = crew.length / 2
  const sides = new Map()
  let ps = 0
  for (const n of crew) {
    if (!SWITCHERS.has(n)) {
      sides.set(n, best[n])
      if (best[n] === 'p') ps++
    }
  }
  for (const n of crew) {
    if (SWITCHERS.has(n)) {
      const side = ps < half ? 'p' : 's'
      sides.set(n, side)
      if (side === 'p') ps++
    }
  }
  return sides
}

// ---- Time reconstruction per piece --------------------------------------
// A date can run Fours and Pairs under the same piece number; those are
// separate races, so pieces are keyed by class as well and renumbered
// uniquely within each date on output.
const pieces = new Map() // key -> {date, cls, n, constraints, crews}
for (const r of rows) {
  const [date, cls, , piece, margin, winners, losers] = r
  const w = crewOf(winners)
  const l = crewOf(losers)
  const key = `${piece}|${cls}`
  if (!pieces.has(key))
    pieces.set(key, { date, cls, n: Number(piece.split(':')[1]), constraints: [], crews: new Map() })
  const p = pieces.get(key)
  const wKey = w.slice().sort().join('|')
  const lKey = l.slice().sort().join('|')
  p.crews.set(wKey, w)
  p.crews.set(lKey, l)
  p.constraints.push([wKey, lKey, Number(margin)])
}

function solveTimes(piece) {
  const keys = [...piece.crews.keys()]
  const index = new Map(keys.map((k, i) => [k, i]))
  const n = keys.length
  // Least squares for t with constraints t_l - t_w = m, ridge-anchored at 0
  // to pin the (per-connected-component) free level.
  const A = Array.from({ length: n }, () => new Float64Array(n))
  const b = new Float64Array(n)
  for (let i = 0; i < n; i++) A[i][i] = 1e-6
  for (const [wKey, lKey, m] of piece.constraints) {
    const w = index.get(wKey)
    const l = index.get(lKey)
    A[l][l] += 1
    A[w][w] += 1
    A[l][w] -= 1
    A[w][l] -= 1
    b[l] += m
    b[w] -= m
  }
  // Gaussian elimination (n is tiny).
  const M = A.map((row, i) => [...row, b[i]])
  for (let col = 0; col < n; col++) {
    let pivot = col
    for (let r2 = col + 1; r2 < n; r2++) if (Math.abs(M[r2][col]) > Math.abs(M[pivot][col])) pivot = r2
    ;[M[col], M[pivot]] = [M[pivot], M[col]]
    for (let r2 = 0; r2 < n; r2++) {
      if (r2 === col || M[col][col] === 0) continue
      const f = M[r2][col] / M[col][col]
      for (let c2 = col; c2 <= n; c2++) M[r2][c2] -= f * M[col][c2]
    }
  }
  const t = keys.map((_, i) => M[i][n] / M[i][i])
  // Shift connected components so each component's fastest boat sits at the
  // class base time. Components found via the constraint graph.
  const parent = keys.map((_, i) => i)
  const find = (x) => (parent[x] === x ? x : (parent[x] = find(parent[x])))
  for (const [wKey, lKey] of piece.constraints) {
    const a = find(index.get(wKey))
    const c = find(index.get(lKey))
    if (a !== c) parent[a] = c
  }
  const minByComp = new Map()
  keys.forEach((k, i) => {
    const c = find(i)
    minByComp.set(c, Math.min(minByComp.get(c) ?? Infinity, t[i]))
  })
  const base = BASE_TIME[piece.cls] ?? 400
  return new Map(keys.map((k, i) => [k, t[i] - minByComp.get(find(i)) + base]))
}

// ---- Emit ---------------------------------------------------------------
const fmtDate = (iso) => {
  const [y, m, d] = iso.split('-').map(Number)
  return `${m}/${d}/${y}`
}
const fmtTime = (s) => {
  const minutes = Math.floor(s / 60)
  const rest = s - minutes * 60
  const restStr = rest < 10 ? `0${rest.toFixed(1)}` : rest.toFixed(1)
  return `${String(minutes).padStart(2, '0')}:${restStr}`
}

const out = ['Race Session (date),Piece,KM,Rigging,Personnel,Result']
let emitted = 0
const sortedPieces = [...pieces.values()].sort((a, b) =>
  a.date !== b.date ? (a.date < b.date ? -1 : 1) : a.cls !== b.cls ? (a.cls < b.cls ? -1 : 1) : a.n - b.n,
)
const pieceCounters = new Map()
for (const piece of sortedPieces) {
  const count = (pieceCounters.get(piece.date) ?? 0) + 1
  pieceCounters.set(piece.date, count)
  const pieceNumber = count
  const times = solveTimes(piece)
  for (const [key, crew] of piece.crews) {
    const sides = boatSides(crew)
    // Alternate sides stroke-to-bow where possible for a realistic pattern.
    const ports = crew.filter((n) => sides.get(n) === 'p')
    const stars = crew.filter((n) => sides.get(n) === 's')
    const ordered = []
    while (ports.length || stars.length) {
      if (ports.length) ordered.push(ports.shift())
      if (stars.length) ordered.push(stars.shift())
    }
    const rigging = ordered.map((n) => sides.get(n)).join('/')
    const personnel = ordered.join('/')
    out.push(
      `${fmtDate(piece.date)},${pieceNumber},${KM},${rigging},${personnel},${fmtTime(times.get(key))}`,
    )
    emitted++
  }
}
writeFileSync(OUT, out.join('\n') + '\n', 'utf8')
console.log(`wrote ${OUT}: ${emitted} boat rows across ${pieces.size} pieces`)
