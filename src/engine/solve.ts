// Weighted least squares via SVD pseudoinverse, matching statsmodels OLS/WLS
// (method="pinv") numerically: the design is rank-deficient by construction
// (piece and shell dummies each sum to the ones vector), and statsmodels
// resolves that with the minimum-norm solution, so we must too.
import { Matrix, SingularValueDecomposition } from 'ml-matrix'
import tQuantile from '@stdlib/stats-base-dists-t-quantile'
import normalQuantile from '@stdlib/stats-base-dists-normal-quantile'

export interface LstsqResult {
  params: Float64Array
  /** Diagonal of the unscaled parameter covariance, pinv(X) pinv(X)'. */
  normalizedCovDiag: Float64Array
  /** V and 1/s (zeroed beyond cutoff) kept for ridge and reuse. */
  rank: number
  singularValues: Float64Array
  fittedWhitened: Float64Array
  residWhitened: Float64Array
}

/**
 * Relative singular-value cutoff. The seat-racing design matrix has exact
 * linear dependencies (piece, shell, and athlete columns each sum to the ones
 * vector), whose singular values surface as implementation-dependent numerical
 * noise (~1e-12 relative and below); the smallest genuine singular values sit
 * around 1e-3 relative. 1e-8 splits that ten-order-of-magnitude gap so rank
 * and the minimum-norm solution are deterministic across SVD implementations
 * (statsmodels' 1e-15 cutoff sits inside the noise band, which makes its rank
 * wobble by one between numpy call sites).
 */
const RCOND = 1e-8

/**
 * Minimum-norm least squares of (X, y), both already whitened.
 */
export function lstsqPinv(xRows: Float64Array[], y: Float64Array): LstsqResult {
  const n = xRows.length
  const k = xRows[0].length
  const X = new Matrix(xRows.map((r) => Array.from(r)))
  const svd = new SingularValueDecomposition(X, { autoTranspose: true })
  const s = svd.diagonal
  const U = svd.leftSingularVectors
  const V = svd.rightSingularVectors
  const sMax = s.length ? Math.max(...s) : 0

  const pinvCut = RCOND * sMax
  let rank = 0
  for (const v of s) if (v > pinvCut) rank++

  // params = V * diag(sInv) * U' * y
  const m = s.length
  const uty = new Float64Array(m)
  for (let j = 0; j < m; j++) {
    let acc = 0
    for (let i = 0; i < n; i++) acc += U.get(i, j) * y[i]
    uty[j] = s[j] > pinvCut ? acc / s[j] : 0
  }
  const params = new Float64Array(k)
  for (let c = 0; c < k; c++) {
    let acc = 0
    for (let j = 0; j < m; j++) acc += V.get(c, j) * uty[j]
    params[c] = acc
  }

  // normalized covariance diagonal: sum_j V[c,j]^2 / s_j^2 over kept j
  const normalizedCovDiag = new Float64Array(k)
  for (let c = 0; c < k; c++) {
    let acc = 0
    for (let j = 0; j < m; j++) {
      if (s[j] > pinvCut) {
        const t = V.get(c, j) / s[j]
        acc += t * t
      }
    }
    normalizedCovDiag[c] = acc
  }

  const fittedWhitened = new Float64Array(n)
  const residWhitened = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let acc = 0
    for (let c = 0; c < k; c++) acc += xRows[i][c] * params[c]
    fittedWhitened[i] = acc
    residWhitened[i] = y[i] - acc
  }

  return {
    params,
    normalizedCovDiag,
    rank,
    singularValues: Float64Array.from(s),
    fittedWhitened,
    residWhitened,
  }
}

export function whiten(
  xRows: Float64Array[],
  y: Float64Array,
  w: Float64Array,
): { xw: Float64Array[]; yw: Float64Array } {
  const xw = xRows.map((row, i) => {
    const sw = Math.sqrt(w[i])
    const out = new Float64Array(row.length)
    for (let c = 0; c < row.length; c++) out[c] = row[c] * sw
    return out
  })
  const yw = new Float64Array(y.length)
  for (let i = 0; i < y.length; i++) yw[i] = y[i] * Math.sqrt(w[i])
  return { xw, yw }
}

export function allCloseToOne(w: Float64Array): boolean {
  for (const v of w) if (Math.abs(v - 1) > 1e-8 + 1e-5) return false
  return true
}

export interface WlsFit {
  params: Float64Array
  bse: Float64Array
  ciLower: Float64Array
  ciUpper: Float64Array
  dfResid: number
  rank: number
  scale: number
  fitted: Float64Array
  lstsq: LstsqResult
}

/**
 * OLS/WLS fit with t-based confidence intervals, statsmodels-equivalent.
 * `weights` of all ones is plain OLS.
 */
export function fitWls(
  xRows: Float64Array[],
  y: Float64Array,
  weights: Float64Array,
  alpha = 0.05,
): WlsFit {
  const useWeights = !allCloseToOne(weights)
  const { xw, yw } = useWeights
    ? whiten(xRows, y, weights)
    : { xw: xRows, yw: y }
  const ls = lstsqPinv(xw, yw)
  const n = xRows.length
  const dfResid = n - ls.rank
  let ssr = 0
  for (const r of ls.residWhitened) ssr += r * r
  const scale = ssr / dfResid
  const k = ls.params.length
  const bse = new Float64Array(k)
  const ciLower = new Float64Array(k)
  const ciUpper = new Float64Array(k)
  const q = tQuantile(1 - alpha / 2, dfResid)
  for (let c = 0; c < k; c++) {
    bse[c] = Math.sqrt(ls.normalizedCovDiag[c] * scale)
    ciLower[c] = ls.params[c] - q * bse[c]
    ciUpper[c] = ls.params[c] + q * bse[c]
  }
  const fitted = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let acc = 0
    for (let c = 0; c < k; c++) acc += xRows[i][c] * ls.params[c]
    fitted[i] = acc
  }
  return { params: ls.params, bse, ciLower, ciUpper, dfResid, rank: ls.rank, scale, fitted, lstsq: ls }
}

/**
 * Gaussian GLM with frequency weights, statsmodels-equivalent: same
 * coefficients as WLS, but df_resid = sum(weights) - rank and normal-based
 * intervals. Kept for parity testing against the old app's default model.
 */
export function fitGlmGaussian(
  xRows: Float64Array[],
  y: Float64Array,
  weights: Float64Array,
  alpha = 0.05,
): WlsFit {
  const { xw, yw } = whiten(xRows, y, weights)
  const ls = lstsqPinv(xw, yw)
  const n = xRows.length
  let wsum = 0
  for (const v of weights) wsum += v
  const dfResid = wsum - ls.rank
  let ssr = 0
  for (const r of ls.residWhitened) ssr += r * r
  const scale = ssr / dfResid
  const k = ls.params.length
  const bse = new Float64Array(k)
  const ciLower = new Float64Array(k)
  const ciUpper = new Float64Array(k)
  const q = normalQuantile(1 - alpha / 2, 0, 1)
  for (let c = 0; c < k; c++) {
    bse[c] = Math.sqrt(ls.normalizedCovDiag[c] * scale)
    ciLower[c] = ls.params[c] - q * bse[c]
    ciUpper[c] = ls.params[c] + q * bse[c]
  }
  const fitted = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    let acc = 0
    for (let c = 0; c < k; c++) acc += xRows[i][c] * ls.params[c]
    fitted[i] = acc
  }
  return { params: ls.params, bse, ciLower, ciUpper, dfResid, rank: ls.rank, scale, fitted, lstsq: ls }
}
