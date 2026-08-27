// Robust regression. Two layers:
//  - fitHuberRlm: a statsmodels-RLM-equivalent Huber fit (fixture-verified).
//  - irlsLp: Lp(p) fitting by iteratively reweighted least squares, used for
//    the user-selectable Lp loss (property-tested; no Python counterpart).
import normalQuantile from '@stdlib/stats-base-dists-normal-quantile'
import { lstsqPinv, whiten } from './solve'

const MAD_C = 0.6744897501960817 // Phi^{-1}(3/4), statsmodels scale.mad default

function median(values: number[]): number {
  if (values.length === 0) return NaN
  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
}

/** Median absolute deviation about zero, scaled to estimate sigma. */
export function madAboutZero(resid: ArrayLike<number>): number {
  return median(Array.from(resid, Math.abs)) / MAD_C
}

export interface HuberOptions {
  t?: number
  maxiter?: number
  tol?: number
}

export interface RobustFit {
  params: Float64Array
  bse: Float64Array
  ciLower: Float64Array
  ciUpper: Float64Array
  scale: number
  iterations: number
  fitted: Float64Array
  dfResid: number
  /** H1 covariance ingredients, exposed so tests can recompute the factor at
   * a different df (statsmodels' rank is numerically flaky by one). */
  h1: { ssPsi: number; mSum: number; varPsiPrime: number; n: number }
  normalizedCovDiag: Float64Array
}

/**
 * Huber M-estimation matching statsmodels RLM(y, X).fit() defaults:
 * HuberT(t=1.345), MAD scale about zero re-estimated each iteration,
 * deviance convergence (tol 1e-8, maxiter 50), H1 covariance, normal CIs.
 * Observation weights are applied by whitening first (the whitened problem
 * is then the RLM problem); pass all-ones weights for exact RLM parity.
 */
export function fitHuberRlm(
  xRows: Float64Array[],
  y: Float64Array,
  obsWeights?: Float64Array,
  options: HuberOptions = {},
  alpha = 0.05,
): RobustFit {
  const t = options.t ?? 1.345
  const maxiter = options.maxiter ?? 50
  const tol = options.tol ?? 1e-8

  let X = xRows
  let yy = y
  if (obsWeights && !obsWeights.every((v) => v === 1)) {
    const w = whiten(xRows, y, obsWeights)
    X = w.xw
    yy = w.yw
  }
  const n = X.length
  const kCols = X[0].length

  const psi = (z: number) => (Math.abs(z) <= t ? z : t * Math.sign(z))
  const psiDeriv = (z: number) => (Math.abs(z) <= t ? 1 : 0)
  const rho = (z: number) =>
    Math.abs(z) <= t ? 0.5 * z * z : Math.abs(z) * t - 0.5 * t * t
  const huberWeight = (z: number) => (Math.abs(z) <= t ? 1 : t / Math.abs(z))

  const residOf = (params: Float64Array): Float64Array => {
    const r = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      let acc = 0
      for (let c = 0; c < kCols; c++) acc += X[i][c] * params[c]
      r[i] = yy[i] - acc
    }
    return r
  }

  // statsmodels MinimalWLS "scale": wresid'wresid / (n - ncols), used only
  // inside the deviance convergence criterion.
  const mwlsScale = (wresid: Float64Array) => {
    let ssr = 0
    for (const v of wresid) ssr += v * v
    return ssr / (n - kCols)
  }
  const devianceOf = (resid: Float64Array, scale: number) => {
    let acc = 0
    for (const r of resid) acc += rho(r / scale)
    return acc
  }

  // Initial unweighted fit.
  let ls = lstsqPinv(X, yy)
  let resid = residOf(ls.params)
  let scale = madAboutZero(resid)
  const deviances: number[] = [Infinity, devianceOf(resid, mwlsScale(ls.residWhitened))]

  let iteration = 1
  let converged = false
  while (!converged) {
    if (scale === 0) break
    const irlsW = new Float64Array(n)
    for (let i = 0; i < n; i++) irlsW[i] = huberWeight(resid[i] / scale)
    const w = whiten(X, yy, irlsW)
    ls = lstsqPinv(w.xw, w.yw)
    resid = residOf(ls.params)
    scale = madAboutZero(resid)
    deviances.push(devianceOf(resid, mwlsScale(ls.residWhitened)))
    iteration += 1
    const delta = Math.abs(deviances[iteration] - deviances[iteration - 1])
    converged = !(delta > tol && iteration < maxiter)
  }

  // H1 covariance on the (whitened) design, following RLMResults.bcov_scaled.
  const base = lstsqPinv(X, yy) // unweighted pinv of X for normalized_cov_params
  const rank = base.rank
  const dfResid = n - rank
  const dfModel = rank - 1
  const sresid = new Float64Array(n)
  for (let i = 0; i < n; i++) sresid[i] = resid[i] / scale
  let mSum = 0
  let ssPsi = 0
  for (let i = 0; i < n; i++) {
    mSum += psiDeriv(sresid[i])
    ssPsi += psi(sresid[i]) ** 2
  }
  const m = mSum / n
  let varAcc = 0
  for (let i = 0; i < n; i++) varAcc += (psiDeriv(sresid[i]) - m) ** 2
  const varPsiPrime = varAcc / n
  const kCorr = 1 + ((dfModel + 1) / n) * (varPsiPrime / (m * m))
  const factor = (kCorr * kCorr * ((1 / dfResid) * ssPsi * scale * scale)) / ((mSum / n) ** 2)

  const k = ls.params.length
  const bse = new Float64Array(k)
  const ciLower = new Float64Array(k)
  const ciUpper = new Float64Array(k)
  const q = normalQuantile(1 - alpha / 2, 0, 1)
  for (let c = 0; c < k; c++) {
    bse[c] = Math.sqrt(base.normalizedCovDiag[c] * factor)
    ciLower[c] = ls.params[c] - q * bse[c]
    ciUpper[c] = ls.params[c] + q * bse[c]
  }
  const fitted = new Float64Array(xRows.length)
  for (let i = 0; i < xRows.length; i++) {
    let acc = 0
    for (let c = 0; c < k; c++) acc += xRows[i][c] * ls.params[c]
    fitted[i] = acc
  }
  return {
    params: ls.params,
    bse,
    ciLower,
    ciUpper,
    scale,
    iterations: iteration,
    fitted,
    dfResid,
    h1: { ssPsi, mSum, varPsiPrime, n },
    normalizedCovDiag: base.normalizedCovDiag,
  }
}

/**
 * Lp regression (minimize sum of |resid|^p) by IRLS, deterministically
 * initialized from the squared-loss solution. For p < 2 the IRLS weight is
 * |r|^(p-2), clamped so near-zero residuals cannot produce infinite weights.
 * p = 2 reduces to least squares; p = 1 approximates median regression.
 */
export function irlsLp(
  xRows: Float64Array[],
  y: Float64Array,
  obsWeights: Float64Array,
  p: number,
  maxiter = 100,
  tol = 1e-10,
): { params: Float64Array; iterations: number; fitted: Float64Array } {
  const { xw, yw } = whiten(xRows, y, obsWeights)
  const n = xw.length
  const k = xw[0].length
  let ls = lstsqPinv(xw, yw)
  if (p === 2) {
    const fitted = new Float64Array(xRows.length)
    for (let i = 0; i < xRows.length; i++) {
      let acc = 0
      for (let c = 0; c < k; c++) acc += xRows[i][c] * ls.params[c]
      fitted[i] = acc
    }
    return { params: ls.params, iterations: 1, fitted }
  }
  let prev = ls.params
  let iterations = 0
  // Residual floor: prevents infinite IRLS weights at interpolated points.
  const scale0 = Math.max(madAboutZero(ls.residWhitened), 1e-8)
  const floor = 1e-6 * scale0
  for (let it = 0; it < maxiter; it++) {
    iterations = it + 1
    const resid = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      let acc = 0
      for (let c = 0; c < k; c++) acc += xw[i][c] * prev[c]
      resid[i] = yw[i] - acc
    }
    const irlsW = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      const r = Math.max(Math.abs(resid[i]), floor)
      irlsW[i] = Math.pow(r, p - 2)
    }
    const w = whiten(xw, yw, irlsW)
    ls = lstsqPinv(w.xw, w.yw)
    let delta = 0
    for (let c = 0; c < k; c++) delta = Math.max(delta, Math.abs(ls.params[c] - prev[c]))
    prev = ls.params
    if (delta < tol) break
  }
  const fitted = new Float64Array(xRows.length)
  for (let i = 0; i < xRows.length; i++) {
    let acc = 0
    for (let c = 0; c < k; c++) acc += xRows[i][c] * ls.params[c]
    fitted[i] = acc
  }
  return { params: ls.params, iterations, fitted }
}
