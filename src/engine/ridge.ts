// Ridge regression with an arbitrary center: minimize
//   sum_i w_i (y_i - x_i b)^2 + lambda * sum_{j in penalized} (b_j - c_j)^2
// Only athlete columns are penalized; shell class and piece columns are free.
// Solved as an augmented least-squares system through the same SVD
// pseudoinverse as the unpenalized fit, so lambda -> 0 recovers it exactly.
//
// Intervals are Bayesian-style credible intervals from the posterior
// covariance sigma^2 (X'WX + P)^+, with sigma^2 = weighted SSR over
// (n - effective degrees of freedom), edf = trace((X'WX + P)^+ X'WX).
import tQuantile from '@stdlib/stats-base-dists-t-quantile'
import { lstsqPinv, whiten } from './solve'

export interface RidgeFit {
  params: Float64Array
  bse: Float64Array
  ciLower: Float64Array
  ciUpper: Float64Array
  scale: number
  edf: number
  fitted: Float64Array
}

export function fitRidge(
  xRows: Float64Array[],
  y: Float64Array,
  obsWeights: Float64Array,
  lambda: number,
  penalized: boolean[],
  centers: Float64Array,
  alpha = 0.05,
): RidgeFit {
  const n = xRows.length
  const k = xRows[0].length
  const { xw, yw } = whiten(xRows, y, obsWeights)

  // Augmented system: one extra row per penalized column.
  const sqrtL = Math.sqrt(lambda)
  const augX: Float64Array[] = [...xw]
  const augYparts: number[] = Array.from(yw)
  if (lambda > 0) {
    for (let c = 0; c < k; c++) {
      if (!penalized[c]) continue
      const row = new Float64Array(k)
      row[c] = sqrtL
      augX.push(row)
      augYparts.push(sqrtL * centers[c])
    }
  }
  const augY = Float64Array.from(augYparts)
  const ls = lstsqPinv(augX, augY)

  const fitted = new Float64Array(n)
  let ssr = 0
  for (let i = 0; i < n; i++) {
    let acc = 0
    for (let c = 0; c < k; c++) acc += xRows[i][c] * ls.params[c]
    fitted[i] = acc
    const r = y[i] - acc
    ssr += obsWeights[i] * r * r
  }

  // Effective degrees of freedom and posterior covariance diagonal:
  // edf = tr(M A) and cov = M, where A = Xw'Xw and M = (A + P)^+, both
  // available from the augmented SVD: M = pinv(augX'augX), and
  // tr(M A) = tr(M (augA - P)) = k_kept - lambda * sum_j penalized M_jj.
  // The augmented lstsq already gives M's diagonal (normalizedCovDiag) and
  // its rank; the trace term needs only the penalized diagonal entries.
  let penalizedCovSum = 0
  for (let c = 0; c < k; c++) if (penalized[c]) penalizedCovSum += ls.normalizedCovDiag[c]
  const edf = lambda > 0 ? ls.rank - lambda * penalizedCovSum : ls.rank
  const dfResid = Math.max(n - edf, 1)
  const scale = ssr / dfResid

  const bse = new Float64Array(k)
  const ciLower = new Float64Array(k)
  const ciUpper = new Float64Array(k)
  const q = tQuantile(1 - alpha / 2, dfResid)
  for (let c = 0; c < k; c++) {
    bse[c] = Math.sqrt(ls.normalizedCovDiag[c] * scale)
    ciLower[c] = ls.params[c] - q * bse[c]
    ciUpper[c] = ls.params[c] + q * bse[c]
  }
  return { params: ls.params, bse, ciLower, ciUpper, scale, edf, fitted }
}
