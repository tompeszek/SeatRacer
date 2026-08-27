// One entrypoint for every loss x shrinkage combination.
import type { Design, FitResult, ModelSpec } from './types'
import { fitWls } from './solve'
import { fitHuberRlm, irlsLp, madAboutZero } from './robust'
import { fitRidge } from './ridge'
import { stripRigging } from './prep'

function centersFor(design: Design, spec: ModelSpec): { penalized: boolean[]; centers: Float64Array } {
  const k = design.columns.length
  const penalized = design.columns.map((_, i) => i < design.athletes.length)
  const centers = new Float64Array(k)
  if (spec.shrinkage.kind === 'ridge' && spec.shrinkage.center === 'erg' && spec.ergCenters) {
    const provided = [...spec.ergCenters.values()]
    const fallback = provided.length ? provided.reduce((a, b) => a + b, 0) / provided.length : 0
    design.athletes.forEach((name, i) => {
      centers[i] = spec.ergCenters!.get(name) ?? spec.ergCenters!.get(stripRigging(name)) ?? fallback
    })
  }
  return { penalized, centers }
}

function toFitResult(
  design: Design,
  params: Float64Array,
  bse: Float64Array,
  ciLower: Float64Array,
  ciUpper: Float64Array,
  dfResid: number,
  rank: number,
  fitted: Float64Array,
): FitResult {
  return {
    columns: design.columns,
    params,
    bse,
    ciLower,
    ciUpper,
    dfResid,
    rank,
    fitted,
    paramMap: new Map(design.columns.map((c, i) => [c, params[i]])),
  }
}

const nan = (k: number) => {
  const a = new Float64Array(k)
  a.fill(NaN)
  return a
}

export function fitModel(design: Design, spec: ModelSpec): FitResult {
  const { x, y, w } = design
  const k = design.columns.length

  if (spec.loss.kind === 'squared') {
    if (spec.shrinkage.kind === 'none') {
      const f = fitWls(x, y, w)
      return toFitResult(design, f.params, f.bse, f.ciLower, f.ciUpper, f.dfResid, f.rank, f.fitted)
    }
    const { penalized, centers } = centersFor(design, spec)
    const f = fitRidge(x, y, w, spec.shrinkage.lambda, penalized, centers)
    return toFitResult(design, f.params, f.bse, f.ciLower, f.ciUpper, x.length - f.edf, Math.round(f.edf), f.fitted)
  }

  if (spec.loss.kind === 'huber') {
    if (spec.shrinkage.kind === 'none') {
      const f = fitHuberRlm(x, y, w, { t: spec.loss.c })
      return toFitResult(design, f.params, f.bse, f.ciLower, f.ciUpper, f.dfResid, x[0].length, f.fitted)
    }
    // Huber + ridge: IRLS with the ridge solve inside. Intervals come from
    // the final ridge step (approximate, labeled as such in the UI).
    const { penalized, centers } = centersFor(design, spec)
    const t = spec.loss.c
    let fit = fitRidge(x, y, w, spec.shrinkage.lambda, penalized, centers)
    for (let it = 0; it < 50; it++) {
      const resid = y.map((v, i) => v - fit.fitted[i])
      const scale = madAboutZero(resid)
      if (scale === 0) break
      const irlsW = new Float64Array(x.length)
      for (let i = 0; i < x.length; i++) {
        const z = resid[i] / scale
        irlsW[i] = w[i] * (Math.abs(z) <= t ? 1 : t / Math.abs(z))
      }
      const next = fitRidge(x, y, irlsW, spec.shrinkage.lambda, penalized, centers)
      let delta = 0
      for (let c = 0; c < k; c++) delta = Math.max(delta, Math.abs(next.params[c] - fit.params[c]))
      fit = next
      if (delta < 1e-8) break
    }
    return toFitResult(
      design, fit.params, fit.bse, fit.ciLower, fit.ciUpper, x.length - fit.edf, Math.round(fit.edf), fit.fitted,
    )
  }

  // Lp loss. Confidence intervals for Lp come from the piece bootstrap in the
  // evaluation layer; the point fit reports NaN intervals.
  const p = spec.loss.p
  if (spec.shrinkage.kind === 'none') {
    const f = irlsLp(x, y, w, p)
    return toFitResult(design, f.params, nan(k), nan(k), nan(k), NaN, x[0].length, f.fitted)
  }
  const { penalized, centers } = centersFor(design, spec)
  let fit = fitRidge(x, y, w, spec.shrinkage.lambda, penalized, centers)
  const floor = 1e-6 * Math.max(madAboutZero(y.map((v, i) => v - fit.fitted[i])), 1e-8)
  for (let it = 0; it < 100; it++) {
    const resid = y.map((v, i) => v - fit.fitted[i])
    const irlsW = new Float64Array(x.length)
    for (let i = 0; i < x.length; i++) {
      const r = Math.max(Math.abs(resid[i]), floor)
      irlsW[i] = w[i] * Math.pow(r, p - 2)
    }
    const next = fitRidge(x, y, irlsW, spec.shrinkage.lambda, penalized, centers)
    let delta = 0
    for (let c = 0; c < k; c++) delta = Math.max(delta, Math.abs(next.params[c] - fit.params[c]))
    fit = next
    if (delta < 1e-10) break
  }
  return toFitResult(design, fit.params, nan(k), nan(k), nan(k), NaN, x[0].length, fit.fitted)
}
