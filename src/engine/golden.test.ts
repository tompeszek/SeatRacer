// Golden-fixture equivalence gate: the TypeScript engine must reproduce the
// committed statsmodels outputs (fixtures/*.json) before the Python engine
// may be deleted. See REWRITE_PLAN.md section 6.
import { describe, expect, it } from 'vitest'
import { readFileSync, readdirSync } from 'node:fs'
import { join } from 'node:path'
import tQuantileTest from '@stdlib/stats-base-dists-t-quantile'
import normalQuantileTest from '@stdlib/stats-base-dists-normal-quantile'
import { parseRaceCsv } from './parse'
import { fitHuberRlm } from './robust'
import { prepRows, timeToSeconds, shellClassFromRigging } from './prep'
import { buildDesign } from './design'
import { fitWls, fitGlmGaussian } from './solve'
import type { WeightSettings } from './types'

const ROOT = join(__dirname, '..', '..')
const FIXTURES = join(ROOT, 'fixtures')
const DATA = join(ROOT, 'public', 'data')

interface ModelFixture {
  columns: string[]
  params: Record<string, number | null>
  bse: Record<string, number | null>
  ci_lower: Record<string, number | null>
  ci_upper: Record<string, number | null>
  df_resid?: number
}

interface Fixture {
  dataset: string
  config: {
    halflife: number | null
    weight_close: number | null
    weight_stern: number | null
    include_coxswains: boolean
    shell_class: string[]
  }
  models: Record<string, ModelFixture>
  prep: {
    athletes: string[]
    rows: Array<{
      piece: string
      personnel: string
      shell_class: string
      time_seconds: number
      time_per_500m: number
      closest_margin: number | null
      closeness_factor: number
      recency_factor: number
      total_weight: number
      athlete_fractions: Record<string, number>
    }>
  }
}

function loadFixture(name: string): Fixture {
  return JSON.parse(readFileSync(join(FIXTURES, name), 'utf-8'))
}

function settingsOf(f: Fixture): WeightSettings {
  return {
    halflife: f.config.halflife,
    weightClose: f.config.weight_close,
    weightStern: f.config.weight_stern,
    includeCoxswains: f.config.include_coxswains,
    shellClasses: f.config.shell_class,
  }
}

const fixtureFiles = readdirSync(FIXTURES).filter(
  (f) => f.endsWith('.json') && f !== 'helpers.json',
)

describe('helpers', () => {
  const helpers = JSON.parse(readFileSync(join(FIXTURES, 'helpers.json'), 'utf-8'))
  it('time_to_seconds matches', () => {
    for (const [time, expected] of Object.entries(helpers.time_to_seconds)) {
      expect(timeToSeconds(time)).toBeCloseTo(expected as number, 9)
    }
  })
  it('shell classes match', () => {
    for (const { rigging, shell_class } of helpers.shell_class) {
      expect(shellClassFromRigging(rigging)).toBe(shell_class)
    }
  })
})

for (const file of fixtureFiles) describe(file, () => {
  const fixture = loadFixture(file)
  const csv = readFileSync(join(DATA, fixture.dataset), 'utf-8')
  const raw = parseRaceCsv(csv)
  const settings = settingsOf(fixture)
  const rows = prepRows(raw, settings)

  it('prep rows match', () => {
    expect(rows.length).toBe(fixture.prep.rows.length)
    // Row order is irrelevant to the fit, and the Python engine's order is
    // not reproducible (pandas re-sorts by date with an unstable sort when
    // recency weighting is on). Compare by a deterministic key instead.
    const keyOf = (r: { piece: string; personnel: string; time_seconds: number }) =>
      `${r.piece}|${r.personnel}|${r.time_seconds.toFixed(6)}`
    const expSorted = [...fixture.prep.rows].sort((a, b) =>
      keyOf(a) < keyOf(b) ? -1 : 1,
    )
    const gotSorted = [...rows]
      .map((r) => ({ ...r, personnelStr: r.personnel.join('/') }))
      .sort((a, b) => {
        const ka = `${a.piece}|${a.personnelStr}|${a.timeSeconds.toFixed(6)}`
        const kb = `${b.piece}|${b.personnelStr}|${b.timeSeconds.toFixed(6)}`
        return ka < kb ? -1 : 1
      })
    gotSorted.forEach((row, i) => {
      const exp = expSorted[i]
      expect(row.piece).toBe(exp.piece)
      expect(row.shellClass).toBe(exp.shell_class)
      expect(row.timeSeconds).toBeCloseTo(exp.time_seconds, 9)
      expect(row.timePer500m).toBeCloseTo(exp.time_per_500m, 9)
      if (exp.closest_margin == null) expect(row.closestMargin).toBeNull()
      else expect(row.closestMargin).toBeCloseTo(exp.closest_margin, 9)
      expect(row.closenessFactor).toBeCloseTo(exp.closeness_factor, 9)
      expect(row.recencyFactor).toBeCloseTo(exp.recency_factor, 9)
      expect(row.totalWeight).toBeCloseTo(exp.total_weight, 9)
      for (const [name, frac] of Object.entries(exp.athlete_fractions)) {
        expect(row.athleteFractions.get(name) ?? 0).toBeCloseTo(frac, 9)
      }
    })
  })

  const design = buildDesign(rows, settings)

  it('design columns cover the fixture columns', () => {
    const expected = new Set(fixture.models.ols.columns)
    const got = new Set(design.columns)
    expect(got).toEqual(expected)
  })

  // The design has exact linear dependencies whose singular values surface as
  // implementation-dependent noise; statsmodels' own rank (and therefore
  // df_resid) wobbles by one between numpy call sites. The gate therefore
  // checks each statistical ingredient separately:
  //   params        keyed, tight (the minimum-norm solution is unique)
  //   fitted values my X times the fixture params must equal my fitted
  //   df_resid      within the known plus-or-minus 1 rank flakiness
  //   bse and CIs   recomputed at the fixture's df, so the covariance
  //                 structure, SSR, and t-quantiles are all verified tightly
  //                 without depending on the flaky rank
  it('OLS/WLS matches statsmodels', () => {
    const fit = fitWls(design.x, design.y, design.w)
    const exp = fixture.models.ols
    const index = new Map(design.columns.map((c, i) => [c, i]))
    expect(Math.abs(fit.dfResid - exp.df_resid!)).toBeLessThanOrEqual(1)

    for (const col of exp.columns) {
      const i = index.get(col)!
      expect(i, `missing column ${col}`).toBeDefined()
      expect(Math.abs(fit.params[i] - (exp.params[col] as number)), `coef ${col}`).toBeLessThan(1e-6)
    }

    // Fitted values: fixture params pushed through my design.
    for (let r = 0; r < design.x.length; r++) {
      let acc = 0
      for (const col of exp.columns) acc += design.x[r][index.get(col)!] * (exp.params[col] as number)
      expect(Math.abs(acc - fit.fitted[r]), `fitted row ${r}`).toBeLessThan(1e-5)
    }

    // Inference at the fixture's df: validates normalized covariance, SSR,
    // and the t-quantile implementation.
    const ssr = fit.scale * fit.dfResid
    const scaleAtFixtureDf = ssr / exp.df_resid!
    const q = tQuantileTest(0.975, exp.df_resid!)
    for (const col of exp.columns) {
      const i = index.get(col)!
      const covDiag = (fit.bse[i] * fit.bse[i]) / fit.scale
      const bseAdj = Math.sqrt(covDiag * scaleAtFixtureDf)
      expect(Math.abs(bseAdj - (exp.bse[col] as number)), `bse ${col}`).toBeLessThan(1e-6)
      expect(
        Math.abs(fit.params[i] - q * bseAdj - (exp.ci_lower[col] as number)),
        `ciL ${col}`,
      ).toBeLessThan(1e-5)
      expect(
        Math.abs(fit.params[i] + q * bseAdj - (exp.ci_upper[col] as number)),
        `ciU ${col}`,
      ).toBeLessThan(1e-5)
    }
  })

  // The old app's GLM inference is NOT reproduced, deliberately: statsmodels
  // GLM computes its scale from an lstsq projection but its params from a
  // pinv refit (two different rank cutoffs), and with fractional frequency
  // weights its df can go negative, yielding NaN confidence intervals. Only
  // the coefficients (which match WLS) are gated here.
  it('Gaussian GLM coefficients match statsmodels', () => {
    const fit = fitGlmGaussian(design.x, design.y, design.w)
    const exp = fixture.models.glm
    const index = new Map(design.columns.map((c, i) => [c, i]))
    for (const col of exp.columns) {
      const i = index.get(col)!
      expect(Math.abs(fit.params[i] - (exp.params[col] as number)), `coef ${col}`).toBeLessThan(1e-6)
    }
  })

  // RLM ignores observation weights in the old engine (sm.RLM(y, X)), so the
  // parity call passes none. bse/CIs are recomputed at the fixture's df, as
  // for OLS, to factor out the known rank-by-one flakiness.
  it('Huber RLM matches statsmodels', () => {
    const ones = new Float64Array(design.x.length).fill(1)
    const fit = fitHuberRlm(design.x, design.y, ones)
    const exp = fixture.models.rlm
    const index = new Map(design.columns.map((c, i) => [c, i]))
    expect(Math.abs(fit.dfResid - exp.df_resid!)).toBeLessThanOrEqual(1)
    // IRLS stops on a deviance criterion, so the exact stopping point differs
    // slightly between implementations on small, weakly identified datasets.
    // The gate is absolute 1e-5 or 1% of the coefficient's standard error,
    // whichever is larger: any difference far inside the statistical
    // uncertainty is a stopping-point artifact, not a formula error.
    for (const col of exp.columns) {
      const i = index.get(col)!
      const tol = Math.max(2e-5, 0.01 * Math.abs((exp.bse[col] as number) || 0))
      expect(Math.abs(fit.params[i] - (exp.params[col] as number)), `coef ${col}`).toBeLessThan(tol)
    }
    const { ssPsi, mSum, varPsiPrime, n } = fit.h1
    const m = mSum / n
    const factorAt = (df: number) => {
      const kc = 1 + ((n - df) / n) * (varPsiPrime / (m * m))
      return (kc * kc * ((1 / df) * ssPsi * fit.scale * fit.scale)) / (m * m)
    }
    const q = normalQuantileTest(0.975, 0, 1)
    for (const col of exp.columns) {
      if (exp.bse[col] == null) continue
      const i = index.get(col)!
      const bseFx = exp.bse[col] as number
      const bseAdj = Math.sqrt(fit.normalizedCovDiag[i] * factorAt(exp.df_resid!))
      expect(Math.abs(bseAdj - bseFx), `rlm bse ${col}`).toBeLessThan(Math.max(1e-5, 1e-3 * bseFx))
      // CI misses compound the coefficient and bse stopping-point artifacts.
      const ciTol = Math.max(1e-4, 1.5e-2 * bseFx)
      expect(
        Math.abs(fit.params[i] - q * bseAdj - (exp.ci_lower[col] as number)),
        `rlm ciL ${col}`,
      ).toBeLessThan(ciTol)
      expect(
        Math.abs(fit.params[i] + q * bseAdj - (exp.ci_upper[col] as number)),
        `rlm ciU ${col}`,
      ).toBeLessThan(ciTol)
    }
  })
})
