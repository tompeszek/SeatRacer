// Property tests for the model options that have no Python counterpart
// (ridge, Lp): known limiting cases and monotonicity, per REWRITE_PLAN.md.
import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { parseRaceCsv } from './parse'
import { prepRows } from './prep'
import { buildDesign } from './design'
import { fitWls } from './solve'
import { fitRidge } from './ridge'
import { irlsLp } from './robust'
import { fitModel } from './model'
import type { WeightSettings } from './types'

const ROOT = join(__dirname, '..', '..')
const SETTINGS: WeightSettings = {
  halflife: null,
  weightClose: null,
  weightStern: null,
  includeCoxswains: true,
  shellClasses: ['1x', '2-', '2x', '4-', '4+', '4x', '8+'],
}

function loadDesign() {
  const csv = readFileSync(join(ROOT, 'public', 'data', 'Olympic Selection - 2012.csv'), 'utf-8')
  const rows = prepRows(parseRaceCsv(csv), SETTINGS)
  return buildDesign(rows, SETTINGS)
}

describe('ridge', () => {
  const design = loadDesign()
  const penalized = design.columns.map((_, i) => i < design.athletes.length)
  const centers = new Float64Array(design.columns.length)

  it('lambda = 0 equals the unpenalized fit', () => {
    const wls = fitWls(design.x, design.y, design.w)
    const ridge = fitRidge(design.x, design.y, design.w, 0, penalized, centers)
    for (let c = 0; c < design.columns.length; c++) {
      expect(Math.abs(ridge.params[c] - wls.params[c])).toBeLessThan(1e-8)
    }
  })

  it('athlete coefficients move monotonically toward the center as lambda grows', () => {
    const distances = [0.01, 1, 100, 10000].map((lambda) => {
      const fit = fitRidge(design.x, design.y, design.w, lambda, penalized, centers)
      let d = 0
      for (let c = 0; c < design.athletes.length; c++) d += fit.params[c] ** 2
      return Math.sqrt(d)
    })
    for (let i = 1; i < distances.length; i++) {
      expect(distances[i]).toBeLessThan(distances[i - 1])
    }
    const heavy = fitRidge(design.x, design.y, design.w, 1e8, penalized, centers)
    for (let c = 0; c < design.athletes.length; c++) {
      expect(Math.abs(heavy.params[c])).toBeLessThan(1e-2)
    }
  })

  it('ridge toward a nonzero center lands on the center under heavy lambda', () => {
    const shifted = new Float64Array(design.columns.length)
    for (let c = 0; c < design.athletes.length; c++) shifted[c] = 5
    const heavy = fitRidge(design.x, design.y, design.w, 1e8, penalized, shifted)
    for (let c = 0; c < design.athletes.length; c++) {
      expect(Math.abs(heavy.params[c] - 5)).toBeLessThan(1e-2)
    }
  })
})

describe('Lp loss', () => {
  it('p = 2 equals least squares', () => {
    const design = loadDesign()
    const wls = fitWls(design.x, design.y, design.w)
    const lp = irlsLp(design.x, design.y, design.w, 2)
    for (let c = 0; c < design.columns.length; c++) {
      expect(Math.abs(lp.params[c] - wls.params[c])).toBeLessThan(1e-8)
    }
  })

  it('p = 1 recovers the median on a location problem', () => {
    const x = [1, 1, 1, 1, 1].map(() => Float64Array.of(1))
    const y = Float64Array.of(1, 2, 3, 4, 100)
    const w = new Float64Array(5).fill(1)
    const lp = irlsLp(x, y, w, 1)
    expect(Math.abs(lp.params[0] - 3)).toBeLessThan(1e-3)
  })

  it('p below 1 resists the outlier harder than squared loss', () => {
    const x = [1, 1, 1, 1, 1].map(() => Float64Array.of(1))
    const y = Float64Array.of(1, 2, 3, 4, 100)
    const w = new Float64Array(5).fill(1)
    const squared = fitWls(x, y, w).params[0] // mean = 22
    const lp = irlsLp(x, y, w, 0.5).params[0]
    expect(lp).toBeLessThan(squared - 15)
  })
})

describe('fitModel dispatch', () => {
  const design = loadDesign()
  it('squared + none equals fitWls', () => {
    const viaSpec = fitModel(design, {
      loss: { kind: 'squared' },
      shrinkage: { kind: 'none' },
      weights: SETTINGS,
    })
    const direct = fitWls(design.x, design.y, design.w)
    for (let c = 0; c < design.columns.length; c++) {
      expect(Math.abs(viaSpec.params[c] - direct.params[c])).toBeLessThan(1e-12)
    }
  })

  it('huber + ridge converges and shrinks athletes', () => {
    const fit = fitModel(design, {
      loss: { kind: 'huber', c: 1.345 },
      shrinkage: { kind: 'ridge', lambda: 1e8, center: 'zero' },
      weights: SETTINGS,
    })
    for (let c = 0; c < design.athletes.length; c++) {
      expect(Math.abs(fit.params[c])).toBeLessThan(1e-2)
    }
  })
})
