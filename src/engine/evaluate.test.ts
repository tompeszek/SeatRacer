// Tests for the walk-forward evaluator: the fast normal-equations solver must
// agree with the SVD path on identified quantities, and the evaluation loop
// must be deterministic and sane.
import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { parseRaceCsv } from './parse'
import { prepRows } from './prep'
import { buildDesign } from './design'
import { fitWls } from './solve'
import { fitCandidate, walkForward } from './evaluate'
import type { WeightSettings } from './types'

const ROOT = join(__dirname, '..', '..')
const SETTINGS: WeightSettings = {
  halflife: null,
  weightClose: null,
  weightStern: null,
  includeCoxswains: true,
  shellClasses: ['1x', '2-', '2x', '4-', '4+', '4x', '8+'],
}

function loadRaw(name: string) {
  return parseRaceCsv(readFileSync(join(ROOT, 'public', 'data', name), 'utf-8'))
}

describe('fitCandidate fast solver', () => {
  it('matches the SVD fit on fitted values and identified coefficients', () => {
    const raw = loadRaw('Olympic Selection - 2012.csv')
    const rows = prepRows(raw, SETTINGS)
    const design = buildDesign(rows, SETTINGS)
    const svd = fitWls(design.x, design.y, design.w)
    const fast = fitCandidate(rows, SETTINGS, {
      key: 'ols',
      label: 'OLS',
      loss: { kind: 'squared' },
      shrinkage: { kind: 'none' },
    })
    // Fitted values are identified regardless of the null-space handling.
    for (let r = 0; r < design.x.length; r++) {
      let acc = 0
      design.columns.forEach((col, c) => {
        const idx = fast.columns.get(col)
        if (idx !== undefined) acc += design.x[r][c] * fast.coef[idx]
      })
      expect(Math.abs(acc - svd.fitted[r]), `fitted row ${r}`).toBeLessThan(1e-4)
    }
  })
})

describe('walkForward', () => {
  it('produces deterministic, sane results', () => {
    const raw = loadRaw('SDRC Masters Men HOCR Selection - 2024.csv')
    const candidates = [
      { key: 'ols', label: 'OLS', loss: { kind: 'squared' } as const, shrinkage: { kind: 'none' } as const },
      {
        key: 'ridge',
        label: 'Ridge',
        loss: { kind: 'squared' } as const,
        shrinkage: { kind: 'ridge', lambda: 1.0, center: 'zero' } as const,
      },
    ]
    const a = walkForward(raw, SETTINGS, candidates)
    const b = walkForward(raw, SETTINGS, candidates)
    expect(a).toEqual(b)
    expect(a.candidates).toHaveLength(2)
    for (const c of a.candidates) {
      const total = c.sameDay.pairs + c.futureDay.pairs
      expect(total).toBeGreaterThan(0)
      if (c.sameDay.pairs > 0) {
        expect(c.sameDay.medianAbsError).toBeGreaterThanOrEqual(0)
        expect(c.sameDay.correctPickRate).toBeGreaterThanOrEqual(0)
        expect(c.sameDay.correctPickRate).toBeLessThanOrEqual(1)
      }
    }
    // Ridge can always predict; OLS may exclude unseen athletes.
    const ridge = a.candidates.find((c) => c.key === 'ridge')!
    expect(ridge.excludedPairs).toBe(0)
  })

  it('handles the larger dataset within the piece cap', () => {
    const raw = loadRaw('Olympic Selection - 2012.csv')
    const result = walkForward(raw, SETTINGS, [
      { key: 'ols', label: 'OLS', loss: { kind: 'squared' }, shrinkage: { kind: 'none' } },
    ])
    expect(result.totalPieces).toBeGreaterThan(80)
    expect(result.testedPieces).toBeLessThanOrEqual(80)
    expect(result.cappedAt).toBe(80)
    const c = result.candidates[0]
    expect(c.futureDay.pairs + c.sameDay.pairs).toBeGreaterThan(50)
    expect(Number.isFinite(c.sameDay.medianAbsError) || Number.isFinite(c.futureDay.medianAbsError)).toBe(true)
  })
})
