// Design matrix construction, matching the Python engine's
// pd.get_dummies(df[['Piece'] + athletes + shell_classes]) layout:
// athlete fraction columns, then shell class dummies, then piece dummies.
import type { Design, PreppedRow, WeightSettings } from './types'
import { collectAthletes, collectShellClasses } from './prep'

export function buildDesign(rows: PreppedRow[], settings: WeightSettings): Design {
  const athletes = collectAthletes(rows, settings.includeCoxswains)
  const shellClasses = collectShellClasses(rows)
  const pieceSet = new Set<string>()
  for (const row of rows) pieceSet.add(row.piece)
  // pandas get_dummies emits dummy columns in sorted category order.
  const pieces = [...pieceSet].sort()

  const columns = [
    ...athletes,
    ...shellClasses,
    ...pieces.map((p) => `Piece_${p}`),
  ]
  const nCols = columns.length
  const pieceIndex = new Map(pieces.map((p, i) => [p, athletes.length + shellClasses.length + i]))
  const shellIndex = new Map(shellClasses.map((s, i) => [s, athletes.length + i]))
  const athleteIndex = new Map(athletes.map((a, i) => [a, i]))

  const x: Float64Array[] = []
  const y = new Float64Array(rows.length)
  const w = new Float64Array(rows.length)
  rows.forEach((row, r) => {
    const line = new Float64Array(nCols)
    for (const [name, frac] of row.athleteFractions) {
      const c = athleteIndex.get(name)
      if (c !== undefined) line[c] = frac
    }
    line[shellIndex.get(row.shellClass)!] = 1
    line[pieceIndex.get(row.piece)!] = 1
    x.push(line)
    y[r] = row.timePer500m
    w[r] = row.totalWeight
  })

  return { columns, athletes, shellClasses, pieces, x, y, w, rows }
}

/**
 * Design row for predicting an arbitrary lineup: athlete fractions (even or
 * stern-biased) plus the shell class dummy; piece dummies stay zero.
 */
export function lineupRow(
  design: Pick<Design, 'columns'>,
  personnel: string[],
  shellClass: string,
  fractions: Map<string, number>,
): Float64Array {
  const line = new Float64Array(design.columns.length)
  design.columns.forEach((col, i) => {
    if (col === shellClass) line[i] = 1
    else if (fractions.has(col)) line[i] = fractions.get(col)!
  })
  void personnel
  return line
}
