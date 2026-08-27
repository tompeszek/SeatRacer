import { useState } from 'react'
import type { FitPayload } from '../../workers/fit.worker'
import type { PairStat } from '../../engine/derived'
import { SortableTable, type Column } from '../SortableTable'

interface Props {
  result: FitPayload | null
}

const MIN_RACES = 5
const P_OPTIONS: Record<string, { value: number; caption: string }> = {
  All: { value: 1.0, caption: 'Showing all pairs, regardless of statistical significance' },
  '0.05': { value: 0.05, caption: 'Showing pairs with p-value at or below 0.05 (statistically significant)' },
  '0.01': { value: 0.01, caption: 'Showing pairs with p-value at or below 0.01 (highly significant)' },
  '0.001': { value: 0.001, caption: 'Showing pairs with p-value at or below 0.001 (extremely significant)' },
}

const fmt = (v: number, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : '')

const COLUMNS: Array<Column<PairStat>> = [
  { key: 'pair', label: 'Pair', value: (r) => `${r.a} + ${r.b}` },
  {
    key: 'delta',
    label: 'Together',
    num: true,
    value: (r) => r.avgDelta,
    render: (r) => `${fmt(r.avgDelta)} ${r.avgDelta < 0 ? '(faster)' : '(slower)'}`,
  },
  { key: 'races', label: 'Races', num: true, value: (r) => r.races },
  { key: 'p', label: 'P-Value', num: true, value: (r) => r.pValue, render: (r) => fmt(r.pValue, 3) },
]

export function SynergyTab({ result }: Props) {
  const [choice, setChoice] = useState('0.05')
  const pairs = (result?.pairs ?? []).filter((p) => p.races >= MIN_RACES)
  const shown = pairs.filter((p) => p.pValue <= P_OPTIONS[choice].value)

  return (
    <>
      <h1>Synergies</h1>
      {pairs.length === 0 ? (
        <div className="empty-state">
          No athlete pairs with at least {MIN_RACES} races together yet.
        </div>
      ) : (
        <>
          <p className="hint">
            Together is the mean gap between actual and modeled pace for boats containing both
            athletes, in seconds per 500m. Negative means the pair's boats keep beating the
            model: the two may be faster together than their individual estimates suggest.
          </p>
          <div className="opt-row" style={{ margin: '8px 0' }}>
            <span className="opt-label">P-Value</span>
            <div className="pills">
              {Object.keys(P_OPTIONS).map((key) => (
                <button
                  key={key}
                  className={`pill${key === choice ? ' active' : ''}`}
                  onClick={() => setChoice(key)}
                >
                  {key}
                </button>
              ))}
            </div>
            <span className="opt-caption">{P_OPTIONS[choice].caption}</span>
          </div>
          {shown.length === 0 ? (
            <div className="empty-state">No pairs meet this significance threshold.</div>
          ) : (
            <SortableTable
              columns={COLUMNS}
              rows={shown}
              defaultSort="delta"
              rowKey={(r) => `${r.a}|${r.b}`}
            />
          )}
        </>
      )}
    </>
  )
}
