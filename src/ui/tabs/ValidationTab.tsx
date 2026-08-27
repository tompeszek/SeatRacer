import { useState } from 'react'
import type { FitPayload } from '../../workers/fit.worker'
import type { FittedRow } from '../../engine/derived'
import { SortableTable, type Column } from '../SortableTable'
import { secondsToTime } from '../../engine/prep'

interface Props {
  result: FitPayload | null
}

const fmt = (v: number, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : '')

const COLUMNS: Array<Column<FittedRow>> = [
  { key: 'piece', label: 'Piece', value: (r) => r.piece },
  { key: 'crew', label: 'Crew', value: (r) => r.crew },
  { key: 'shell', label: 'Shell', value: (r) => r.shellClass },
  { key: 'actual', label: 'Actual', num: true, value: (r) => r.actual, render: (r) => secondsToTime(r.actual) },
  { key: 'model', label: 'Model', num: true, value: (r) => r.fitted, render: (r) => secondsToTime(r.fitted) },
  { key: 'delta', label: 'Delta', num: true, value: (r) => r.delta, render: (r) => fmt(r.delta) },
]

export function ValidationTab({ result }: Props) {
  const [filter, setFilter] = useState('')
  const fitted = result?.fitted ?? []
  const duplicates = result?.duplicates ?? []

  const abs = fitted.map((r) => Math.abs(r.delta))
  const meanAbs = abs.length ? abs.reduce((s, v) => s + v, 0) / abs.length : NaN
  const maxAbs = abs.length ? Math.max(...abs) : NaN

  const shown = filter
    ? fitted.filter((r) => r.crew.toLowerCase().includes(filter.toLowerCase()))
    : fitted

  return (
    <>
      <h1>Validation</h1>
      {fitted.length === 0 ? (
        <div className="empty-state">Load a dataset on the Data tab to check the model fit.</div>
      ) : (
        <>
          <p className="hint">
            On the data it was fit to, the model misses each boat's pace by {fmt(meanAbs)} seconds
            per 500m on average, and by {fmt(maxAbs)} at worst. Delta is actual minus model:
            positive means the boat went slower than modeled. These are in-sample numbers; the
            Model Lab measures honest forward prediction.
          </p>
          <div className="controls">
            <label className="form-field">
              Filter by athlete
              <input
                className="erg-input wide"
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                placeholder="Name"
              />
            </label>
          </div>
          <SortableTable
            columns={COLUMNS}
            rows={shown}
            defaultSort="delta"
            defaultDesc
            rowKey={(r, i) => `${r.piece}|${r.crew}|${i}`}
            groupKey={(r) => r.piece}
          />
          <h2>Possible Errors</h2>
          {duplicates.length === 0 ? (
            <p className="hint">No athlete appears in more than one boat within the same piece.</p>
          ) : (
            <>
              <p className="warn-note">
                {duplicates.length} athlete entries appear in more than one boat in the same piece;
                check for data entry mistakes.
              </p>
              <SortableTable
                columns={[
                  { key: 'piece', label: 'Piece', value: (r: (typeof duplicates)[number]) => r.piece },
                  { key: 'athlete', label: 'Athlete', value: (r) => r.athlete },
                  { key: 'boats', label: 'Boats', num: true, value: (r) => r.boats },
                ]}
                rows={duplicates}
                rowKey={(r) => `${r.piece}|${r.athlete}`}
              />
            </>
          )}
        </>
      )}
    </>
  )
}
