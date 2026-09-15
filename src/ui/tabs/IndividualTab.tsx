import { useState } from 'react'
import type { AthleteInfluence } from '../../engine/influence'
import { SortableTable, type Column } from '../SortableTable'

interface Props {
  influence: AthleteInfluence[] | null
  running: boolean
  progress: [number, number] | null
  onRun: () => void
  hasData: boolean
}

const fmt = (v: number) => (Number.isFinite(v) ? (v > 0 ? `+${v.toFixed(2)}` : v.toFixed(2)) : '')

export function IndividualTab({ influence, running, progress, onRun, hasData }: Props) {
  const [selected, setSelected] = useState('')
  const athletes = influence?.map((a) => a.name) ?? []
  const active = influence?.find((a) => a.name === (selected || athletes[0])) ?? null

  const columns: Array<Column<{ piece: string; delta: number }>> = [
    { key: 'piece', label: 'Piece Removed', value: (r) => r.piece },
    {
      key: 'delta',
      label: 'Estimate Change',
      num: true,
      value: (r) => Math.abs(r.delta),
      render: (r) => fmt(r.delta),
    },
  ]

  return (
    <>
      <div className="page-header">
        <h1>Individual</h1>
        <button className="btn-primary" onClick={onRun} disabled={running || !hasData}>
          {running ? 'Computing...' : 'Compute Influence'}
        </button>
      </div>
      <ul className="hint-list">
        <li>How much the athlete's estimate moves when one piece is left out, in seconds per 500m.</li>
        <li>A big move means the estimate leans on that one race.</li>
        <li>Negative: the athlete looks faster without that piece.</li>
      </ul>
      {running && progress && (
        <p className="hint">
          Computing: {progress[0]} of {progress[1]} refits complete.
        </p>
      )}
      {!influence && !running && (
        <div className="empty-state">
          {hasData ? 'Run the computation to see per-piece influence.' : 'Load a dataset first.'}
        </div>
      )}
      {influence && (
        <>
          <div className="opt-row" style={{ margin: '8px 0' }}>
            <span className="opt-label">Athlete</span>
            <select
              className="plain"
              value={selected || athletes[0] || ''}
              onChange={(e) => setSelected(e.target.value)}
            >
              {athletes.map((a) => (
                <option key={a} value={a}>
                  {a}
                </option>
              ))}
            </select>
          </div>
          {active && (
            <SortableTable
              columns={columns}
              rows={active.entries}
              defaultSort="delta"
              defaultDesc
              rowKey={(r) => r.piece}
            />
          )}
        </>
      )}
    </>
  )
}
