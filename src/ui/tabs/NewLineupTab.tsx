import { useMemo, useState } from 'react'
import type { FitPayload } from '../../workers/fit.worker'
import { predictLineup } from '../../engine/derived'
import { secondsToTime } from '../../engine/prep'
import { SortableTable, type Column } from '../SortableTable'

interface Props {
  result: FitPayload | null
  sternWeight: number
}

interface SavedLineup {
  name: string
  shellClass: string
  personnel: string[]
  pace: number
}

function seatCount(shellClass: string): number {
  const rowers = parseInt(shellClass, 10)
  return rowers + (shellClass.includes('+') ? 1 : 0)
}

export function NewLineupTab({ result, sternWeight }: Props) {
  const shellClasses = result?.shellClasses ?? []
  const [shellClass, setShellClass] = useState<string>('')
  const [seats, setSeats] = useState<string[]>([])
  const [saved, setSaved] = useState<SavedLineup[]>([])

  const activeShell = shellClass || shellClasses[0] || ''
  const nSeats = activeShell ? seatCount(activeShell) : 0
  const paramMap = useMemo(() => new Map(result?.params ?? []), [result])

  const chosen = seats.slice(0, nSeats)
  const complete = chosen.length === nSeats && chosen.every((s) => s !== '')
  const pace = complete ? predictLineup(paramMap, chosen, activeShell, sternWeight) : NaN

  const savedColumns: Array<Column<SavedLineup>> = [
    { key: 'name', label: 'Lineup', value: (r) => r.name },
    { key: 'shell', label: 'Shell', value: (r) => r.shellClass },
    { key: 'crew', label: 'Personnel', value: (r) => r.personnel.join(' / ') },
    { key: 'pace', label: 'Pace / 500m', num: true, value: (r) => r.pace, render: (r) => secondsToTime(r.pace) },
    {
      key: 'gap',
      label: 'Gap to Best',
      num: true,
      value: (r) => r.pace,
      render: (r) => {
        const best = Math.min(...saved.map((s) => s.pace))
        const gap = r.pace - best
        return gap > 0 ? `+${gap.toFixed(1)}` : ''
      },
    },
  ]

  if (!result || result.athleteNames.length === 0) {
    return (
      <>
        <h1>New Lineup</h1>
        <div className="empty-state">Load a dataset on the Data tab to build lineups.</div>
      </>
    )
  }

  return (
    <>
      <h1>New Lineup</h1>
      <p className="hint">
        Predicted pace combines the shell class effect and each athlete's coefficient. Piece
        conditions are unknown for a future race, so compare lineups against each other rather
        than reading the pace as an absolute time.
      </p>
      <div className="opt-row" style={{ margin: '8px 0' }}>
        <span className="opt-label">Shell</span>
        <div className="pills">
          {shellClasses.map((s) => (
            <button
              key={s}
              className={`pill${s === activeShell ? ' active' : ''}`}
              onClick={() => {
                setShellClass(s)
                setSeats([])
              }}
            >
              {s}
            </button>
          ))}
        </div>
      </div>
      <div className="form-row">
        {Array.from({ length: nSeats }, (_, i) => (
          <label key={i} className="form-field">
            {activeShell.includes('+') && i === 0 ? 'Coxswain' : `Seat ${activeShell.includes('+') ? i : i + 1}`}
            <select
              className="plain"
              value={chosen[i] ?? ''}
              onChange={(e) => {
                const next = [...chosen]
                while (next.length < nSeats) next.push('')
                next[i] = e.target.value
                setSeats(next)
              }}
            >
              <option value="">Choose</option>
              {result.athleteNames
                .filter((a) => !chosen.includes(a) || chosen[i] === a)
                .map((a) => (
                  <option key={a} value={a}>
                    {a}
                  </option>
                ))}
            </select>
          </label>
        ))}
      </div>
      <div className="controls">
        <span className="count-pill">
          {complete ? `Predicted pace ${secondsToTime(pace)} per 500m` : 'Fill every seat to predict'}
        </span>
        <button
          className="btn-primary"
          disabled={!complete}
          onClick={() =>
            setSaved([
              ...saved,
              { name: `Lineup ${saved.length + 1}`, shellClass: activeShell, personnel: chosen, pace },
            ])
          }
        >
          Add to Comparison
        </button>
        {saved.length > 0 && (
          <button className="btn-outline" onClick={() => setSaved([])}>
            Clear Comparison
          </button>
        )}
      </div>
      {saved.length > 0 && (
        <SortableTable columns={savedColumns} rows={saved} defaultSort="pace" rowKey={(r) => r.name} />
      )}
    </>
  )
}
