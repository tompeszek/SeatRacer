import { useMemo, useState } from 'react'
import type { FitPayload } from '../../workers/fit.worker'
import { predictLineup } from '../../engine/derived'
import { secondsToTime, RIG_SUPERSCRIPTS } from '../../engine/prep'
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

/** Rigging patterns for a shell class: cox first, then stroke to bow. */
function riggingOptionsFor(shellClass: string): string[] {
  const rowers = parseInt(shellClass, 10)
  const cox = shellClass.includes('+')
  const scull = shellClass.includes('x')
  if (scull) {
    const seats = Array(rowers).fill('x')
    return [cox ? ['c', ...seats].join('/') : seats.join('/')]
  }
  const startP: string[] = []
  const startS: string[] = []
  for (let i = 0; i < rowers; i++) {
    startP.push(i % 2 === 0 ? 'p' : 's')
    startS.push(i % 2 === 0 ? 's' : 'p')
  }
  const patterns = [startP.join('/'), startS.join('/')]
  return patterns.map((p) => (cox ? `c/${p}` : p))
}

function seatLabel(rig: string[], index: number): string {
  if (rig[index] === 'c') return 'Coxswain'
  const rowerSeats = rig.filter((r) => r !== 'c').length
  const rowerIndex = rig.slice(0, index + 1).filter((r) => r !== 'c').length // 1-based from stroke
  if (rowerIndex === 1) return rowerSeats > 1 ? 'Stroke' : 'Sculler'
  if (rowerIndex === rowerSeats) return 'Bow'
  return `Seat ${rowerSeats - rowerIndex + 1}`
}

export function NewLineupTab({ result, sternWeight }: Props) {
  const shellClasses = result?.shellClasses ?? []
  const [shellClass, setShellClass] = useState<string>('')
  const [riggingIndex, setRiggingIndex] = useState(0)
  const [seats, setSeats] = useState<string[]>([])
  const [saved, setSaved] = useState<SavedLineup[]>([])

  const activeShell = shellClass || shellClasses[0] || ''
  const riggingOptions = activeShell ? riggingOptionsFor(activeShell) : []
  const rigging = riggingOptions[Math.min(riggingIndex, riggingOptions.length - 1)] ?? ''
  const rig = rigging ? rigging.split('/') : []
  const paramMap = useMemo(() => new Map(result?.params ?? []), [result])
  const coxswainsExcluded = (result?.athleteNames ?? []).every((a) => !a.endsWith('ᶜ'))

  const chosen = rig.map((seat, i) => {
    if (seat === 'c' && coxswainsExcluded) return 'Coxᶜ'
    return seats[i] ?? ''
  })
  const complete = rig.length > 0 && chosen.every((s) => s !== '')
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
        Predicted pace combines the shell class effect and each athlete's coefficient. Each seat
        only offers athletes who row that side: port and starboard estimates are separate and
        cannot stand in for each other. Piece conditions are unknown for a future race, so
        compare lineups against each other rather than reading the pace as an absolute time.
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
                setRiggingIndex(0)
                setSeats([])
              }}
            >
              {s}
            </button>
          ))}
        </div>
      </div>
      {riggingOptions.length > 1 && (
        <div className="opt-row" style={{ margin: '8px 0' }}>
          <span className="opt-label">Rigging</span>
          <div className="pills">
            {riggingOptions.map((opt, i) => (
              <button
                key={opt}
                className={`pill${i === riggingIndex ? ' active' : ''}`}
                onClick={() => {
                  setRiggingIndex(i)
                  setSeats([])
                }}
              >
                {opt}
              </button>
            ))}
          </div>
        </div>
      )}
      <div className="form-row">
        {rig.map((seat, i) => {
          const suffix = RIG_SUPERSCRIPTS[seat]
          if (seat === 'c' && coxswainsExcluded) {
            return (
              <label key={i} className="form-field">
                {seatLabel(rig, i)}
                <select className="plain" value="Cox" disabled>
                  <option>Cox</option>
                </select>
              </label>
            )
          }
          const eligible = result.athleteNames.filter(
            (a) => a.endsWith(suffix) && (!chosen.includes(a) || chosen[i] === a),
          )
          return (
            <label key={i} className="form-field">
              {seatLabel(rig, i)}
              <select
                className="plain"
                value={chosen[i]}
                onChange={(e) => {
                  const next = [...seats]
                  while (next.length < rig.length) next.push('')
                  next[i] = e.target.value
                  setSeats(next)
                }}
              >
                <option value="">Choose</option>
                {eligible.map((a) => (
                  <option key={a} value={a}>
                    {a}
                  </option>
                ))}
              </select>
            </label>
          )
        })}
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
