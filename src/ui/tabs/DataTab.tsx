import { useRef } from 'react'
import type { RaceRow } from '../../engine/types'
import { SortableTable, type Column } from '../SortableTable'
import { parseDate, timeToSeconds } from '../../engine/prep'
import { OptionsSection } from '../OptionsPanel'
import type { ControlState } from '../options'

interface Props {
  rows: RaceRow[]
  datasetNames: string[]
  selected: string
  onSelect: (name: string) => void
  onUpload: (name: string, text: string) => void
  controls: ControlState
  defaults: ControlState
  allShells: string[]
  onControls: (c: ControlState) => void
}

// Sort values that tolerate malformed cells; NaN sorts last.
function dateValue(r: RaceRow): number {
  try {
    return parseDate(r.dateRaw).getTime()
  } catch {
    return NaN
  }
}

function resultValue(r: RaceRow): number {
  try {
    return timeToSeconds(r.result)
  } catch {
    return NaN
  }
}

const orDefault = (n: number) => (Number.isNaN(n) ? Infinity : n)

/** Default order: date, then piece number, then fastest result first. */
function compareRows(a: RaceRow, b: RaceRow): number {
  return (
    orDefault(dateValue(a)) - orDefault(dateValue(b)) ||
    orDefault(a.pieceNumber) - orDefault(b.pieceNumber) ||
    orDefault(resultValue(a)) - orDefault(resultValue(b))
  )
}

type IndexedRow = RaceRow & { index: number }

function columnsFor(hasShell: boolean): Array<Column<IndexedRow>> {
  return [
    { key: 'date', label: 'Session', value: dateValue, render: (r) => r.dateRaw },
    { key: 'piece', label: 'Piece', num: true, value: (r) => r.pieceNumber },
    { key: 'km', label: 'KM', num: true, value: (r) => r.km },
    { key: 'rigging', label: 'Rigging', value: (r) => r.rigging },
    ...(hasShell ? [{ key: 'shell', label: 'Shell', value: (r: IndexedRow) => r.shell ?? '' }] : []),
    { key: 'personnel', label: 'Personnel', value: (r) => r.personnel },
    { key: 'result', label: 'Result', num: true, value: resultValue, render: (r) => r.result },
  ]
}

export function DataTab({ rows, datasetNames, selected, onSelect, onUpload, controls, defaults, allShells, onControls }: Props) {
  const fileInput = useRef<HTMLInputElement>(null)
  const indexed = rows.map((r, index) => ({ ...r, index })).sort(compareRows)
  const columns = columnsFor(rows.some((r) => r.shell))

  return (
    <>
      <div className="page-header">
        <h1>Data</h1>
        <button className="btn-outline" onClick={() => fileInput.current?.click()}>
          Upload CSV
        </button>
        <input
          ref={fileInput}
          type="file"
          accept=".csv"
          style={{ display: 'none' }}
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (!file) return
            file.text().then((text) => onUpload(file.name, text))
            e.target.value = ''
          }}
        />
      </div>
      <div className="controls">
        <label className="form-field">
          Dataset
          <select className="plain" value={selected} onChange={(e) => onSelect(e.target.value)}>
            {datasetNames.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </label>
        <span className="count-pill">{rows.length} results</span>
      </div>
      <OptionsSection controls={controls} defaults={defaults} allShells={allShells} onControls={onControls} />
      <ul className="hint-list">
        <li>Each row is one boat's result in one piece.</li>
        <li>Uploads need the columns Race Session (date), Piece, KM, Rigging, Personnel, Result.</li>
        <li>An optional Shell column names the boat.</li>
      </ul>
      {rows.length === 0 ? (
        <div className="empty-state">Select or upload a dataset to begin.</div>
      ) : (
        <SortableTable
          columns={columns}
          rows={indexed}
          rowKey={(r) => String(r.index)}
          groupKey={(r) => r.dateRaw}
        />
      )}
    </>
  )
}
