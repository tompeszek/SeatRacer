import { useRef } from 'react'
import type { RaceRow } from '../../engine/types'
import { SortableTable, type Column } from '../SortableTable'
import { OptionsSection } from '../OptionsPanel'
import type { ControlState } from '../options'

interface Props {
  rows: RaceRow[]
  datasetNames: string[]
  selected: string
  onSelect: (name: string) => void
  onUpload: (name: string, text: string) => void
  controls: ControlState
  allShells: string[]
  onControls: (c: ControlState) => void
}

const COLUMNS: Array<Column<RaceRow & { index: number }>> = [
  { key: 'date', label: 'Session', value: (r) => r.dateRaw },
  { key: 'piece', label: 'Piece', num: true, value: (r) => r.pieceNumber },
  { key: 'km', label: 'KM', num: true, value: (r) => r.km },
  { key: 'rigging', label: 'Rigging', value: (r) => r.rigging },
  { key: 'personnel', label: 'Personnel', value: (r) => r.personnel },
  { key: 'result', label: 'Result', num: true, value: (r) => r.result },
]

export function DataTab({ rows, datasetNames, selected, onSelect, onUpload, controls, allShells, onControls }: Props) {
  const fileInput = useRef<HTMLInputElement>(null)
  const indexed = rows.map((r, index) => ({ ...r, index }))

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
      <OptionsSection controls={controls} allShells={allShells} onControls={onControls} />
      <p className="hint">
        Each row is one boat's result in one piece. Uploaded files need the columns Race Session
        (date), Piece, KM, Rigging, Personnel, and Result.
      </p>
      {rows.length === 0 ? (
        <div className="empty-state">Select or upload a dataset to begin.</div>
      ) : (
        <SortableTable
          columns={COLUMNS}
          rows={indexed}
          rowKey={(r) => String(r.index)}
          groupKey={(r) => r.dateRaw}
        />
      )}
    </>
  )
}
