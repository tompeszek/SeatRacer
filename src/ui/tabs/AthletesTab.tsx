import { useRef } from 'react'
import type { RaceRow } from '../../engine/types'
import { timeToSeconds } from '../../engine/prep'

interface Props {
  rows: RaceRow[]
  ergs: Record<string, string>
  onErgs: (next: Record<string, string>) => void
}

export function ergToCenter(time: string): number | null {
  try {
    const seconds = timeToSeconds(time)
    // 2k erg time to split per 500m: the coefficient prior is the athlete's
    // own pace scale; only differences between athletes matter to the fit.
    return seconds / 4
  } catch {
    return null
  }
}

function roster(rows: RaceRow[]): string[] {
  const names = new Set<string>()
  for (const row of rows) {
    for (const name of row.personnel.split('/')) {
      if (name) names.add(name)
    }
  }
  return [...names].sort()
}

export function AthletesTab({ rows, ergs, onErgs }: Props) {
  const fileInput = useRef<HTMLInputElement>(null)
  const names = roster(rows)

  const importCsv = (text: string) => {
    const next = { ...ergs }
    for (const line of text.split(/\r?\n/).slice(1)) {
      const [athlete, erg] = line.split(',').map((s) => s.trim())
      if (athlete && erg && ergToCenter(erg) != null) next[athlete] = erg
    }
    onErgs(next)
  }

  if (names.length === 0) {
    return (
      <>
        <h1>Athletes</h1>
        <div className="empty-state">Load racing data first; the roster comes from the race data.</div>
      </>
    )
  }

  return (
    <>
      <div className="page-header">
        <h1>Athletes</h1>
        <button className="btn-outline" onClick={() => fileInput.current?.click()}>
          Upload Erg CSV
        </button>
        <input
          ref={fileInput}
          type="file"
          accept=".csv"
          style={{ display: 'none' }}
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (!file) return
            file.text().then(importCsv)
            e.target.value = ''
          }}
        />
      </div>
      <p className="hint">
        2k erg times, format m:ss.s, one per athlete. They are used only when Shrinkage is set to
        Toward Ergs: an athlete's estimate is pulled toward their erg-implied pace, so ergs break
        ties the racing data cannot settle and stand in for athletes with little racing. A CSV
        upload needs the columns Athlete and 2k Erg.
      </p>
      <div className="table-card" style={{ maxWidth: 420 }}>
        <table>
          <thead>
            <tr>
              <th>Athlete</th>
              <th className="num">2k Erg</th>
            </tr>
          </thead>
          <tbody>
            {names.map((name) => (
              <tr key={name}>
                <td>{name}</td>
                <td className="num">
                  <input
                    className="erg-input"
                    value={ergs[name] ?? ''}
                    placeholder="7:00.0"
                    onChange={(e) => onErgs({ ...ergs, [name]: e.target.value })}
                  />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="hint tiny">
        Entries save as you type. An entry that does not parse as m:ss.s is ignored by the model.
      </p>
    </>
  )
}
