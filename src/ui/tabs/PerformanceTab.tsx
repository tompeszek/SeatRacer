import type { FitPayload } from '../../workers/fit.worker'
import type { AthleteStat, ShellStat } from '../../engine/derived'
import { SortableTable, type Column } from '../SortableTable'
import { OptionsSection } from '../OptionsPanel'
import type { ControlState } from '../options'

interface Props {
  result: FitPayload | null
  fitting: boolean
  controls: ControlState
  allShells: string[]
  onControls: (c: ControlState) => void
}

const fmt = (v: number, digits = 1) => (Number.isFinite(v) ? v.toFixed(digits) : '')

const SIDE_NAMES: Record<string, string> = {
  'ᵖ': 'Port',
  'ˢ': 'Starboard',
  'ˣ': 'Scull',
  'ᶜ': 'Coxswain',
}
const SIDE_ORDER = ['Port', 'Starboard', 'Scull', 'Coxswain']

const ATHLETE_COLUMNS: Array<Column<AthleteStat>> = [
  { key: 'name', label: 'Rower', value: (r) => r.name },
  {
    key: 'behind',
    label: 'Behind',
    num: true,
    value: (r) => r.speedBehind,
    render: (r) => (r.speedBehind > 0 ? `+${fmt(r.speedBehind)}` : 'Fastest'),
  },
  {
    key: 'ci',
    label: 'Uncertainty',
    num: true,
    value: (r) => (r.upper - r.lower) / 2,
    render: (r) => (Number.isFinite(r.lower) ? `±${fmt((r.upper - r.lower) / 2)}` : ''),
  },
  { key: 'rank', label: 'Rank', num: true, value: (r) => r.rank },
  { key: 'races', label: 'Races', num: true, value: (r) => r.races },
  {
    key: 'maxcorr',
    label: 'Most Confounded With',
    value: (r) => r.maxCorrelation,
    render: (r) => (r.maxCorrelatedWith ? `${r.maxCorrelatedWith} (${fmt(r.maxCorrelation, 2)})` : ''),
  },
]

function shellColumns(shells: ShellStat[]): Array<Column<ShellStat>> {
  const fastest = Math.min(...shells.map((s) => s.coefficient))
  return [
    { key: 'shell', label: 'Shell Class', value: (r) => r.shellClass },
    {
      key: 'behind',
      label: 'Behind',
      num: true,
      value: (r) => r.coefficient - fastest,
      render: (r) => (r.coefficient - fastest > 0 ? `+${fmt(r.coefficient - fastest)}` : 'Fastest'),
    },
    {
      key: 'ci',
      label: 'Uncertainty',
      num: true,
      value: (r) => (r.upper - r.lower) / 2,
      render: (r) => (Number.isFinite(r.lower) ? `±${fmt((r.upper - r.lower) / 2)}` : ''),
    },
  ]
}

export function PerformanceTab({ result, fitting, controls, allShells, onControls }: Props) {
  const bySide = new Map<string, AthleteStat[]>()
  for (const a of result?.athletes ?? []) {
    const side = SIDE_NAMES[a.suffix] ?? 'Other'
    if (!bySide.has(side)) bySide.set(side, [])
    bySide.get(side)!.push(a)
  }
  const sides = SIDE_ORDER.filter((s) => bySide.has(s)).concat(
    [...bySide.keys()].filter((s) => !SIDE_ORDER.includes(s)),
  )

  return (
    <>
      <div className="page-header">
        <h1>Performance</h1>
        {fitting && <span className="hint">Fitting...</span>}
      </div>
      <OptionsSection controls={controls} allShells={allShells} onControls={onControls} />
      {!result || result.athletes.length === 0 ? (
        <div className="empty-state">Load a dataset on the Data tab to see athlete estimates.</div>
      ) : (
        <>
          <p className="hint">
            Behind is each athlete's estimated cost in boat pace, in seconds per 500m, relative
            to the fastest athlete on the same side; only these gaps are meaningful, never an
            absolute number. Port and starboard are estimated separately and are not comparable
            to each other, so each side has its own table. Uncertainty is the give-or-take on
            the estimate; when an athlete almost always rows with the same partners, the data
            cannot separate them and the uncertainty widens (see Most Confounded With).
          </p>
          <div className="side-cols">
            {sides.map((side) => (
              <div className="side-col" key={side}>
                <h2>{side}</h2>
                <SortableTable
                  columns={ATHLETE_COLUMNS}
                  rows={bySide.get(side)!}
                  defaultSort="behind"
                  rowKey={(r) => r.name}
                />
              </div>
            ))}
          </div>
          <h2>Shell Classes</h2>
          <p className="hint">
            How much pace each boat type gives up to the fastest type, in seconds per 500m.
          </p>
          <SortableTable
            columns={shellColumns(result.shells)}
            rows={result.shells}
            defaultSort="behind"
            rowKey={(r) => r.shellClass}
          />
        </>
      )}
    </>
  )
}
