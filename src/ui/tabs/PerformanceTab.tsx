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
  {
    key: 'rankrange',
    label: 'Rank 80%',
    num: true,
    value: (r) => (r.rankLow == null ? NaN : r.rankLow * 100 + (r.rankHigh ?? 0)),
    render: (r) =>
      r.rankLow == null ? '' : r.rankLow === r.rankHigh ? String(r.rankLow) : `${r.rankLow}-${r.rankHigh}`,
  },
  { key: 'races', label: 'Races', num: true, value: (r) => r.races },
  {
    key: 'maxcorr',
    label: 'Confounded With',
    value: (r) => r.maxCorrelation,
    render: (r) => (r.maxCorrelatedWith ? `${r.maxCorrelatedWith} (${fmt(r.maxCorrelation, 2)})` : ''),
  },
]

function shellColumns(shells: ShellStat[]): Array<Column<ShellStat>> {
  const comparable = (s: ShellStat) => s.crossClassPieces > 0 || shells.length === 1
  const comparableShells = shells.filter(comparable)
  const fastestCrew = Math.min(...comparableShells.map((s) => s.averageCrewPace))
  return [
    { key: 'shell', label: 'Shell Class', value: (r) => r.shellClass },
    {
      key: 'crewBehind',
      label: 'Behind (Average Crew)',
      num: true,
      value: (r) => (comparable(r) ? r.averageCrewPace - fastestCrew : Infinity),
      render: (r) =>
        !comparable(r)
          ? 'Not comparable'
          : r.averageCrewPace - fastestCrew > 0.05
            ? `+${fmt(r.averageCrewPace - fastestCrew)}`
            : 'Fastest',
    },
    {
      key: 'cross',
      label: 'Cross-Class Pieces',
      num: true,
      value: (r) => r.crossClassPieces,
    },
    {
      key: 'ci',
      label: 'Uncertainty',
      num: true,
      value: (r) => (comparable(r) ? (r.upper - r.lower) / 2 : Infinity),
      // An unidentified class has unbounded true uncertainty; the computed
      // number is the spread of an arbitrary representative, so hide it.
      render: (r) =>
        comparable(r) && Number.isFinite(r.lower) ? `±${fmt((r.upper - r.lower) / 2)}` : '',
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
            cannot separate them and the uncertainty widens (see Confounded With). Rank 80% is
            the range of ranks the athlete plausibly holds once everyone's uncertainty is
            accounted for together: 1000 simulated redraws of all estimates at once, including
            how they move together, keeping the middle 80% of each athlete's simulated ranks.
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
            Behind (Average Crew) compares boat types fairly: the predicted pace of each class
            with an average rower in every seat, as a gap behind the fastest class. These gaps
            are learned only from pieces where different classes race each other, counted in
            Cross-Class Pieces; a class that never races another class shows Not comparable,
            because the data cannot separate that boat's speed from the quality of the crews who
            happened to row it.
          </p>
          <div style={{ maxWidth: 520 }}>
            <SortableTable
              columns={shellColumns(result.shells)}
              rows={result.shells}
              defaultSort="crewBehind"
              rowKey={(r) => r.shellClass}
            />
          </div>
        </>
      )}
    </>
  )
}
