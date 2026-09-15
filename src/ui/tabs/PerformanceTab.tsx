import type { FitPayload } from '../../workers/fit.worker'
import type { AthleteStat, NamedShellStat, ShellStat } from '../../engine/derived'
import { SortableTable, type Column } from '../SortableTable'
import { OptionsSection } from '../OptionsPanel'
import type { ControlState } from '../options'

interface Props {
  result: FitPayload | null
  fitting: boolean
  controls: ControlState
  defaults: ControlState
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
    render: (r) => (!r.comparable ? '' : r.speedBehind > 0 ? `+${fmt(r.speedBehind)}` : 'Fastest'),
  },
  {
    key: 'ci',
    label: 'Uncertainty',
    num: true,
    value: (r) => (r.upper - r.lower) / 2,
    render: (r) =>
      !r.comparable ? '∞' : Number.isFinite(r.lower) ? `±${fmt((r.upper - r.lower) / 2)}` : '',
  },
  { key: 'rank', label: 'Rank', num: true, value: (r) => r.rank || NaN, render: (r) => (r.rank || '') },
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
    render: (r) =>
      r.maxCorrelatedWith
        ? `${r.maxCorrelatedWith}${r.maxCorrelatedOthers ? ` +${r.maxCorrelatedOthers}` : ''} (${fmt(r.maxCorrelation, 2)})`
        : '',
  },
]

const NAMED_SHELL_COLUMNS: Array<Column<NamedShellStat>> = [
  { key: 'shell', label: 'Shell', value: (r) => r.shell },
  {
    key: 'behind',
    label: 'Behind',
    num: true,
    value: (r) => r.behind,
    render: (r) => (!r.comparable ? '' : r.behind > 0.05 ? `+${fmt(r.behind)}` : 'Fastest'),
  },
  {
    key: 'ci',
    label: 'Uncertainty',
    num: true,
    value: (r) => (r.upper - r.lower) / 2,
    render: (r) =>
      !r.comparable ? '∞' : Number.isFinite(r.lower) ? `±${fmt((r.upper - r.lower) / 2)}` : '',
  },
  { key: 'races', label: 'Races', num: true, value: (r) => r.races },
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

export function PerformanceTab({ result, fitting, controls, defaults, allShells, onControls }: Props) {
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
      <OptionsSection controls={controls} defaults={defaults} allShells={allShells} onControls={onControls} />
      {!result || result.athletes.length === 0 ? (
        <div className="empty-state">Load a dataset on the Data tab to see athlete estimates.</div>
      ) : (
        <>
          <ul className="hint-list">
            <li>Behind: seconds per 500m slower than the fastest rower on the same side.</li>
            <li>Port and starboard are never compared with each other.</li>
            <li>∞: the data cannot tell this rower apart from anyone.</li>
            <li>Rank 80%: the range of ranks the data supports.</li>
          </ul>
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
          {result.namedShells.length > 0 && (
            <>
              <h2>Shells</h2>
              <ul className="hint-list">
                <li>Behind: seconds per 500m slower than the fastest boat, crews accounted for.</li>
                <li>∞: the boat always had the same crew, so the data cannot tell them apart.</li>
              </ul>
              <div style={{ maxWidth: 520 }}>
                <SortableTable
                  columns={NAMED_SHELL_COLUMNS}
                  rows={result.namedShells}
                  defaultSort="behind"
                  rowKey={(r) => r.shell}
                />
              </div>
            </>
          )}
          <h2>Shell Classes</h2>
          <ul className="hint-list">
            <li>Behind: seconds per 500m slower than the fastest boat type, with an average crew.</li>
            <li>Only pieces where different boat types raced each other count.</li>
            <li>Not comparable: this boat type never raced another type.</li>
          </ul>
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
