import type { FitPayload } from '../../workers/fit.worker'
import type { AthleteStat, ShellStat } from '../../engine/derived'
import { SortableTable, type Column } from '../SortableTable'
import { OptionsSection } from '../OptionsPanel'
import type { ControlState } from '../options'
import { secondsToTime } from '../../engine/prep'

interface Props {
  result: FitPayload | null
  fitting: boolean
  controls: ControlState
  allShells: string[]
  onControls: (c: ControlState) => void
}

const fmt = (v: number, digits = 1) => (Number.isFinite(v) ? v.toFixed(digits) : '')

const ATHLETE_COLUMNS: Array<Column<AthleteStat>> = [
  { key: 'name', label: 'Rower', value: (r) => r.name },
  { key: 'coef', label: 'Coefficient', num: true, value: (r) => r.coefficient, render: (r) => fmt(r.coefficient) },
  {
    key: 'ci',
    label: '95% Interval',
    num: true,
    value: (r) => r.upper - r.lower,
    render: (r) =>
      Number.isFinite(r.lower) ? `${fmt(r.lower)} to ${fmt(r.upper)}` : '',
  },
  {
    key: 'behind',
    label: 'Behind',
    num: true,
    value: (r) => r.speedBehind,
    render: (r) => (r.speedBehind > 0 ? `+${fmt(r.speedBehind)}` : ''),
  },
  { key: 'rank', label: 'Rank', num: true, value: (r) => r.rank },
  { key: 'of', label: 'Of', num: true, value: (r) => r.totalInPosition },
  { key: 'races', label: 'Races', num: true, value: (r) => r.races },
  {
    key: 'maxcorr',
    label: 'Most Confounded With',
    value: (r) => r.maxCorrelation,
    render: (r) => (r.maxCorrelatedWith ? `${r.maxCorrelatedWith} (${fmt(r.maxCorrelation, 2)})` : ''),
  },
]

const SHELL_COLUMNS: Array<Column<ShellStat>> = [
  { key: 'shell', label: 'Shell Class', value: (r) => r.shellClass },
  { key: 'coef', label: 'Pace Effect', num: true, value: (r) => r.coefficient, render: (r) => fmt(r.coefficient) },
  {
    key: 'ci',
    label: '95% Interval',
    num: true,
    value: (r) => r.upper - r.lower,
    render: (r) => (Number.isFinite(r.lower) ? `${fmt(r.lower)} to ${fmt(r.upper)}` : ''),
  },
]

export function PerformanceTab({ result, fitting, controls, allShells, onControls }: Props) {
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
            The coefficient is each athlete's estimated effect on boat pace, in seconds per 500m:
            lower is faster. Behind is the gap to the fastest athlete rowing the same side. The
            interval covers where the true effect plausibly lies; when an athlete almost always
            rows with the same partners, the data cannot separate them and the interval widens
            (see Most Confounded With).
          </p>
          <SortableTable
            columns={ATHLETE_COLUMNS}
            rows={result.athletes}
            defaultSort="coef"
            rowKey={(r) => r.name}
          />
          <h2>Shell Classes</h2>
          <p className="hint">
            Base pace by boat type, seconds per 500m. Example: a predicted 8+ at{' '}
            {result.shells.length > 0 ? secondsToTime(Math.min(...result.shells.map((s) => s.coefficient))) : ''}{' '}
            per 500m before athlete effects.
          </p>
          <SortableTable
            columns={SHELL_COLUMNS}
            rows={result.shells}
            defaultSort="coef"
            rowKey={(r) => r.shellClass}
          />
        </>
      )}
    </>
  )
}
