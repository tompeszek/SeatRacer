import type { FitPayload } from '../../workers/fit.worker'
import type { CorrelationPair } from '../../engine/derived'
import { SortableTable, type Column } from '../SortableTable'

interface Props {
  result: FitPayload | null
}

const fmt = (v: number) => (Number.isFinite(v) ? v.toFixed(2) : '')

const COLUMNS: Array<Column<CorrelationPair>> = [
  { key: 'a', label: 'Athlete', value: (r) => r.a },
  { key: 'b', label: 'Athlete', value: (r) => r.b },
  {
    key: 'corr',
    label: 'Correlation',
    num: true,
    value: (r) => r.correlation,
    render: (r) => fmt(r.correlation),
  },
  { key: 'together', label: 'Races Together', num: true, value: (r) => r.racesTogether },
]

export function CorrelationsTab({ result }: Props) {
  const pairs = (result?.correlations ?? []).filter((p) => Math.abs(p.correlation) >= 0.2)

  return (
    <>
      <h1>Correlations</h1>
      {!result || result.correlations.length === 0 ? (
        <div className="empty-state">Load a dataset on the Data tab to see athlete correlations.</div>
      ) : (
        <>
          <ul className="hint-list">
            <li>How often two athletes rowed in the same boat.</li>
            <li>Near 1: they almost always rowed together, so the model cannot tell them apart.</li>
            <li>Racing apart is the only fix. Pairs below 0.2 are not listed.</li>
          </ul>
          <SortableTable
            columns={COLUMNS}
            rows={pairs}
            defaultSort="corr"
            defaultDesc
            rowKey={(r) => `${r.a}|${r.b}`}
          />
        </>
      )}
    </>
  )
}
