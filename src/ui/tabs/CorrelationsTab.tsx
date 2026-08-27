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
          <p className="hint">
            How entangled two athletes' racing histories are. A correlation near 1 means they
            almost always rowed together, so their individual contributions cannot be separated:
            the model can only estimate their sum, and both estimates carry wide uncertainty.
            Shrinkage keeps such estimates tame; more racing apart is what actually separates
            them. Pairs below 0.2 are not listed.
          </p>
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
