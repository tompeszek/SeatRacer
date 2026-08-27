import { useRef, useState } from 'react'
import type { EvalResult, CandidateResult } from '../../engine/evaluate'
import type { SerializableCandidate } from '../../workers/eval.worker'
import type { WeightSettings } from '../../engine/types'
import { EvalClient } from '../evalClient'
import { SortableTable, type Column } from '../SortableTable'
import { STRENGTH_OPTIONS, type ControlState } from '../options'

interface Props {
  csvText: string | null
  settings: WeightSettings
  controls: ControlState
}

const fmt = (v: number, digits = 2) => (Number.isFinite(v) ? v.toFixed(digits) : '')
const pct = (v: number) => (Number.isFinite(v) ? `${Math.round(v * 100)}%` : '')

function defaultCandidates(controls: ControlState): SerializableCandidate[] {
  const strength = STRENGTH_OPTIONS[controls.strength].value
  const list: SerializableCandidate[] = [
    { key: 'ols', label: 'Squared', loss: { kind: 'squared' }, shrinkage: { kind: 'none' } },
    { key: 'huber', label: 'Huber', loss: { kind: 'huber', c: 1.345 }, shrinkage: { kind: 'none' } },
    { key: 'lp05', label: 'Lp p=0.5', loss: { kind: 'lp', p: 0.5 }, shrinkage: { kind: 'none' } },
    {
      key: 'ols-ridge',
      label: 'Squared + Ridge',
      loss: { kind: 'squared' },
      shrinkage: { kind: 'ridge', lambda: strength, center: 'zero' },
    },
    {
      key: 'huber-ridge',
      label: 'Huber + Ridge',
      loss: { kind: 'huber', c: 1.345 },
      shrinkage: { kind: 'ridge', lambda: strength, center: 'zero' },
    },
    {
      key: 'lp05-ridge',
      label: 'Lp p=0.5 + Ridge',
      loss: { kind: 'lp', p: 0.5 },
      shrinkage: { kind: 'ridge', lambda: strength, center: 'zero' },
    },
  ]
  return list
}

export function ModelLabTab({ csvText, settings, controls }: Props) {
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState<[number, number] | null>(null)
  const [result, setResult] = useState<EvalResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [detailKey, setDetailKey] = useState<string | null>(null)
  const client = useRef<EvalClient>()
  if (!client.current) client.current = new EvalClient()

  const run = () => {
    if (!csvText) return
    setRunning(true)
    setResult(null)
    setError(null)
    client
      .current!.evaluate(csvText, settings, defaultCandidates(controls), (done, total) =>
        setProgress([done, total]),
      )
      .then((r) => {
        setResult(r)
        setRunning(false)
        setProgress(null)
      })
      .catch((err) => {
        setError(String(err))
        setRunning(false)
        setProgress(null)
      })
  }

  const columns: Array<Column<CandidateResult>> = [
    { key: 'label', label: 'Model', value: (r) => r.label },
    { key: 'fmed', label: 'Future-Day Miss', num: true, value: (r) => r.futureDay.medianAbsError, render: (r) => fmt(r.futureDay.medianAbsError) },
    { key: 'fpick', label: 'Future-Day Picks', num: true, value: (r) => r.futureDay.correctPickRate, render: (r) => pct(r.futureDay.correctPickRate) },
    { key: 'fpairs', label: 'Future Pairs', num: true, value: (r) => r.futureDay.pairs },
    { key: 'smed', label: 'Same-Day Miss', num: true, value: (r) => r.sameDay.medianAbsError, render: (r) => fmt(r.sameDay.medianAbsError) },
    { key: 'spick', label: 'Same-Day Picks', num: true, value: (r) => r.sameDay.correctPickRate, render: (r) => pct(r.sameDay.correctPickRate) },
    { key: 'spairs', label: 'Same Pairs', num: true, value: (r) => r.sameDay.pairs },
    { key: 'excl', label: 'Excluded', num: true, value: (r) => r.excludedPairs },
  ]

  const detail = result?.candidates.find((c) => c.key === detailKey) ?? null

  return (
    <>
      <div className="page-header">
        <h1>Model Lab</h1>
        <button className="btn-primary" onClick={run} disabled={running || !csvText}>
          {running ? 'Evaluating...' : 'Evaluate Models'}
        </button>
      </div>
      <p className="hint">
        Every model is tested forward in time only: for each race piece, the model is fit on all
        earlier pieces and asked to predict the margins between boats in that piece. Miss is the
        median absolute margin error in seconds per 500m (lower is better). Picks is how often
        the model named the faster boat of a pair (50% is a coin flip). Future-Day scores
        predictions of a day the model has not seen at all; Same-Day scores later pieces of a day
        partly seen, which favors models that track day-of form. Excluded counts boat pairs a
        model could not predict because an athlete had never been seen; ridge models predict
        newcomers at their prior instead. All models use the weighting options currently set on
        the Data tab.
      </p>
      {running && progress && (
        <p className="hint">
          Evaluating: {progress[0]} of {progress[1]} fits complete.
        </p>
      )}
      {error && <p className="error">{error}</p>}
      {result && (
        <>
          {result.cappedAt != null && (
            <p className="warn-note">
              Evaluation tested the most recent {result.cappedAt} pieces of {result.totalPieces};
              earlier pieces were used as training data only.
            </p>
          )}
          <SortableTable
            columns={columns}
            rows={result.candidates}
            defaultSort="fmed"
            rowKey={(r) => r.key}
          />
          <div className="controls">
            <span className="opt-label">Worst Pieces</span>
            <div className="pills">
              {result.candidates.map((c) => (
                <button
                  key={c.key}
                  className={`pill${detailKey === c.key ? ' active' : ''}`}
                  onClick={() => setDetailKey(detailKey === c.key ? null : c.key)}
                >
                  {c.label}
                </button>
              ))}
            </div>
          </div>
          {detail && (
            <>
              <p className="hint">
                The pieces {detail.label} predicted worst, by mean absolute margin error. A piece
                that every model misses badly usually had something unusual happen in it.
              </p>
              <SortableTable
                columns={[
                  { key: 'piece', label: 'Piece', value: (r: (typeof detail.worstPieces)[number]) => r.piece },
                  { key: 'err', label: 'Mean Miss', num: true, value: (r) => r.meanAbsError, render: (r) => fmt(r.meanAbsError) },
                  { key: 'pairs', label: 'Pairs', num: true, value: (r) => r.pairs },
                ]}
                rows={detail.worstPieces}
                defaultSort="err"
                defaultDesc
                rowKey={(r) => r.piece}
              />
            </>
          )}
        </>
      )}
      {!result && !running && (
        <div className="empty-state">
          Run the evaluation to compare model options on this dataset.
        </div>
      )}
    </>
  )
}
