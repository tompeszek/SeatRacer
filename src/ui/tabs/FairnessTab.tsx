import { useCallback } from 'react'
import type { FitPayload } from '../../workers/fit.worker'
import type { BiasStat } from '../../engine/derived'
import { SortableTable, type Column } from '../SortableTable'
import { EChart } from '../EChart'
import type { VizTheme } from '../vizPalette'

interface Props {
  result: FitPayload | null
}

const fmt = (v: number, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : '')

const COLUMNS: Array<Column<BiasStat>> = [
  { key: 'name', label: 'Athlete', value: (r) => r.name },
  {
    key: 'avg',
    label: 'Average Error',
    num: true,
    value: (r) => r.avgDelta,
    render: (r) => fmt(r.avgDelta),
  },
  { key: 'sd', label: 'Spread', num: true, value: (r) => r.sd, render: (r) => fmt(r.sd) },
  { key: 'races', label: 'Races', num: true, value: (r) => r.races },
  { key: 'p', label: 'P-Value', num: true, value: (r) => r.pValue, render: (r) => fmt(r.pValue, 3) },
  {
    key: 'verdict',
    label: 'Reading',
    value: (r) =>
      !r.significant ? 'No consistent bias' : r.avgDelta < 0 ? 'Faster than modeled' : 'Slower than modeled',
  },
]

export function FairnessTab({ result }: Props) {
  const bias = result?.bias ?? []
  const significant = bias.filter((b) => b.significant)
  const chartRows = significant.length > 0 ? significant : bias

  const option = useCallback(
    (viz: VizTheme) => ({
      backgroundColor: 'transparent',
      grid: { left: 8, right: 16, top: 24, bottom: 60, containLabel: true },
      tooltip: {
        trigger: 'axis' as const,
        backgroundColor: viz.surface,
        borderColor: viz.gridline,
        textStyle: { color: viz.primaryInk, fontSize: 12 },
        valueFormatter: (v: unknown) => `${Number(v).toFixed(2)} seconds per 500m`,
      },
      xAxis: {
        type: 'category' as const,
        data: chartRows.map((r) => r.name),
        axisLabel: { rotate: 45, interval: 0, fontSize: 10, color: viz.secondaryInk },
        axisLine: { lineStyle: { color: viz.baseline } },
        axisTick: { show: false },
      },
      yAxis: {
        type: 'value' as const,
        name: 'Average error (s / 500m)',
        nameTextStyle: { color: viz.muted, fontSize: 10 },
        axisLabel: { color: viz.muted, fontSize: 10 },
        splitLine: { lineStyle: { color: viz.gridline } },
      },
      series: [
        {
          type: 'bar' as const,
          barMaxWidth: 18,
          itemStyle: {
            borderRadius: [4, 4, 0, 0],
            color: (params: { dataIndex: number }) => {
              const row = chartRows[params.dataIndex]
              if (!row.significant) return viz.neutral
              return row.avgDelta < 0 ? viz.divergingCool : viz.divergingWarm
            },
          },
          data: chartRows.map((r) => Number(r.avgDelta.toFixed(3))),
        },
      ],
    }),
    [chartRows],
  )

  return (
    <>
      <h1>Fairness</h1>
      {bias.length === 0 ? (
        <div className="empty-state">Load a dataset on the Data tab to check prediction bias.</div>
      ) : (
        <>
          <p className="hint">
            Average Error is the mean gap between a boat's actual pace and the model's, across
            every boat the athlete sat in, in seconds per 500m. Negative (blue) means their boats
            keep going faster than the model expects; positive (red) means slower. Gray bars show
            no statistically consistent bias. A consistent bias suggests the model is crediting
            or blaming the athlete's usual crewmates for something this athlete brings.
          </p>
          {significant.length === 0 && (
            <p className="hint">No statistically consistent bias detected; showing every athlete.</p>
          )}
          <EChart option={option} height={340} />
          <SortableTable columns={COLUMNS} rows={bias} defaultSort="avg" rowKey={(r) => r.name} />
        </>
      )}
    </>
  )
}
