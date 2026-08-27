import { useCallback, useMemo, useState } from 'react'
import type { TimeSeriesResult } from '../../engine/influence'
import { EChart } from '../EChart'
import type { VizTheme } from '../vizPalette'
import { SortableTable, type Column } from '../SortableTable'

interface Props {
  series: TimeSeriesResult | null
  running: boolean
  progress: [number, number] | null
  onRun: () => void
  hasData: boolean
}

const SIDES: Record<string, string> = { 'ᵖ': 'Port', 'ˢ': 'Starboard', 'ˣ': 'Scull', 'ᶜ': 'Coxswain' }
const MAX_SERIES = 8

export function OverTimeTab({ series, running, progress, onRun, hasData }: Props) {
  const [side, setSide] = useState('ᵖ')
  const [chosen, setChosen] = useState<string[] | null>(null)

  const sidesPresent = useMemo(
    () => Object.keys(SIDES).filter((s) => series?.series.some((x) => x.suffix === s)),
    [series],
  )
  const activeSide = sidesPresent.includes(side) ? side : sidesPresent[0]
  const sideSeries = useMemo(
    () => (series?.series ?? []).filter((s) => s.suffix === activeSide),
    [series, activeSide],
  )
  // Default: the athletes with the most datapoints, capped at the palette's
  // series limit. Colors are assigned by roster order within the selection.
  const defaultChosen = useMemo(
    () =>
      [...sideSeries]
        .sort((a, b) => b.values.filter((v) => v != null).length - a.values.filter((v) => v != null).length)
        .slice(0, Math.min(6, MAX_SERIES))
        .map((s) => s.name),
    [sideSeries],
  )
  const selection = chosen?.filter((c) => sideSeries.some((s) => s.name === c)) ?? defaultChosen
  const shown = sideSeries.filter((s) => selection.includes(s.name))

  const option = useCallback(
    (viz: VizTheme) => ({
      backgroundColor: 'transparent',
      grid: { left: 8, right: 90, top: 30, bottom: 40, containLabel: true },
      legend: {
        top: 0,
        textStyle: { color: viz.secondaryInk, fontSize: 11 },
        icon: 'roundRect',
        itemWidth: 12,
        itemHeight: 4,
      },
      tooltip: {
        trigger: 'axis' as const,
        backgroundColor: viz.surface,
        borderColor: viz.gridline,
        textStyle: { color: viz.primaryInk, fontSize: 12 },
        valueFormatter: (v: unknown) =>
          v == null ? '' : `${Number(v) > 0 ? '+' : ''}${Number(v).toFixed(2)} s / 500m vs side average`,
      },
      xAxis: {
        type: 'category' as const,
        data: series?.dates ?? [],
        axisLabel: { color: viz.muted, fontSize: 10 },
        axisLine: { lineStyle: { color: viz.baseline } },
        axisTick: { show: false },
      },
      yAxis: {
        type: 'value' as const,
        name: 'Vs side average (s / 500m)',
        nameTextStyle: { color: viz.muted, fontSize: 10 },
        axisLabel: { color: viz.muted, fontSize: 10 },
        splitLine: { lineStyle: { color: viz.gridline } },
      },
      series: shown.map((s, i) => ({
        name: s.name,
        type: 'line' as const,
        data: s.values.map((v) => (v == null ? null : Number(v.toFixed(3)))),
        connectNulls: true,
        symbolSize: 5,
        lineStyle: { width: 2, color: viz.categorical[i % viz.categorical.length] },
        itemStyle: { color: viz.categorical[i % viz.categorical.length] },
        endLabel: {
          show: shown.length <= 4,
          formatter: '{a}',
          color: viz.secondaryInk,
          fontSize: 10,
        },
      })),
    }),
    [series, shown],
  )

  const latestColumns: Array<Column<(typeof sideSeries)[number]>> = [
    { key: 'name', label: 'Athlete', value: (r) => r.name },
    {
      key: 'latest',
      label: 'Latest Vs Side Average',
      num: true,
      value: (r) => {
        const last = [...r.values].reverse().find((v) => v != null)
        return last ?? NaN
      },
      render: (r) => {
        const last = [...r.values].reverse().find((v) => v != null)
        return last == null ? '' : `${last > 0 ? '+' : ''}${last.toFixed(2)}`
      },
    },
  ]

  return (
    <>
      <div className="page-header">
        <h1>Over Time</h1>
        <button className="btn-primary" onClick={onRun} disabled={running || !hasData}>
          {running ? 'Computing...' : 'Compute Trends'}
        </button>
      </div>
      <p className="hint">
        The model is refit on all data up to each session date. Each line is an athlete's
        estimate relative to the average of their side at that date, in seconds per 500m: below
        zero is faster than the side's average. Absolute estimates from different dates are not
        comparable, which is why the chart shows relative standing.
      </p>
      {running && progress && (
        <p className="hint">
          Computing: {progress[0]} of {progress[1]} refits complete.
        </p>
      )}
      {!series && !running && (
        <div className="empty-state">
          {hasData ? 'Run the computation to see trends over time.' : 'Load a dataset first.'}
        </div>
      )}
      {series && (
        <>
          <div className="opt-row" style={{ margin: '8px 0' }}>
            <span className="opt-label">Side</span>
            <div className="pills">
              {sidesPresent.map((s) => (
                <button
                  key={s}
                  className={`pill${s === activeSide ? ' active' : ''}`}
                  onClick={() => {
                    setSide(s)
                    setChosen(null)
                  }}
                >
                  {SIDES[s]}
                </button>
              ))}
            </div>
          </div>
          <div className="opt-row" style={{ margin: '8px 0', flexWrap: 'wrap' }}>
            <span className="opt-label">Athletes</span>
            <div className="pills">
              {sideSeries.map((s) => {
                const active = selection.includes(s.name)
                return (
                  <button
                    key={s.name}
                    className={`pill${active ? ' active' : ''}`}
                    disabled={!active && selection.length >= MAX_SERIES}
                    onClick={() =>
                      setChosen(
                        active ? selection.filter((n) => n !== s.name) : [...selection, s.name],
                      )
                    }
                  >
                    {s.name}
                  </button>
                )
              })}
            </div>
            <span className="opt-caption">At most {MAX_SERIES} athletes at once</span>
          </div>
          <EChart option={option} height={380} />
          <h2>Latest Standing</h2>
          <SortableTable
            columns={latestColumns}
            rows={sideSeries}
            defaultSort="latest"
            rowKey={(r) => r.name}
          />
        </>
      )}
    </>
  )
}
