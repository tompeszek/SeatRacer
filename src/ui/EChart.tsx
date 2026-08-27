import { useEffect, useRef } from 'react'
import * as echarts from 'echarts'
import { currentVizTheme, type VizTheme } from './vizPalette'

interface Props {
  /** Build the option from the current viz theme (re-run on theme change). */
  option: (viz: VizTheme) => echarts.EChartsOption
  height: number
}

export function EChart({ option, height }: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const chart = useRef<echarts.ECharts>()

  useEffect(() => {
    if (!ref.current) return
    chart.current = echarts.init(ref.current)
    const render = () => chart.current!.setOption(option(currentVizTheme()), true)
    render()
    const resize = () => chart.current?.resize()
    window.addEventListener('resize', resize)
    const observer = new MutationObserver(render)
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })
    return () => {
      window.removeEventListener('resize', resize)
      observer.disconnect()
      chart.current?.dispose()
    }
  }, [option])

  return <div ref={ref} style={{ width: '100%', height }} />
}
