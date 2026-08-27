import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { parseRaceCsv } from '../engine/parse'
import { shellClassFromRigging } from '../engine/prep'
import type { RaceRow, WeightSettings } from '../engine/types'
import type { AthleteInfluence, TimeSeriesResult } from '../engine/influence'
import type { FitPayload, SerializableSpec } from '../workers/fit.worker'
import { FitClient } from './fitClient'
import { useTheme } from './theme'
import {
  BUNDLED_DATASETS,
  CLOSE_RACES_OPTIONS,
  DEFAULT_CONTROLS,
  RECENCY_OPTIONS,
  STERN_BIAS_OPTIONS,
  STRENGTH_OPTIONS,
  type ControlState,
} from './options'
import { DataTab } from './tabs/DataTab'
import { PerformanceTab } from './tabs/PerformanceTab'
import { NewLineupTab } from './tabs/NewLineupTab'
import { ModelLabTab } from './tabs/ModelLabTab'
import { AthletesTab, ergToCenter } from './tabs/AthletesTab'
import { FairnessTab } from './tabs/FairnessTab'
import { SynergyTab } from './tabs/SynergyTab'
import { CorrelationsTab } from './tabs/CorrelationsTab'
import { ValidationTab } from './tabs/ValidationTab'
import { IndividualTab } from './tabs/IndividualTab'
import { OverTimeTab } from './tabs/OverTimeTab'

const TABS = [
  'Data',
  'Athletes',
  'Performance',
  'New Lineup',
  'Individual',
  'Synergies',
  'Fairness',
  'Correlations',
  'Validation',
  'Over Time',
  'Model Lab',
] as const
type Tab = (typeof TABS)[number]

interface Upload {
  name: string
  text: string
}

function loadStored<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key)
    return raw ? { ...fallback, ...(JSON.parse(raw) as T) } : fallback
  } catch {
    return fallback
  }
}

function store(key: string, value: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(value))
  } catch {
    // storage unavailable; state stays session-only
  }
}

export function App() {
  const [theme, setTheme] = useTheme()
  const [menuOpen, setMenuOpen] = useState(false)
  const [tab, setTab] = useState<Tab>('Data')
  const [datasetName, setDatasetName] = useState<string>(
    () => loadStored('dataset', { name: BUNDLED_DATASETS[0] }).name,
  )
  const [uploads, setUploads] = useState<Upload[]>(() => loadStored('uploads', { list: [] as Upload[] }).list)
  const [csvText, setCsvText] = useState<string | null>(null)
  const [controls, setControls] = useState<ControlState>(() => loadStored('controls', DEFAULT_CONTROLS))
  const [ergs, setErgs] = useState<Record<string, string>>(() => loadStored('ergs', { map: {} as Record<string, string> }).map)
  const [result, setResult] = useState<FitPayload | null>(null)
  const [fitting, setFitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [influence, setInfluence] = useState<AthleteInfluence[] | null>(null)
  const [influenceRunning, setInfluenceRunning] = useState(false)
  const [influenceProgress, setInfluenceProgress] = useState<[number, number] | null>(null)
  const [timeSeries, setTimeSeries] = useState<TimeSeriesResult | null>(null)
  const [timeRunning, setTimeRunning] = useState(false)
  const [timeProgress, setTimeProgress] = useState<[number, number] | null>(null)
  const client = useRef<FitClient>()
  if (!client.current) client.current = new FitClient()
  const menuRef = useRef<HTMLDivElement>(null)

  // The menu closes on outside click and Escape (guide Section 21.2).
  useEffect(() => {
    if (!menuOpen) return
    const onDown = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setMenuOpen(false)
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setMenuOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [menuOpen])

  // Load the selected dataset's text (bundled via fetch, uploads from state).
  useEffect(() => {
    store('dataset', { name: datasetName })
    const upload = uploads.find((u) => u.name === datasetName)
    if (upload) {
      setCsvText(upload.text)
      return
    }
    let cancelled = false
    fetch(encodeURI(`/data/${datasetName}`))
      .then((r) => {
        if (!r.ok) throw new Error(`Failed to load dataset (${r.status})`)
        return r.text()
      })
      .then((text) => {
        if (!cancelled) setCsvText(text)
      })
      .catch((err) => {
        if (!cancelled) setError(String(err))
      })
    return () => {
      cancelled = true
    }
  }, [datasetName, uploads])

  const rawRows: RaceRow[] = useMemo(() => {
    if (!csvText) return []
    try {
      return parseRaceCsv(csvText)
    } catch (err) {
      setError(String(err))
      return []
    }
  }, [csvText])

  const allShells = useMemo(
    () => [...new Set(rawRows.map((r) => shellClassFromRigging(r.rigging)))].sort(),
    [rawRows],
  )

  const settings: WeightSettings = useMemo(
    () => ({
      halflife: RECENCY_OPTIONS[controls.recency].value,
      weightClose: CLOSE_RACES_OPTIONS[controls.close].value,
      weightStern: STERN_BIAS_OPTIONS[controls.stern].value,
      includeCoxswains: controls.coxswains,
      shellClasses: controls.shells ?? allShells,
    }),
    [controls, allShells],
  )

  const spec: SerializableSpec = useMemo(() => {
    const loss =
      controls.loss === 'Squared'
        ? ({ kind: 'squared' } as const)
        : controls.loss === 'Huber'
          ? ({ kind: 'huber', c: 1.345 } as const)
          : ({ kind: 'lp', p: Number(controls.lpP) } as const)
    const shrinkage =
      controls.shrinkage === 'Off'
        ? ({ kind: 'none' } as const)
        : ({
            kind: 'ridge',
            lambda: STRENGTH_OPTIONS[controls.strength].value,
            center: controls.shrinkage === 'Toward Ergs' ? 'erg' : 'zero',
          } as const)
    const ergCenters: Array<[string, number]> = []
    if (shrinkage.kind === 'ridge' && shrinkage.center === 'erg') {
      for (const [name, time] of Object.entries(ergs)) {
        const center = ergToCenter(time)
        if (center != null) ergCenters.push([name, center])
      }
    }
    return { loss, shrinkage, ergCenters: ergCenters.length ? ergCenters : undefined }
  }, [controls, ergs])

  // Refit on any input change, debounced. Slow derived computations
  // (influence, trends) are invalidated and rerun on demand.
  useEffect(() => {
    store('controls', controls)
    setInfluence(null)
    setTimeSeries(null)
    if (!csvText || rawRows.length === 0) {
      setResult(null)
      return
    }
    const timer = setTimeout(() => {
      setFitting(true)
      client
        .current!.fit(csvText, settings, spec)
        .then((payload) => {
          setResult(payload)
          setError(null)
          setFitting(false)
        })
        .catch((err) => {
          setError(String(err))
          setFitting(false)
        })
    }, 120)
    return () => clearTimeout(timer)
  }, [csvText, rawRows, settings, spec, controls])

  const runInfluence = useCallback(() => {
    if (!csvText) return
    setInfluenceRunning(true)
    setInfluenceProgress(null)
    client
      .current!.leaveOneOut(csvText, settings, spec, (d, t) => setInfluenceProgress([d, t]))
      .then((r) => {
        setInfluence(r)
        setInfluenceRunning(false)
      })
      .catch((err) => {
        setError(String(err))
        setInfluenceRunning(false)
      })
  }, [csvText, settings, spec])

  const runTime = useCallback(() => {
    if (!csvText) return
    setTimeRunning(true)
    setTimeProgress(null)
    client
      .current!.overTime(csvText, settings, spec, (d, t) => setTimeProgress([d, t]))
      .then((r) => {
        setTimeSeries(r)
        setTimeRunning(false)
      })
      .catch((err) => {
        setError(String(err))
        setTimeRunning(false)
      })
  }, [csvText, settings, spec])

  const onUpload = useCallback(
    (name: string, text: string) => {
      const next = [...uploads.filter((u) => u.name !== name), { name, text }]
      setUploads(next)
      store('uploads', { list: next })
      setDatasetName(name)
    },
    [uploads],
  )

  const onErgs = useCallback((next: Record<string, string>) => {
    setErgs(next)
    store('ergs', { map: next })
  }, [])

  const datasetNames = [...BUNDLED_DATASETS, ...uploads.map((u) => u.name)]

  return (
    <>
      <header className="app-header">
        <div>
          <div className="header-name">SeatRacer</div>
          <div className="header-sub">Rowing lineup analysis</div>
        </div>
        <div className="spacer" />
        <div className="header-menu-wrap" ref={menuRef}>
          <button className="header-text-btn" onClick={() => setMenuOpen(!menuOpen)}>
            Menu
          </button>
          {menuOpen && (
            <div className="header-menu">
              <button
                className="menu-item"
                onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
              >
                Dark Mode
                <span className={`menu-switch${theme === 'dark' ? ' on' : ''}`}>
                  <span className="menu-knob" />
                </span>
              </button>
            </div>
          )}
        </div>
      </header>
      <div className="tab-bar-wrap">
        <nav className="tab-bar">
          {TABS.map((t) => (
            <button key={t} className={`tab${t === tab ? ' active' : ''}`} onClick={() => setTab(t)}>
              {t}
            </button>
          ))}
        </nav>
      </div>
      <main className={`container${tab === 'Performance' ? ' wide' : ''}`}>
        {error && <p className="error">{error}</p>}
        {tab === 'Data' && (
          <DataTab
            rows={rawRows}
            datasetNames={datasetNames}
            selected={datasetName}
            onSelect={setDatasetName}
            onUpload={onUpload}
            controls={controls}
            allShells={allShells}
            onControls={setControls}
          />
        )}
        {tab === 'Athletes' && <AthletesTab rows={rawRows} ergs={ergs} onErgs={onErgs} />}
        {tab === 'Performance' && (
          <PerformanceTab
            result={result}
            fitting={fitting}
            controls={controls}
            allShells={allShells}
            onControls={setControls}
          />
        )}
        {tab === 'New Lineup' && (
          <NewLineupTab result={result} sternWeight={STERN_BIAS_OPTIONS[controls.stern].value} />
        )}
        {tab === 'Individual' && (
          <IndividualTab
            influence={influence}
            running={influenceRunning}
            progress={influenceProgress}
            onRun={runInfluence}
            hasData={rawRows.length > 0}
          />
        )}
        {tab === 'Synergies' && <SynergyTab result={result} />}
        {tab === 'Fairness' && <FairnessTab result={result} />}
        {tab === 'Correlations' && <CorrelationsTab result={result} />}
        {tab === 'Validation' && <ValidationTab result={result} />}
        {tab === 'Over Time' && (
          <OverTimeTab
            series={timeSeries}
            running={timeRunning}
            progress={timeProgress}
            onRun={runTime}
            hasData={rawRows.length > 0}
          />
        )}
        {tab === 'Model Lab' && <ModelLabTab csvText={csvText} settings={settings} controls={controls} />}
      </main>
    </>
  )
}
