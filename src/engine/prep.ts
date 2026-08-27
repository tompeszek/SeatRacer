// Row preparation: time parsing, shell class detection, rigging suffixes,
// piece labels, sorting. Ports seatracer/utils/helpers.py and the prep steps
// in analysis_base.py 1:1 (fixture-verified).
import type { RaceRow, PreppedRow, WeightSettings } from './types'
import { applyWeights, closestMargins, athleteFractions } from './weights'

export const RIG_SUPERSCRIPTS: Record<string, string> = {
  p: 'ᵖ',
  s: 'ˢ',
  c: 'ᶜ',
  x: 'ˣ',
}
const SUPERSCRIPT_SET = new Set(Object.values(RIG_SUPERSCRIPTS))

/** "13:05" -> 785; "03:28.5" -> 208.5. */
export function timeToSeconds(time: string): number {
  const m = time.trim().match(/^(\d+):(\d+(?:\.\d+)?)$/)
  if (!m) throw new Error(`Unparseable time: "${time}"`)
  return Number(m[1]) * 60 + Number(m[2])
}

/** 98.125 -> "01:38.1" (MM:SS.d, matching the Python seconds_to_time). */
export function secondsToTime(seconds: number): string {
  const minutes = Math.floor(seconds / 60)
  const rest = seconds % 60
  const restStr = rest < 10 ? `0${rest.toFixed(1)}` : rest.toFixed(1)
  return `${String(minutes).padStart(2, '0')}:${restStr}`
}

/** Shell class from the rigging string: "c/p/s/p/s" -> "4+", "x" -> "1x". */
export function shellClassFromRigging(rigging: string): string {
  const seats = rigging.split('/')
  const athletes = seats.length
  const rowers = seats.filter((s) => !s.includes('c')).length
  const isSculling = rigging.includes('x')
  const hasCox = athletes !== rowers
  let cls = String(rowers)
  if (isSculling) cls += 'x'
  if (hasCox) cls += '+'
  if (!isSculling && !hasCox) cls += '-'
  return cls
}

/** "1/5/2012" (M/D/YYYY) -> local-midnight Date. */
export function parseDate(raw: string): Date {
  const m = raw.trim().match(/^(\d{1,2})\/(\d{1,2})\/(\d{4})$/)
  if (!m) throw new Error(`Unparseable date: "${raw}"`)
  return new Date(Number(m[3]), Number(m[1]) - 1, Number(m[2]))
}

export function daysBetween(a: Date, b: Date): number {
  return Math.round((a.getTime() - b.getTime()) / 86_400_000)
}

/**
 * Append rigging superscripts to personnel names; insert "Cox" when the
 * rigging has one more seat than the personnel list (odd-length riggings).
 */
export function suffixPersonnel(rigging: string, personnel: string): string[] {
  const rigs = rigging.split('/')
  let names = personnel.split('/')
  if (rigs.length - 1 === names.length) {
    names = ['Cox', ...names]
  } else if (rigs.length !== names.length) {
    throw new Error(`Rigging and personnel differ in length: "${rigging}" vs "${personnel}"`)
  }
  return names.map((name, i) => {
    if (name && SUPERSCRIPT_SET.has(name[name.length - 1])) return name
    return name + (RIG_SUPERSCRIPTS[rigs[i]] ?? '')
  })
}

export function stripRigging(name: string): string {
  let end = name.length
  while (end > 0 && SUPERSCRIPT_SET.has(name[end - 1])) end--
  return name.slice(0, end)
}

/**
 * Full prep pipeline for one dataset under the given settings. Returns rows
 * sorted the way the Python engine sorts them (lexically by raw date string,
 * then by piece number, stable), filtered to the selected shell classes, with
 * weights and athlete fractions attached.
 */
export function prepRows(raw: RaceRow[], settings: WeightSettings): PreppedRow[] {
  const kept = raw.filter((r) => settings.shellClasses.includes(shellClassFromRigging(r.rigging)))
  const sorted = [...kept].sort((a, b) => {
    if (a.dateRaw !== b.dateRaw) return a.dateRaw < b.dateRaw ? -1 : 1
    return a.pieceNumber - b.pieceNumber
  })
  const rows: PreppedRow[] = sorted.map((r) => {
    const timeSeconds = timeToSeconds(r.result)
    return {
      dateRaw: r.dateRaw,
      date: parseDate(r.dateRaw),
      pieceNumber: r.pieceNumber,
      piece: `${r.dateRaw} #${r.pieceNumber}`,
      km: r.km,
      rigging: r.rigging,
      personnel: suffixPersonnel(r.rigging, r.personnel),
      shellClass: shellClassFromRigging(r.rigging),
      timeSeconds,
      timePer500m: timeSeconds / (r.km * 2.0),
      closestMargin: null,
      closenessFactor: 1,
      recencyFactor: 1,
      totalWeight: 1,
      athleteFractions: new Map(),
    }
  })
  closestMargins(rows)
  applyWeights(rows, settings)
  athleteFractions(rows, settings)
  return rows
}

/** Athletes in first-appearance order, optionally without coxswains. */
export function collectAthletes(rows: PreppedRow[], includeCoxswains: boolean): string[] {
  const seen = new Set<string>()
  const out: string[] = []
  for (const row of rows) {
    for (const name of row.personnel) {
      if (!seen.has(name)) {
        seen.add(name)
        if (includeCoxswains || !name.endsWith('ᶜ')) out.push(name)
      }
    }
  }
  return out
}

/** Shell classes in first-appearance order. */
export function collectShellClasses(rows: PreppedRow[]): string[] {
  const seen = new Set<string>()
  const out: string[] = []
  for (const row of rows) {
    if (!seen.has(row.shellClass)) {
      seen.add(row.shellClass)
      out.push(row.shellClass)
    }
  }
  return out
}
