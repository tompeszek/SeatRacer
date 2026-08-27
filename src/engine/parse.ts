// CSV parsing for the SeatRacer schema:
//   Race Session (date), Piece, KM, Rigging, Personnel, Result
import type { RaceRow } from './types'

/** Minimal CSV parser: handles quoted fields, CRLF, and a UTF-8 BOM. */
export function parseCsv(text: string): string[][] {
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1)
  const rows: string[][] = []
  let field = ''
  let row: string[] = []
  let inQuotes = false
  for (let i = 0; i < text.length; i++) {
    const ch = text[i]
    if (inQuotes) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          field += '"'
          i++
        } else inQuotes = false
      } else field += ch
    } else if (ch === '"') {
      inQuotes = true
    } else if (ch === ',') {
      row.push(field)
      field = ''
    } else if (ch === '\n' || ch === '\r') {
      if (ch === '\r' && text[i + 1] === '\n') i++
      row.push(field)
      field = ''
      rows.push(row)
      row = []
    } else field += ch
  }
  if (field !== '' || row.length > 0) {
    row.push(field)
    rows.push(row)
  }
  // Drop fully empty trailing lines.
  return rows.filter((r) => r.some((c) => c.trim() !== ''))
}

const REQUIRED = ['Race Session (date)', 'Piece', 'KM', 'Rigging', 'Personnel', 'Result']

export function parseRaceCsv(text: string): RaceRow[] {
  const grid = parseCsv(text)
  if (grid.length < 2) throw new Error('CSV has no data rows')
  const header = grid[0].map((h) => h.trim())
  const idx: Record<string, number> = {}
  for (const col of REQUIRED) {
    const i = header.indexOf(col)
    if (i < 0) throw new Error(`CSV is missing the "${col}" column`)
    idx[col] = i
  }
  const rows: RaceRow[] = []
  for (const line of grid.slice(1)) {
    const get = (c: string) => (line[idx[c]] ?? '').trim()
    if (get('Personnel') === '' && get('Result') === '') continue
    rows.push({
      dateRaw: get('Race Session (date)'),
      pieceNumber: Number(get('Piece')),
      km: Number(get('KM')),
      rigging: get('Rigging'),
      personnel: get('Personnel'),
      result: get('Result'),
    })
  }
  return rows
}
