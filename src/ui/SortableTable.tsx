import { useMemo, useState, type ReactNode } from 'react'

export interface Column<T> {
  key: string
  label: string
  num?: boolean
  /** Value used for sorting; also the display unless render is given. */
  value: (row: T) => string | number
  render?: (row: T) => ReactNode
}

interface Props<T> {
  columns: Array<Column<T>>
  rows: T[]
  defaultSort?: string
  defaultDesc?: boolean
  rowKey: (row: T, index: number) => string
  /** When set, a heavier separator is drawn where this key changes. */
  groupKey?: (row: T) => string
}

export function SortableTable<T>({ columns, rows, defaultSort, defaultDesc, rowKey, groupKey }: Props<T>) {
  const [sortKey, setSortKey] = useState<string | null>(defaultSort ?? null)
  const [desc, setDesc] = useState(defaultDesc ?? false)

  const sorted = useMemo(() => {
    if (!sortKey) return rows
    const col = columns.find((c) => c.key === sortKey)
    if (!col) return rows
    const withIndex = rows.map((row, i) => ({ row, i }))
    withIndex.sort((a, b) => {
      const va = col.value(a.row)
      const vb = col.value(b.row)
      let cmp: number
      if (typeof va === 'number' && typeof vb === 'number') {
        const na = Number.isNaN(va) ? Infinity : va
        const nb = Number.isNaN(vb) ? Infinity : vb
        cmp = na - nb
      } else {
        cmp = String(va) < String(vb) ? -1 : String(va) > String(vb) ? 1 : 0
      }
      if (cmp === 0) cmp = a.i - b.i
      return desc ? -cmp : cmp
    })
    return withIndex.map((x) => x.row)
  }, [rows, columns, sortKey, desc])

  const onSort = (key: string) => {
    if (sortKey === key) setDesc(!desc)
    else {
      setSortKey(key)
      setDesc(false)
    }
  }

  return (
    <div className="table-card">
      <table>
        <thead>
          <tr>
            {columns.map((c) => (
              <th
                key={c.key}
                className={`sortable${c.num ? ' num' : ''}`}
                onClick={() => onSort(c.key)}
              >
                {c.label}
                {sortKey === c.key && <span className="sort-arrow">{desc ? '▼' : '▲'}</span>}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((row, i) => {
            const sep =
              groupKey && i > 0 && groupKey(row) !== groupKey(sorted[i - 1]) ? 'day-sep' : ''
            return (
              <tr key={rowKey(row, i)} className={sep}>
                {columns.map((c) => (
                  <td key={c.key} className={c.num ? 'num' : ''}>
                    {c.render ? c.render(row) : c.value(row)}
                  </td>
                ))}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
