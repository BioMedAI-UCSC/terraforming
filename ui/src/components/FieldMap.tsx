import { useEffect, useMemo, useRef, useState } from 'react'
import { getRunFields, getRunSnapshots } from '../api'
import type { FieldGrid, RunFields } from '../types'

// ── Colormaps (control-point RGB stops, sampled by interpolation) ─────────────
const COLORMAPS: Record<string, number[][]> = {
  inferno: [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99],
            [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]],
  viridis: [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]],
  RdBu:    [[103, 0, 31], [244, 165, 130], [247, 247, 247], [146, 197, 222], [5, 48, 97]],
  gray:    [[0, 0, 0], [255, 255, 255]],
}

function sample(stops: number[][], t: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, t)) * (stops.length - 1)
  const i = Math.min(stops.length - 2, Math.floor(x))
  const f = x - i
  const a = stops[i], b = stops[i + 1]
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]
}

interface Selected { key: string; grid: FieldGrid; kind: 'map' | 'section' }

const CANVAS_W = 760
const CANVAS_H = 400

export function FieldMap({ runId }: { runId: string }) {
  const [fields, setFields] = useState<RunFields | null>(null)
  const [error, setError]   = useState<string | null>(null)
  const [key, setKey]       = useState<string>('map:surface_temperature')
  const [cmapName, setCmap] = useState<string>('inferno')
  const [vmin, setVmin]     = useState<number | null>(null)
  const [vmax, setVmax]     = useState<number | null>(null)
  const [years, setYears]   = useState<number[]>([])
  const [year, setYear]     = useState<number | null>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // Discover snapshot years (terraforming timeline). Default to the last (final).
  useEffect(() => {
    let ok = true
    getRunSnapshots(runId)
      .then(s => { if (ok && s.years.length) { setYears(s.years); setYear(s.years[s.years.length - 1]) } })
      .catch(() => { /* no timeline: single-snapshot run */ })
    return () => { ok = false }
  }, [runId])

  // Fetch the field grids for the selected year (or the headline snapshot).
  useEffect(() => {
    let ok = true
    getRunFields(runId, year ?? undefined)
      .then(f => { if (ok) { setFields(f); setError(null) } })
      .catch(() => { if (ok) setError('No field data for this run yet.') })
    return () => { ok = false }
  }, [runId, year])

  // Flatten maps + sections into a single option list.
  const options = useMemo(() => {
    if (!fields) return [] as { key: string; label: string }[]
    return [
      ...Object.entries(fields.maps).map(([k, g]) => ({ key: `map:${k}`, label: g.label })),
      ...Object.entries(fields.sections).map(([k, g]) => ({ key: `section:${k}`, label: g.label })),
    ]
  }, [fields])

  const selected: Selected | null = useMemo(() => {
    if (!fields) return null
    const [kind, name] = key.split(':') as ['map' | 'section', string]
    const grid = kind === 'map' ? fields.maps[name] : fields.sections[name]
    return grid ? { key, grid, kind } : null
  }, [fields, key])

  // Reset the value range to the field's own min/max when the field changes.
  useEffect(() => {
    if (selected) { setVmin(selected.grid.min); setVmax(selected.grid.max) }
  }, [key, selected])

  // Draw the heatmap + colorbar + axes onto the canvas.
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !fields || !selected || vmin == null || vmax == null) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const stops = COLORMAPS[cmapName]
    const lo = vmin, hi = vmax, span = hi - lo || 1

    // Rows top→bottom; for maps flip so north is on top.
    const raw = selected.grid.data
    const grid = selected.kind === 'map' ? [...raw].reverse() : raw
    const rows = grid.length, cols = grid[0].length

    const mL = 54, mR = 96, mT = 34, mB = 44
    const pw = CANVAS_W - mL - mR, ph = CANVAS_H - mT - mB

    ctx.fillStyle = '#0e1116'; ctx.fillRect(0, 0, CANVAS_W, CANVAS_H)

    // Heatmap cells.
    const cw = pw / cols, ch = ph / rows
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const [r, g, b] = sample(stops, (grid[i][j] - lo) / span)
        ctx.fillStyle = `rgb(${r | 0},${g | 0},${b | 0})`
        ctx.fillRect(mL + j * cw, mT + i * ch, cw + 1, ch + 1)
      }
    }
    ctx.strokeStyle = '#3a3f47'; ctx.strokeRect(mL, mT, pw, ph)

    // Colorbar.
    const cbX = CANVAS_W - mR + 24, cbW = 14
    for (let k = 0; k < ph; k++) {
      const [r, g, b] = sample(stops, 1 - k / ph)
      ctx.fillStyle = `rgb(${r | 0},${g | 0},${b | 0})`
      ctx.fillRect(cbX, mT + k, cbW, 1)
    }
    ctx.strokeRect(cbX, mT, cbW, ph)

    // Text (axes, ticks, title).
    ctx.fillStyle = '#c9d1d9'; ctx.font = '12px system-ui, sans-serif'
    ctx.textBaseline = 'middle'
    const isMap = selected.kind === 'map'
    const xlab = isMap ? 'Longitude (°E)' : 'Latitude (°)'
    const ylab = isMap ? 'Latitude (°)' : 'sigma'
    ctx.textAlign = 'center'
    ctx.fillText(`${selected.grid.label} (${selected.grid.units})`, mL + pw / 2, 14)
    ctx.fillText(xlab, mL + pw / 2, CANVAS_H - 12)

    // Colorbar ticks (hi at top, lo at bottom).
    ctx.textAlign = 'left'
    ctx.fillText(hi.toFixed(1), cbX + cbW + 4, mT + 2)
    ctx.fillText(((hi + lo) / 2).toFixed(1), cbX + cbW + 4, mT + ph / 2)
    ctx.fillText(lo.toFixed(1), cbX + cbW + 4, mT + ph - 2)

    // Axis tick labels.
    ctx.fillStyle = '#8b949e'
    ctx.textAlign = 'right'
    const yTop = isMap ? '90' : '0';  const yBot = isMap ? '-90' : '1'
    ctx.fillText(yTop, mL - 6, mT + 4); ctx.fillText(yBot, mL - 6, mT + ph - 4)
    ctx.save(); ctx.translate(14, mT + ph / 2); ctx.rotate(-Math.PI / 2)
    ctx.textAlign = 'center'; ctx.fillStyle = '#c9d1d9'; ctx.fillText(ylab, 0, 0); ctx.restore()
    ctx.textAlign = 'center'; ctx.fillStyle = '#8b949e'
    const xLo = isMap ? '0' : '-90'; const xHi = isMap ? '360' : '90'
    ctx.fillText(xLo, mL, CANVAS_H - 28); ctx.fillText(xHi, mL + pw, CANVAS_H - 28)
  }, [fields, selected, cmapName, vmin, vmax, key])

  function exportPNG() {
    const canvas = canvasRef.current
    if (!canvas) return
    const a = document.createElement('a')
    a.href = canvas.toDataURL('image/png')
    a.download = `marsgcm_${key.replace(':', '_')}.png`
    a.click()
  }

  function exportCSV() {
    if (!fields || !selected) return
    const g = selected.grid.data
    const csv = g.map(row => row.join(',')).join('\n')
    const a = document.createElement('a')
    a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }))
    a.download = `marsgcm_${key.replace(':', '_')}.csv`
    a.click()
  }

  if (error) return <p style={{ color: '#8b949e', padding: 12 }}>{error}</p>
  if (!fields || !selected || vmin == null || vmax == null)
    return <p style={{ color: '#8b949e', padding: 12 }}>Loading fields…</p>

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10, padding: 12 }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center' }}>
        <label style={lbl}>Field&nbsp;
          <select value={key} onChange={e => setKey(e.target.value)} style={sel}>
            {options.map(o => <option key={o.key} value={o.key}>{o.label}</option>)}
          </select>
        </label>
        <label style={lbl}>Colormap&nbsp;
          <select value={cmapName} onChange={e => setCmap(e.target.value)} style={sel}>
            {Object.keys(COLORMAPS).map(c => <option key={c} value={c}>{c}</option>)}
          </select>
        </label>
        <label style={lbl}>Min&nbsp;
          <input type="number" step="any" value={vmin}
                 onChange={e => setVmin(parseFloat(e.target.value))} style={num} />
        </label>
        <label style={lbl}>Max&nbsp;
          <input type="number" step="any" value={vmax}
                 onChange={e => setVmax(parseFloat(e.target.value))} style={num} />
        </label>
        <button onClick={() => { setVmin(selected.grid.min); setVmax(selected.grid.max) }} style={btn}>Auto</button>
        <button onClick={exportPNG} style={btn}>Export PNG</button>
        <button onClick={exportCSV} style={btn}>Export CSV</button>
      </div>
      {years.length > 1 && year != null && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ ...lbl, minWidth: 96 }}>Year&nbsp;<b>{year}</b></span>
          <input type="range" min={0} max={years.length - 1}
                 value={years.indexOf(year)}
                 onChange={e => setYear(years[parseInt(e.target.value)])}
                 style={{ flex: 1, maxWidth: 520, accentColor: '#d1552b' }} />
          <span style={{ color: '#8b949e', fontSize: 12 }}>
            {years[0]}–{years[years.length - 1]} yr · {years.length} snapshots
          </span>
        </div>
      )}
      <canvas ref={canvasRef} width={CANVAS_W} height={CANVAS_H}
              style={{ width: '100%', maxWidth: CANVAS_W, borderRadius: 6, border: '1px solid #30363d' }} />
    </div>
  )
}

const lbl: React.CSSProperties = { color: '#c9d1d9', fontSize: 13 }
const sel: React.CSSProperties = { background: '#161b22', color: '#c9d1d9', border: '1px solid #30363d', borderRadius: 4, padding: '3px 6px' }
const num: React.CSSProperties = { ...sel, width: 80 }
const btn: React.CSSProperties = { ...sel, cursor: 'pointer' }
