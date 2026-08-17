import { useEffect, useRef, useState } from 'react'
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

const TILE_W = 360
const TILE_H = 220

// Draw one field (heatmap + colorbar + axis ticks) onto a canvas, auto-scaled
// to its own min/max.
function drawField(canvas: HTMLCanvasElement, grid: FieldGrid, cmapName: string,
                   kind: 'map' | 'section') {
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  const stops = COLORMAPS[cmapName]
  const lo = grid.min, hi = grid.max, span = hi - lo || 1

  // Rows top→bottom; for maps flip so north is on top.
  const data = kind === 'map' ? [...grid.data].reverse() : grid.data
  const rows = data.length, cols = data[0].length

  const mL = 34, mR = 58, mT = 8, mB = 22
  const pw = TILE_W - mL - mR, ph = TILE_H - mT - mB

  ctx.fillStyle = '#0e1116'; ctx.fillRect(0, 0, TILE_W, TILE_H)

  const cw = pw / cols, ch = ph / rows
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      const [r, g, b] = sample(stops, (data[i][j] - lo) / span)
      ctx.fillStyle = `rgb(${r | 0},${g | 0},${b | 0})`
      ctx.fillRect(mL + j * cw, mT + i * ch, cw + 1, ch + 1)
    }
  }
  ctx.strokeStyle = '#3a3f47'; ctx.strokeRect(mL, mT, pw, ph)

  // Colorbar.
  const cbX = TILE_W - mR + 16, cbW = 10
  for (let k = 0; k < ph; k++) {
    const [r, g, b] = sample(stops, 1 - k / ph)
    ctx.fillStyle = `rgb(${r | 0},${g | 0},${b | 0})`
    ctx.fillRect(cbX, mT + k, cbW, 1)
  }
  ctx.strokeRect(cbX, mT, cbW, ph)

  ctx.font = '10px system-ui, sans-serif'
  ctx.textBaseline = 'middle'
  const isMap = kind === 'map'

  // Colorbar ticks.
  ctx.fillStyle = '#c9d1d9'; ctx.textAlign = 'left'
  ctx.fillText(hi.toFixed(1), cbX + cbW + 3, mT + 4)
  ctx.fillText(lo.toFixed(1), cbX + cbW + 3, mT + ph - 4)

  // Axis tick labels.
  ctx.fillStyle = '#8b949e'; ctx.textAlign = 'right'
  ctx.fillText(isMap ? '90' : '0', mL - 4, mT + 5)
  ctx.fillText(isMap ? '-90' : '1', mL - 4, mT + ph - 5)
  ctx.textAlign = 'center'
  ctx.fillText(isMap ? '0' : '-90', mL + 6, TILE_H - 8)
  ctx.fillText(isMap ? '360' : '90', mL + pw - 10, TILE_H - 8)
}

// One field tile: title/units header + heatmap canvas + PNG export.
function MapTile({ name, grid, cmapName, kind }:
                 { name: string; grid: FieldGrid; cmapName: string; kind: 'map' | 'section' }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    if (ref.current) drawField(ref.current, grid, cmapName, kind)
  }, [grid, cmapName, kind])

  function exportPNG() {
    if (!ref.current) return
    const a = document.createElement('a')
    a.href = ref.current.toDataURL('image/png')
    a.download = `marsgcm_${name}.png`
    a.click()
  }

  return (
    <div style={tile}>
      <div style={tileHead}>
        <span style={{ color: '#c9d1d9', fontSize: 12, fontWeight: 600 }}>
          {grid.label} <span style={{ color: '#8b949e', fontWeight: 400 }}>({grid.units})</span>
        </span>
        <button onClick={exportPNG} style={miniBtn} title="Export PNG">PNG</button>
      </div>
      <canvas ref={ref} width={TILE_W} height={TILE_H}
              style={{ width: '100%', display: 'block', borderRadius: 4 }} />
    </div>
  )
}

export function FieldMap({ runId }: { runId: string }) {
  const [fields, setFields] = useState<RunFields | null>(null)
  const [error, setError]   = useState<string | null>(null)
  const [cmapName, setCmap] = useState<string>('inferno')
  const [years, setYears]   = useState<number[]>([])
  const [year, setYear]     = useState<number | null>(null)
  const [source, setSource] = useState<'gcm' | 'mcd' | 'difference'>('gcm')

  useEffect(() => { setSource('gcm') }, [runId])

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

  if (error) return <p style={{ color: '#8b949e', padding: 12 }}>{error}</p>
  if (!fields) return <p style={{ color: '#8b949e', padding: 12 }}>Loading fields…</p>

  const selectedMaps = source === 'gcm'
    ? fields.maps
    : source === 'mcd'
      ? fields.comparison?.mcd ?? {}
      : fields.comparison?.difference ?? {}
  const mapEntries = Object.entries(selectedMaps)
  const sectionEntries = Object.entries(fields.sections)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, padding: 12 }}>
      {fields.metadata?.is_transient && (
        <div style={{ color: '#f0b45a', background: '#2a2115', border: '1px solid #6b4b1f',
                      borderRadius: 5, padding: '8px 10px', fontSize: 12 }}>
          Diagnostic transient ({fields.metadata.duration_sols.toFixed(1)} sols), not an
          equilibrated Mars climatology. {fields.metadata.fidelity}
        </div>
      )}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center' }}>
        <label style={lbl}>Output&nbsp;
          <select value={source} onChange={e => setSource(e.target.value as typeof source)} style={sel}>
            <option value="gcm">GCM</option>
            {fields.comparison && <option value="mcd">MCD</option>}
            {fields.comparison && <option value="difference">GCM − MCD</option>}
          </select>
        </label>
        <label style={lbl}>Colormap&nbsp;
          <select value={cmapName} onChange={e => setCmap(e.target.value)} style={sel}>
            {Object.keys(COLORMAPS).map(c => <option key={c} value={c}>{c}</option>)}
          </select>
        </label>
        <span style={{ color: '#8b949e', fontSize: 12 }}>
          {mapEntries.length + sectionEntries.length} fields · each auto-scaled to its own range
        </span>
      </div>

      {fields.comparison && (
        <div style={{ background: '#0e1116', border: '1px solid #30363d', borderRadius: 6, padding: 10 }}>
          <div style={{ color: '#c9d1d9', fontWeight: 600, fontSize: 12, marginBottom: 8 }}>
            Matched MCD v{fields.comparison.metadata.mcd_version} diagnostics · Ls {fields.comparison.metadata.ls_deg.toFixed(1)}° · wind {fields.comparison.metadata.wind_altitude_m.toFixed(0)} m
          </div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11, color: '#aeb6c0' }}>
              <thead><tr>{['Field', 'Bias', 'MAE', 'RMSE', 'Correlation', 'GCM mean', 'MCD mean'].map(h =>
                <th key={h} style={{ textAlign: h === 'Field' ? 'left' : 'right', padding: '4px 6px', borderBottom: '1px solid #30363d' }}>{h}</th>)}</tr></thead>
              <tbody>{Object.entries(fields.comparison.metrics).map(([name, m]) =>
                <tr key={name}>
                  <td style={{ padding: '4px 6px' }}>{name.replace(/_/g, ' ')}</td>
                  {[m.bias, m.mae, m.rmse, m.spatial_correlation, m.model_area_mean, m.mcd_area_mean].map((v, i) =>
                    <td key={i} style={{ textAlign: 'right', padding: '4px 6px', fontVariantNumeric: 'tabular-nums' }}>{v.toFixed(3)}</td>)}
                </tr>)}</tbody>
            </table>
          </div>
        </div>
      )}

      {years.length > 1 && year != null && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ ...lbl, minWidth: 84 }}>Year&nbsp;<b>{year}</b></span>
          <input type="range" min={0} max={years.length - 1}
                 value={years.indexOf(year)}
                 onChange={e => setYear(years[parseInt(e.target.value)])}
                 style={{ flex: 1, maxWidth: 520, accentColor: '#d1552b' }} />
          <span style={{ color: '#8b949e', fontSize: 12 }}>
            {years[0]}–{years[years.length - 1]} yr · {years.length} snapshots
          </span>
        </div>
      )}

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))',
        gap: 12,
      }}>
        {mapEntries.map(([k, g]) => (
          <MapTile key={k} name={k} grid={g} cmapName={cmapName} kind="map" />
        ))}
        {sectionEntries.map(([k, g]) => (
          <MapTile key={k} name={k} grid={g} cmapName={cmapName} kind="section" />
        ))}
      </div>
    </div>
  )
}

const lbl: React.CSSProperties = { color: '#c9d1d9', fontSize: 13 }
const sel: React.CSSProperties = { background: '#161b22', color: '#c9d1d9', border: '1px solid #30363d', borderRadius: 4, padding: '3px 6px' }
const tile: React.CSSProperties = { background: '#0e1116', border: '1px solid #30363d', borderRadius: 6, padding: 8 }
const tileHead: React.CSSProperties = { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6, gap: 8 }
const miniBtn: React.CSSProperties = { ...sel, cursor: 'pointer', fontSize: 11, padding: '2px 6px' }
