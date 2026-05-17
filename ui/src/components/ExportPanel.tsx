import { useRef, useState } from 'react'
import {
  AreaChart, Area, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend,
} from 'recharts'
import { downloadAsPng } from '../utils/exportChart'
import type { DataPoint, RunSummary } from '../types'
import { genTicks, getAxisInfo } from '../utils/chartAxis'

// ── Color schemes ─────────────────────────────────────────────────────────────

type Scheme = 'mars' | 'science' | 'mono'
type Bg     = 'dark' | 'light'

const SCHEMES: Record<Scheme, { label: string; temp: string; pressure: string; ice: string; dF: string; ghf: string; band: string }> = {
  mars: {
    label:    'Mars',
    temp:     '#d62728',
    pressure: '#c1440e',
    ice:      '#ff7f0e',
    dF:       '#9467bd',
    ghf:      '#e377c2',
    band:     '#3a0f0f',
  },
  science: {
    label:    'Science',
    temp:     '#d62728',
    pressure: '#1f77b4',
    ice:      '#2ca02c',
    dF:       '#ff7f0e',
    ghf:      '#9467bd',
    band:     '#ffeaea',
  },
  mono: {
    label:    'Mono',
    temp:     '#cccccc',
    pressure: '#999999',
    ice:      '#666666',
    dF:       '#888888',
    ghf:      '#555555',
    band:     '#2a2a2a',
  },
}

const BG_STYLES: Record<Bg, { bg: string; text: string; grid: string; axis: string; border: string }> = {
  dark:  { bg: '#0a0a0a', text: '#e0e0e0', grid: '#1e1e1e', axis: '#444444', border: '#2a2a2a' },
  light: { bg: '#ffffff', text: '#111111', grid: '#e8e8e8', axis: '#888888', border: '#dddddd' },
}

// ── Chart definitions ─────────────────────────────────────────────────────────

interface ChartDef {
  id:        'temperature' | 'pressure' | 'ice' | 'deltaF'
  label:     string
  ivOnly:    boolean
}

const CHART_DEFS: ChartDef[] = [
  { id: 'temperature', label: 'Temperature (K)',          ivOnly: false },
  { id: 'pressure',    label: 'Pressure (Pa)',            ivOnly: false },
  { id: 'ice',         label: 'Ice Mass (kg)',            ivOnly: false },
  { id: 'deltaF',      label: 'ΔF Radiative Forcing',    ivOnly: true  },
]

// ── Export options state ───────────────────────────────────────────────────────

interface ExportOptions {
  title:    string
  scheme:   Scheme
  bg:       Bg
  legend:   boolean
  included: Set<ChartDef['id']>
}

const DEFAULT_PRESET: ExportOptions = {
  title:    '',
  scheme:   'mars',
  bg:       'dark',
  legend:   true,
  included: new Set(['temperature', 'pressure', 'ice', 'deltaF']),
}

// ── Component ─────────────────────────────────────────────────────────────────

interface Props {
  data:  DataPoint[]
  run:   RunSummary
  onClose: () => void
}

export function ExportPanel({ data, run, onClose }: Props) {
  const isIV             = run.config.exp_type === 'intervention'
  const { xKey, xLabel } = getAxisInfo(run)

  const [opts, setOpts] = useState<ExportOptions>({
    ...DEFAULT_PRESET,
    title:    run.label,
    included: new Set(
      CHART_DEFS.filter(c => !c.ivOnly || isIV).map(c => c.id)
    ),
  })
  const [exporting, setExporting] = useState(false)
  const previewRef = useRef<HTMLDivElement>(null)

  const set = <K extends keyof ExportOptions>(k: K, v: ExportOptions[K]) =>
    setOpts(prev => ({ ...prev, [k]: v }))

  const toggleChart = (id: ChartDef['id']) =>
    setOpts(prev => {
      const next = new Set(prev.included)
      next.has(id) ? next.delete(id) : next.add(id)
      return { ...prev, included: next }
    })

  const handleDownload = async () => {
    if (!previewRef.current) return
    setExporting(true)
    try {
      const slug = opts.title.replace(/\s+/g, '-').toLowerCase() || run.id
      await downloadAsPng(previewRef.current, { title: opts.title, background: opts.bg }, `tform-${slug}.png`)
    } finally {
      setExporting(false)
    }
  }

  const c   = SCHEMES[opts.scheme]
  const bg  = BG_STYLES[opts.bg]
  const defs = CHART_DEFS.filter(d => !d.ivOnly || isIV)
  const tick = { fontSize: 10, fill: bg.axis }
  const tooltipStyle: React.CSSProperties = {
    background: opts.bg === 'dark' ? '#181818' : '#f5f5f5',
    border: `1px solid ${bg.border}`,
    borderRadius: 4,
    fontSize: 11,
    color: bg.text,
  }

  return (
    <div style={s.overlay} onClick={e => { if (e.target === e.currentTarget) onClose() }}>
      <div style={s.modal}>

        {/* Header */}
        <div style={s.header}>
          <span style={s.heading}>Export Charts</span>
          <button style={s.closeBtn} onClick={onClose}>×</button>
        </div>

        <div style={s.body}>

          {/* ── Options panel ── */}
          <div style={s.options}>

            <div style={s.group}>
              <div style={s.groupLabel}>Title</div>
              <input
                style={s.input}
                value={opts.title}
                onChange={e => set('title', e.target.value)}
                placeholder="Chart title"
              />
            </div>

            <div style={s.group}>
              <div style={s.groupLabel}>Charts</div>
              {defs.map(d => (
                <label key={d.id} style={s.checkRow}>
                  <input
                    type="checkbox"
                    checked={opts.included.has(d.id)}
                    onChange={() => toggleChart(d.id)}
                    style={{ accentColor: '#c1440e' }}
                  />
                  <span style={{ fontSize: 12, color: opts.included.has(d.id) ? '#e0e0e0' : '#555' }}>
                    {d.label}
                  </span>
                </label>
              ))}
            </div>

            <div style={s.group}>
              <div style={s.groupLabel}>Color scheme</div>
              <div style={s.chipRow}>
                {(Object.keys(SCHEMES) as Scheme[]).map(k => (
                  <button
                    key={k}
                    style={{ ...s.chip, ...(opts.scheme === k ? s.chipActive : {}) }}
                    onClick={() => set('scheme', k)}
                  >
                    <span style={{ width: 8, height: 8, borderRadius: '50%', background: SCHEMES[k].temp, display: 'inline-block', marginRight: 5 }} />
                    {SCHEMES[k].label}
                  </button>
                ))}
              </div>
            </div>

            <div style={s.group}>
              <div style={s.groupLabel}>Background</div>
              <div style={s.chipRow}>
                {(['dark', 'light'] as Bg[]).map(b => (
                  <button
                    key={b}
                    style={{ ...s.chip, ...(opts.bg === b ? s.chipActive : {}) }}
                    onClick={() => set('bg', b)}
                  >{b}</button>
                ))}
              </div>
            </div>

            <div style={s.group}>
              <div style={s.groupLabel}>Legend</div>
              <label style={s.checkRow}>
                <input
                  type="checkbox"
                  checked={opts.legend}
                  onChange={e => set('legend', e.target.checked)}
                  style={{ accentColor: '#c1440e' }}
                />
                <span style={{ fontSize: 12, color: '#aaa' }}>Show legend</span>
              </label>
            </div>

            <button
              style={{ ...s.downloadBtn, opacity: exporting ? 0.6 : 1 }}
              onClick={handleDownload}
              disabled={exporting || opts.included.size === 0}
            >
              {exporting ? 'Exporting…' : 'Download PNG'}
            </button>
          </div>

          {/* ── Preview ── */}
          <div style={s.previewWrap}>
            <div style={s.previewLabel}>Preview</div>
            <div
              ref={previewRef}
              style={{ ...s.preview, background: bg.bg }}
            >
              {/* Title row */}
              {opts.title && (
                <div style={{ padding: '14px 16px 2px', color: bg.text, fontSize: 13, fontWeight: 700 }}>
                  {opts.title}
                </div>
              )}

              {/* Selected charts */}
              {defs.filter(d => opts.included.has(d.id)).map(d => (
                <div key={d.id} style={{ padding: '4px 0' }}>
                  <div style={{ fontSize: 10, color: bg.axis, padding: '0 16px 2px' }}>{d.label}</div>
                  <ResponsiveContainer width="100%" height={130}>
                    {d.id === 'temperature' && isIV && data[0]?.temp_min_k != null
                      ? <AreaChart data={data} syncId="export">
                          <CartesianGrid strokeDasharray="3 3" stroke={bg.grid} />
                          <XAxis dataKey={xKey} type="number" domain={[0, 'dataMax']} ticks={genTicks(data, xKey)} stroke={bg.axis} tick={tick} tickFormatter={(v: number) => Number.isInteger(v) ? String(v) : parseFloat(v.toFixed(2)).toString()} label={{ value: xLabel, position: 'insideBottom', offset: -3, style: { fontSize: 9, fill: bg.axis } }} />
                          <YAxis stroke={bg.axis} tick={tick} width={62} />
                          <Tooltip contentStyle={tooltipStyle} />
                          {opts.legend && <Legend wrapperStyle={{ fontSize: 10, color: bg.axis }} />}
                          <Area type="monotone" dataKey="temp_max_k" stroke="none" fill={c.band} fillOpacity={0.7} legendType="none" name="Max" />
                          <Area type="monotone" dataKey="temp_min_k" stroke="none" fill={bg.bg}  fillOpacity={1}   legendType="none" name="Min" />
                          <Line type="monotone" dataKey="temperature_k" stroke={c.temp} strokeWidth={2} dot={false} name="Avg T (K)" />
                        </AreaChart>
                      : <LineChart data={data} syncId="export">
                          <CartesianGrid strokeDasharray="3 3" stroke={bg.grid} />
                          <XAxis dataKey={xKey} type="number" domain={[0, 'dataMax']} ticks={genTicks(data, xKey)} stroke={bg.axis} tick={tick} tickFormatter={(v: number) => Number.isInteger(v) ? String(v) : parseFloat(v.toFixed(2)).toString()} label={{ value: xLabel, position: 'insideBottom', offset: -3, style: { fontSize: 9, fill: bg.axis } }} />
                          <YAxis stroke={bg.axis} tick={tick} width={d.id === 'ice' ? 76 : 62}
                            tickFormatter={d.id === 'ice' ? v => v.toExponential(1) : undefined}
                          />
                          <Tooltip contentStyle={tooltipStyle} />
                          {opts.legend && <Legend wrapperStyle={{ fontSize: 10, color: bg.axis }} />}
                          <Line
                            type="monotone"
                            dataKey={d.id === 'temperature' ? 'temperature_k' : d.id === 'pressure' ? 'pressure_pa' : d.id === 'ice' ? 'ice_mass_kg' : 'delta_F'}
                            stroke={d.id === 'temperature' ? c.temp : d.id === 'pressure' ? c.pressure : d.id === 'ice' ? c.ice : c.dF}
                            strokeWidth={2}
                            dot={false}
                            name={d.label}
                          />
                          {d.id === 'deltaF' && (
                            <Line type="monotone" dataKey="greenhouse_factor" stroke={c.ghf} strokeWidth={1.5} dot={false} name="GHF" strokeDasharray="4 2" yAxisId="ghf" />
                          )}
                          {d.id === 'deltaF' && (
                            <YAxis yAxisId="ghf" orientation="right" stroke={c.ghf} tick={{ ...tick, fill: c.ghf }} width={48} />
                          )}
                        </LineChart>
                    }
                  </ResponsiveContainer>
                </div>
              ))}

              {opts.included.size === 0 && (
                <div style={{ color: bg.axis, fontSize: 12, padding: 24, textAlign: 'center' }}>
                  Select at least one chart.
                </div>
              )}
            </div>
          </div>

        </div>
      </div>
    </div>
  )
}

// ── Styles ────────────────────────────────────────────────────────────────────

const s: Record<string, React.CSSProperties> = {
  overlay:     { position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.75)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 100 },
  modal:       { background: '#111', border: '1px solid #2a2a2a', borderRadius: 8, width: 900, maxWidth: '95vw', maxHeight: '90vh', display: 'flex', flexDirection: 'column', overflow: 'hidden' },
  header:      { display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px 16px', borderBottom: '1px solid #1e1e1e', flexShrink: 0 },
  heading:     { fontSize: 14, fontWeight: 700, color: '#e0e0e0' },
  closeBtn:    { background: 'none', border: 'none', color: '#666', fontSize: 20, cursor: 'pointer', lineHeight: 1, padding: '0 4px' },
  body:        { display: 'flex', flex: 1, overflow: 'hidden' },
  options:     { width: 200, flexShrink: 0, borderRight: '1px solid #1e1e1e', padding: '12px 14px', display: 'flex', flexDirection: 'column', gap: 16, overflowY: 'auto' },
  group:       { display: 'flex', flexDirection: 'column', gap: 6 },
  groupLabel:  { fontSize: 10, fontWeight: 700, color: '#c1440e', textTransform: 'uppercase', letterSpacing: '0.07em' },
  input:       { background: '#181818', border: '1px solid #2a2a2a', color: '#e0e0e0', padding: '5px 8px', borderRadius: 4, fontSize: 12, fontFamily: 'inherit', width: '100%' },
  checkRow:    { display: 'flex', alignItems: 'center', gap: 7, cursor: 'pointer' },
  chipRow:     { display: 'flex', gap: 5, flexWrap: 'wrap' },
  chip:        { background: '#1a1a1a', border: '1px solid #2a2a2a', color: '#777', padding: '4px 9px', borderRadius: 4, fontSize: 11, cursor: 'pointer', display: 'flex', alignItems: 'center' },
  chipActive:  { background: '#2a1008', border: '1px solid #c1440e', color: '#c1440e' },
  downloadBtn: { background: '#c1440e', border: 'none', color: '#fff', padding: '8px 0', borderRadius: 4, fontSize: 13, fontWeight: 700, cursor: 'pointer', fontFamily: 'inherit', marginTop: 'auto' },
  previewWrap: { flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' },
  previewLabel:{ fontSize: 10, color: '#444', padding: '8px 16px 0', textTransform: 'uppercase', letterSpacing: '0.06em', flexShrink: 0 },
  preview:     { flex: 1, overflowY: 'auto', padding: '0 0 12px' },
}
