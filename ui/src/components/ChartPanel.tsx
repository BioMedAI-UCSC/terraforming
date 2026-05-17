import { useEffect, useRef, useState } from 'react'
import {
  LineChart, Line, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend,
} from 'recharts'
import type { DataPoint, RunSummary } from '../types'
import { genTicks, getAxisInfo } from '../utils/chartAxis'
import { ExportPanel } from './ExportPanel'

const SYNC = 'tform'

interface Props {
  data: DataPoint[]
  run: RunSummary
}

export function ChartPanel({ data, run }: Props) {
  const [showExport, setShowExport] = useState(false)
  const [zoom, setZoom]             = useState(1)
  // measureRef is on a stable wrapper that always renders — ensures the
  // ResizeObserver fires even before the first data point arrives.
  const measureRef  = useRef<HTMLDivElement>(null)
  const [containerW, setContainerW] = useState(600)
  // Three scroll boxes: 0=temp, 1=pressure, 2=bottom row
  const scrollRefs = useRef<(HTMLDivElement | null)[]>([null, null, null])

  const isIV          = run.config.exp_type === 'intervention'
  const { xKey, xLabel } = getAxisInfo(run)
  const pct           = Math.round(run.progress * 100)

  // Track container width for zoom math.
  // The measured div has no padding; the inner charts div has 16px on each side.
  useEffect(() => {
    const el = measureRef.current
    if (!el) return
    const obs = new ResizeObserver(es => {
      setContainerW(Math.max(200, es[0].contentRect.width - 32))
    })
    obs.observe(el)
    return () => obs.disconnect()
  }, [])

  // Sync horizontal scroll across all three scroll boxes
  const syncScroll = (idx: number) => (e: React.UIEvent<HTMLDivElement>) => {
    const sl = e.currentTarget.scrollLeft
    scrollRefs.current.forEach((el, i) => { if (el && i !== idx) el.scrollLeft = sl })
  }

  const chartW = Math.round(containerW * zoom)

  return (
    <div style={s.panel}>

      {/* ── Header ── */}
      <div style={s.header}>
        <div style={s.headerLeft}>
          <span style={s.runTitle}>{run.label}</span>
          <span style={s.runMeta}>
            {run.config.preset} · {run.config.accuracy}
            {isIV && ` · ${run.config.years} yr`}
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {data.length > 0 && (
            <div style={s.zoomGroup}>
              <button style={s.zoomBtn} onClick={() => setZoom(z => Math.max(1, z / 2))}>−</button>
              <span style={s.zoomLabel}>{zoom}×</span>
              <button style={s.zoomBtn} onClick={() => setZoom(z => Math.min(16, z * 2))}>+</button>
            </div>
          )}
          {data.length > 0 && run.status !== 'running' && (
            <button style={s.exportBtn} onClick={() => setShowExport(true)}>Export</button>
          )}
          <StatusBadge status={run.status} pct={pct} />
        </div>
      </div>

      {showExport && (
        <ExportPanel data={data} run={run} onClose={() => setShowExport(false)} />
      )}

      {run.status === 'running' && (
        <div style={s.progressTrack}>
          <div style={{ ...s.progressBar, width: `${pct}%` }} />
        </div>
      )}

      {/* measureRef wrapper always present so ResizeObserver fires immediately */}
      <div ref={measureRef} style={s.chartsOuter}>
      {data.length === 0
        ? <div style={s.waiting}>Waiting for first data point…</div>
        : <div style={s.charts}>

            {/* Temperature — full width */}
            <div
              ref={el => { scrollRefs.current[0] = el }}
              style={s.scrollBox}
              onScroll={syncScroll(0)}
            >
              <div style={{ width: chartW, minWidth: chartW }}>
                <ChartRow label="Temperature (K)" height={200}>
                  {isIV && data[0]?.temp_min_k != null
                    ? <AreaChart data={data} syncId={SYNC}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e1e1e" />
                        <XAxis dataKey={xKey} type="number" domain={[0, 'dataMax']} ticks={genTicks(data, xKey, 10 * zoom)} stroke="#444" tick={tick} tickFormatter={xTickFmt} label={xAxisLabel(xLabel)} />
                        <YAxis stroke="#444" tick={tick} width={70} />
                        <Tooltip contentStyle={tooltipStyle} />
                        <Area type="monotone" dataKey="temp_max_k" stroke="none" fill="#3a0f0f" fillOpacity={0.6} legendType="none" name="Max" />
                        <Area type="monotone" dataKey="temp_min_k" stroke="none" fill="#0a0a0a"  fillOpacity={1}   legendType="none" name="Min" />
                        <Line type="monotone" dataKey="temperature_k" stroke="#d62728" strokeWidth={2} dot={false} name="Avg T (K)" />
                      </AreaChart>
                    : <LineChart data={data} syncId={SYNC}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e1e1e" />
                        <XAxis dataKey={xKey} type="number" domain={[0, 'dataMax']} ticks={genTicks(data, xKey, 10 * zoom)} stroke="#444" tick={tick} tickFormatter={xTickFmt} label={xAxisLabel(xLabel)} />
                        <YAxis stroke="#444" tick={tick} width={70} />
                        <Tooltip contentStyle={tooltipStyle} />
                        <Line type="monotone" dataKey="temperature_k" stroke="#d62728" strokeWidth={2} dot={false} name="T (K)" />
                      </LineChart>
                  }
                </ChartRow>
              </div>
            </div>

            {/* Pressure — full width */}
            <div
              ref={el => { scrollRefs.current[1] = el }}
              style={s.scrollBox}
              onScroll={syncScroll(1)}
            >
              <div style={{ width: chartW, minWidth: chartW }}>
                <ChartRow label="Pressure (Pa)" height={180}>
                  <LineChart data={data} syncId={SYNC}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e1e1e" />
                    <XAxis dataKey={xKey} type="number" domain={[0, 'dataMax']} ticks={genTicks(data, xKey, 10 * zoom)} stroke="#444" tick={tick} tickFormatter={xTickFmt} label={xAxisLabel(xLabel)} />
                    <YAxis stroke="#444" tick={tick} width={70} />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Line type="monotone" dataKey="pressure_pa" stroke="#1f77b4" strokeWidth={2} dot={false} name="P (Pa)" />
                  </LineChart>
                </ChartRow>
              </div>
            </div>

            {/* Bottom row: Ice + ΔF — share one scroll box */}
            <div
              ref={el => { scrollRefs.current[2] = el }}
              style={{ ...s.scrollBox, flex: 1 }}
              onScroll={syncScroll(2)}
            >
              <div style={{ width: chartW, minWidth: chartW, display: 'flex', gap: 16, height: '100%' }}>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <ChartRow label="Ice Mass (kg)" height={180} flex>
                    <LineChart data={data} syncId={SYNC}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e1e1e" />
                      <XAxis dataKey={xKey} type="number" domain={[0, 'dataMax']} ticks={genTicks(data, xKey, 10 * zoom)} stroke="#444" tick={tick} tickFormatter={xTickFmt} label={xAxisLabel(xLabel)} />
                      <YAxis stroke="#444" tick={tick} width={78} tickFormatter={sciTick} />
                      <Tooltip contentStyle={tooltipStyle} formatter={sciFormatter} />
                      <Line type="monotone" dataKey="ice_mass_kg" stroke="#2ca02c" strokeWidth={2} dot={false} name="Ice (kg)" />
                    </LineChart>
                  </ChartRow>
                </div>

                {isIV && (
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <ChartRow label="ΔF Radiative Forcing (W/m²)" height={180} flex>
                      <LineChart data={data} syncId={SYNC}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e1e1e" />
                        <XAxis dataKey={xKey} type="number" domain={[0, 'dataMax']} ticks={genTicks(data, xKey, 10 * zoom)} stroke="#444" tick={tick} tickFormatter={xTickFmt} label={xAxisLabel(xLabel)} />
                        <YAxis stroke="#444" tick={tick} width={58} />
                        <Tooltip contentStyle={tooltipStyle} />
                        <Line type="monotone" dataKey="delta_F" stroke="#ff7f0e" strokeWidth={2} dot={false} name="ΔF (W/m²)" />
                        <Line type="monotone" dataKey="greenhouse_factor" stroke="#9467bd" strokeWidth={1.5} dot={false} name="GHF" strokeDasharray="4 2" yAxisId="ghf" />
                        <YAxis yAxisId="ghf" orientation="right" stroke="#9467bd" tick={{ ...tick, fill: '#9467bd' }} width={52} />
                        <Legend wrapperStyle={{ fontSize: 11, color: '#888' }} />
                      </LineChart>
                    </ChartRow>
                  </div>
                )}
              </div>
            </div>

          </div>
      }
      </div>{/* /measureRef */}
    </div>
  )
}

// ── Sub-components ─────────────────────────────────────────────────────────────

function ChartRow({
  label, height, flex, children,
}: {
  label: string; height: number; flex?: boolean; children: React.ReactNode
}) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', flex: flex ? 1 : undefined, minWidth: 0 }}>
      <div style={s.chartLabel}>{label}</div>
      <ResponsiveContainer width="100%" height={height}>
        {children as React.ReactElement}
      </ResponsiveContainer>
    </div>
  )
}

function StatusBadge({ status, pct }: { status: RunSummary['status']; pct: number }) {
  const colors: Record<string, string> = { running: '#f5a623', done: '#4caf50', error: '#e53935' }
  const c = colors[status] ?? '#888'
  return (
    <span style={{ fontSize: 11, color: c, border: `1px solid ${c}`, borderRadius: 4, padding: '2px 8px' }}>
      {status === 'running' ? `${pct}%` : status}
    </span>
  )
}

// ── Chart helpers ─────────────────────────────────────────────────────────────

const tick = { fontSize: 10, fill: '#666' }

const tooltipStyle: React.CSSProperties = {
  background: '#181818', border: '1px solid #2a2a2a', borderRadius: 4, fontSize: 11,
}

const xAxisLabel = (v: string) => ({
  value: v, position: 'insideBottom' as const, offset: -4, style: { fontSize: 10, fill: '#555' },
})

// Format x-axis tick labels — integers stay clean, small decimals get 2dp
const xTickFmt = (v: number) => Number.isInteger(v) ? String(v) : parseFloat(v.toFixed(2)).toString()

const sciTick = (v: number) => v.toExponential(1)

const sciFormatter = (v: unknown) =>
  typeof v === 'number' ? [v.toExponential(3), ''] : [String(v), '']

// ── Styles ────────────────────────────────────────────────────────────────────

const s: Record<string, React.CSSProperties> = {
  exportBtn:    { background: 'none', border: '1px solid #2a2a2a', color: '#888', padding: '3px 12px', borderRadius: 4, fontSize: 11, cursor: 'pointer', fontFamily: 'inherit' },
  panel:        { display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' },
  header:       { display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', padding: '14px 20px 8px', borderBottom: '1px solid #1a1a1a', flexShrink: 0 },
  headerLeft:   { display: 'flex', flexDirection: 'column', gap: 3 },
  runTitle:     { fontSize: 15, fontWeight: 700, color: '#e0e0e0' },
  runMeta:      { fontSize: 11, color: '#555' },
  progressTrack:{ height: 2, background: '#1a1a1a', flexShrink: 0 },
  progressBar:  { height: '100%', background: '#c1440e', transition: 'width 0.4s ease' },
  chartsOuter:  { flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' },
  waiting:      { flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#444', fontSize: 13 },
  charts:       { flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 0, padding: '12px 16px 16px' },
  chartLabel:   { fontSize: 11, color: '#666', marginBottom: 4, marginTop: 8 },
  scrollBox:    { overflowX: 'auto', overflowY: 'hidden' },
  zoomGroup:    { display: 'flex', alignItems: 'center', gap: 0, border: '1px solid #2a2a2a', borderRadius: 4, overflow: 'hidden' },
  zoomBtn:      { background: 'none', border: 'none', color: '#888', padding: '3px 9px', fontSize: 14, cursor: 'pointer', fontFamily: 'inherit', lineHeight: 1 },
  zoomLabel:    { fontSize: 11, color: '#666', padding: '0 6px', borderLeft: '1px solid #2a2a2a', borderRight: '1px solid #2a2a2a', lineHeight: '24px' },
}
