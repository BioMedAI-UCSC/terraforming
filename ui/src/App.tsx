import { useCallback, useEffect, useRef, useState } from 'react'
import { createRun, getRun, listRuns, subscribeToRun } from './api'
import { ChartPanel } from './components/ChartPanel'
import { RunForm } from './components/RunForm'
import { RunList } from './components/RunList'
import type { DataPoint, RunConfig, RunSummary } from './types'

const cacheKey = (id: string) => `tform-run-${id}`

const readCache = (id: string): DataPoint[] | null => {
  try {
    const raw = sessionStorage.getItem(cacheKey(id))
    return raw ? (JSON.parse(raw) as DataPoint[]) : null
  } catch { return null }
}

const writeCache = (id: string, data: DataPoint[]) => {
  try { sessionStorage.setItem(cacheKey(id), JSON.stringify(data)) } catch {}
}

export function App() {
  const [runs, setRuns]           = useState<RunSummary[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [viewData, setViewData]   = useState<DataPoint[]>([])
  const [viewRun, setViewRun]     = useState<RunSummary | null>(null)
  const unsubRef = useRef<(() => void) | null>(null)

  useEffect(() => {
    listRuns().then(setRuns).catch(() => {})
  }, [])

  const handleCreate = useCallback(async (config: Partial<RunConfig>) => {
    const { run_id } = await createRun(config)

    const stub: RunSummary = {
      id:           run_id,
      status:       'running',
      progress:     0,
      config:       config as RunConfig,
      label:        config.label || buildLabel(config),
      error:        null,
      created_at:   new Date().toISOString(),
      completed_at: null,
    }

    setRuns(prev => [stub, ...prev])
    setSelectedId(run_id)
    setViewRun(stub)
    setViewData([])

    unsubRef.current?.()

    const accumulated: DataPoint[] = []
    unsubRef.current = subscribeToRun(
      run_id,
      (pt) => {
        accumulated.push(pt)
        setViewData([...accumulated])
        const prog = pt.year != null && config.years
          ? pt.year / config.years
          : 0
        setRuns(prev => prev.map(r =>
          r.id === run_id ? { ...r, progress: prog } : r,
        ))
        setViewRun(prev => prev?.id === run_id ? { ...prev, progress: prog } : prev)
      },
      (status) => {
        setRuns(prev => prev.map(r =>
          r.id === run_id
            ? { ...r, status: status as RunSummary['status'], progress: 1, completed_at: new Date().toISOString() }
            : r,
        ))
        setViewRun(prev =>
          prev?.id === run_id
            ? { ...prev, status: status as RunSummary['status'], progress: 1 }
            : prev,
        )
        if (status === 'done') writeCache(run_id, accumulated)
      },
    )
  }, [])

  const handleSelect = useCallback(async (id: string) => {
    if (id === selectedId) return
    unsubRef.current?.()
    unsubRef.current = null
    setSelectedId(id)
    setViewRun(runs.find(r => r.id === id) ?? null)

    // Serve from session cache when available — avoids re-fetching large payloads
    const cached = readCache(id)
    if (cached) {
      setViewData(cached)
      getRun(id).then(setViewRun).catch(() => {})
      return
    }

    setViewData([])
    const run = await getRun(id)
    setViewData(run.data)
    setViewRun(run)
    if (run.status === 'done' && run.data.length > 0) writeCache(id, run.data)
  }, [selectedId, runs])

  return (
    <div style={s.root}>
      <aside style={s.sidebar}>
        <div style={s.brand}>
          <span style={s.brandName}>tform</span>
          <span style={s.brandTag}>visualizer</span>
        </div>
        <div style={s.sideScroll}>
          <RunForm onSubmit={handleCreate} />
          <RunList runs={runs} selectedId={selectedId} onSelect={handleSelect} />
        </div>
      </aside>

      <main style={s.main}>
        {viewRun
          ? <ChartPanel data={viewData} run={viewRun} />
          : <EmptyState />
        }
      </main>
    </div>
  )
}

function EmptyState() {
  return (
    <div style={s.empty}>
      <div style={s.emptyInner}>
        <div style={s.emptyGlyph}>◉</div>
        <p style={s.emptyText}>Configure a simulation on the left and click Run.</p>
        <p style={s.emptyHint}>Charts update live as each year completes.</p>
      </div>
    </div>
  )
}

function buildLabel(config: Partial<RunConfig>): string {
  if (config.exp_type === 'intervention') {
    const cmpds = Object.keys(config.inject ?? {}).join('+') || 'baseline'
    return `${config.years ?? '?'}yr ${cmpds}`
  }
  return `${config.exp_type ?? 'run'} — ${config.preset ?? ''}`
}

const s: Record<string, React.CSSProperties> = {
  root:       { display: 'flex', height: '100vh', overflow: 'hidden', background: '#0a0a0a' },
  sidebar:    { width: 280, flexShrink: 0, borderRight: '1px solid #1e1e1e', display: 'flex', flexDirection: 'column', overflow: 'hidden' },
  brand:      { display: 'flex', alignItems: 'baseline', gap: 8, padding: '14px 14px 10px', borderBottom: '1px solid #1e1e1e', flexShrink: 0 },
  brandName:  { fontSize: 18, fontWeight: 800, color: '#c1440e', letterSpacing: '-0.03em' },
  brandTag:   { fontSize: 11, color: '#555' },
  sideScroll: { flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column' },
  main:       { flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' },
  empty:      { flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' },
  emptyInner: { textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 },
  emptyGlyph: { fontSize: 40, color: '#2a2a2a' },
  emptyText:  { fontSize: 14, color: '#555' },
  emptyHint:  { fontSize: 12, color: '#333' },
}
