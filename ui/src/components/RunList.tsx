import type { RunSummary } from '../types'

interface Props {
  runs: RunSummary[]
  selectedId: string | null
  onSelect: (id: string) => void
}

export function RunList({ runs, selectedId, onSelect }: Props) {
  if (runs.length === 0) return (
    <div style={s.empty}>No runs yet.</div>
  )

  return (
    <div style={s.list}>
      <div style={s.heading}>Past Runs</div>
      {runs.map(run => (
        <button
          key={run.id}
          style={{ ...s.item, ...(run.id === selectedId ? s.itemActive : {}) }}
          onClick={() => onSelect(run.id)}
        >
          <div style={s.topRow}>
            <span style={s.runLabel}>{run.label}</span>
            <StatusDot status={run.status} />
          </div>
          {run.status === 'running' && (
            <div style={s.track}>
              <div style={{ ...s.bar, width: `${Math.round(run.progress * 100)}%` }} />
            </div>
          )}
          <div style={s.meta}>
            {run.config.preset} · {run.config.accuracy}
            {run.status === 'running' && ` · ${Math.round(run.progress * 100)}%`}
            {run.status === 'done' && run.config.exp_type === 'intervention' && ` · ${run.config.years}yr`}
          </div>
        </button>
      ))}
    </div>
  )
}

function StatusDot({ status }: { status: RunSummary['status'] }) {
  const color = status === 'running' ? '#f5a623' : status === 'done' ? '#4caf50' : '#e53935'
  return (
    <span style={{ width: 7, height: 7, borderRadius: '50%', background: color, flexShrink: 0,
      boxShadow: status === 'running' ? `0 0 6px ${color}` : 'none' }} />
  )
}

const s: Record<string, React.CSSProperties> = {
  list:      { display: 'flex', flexDirection: 'column', overflowY: 'auto', flex: 1 },
  heading:   { fontSize: 11, fontWeight: 700, color: '#c1440e', letterSpacing: '0.08em', textTransform: 'uppercase', padding: '10px 14px 4px' },
  empty:     { fontSize: 12, color: '#444', padding: '12px 14px' },
  item:      { display: 'flex', flexDirection: 'column', gap: 4, padding: '8px 14px', background: 'none', border: 'none', cursor: 'pointer', textAlign: 'left', borderLeft: '2px solid transparent' },
  itemActive:{ background: '#111', borderLeftColor: '#c1440e' },
  topRow:    { display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 },
  runLabel:  { fontSize: 12, color: '#e0e0e0', fontWeight: 600, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' },
  track:     { height: 2, background: '#222', borderRadius: 1, overflow: 'hidden' },
  bar:       { height: '100%', background: '#c1440e', borderRadius: 1, transition: 'width 0.3s' },
  meta:      { fontSize: 10, color: '#555' },
}
