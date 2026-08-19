import { useState } from 'react'
import type { RunConfig } from '../types'

interface Props {
  onSubmit: (config: Partial<RunConfig>) => Promise<void>
  benchmarkMode: boolean
}

export function RunForm({ onSubmit, benchmarkMode }: Props) {
  const [sols, setSols] = useState(1)
  const [scale, setScale] = useState('fast')
  const [label, setLabel] = useState('')
  const [ls, setLs] = useState('0')
  const [diurnal, setDiurnal] = useState(false)
  const [mcdLocalTime, setMcdLocalTime] = useState('')
  const [mcdDust, setMcdDust] = useState(1)
  const [running, setRunning] = useState(false)

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    setRunning(true)
    try {
      await onSubmit({
        exp_type: 'sol', accuracy: 'gcm', preset: 'current-mars', sols, scale,
        label: label.trim() || `DMGCM ${sols} sol ${scale}`,
        ls: Number(ls) || 0, diurnal,
        compare_mcd: benchmarkMode,
        mcd_local_time: mcdLocalTime.trim() ? Number(mcdLocalTime) : null,
        mcd_dust: mcdDust,
      })
    } finally { setRunning(false) }
  }

  return <form onSubmit={submit} style={s.form}>
    <div style={s.title}>New DMGCM run</div>
    <label style={s.label}>Name</label>
    <input style={s.input} value={label} onChange={e => setLabel(e.target.value)}
           placeholder="MY24 baseline" />

    <label style={s.label}>Type</label>
    <div style={s.readonly}>Sol</div>

    <label style={s.label}>Duration</label>
    <div style={s.row}>
      {[0.25, 1, 7, 30, 668].map(v => <button type="button" key={v}
        style={{ ...s.chip, ...(sols === v ? s.active : {}) }} onClick={() => setSols(v)}>
        {v === 668 ? '1 Mars yr' : `${v} sol`}
      </button>)}
    </div>
    <input style={s.input} type="number" min="0.01" step="any" value={sols}
           onChange={e => setSols(Math.max(.01, Number(e.target.value)))} />

    <label style={s.label}>GCM resolution</label>
    <select style={s.input} value={scale} onChange={e => setScale(e.target.value)}>
      <option value="fast">T42 · 12 levels</option>
      <option value="balanced">T85 · 20 levels</option>
      <option value="high">T106 · 30 levels</option>
      <option value="ultra">T170 · 40 levels</option>
    </select>

    <label style={s.label}>Season (Ls °)</label>
    <input style={s.input} type="number" min="0" max="360" value={ls}
           onChange={e => setLs(e.target.value)} />
    <label style={s.check}><input type="checkbox" checked={diurnal}
      onChange={e => setDiurnal(e.target.checked)} /> Resolve day/night cycle</label>

    {benchmarkMode && <div style={s.benchmark}>
      <div style={s.title}>MCD benchmark</div>
      <div style={s.hint}>Season and model wind height are matched automatically.</div>
      <label style={s.label}>Local time</label>
      <input style={s.input} value={mcdLocalTime} onChange={e => setMcdLocalTime(e.target.value)}
             placeholder="blank = 12-time diurnal mean" />
      <label style={s.label}>Dust scenario</label>
      <select style={s.input} value={mcdDust} onChange={e => setMcdDust(Number(e.target.value))}>
        {[1,2,3,4,5,6,7,8].map(v => <option key={v}>{v}</option>)}
      </select>
    </div>}

    <button style={s.run} disabled={running}>{running ? 'Starting…' : 'Run simulation'}</button>
  </form>
}

const s: Record<string, React.CSSProperties> = {
  form: { display:'flex', flexDirection:'column', gap:7, padding:14, borderBottom:'1px solid #252525' },
  title: { color:'#df6b3d', fontSize:11, fontWeight:800, textTransform:'uppercase', letterSpacing:'.08em' },
  label: { color:'#8b949e', fontSize:11, marginTop:3 },
  input: { width:'100%', background:'#15181d', color:'#e6edf3', border:'1px solid #30363d', borderRadius:5, padding:'7px 8px' },
  readonly: { ...({} as React.CSSProperties), background:'#111318', color:'#9da7b1', border:'1px solid #252a31', borderRadius:5, padding:'7px 8px', fontSize:12 },
  row: { display:'flex', flexWrap:'wrap', gap:5 },
  chip: { background:'#15181d', color:'#8b949e', border:'1px solid #30363d', borderRadius:5, padding:'5px 7px', cursor:'pointer', fontSize:10 },
  active: { color:'#ff7849', borderColor:'#b94820', background:'#2a1510' },
  check: { color:'#9da7b1', fontSize:11, display:'flex', gap:7, alignItems:'center' },
  benchmark: { display:'flex', flexDirection:'column', gap:6, padding:10, marginTop:6, background:'#10151a', border:'1px solid #26313c', borderRadius:6 },
  hint: { color:'#66717d', fontSize:10, lineHeight:1.4 },
  run: { marginTop:6, background:'#c64d22', color:'white', border:0, borderRadius:5, padding:9, fontWeight:700, cursor:'pointer' },
}
