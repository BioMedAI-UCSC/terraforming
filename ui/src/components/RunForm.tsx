import { useEffect, useState } from 'react'
import { getPresetConfig, getPresets } from '../api'
import type { PresetValues, RunConfig } from '../types'

const KNOWN_COMPOUNDS = ['CF4', 'C2F6', 'C3F8', 'SF6', 'NF3', 'C4F10', 'C6F14']

interface Props {
  onSubmit: (config: Partial<RunConfig>) => Promise<void>
}

interface InjectEntry {
  compound: string
  kgPerYear: string
}

const parseOpt = (v: string): number | undefined => {
  const n = parseFloat(v)
  return v.trim() && !isNaN(n) ? n : undefined
}

const fmt = (v: number, exp?: boolean) =>
  exp ? v.toExponential(2) : String(v)

export function RunForm({ onSubmit }: Props) {
  const [presets, setPresets]       = useState<string[]>(['current-mars'])
  const [preset, setPreset]         = useState('current-mars')
  const [presetCfg, setPresetCfg]   = useState<PresetValues | null>(null)
  const [showPreview, setShowPreview] = useState(false)
  const [expType, setExpType]       = useState('intervention')
  const [years, setYears]           = useState(100)
  const [sols, setSols]             = useState(1)
  const [accuracy, setAccuracy]     = useState('fast')
  const [inject, setInject]         = useState<InjectEntry[]>([{ compound: 'SF6', kgPerYear: '1e9' }])
  const [label, setLabel]           = useState('')
  const [running, setRunning]       = useState(false)
  const [showAdv, setShowAdv]       = useState(false)

  // Advanced planet / engine params — pre-populated from preset, user-editable
  const [surfaceTemp, setSurfaceTemp]           = useState('')
  const [surfacePressure, setSurfacePressure]   = useState('')
  const [albedo, setAlbedo]                     = useState('')
  const [greenhouse, setGreenhouse]             = useState('')
  const [iceMass, setIceMass]                   = useState('')
  const [lat, setLat]                           = useState('')
  const [lon, setLon]                           = useState('')
  const [elevation, setElevation]               = useState('')
  const [ls, setLs]                             = useState('')
  const [dt, setDt]                             = useState('')

  useEffect(() => {
    getPresets().then(setPresets).catch(() => {})
  }, [])

  // Fetch preset config and reflect values into advanced fields whenever preset changes
  useEffect(() => {
    getPresetConfig(preset).then(cfg => {
      setPresetCfg(cfg)
      const p = cfg.planet
      setSurfaceTemp(String(p.surface_temperature))
      setSurfacePressure(String(p.surface_pressure))
      setAlbedo(String(p.albedo))
      setGreenhouse(String(p.greenhouse_factor))
      setIceMass(String(p.ice_mass))
      setLat(String(p.latitude))
      setLon(String(p.longitude))
      setElevation(String(p.elevation_m))
      setLs(String(p.initial_ls_deg))
      setDt(String(cfg.engine.dt))
    }).catch(() => {})
  }, [preset])

  // Reset duration defaults when switching experiment type
  useEffect(() => {
    if (expType === 'intervention') setYears(100)
    else if (expType === 'year') setYears(1)
  }, [expType])

  const addInject = () =>
    setInject(prev => [...prev, { compound: KNOWN_COMPOUNDS[0], kgPerYear: '1e9' }])

  const removeInject = (i: number) =>
    setInject(prev => prev.filter((_, idx) => idx !== i))

  const updateInject = (i: number, field: keyof InjectEntry, val: string) =>
    setInject(prev => prev.map((e, idx) => idx === i ? { ...e, [field]: val } : e))

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setRunning(true)
    try {
      const injectMap: Record<string, number> = {}
      if (expType === 'intervention') {
        for (const { compound, kgPerYear } of inject) {
          const v = parseFloat(kgPerYear)
          if (compound && !isNaN(v)) injectMap[compound] = v
        }
      }
      await onSubmit({
        preset,
        exp_type: expType,
        years,
        sols,
        accuracy,
        dt: parseOpt(dt) ?? 3600,
        lat: parseOpt(lat) ?? null,
        lon: parseOpt(lon) ?? null,
        elevation: parseOpt(elevation) ?? null,
        ls: parseOpt(ls) ?? null,
        surface_temp: parseOpt(surfaceTemp) ?? null,
        surface_pressure: parseOpt(surfacePressure) ?? null,
        albedo: parseOpt(albedo) ?? null,
        greenhouse_factor: parseOpt(greenhouse) ?? null,
        ice_mass: parseOpt(iceMass) ?? null,
        inject: injectMap,
        label: label.trim() || undefined,
      })
    } finally {
      setRunning(false)
    }
  }

  const p = presetCfg?.planet

  return (
    <form onSubmit={handleSubmit} style={s.form}>
      <div style={s.sectionTitle}>New Run</div>

      {/* Preset row with preview toggle */}
      <label style={s.label}>Preset</label>
      <div style={s.presetRow}>
        <select style={{ ...s.select, flex: 1 }} value={preset} onChange={e => setPreset(e.target.value)}>
          {presets.map(pr => <option key={pr}>{pr}</option>)}
        </select>
        <button
          type="button"
          style={{ ...s.previewBtn, ...(showPreview ? s.previewBtnActive : {}) }}
          onClick={() => setShowPreview(v => !v)}
          title="Show preset values"
        >⊙</button>
      </div>

      {showPreview && p && (
        <div style={s.previewBox}>
          <PRow label="T" value={`${p.surface_temperature} K`} />
          <PRow label="P" value={`${p.surface_pressure} Pa`} />
          <PRow label="Albedo" value={p.albedo} />
          <PRow label="GHF" value={p.greenhouse_factor} />
          <PRow label="Ice" value={`${fmt(p.ice_mass, true)} kg`} />
          <PRow label="Lat / Lon" value={`${p.latitude}° / ${p.longitude}°`} />
          <PRow label="Elev" value={`${p.elevation_m} m`} />
          <PRow label="Ls₀" value={`${p.initial_ls_deg}°`} />
          <PRow label="dt" value={`${presetCfg.engine.dt} s`} />
        </div>
      )}

      {/* Experiment type */}
      <label style={s.label}>Type</label>
      <select style={s.select} value={expType} onChange={e => setExpType(e.target.value)}>
        <option value="intervention">intervention</option>
        <option value="sol">sol</option>
        <option value="year">year</option>
      </select>

      {/* Duration — intervention */}
      {expType === 'intervention' && <>
        <label style={s.label}>Years</label>
        <div style={s.row}>
          {[100, 500, 1000].map(y => (
            <button key={y} type="button"
              style={{ ...s.chip, ...(years === y ? s.chipActive : {}) }}
              onClick={() => setYears(y)}
            >{y}yr</button>
          ))}
          <input style={{ ...s.input, width: 60 }} type="number" value={years} min={1}
            onChange={e => setYears(parseInt(e.target.value) || 1)} />
        </div>

        <label style={s.label}>Compounds</label>
        {inject.map((entry, i) => (
          <div key={i} style={s.injectRow}>
            <select style={{ ...s.select, flex: 1 }} value={entry.compound}
              onChange={e => updateInject(i, 'compound', e.target.value)}>
              {KNOWN_COMPOUNDS.map(c => <option key={c}>{c}</option>)}
            </select>
            <input style={{ ...s.input, width: 76 }} value={entry.kgPerYear} placeholder="kg/yr"
              onChange={e => updateInject(i, 'kgPerYear', e.target.value)} />
            <button type="button" style={s.removeBtn} onClick={() => removeInject(i)}>×</button>
          </div>
        ))}
        <button type="button" style={s.addBtn} onClick={addInject}>+ compound</button>
      </>}

      {/* Duration — sol */}
      {expType === 'sol' && <>
        <label style={s.label}>Duration (sols)</label>
        <div style={s.row}>
          {[0.25, 1, 7, 30].map(v => (
            <button key={v} type="button"
              style={{ ...s.chip, ...(sols === v ? s.chipActive : {}) }}
              onClick={() => setSols(v)}
            >{v}</button>
          ))}
          <input style={{ ...s.input, width: 60 }} type="number" step="any" min={0.01}
            value={sols} onChange={e => setSols(parseFloat(e.target.value) || 1)} />
        </div>
      </>}

      {/* Duration — year (custom years) */}
      {expType === 'year' && <>
        <label style={s.label}>Mars years</label>
        <div style={s.row}>
          {[1, 5, 10, 50].map(y => (
            <button key={y} type="button"
              style={{ ...s.chip, ...(years === y ? s.chipActive : {}) }}
              onClick={() => setYears(y)}
            >{y}yr</button>
          ))}
          <input style={{ ...s.input, width: 60 }} type="number" value={years} min={1}
            onChange={e => setYears(parseInt(e.target.value) || 1)} />
        </div>
      </>}

      {/* Accuracy */}
      <label style={s.label}>Accuracy</label>
      <div style={s.row}>
        {(['fast', 'accurate'] as const).map(a => (
          <button key={a} type="button"
            style={{ ...s.chip, ...(accuracy === a ? s.chipActive : {}) }}
            onClick={() => setAccuracy(a)}
          >{a}</button>
        ))}
      </div>

      {/* Label */}
      <label style={s.label}>Label <span style={s.optional}>(optional)</span></label>
      <input style={s.input} value={label} placeholder="e.g. 500yr SF6 aggressive"
        onChange={e => setLabel(e.target.value)} />

      {/* Advanced toggle */}
      <button type="button" style={s.advBtn} onClick={() => setShowAdv(v => !v)}>
        {showAdv ? '▾' : '▸'} Advanced parameters
      </button>

      {showAdv && (
        <div style={s.advGrid}>
          <AdvField label="Temp (K)"      placeholder="210"  value={surfaceTemp}      onChange={setSurfaceTemp} />
          <AdvField label="Pressure (Pa)" placeholder="610"  value={surfacePressure}  onChange={setSurfacePressure} />
          <AdvField label="Albedo"        placeholder="0.25" value={albedo}           onChange={setAlbedo} />
          <AdvField label="GHF (≥1)"      placeholder="1.02" value={greenhouse}       onChange={setGreenhouse} />
          <AdvField label="Ice (kg)"      placeholder="5e15" value={iceMass}          onChange={setIceMass} />
          <AdvField label="Lat (°)"       placeholder="22"   value={lat}              onChange={setLat} />
          <AdvField label="Lon (°)"       placeholder="0"    value={lon}              onChange={setLon} />
          <AdvField label="Elev (m)"      placeholder="0"    value={elevation}        onChange={setElevation} />
          <AdvField label="Init Ls (°)"   placeholder="251"  value={ls}              onChange={setLs} />
          <AdvField label="dt (s)"        placeholder="3600" value={dt}              onChange={setDt} />
        </div>
      )}

      <button type="submit" style={s.runBtn} disabled={running}>
        {running ? 'Starting…' : 'Run Simulation'}
      </button>
    </form>
  )
}

// ── Sub-components ────────────────────────────────────────────────────────────

function AdvField({ label, placeholder, value, onChange }: {
  label: string; placeholder: string; value: string; onChange: (v: string) => void
}) {
  return (
    <div>
      <div style={{ fontSize: 10, color: '#666', marginBottom: 2 }}>{label}</div>
      <input
        style={{ background: '#181818', border: '1px solid #222', color: '#bbb', padding: '4px 6px', borderRadius: 4, fontSize: 11, width: '100%', fontFamily: 'inherit', boxSizing: 'border-box' }}
        value={value}
        placeholder={placeholder}
        onChange={e => onChange(e.target.value)}
      />
    </div>
  )
}

function PRow({ label, value }: { label: string; value: string | number }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10 }}>
      <span style={{ color: '#555' }}>{label}</span>
      <span style={{ color: '#aaa', fontVariantNumeric: 'tabular-nums' }}>{value}</span>
    </div>
  )
}

// ── Styles ────────────────────────────────────────────────────────────────────

const s: Record<string, React.CSSProperties> = {
  form:           { display: 'flex', flexDirection: 'column', gap: 6, padding: '12px 14px', borderBottom: '1px solid #222' },
  sectionTitle:   { fontSize: 11, fontWeight: 700, color: '#c1440e', letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 4 },
  label:          { fontSize: 11, color: '#888', marginTop: 4 },
  optional:       { color: '#555' },
  select:         { background: '#181818', border: '1px solid #2a2a2a', color: '#e0e0e0', padding: '5px 8px', borderRadius: 4, fontSize: 12, width: '100%' },
  input:          { background: '#181818', border: '1px solid #2a2a2a', color: '#e0e0e0', padding: '5px 8px', borderRadius: 4, fontSize: 12, width: '100%', fontFamily: 'inherit' },
  row:            { display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap' },
  chip:           { background: '#1a1a1a', border: '1px solid #2a2a2a', color: '#888', padding: '4px 10px', borderRadius: 4, fontSize: 11, cursor: 'pointer' },
  chipActive:     { background: '#2a1008', border: '1px solid #c1440e', color: '#c1440e' },
  injectRow:      { display: 'flex', gap: 6, alignItems: 'center' },
  removeBtn:      { background: 'none', border: 'none', color: '#555', fontSize: 16, cursor: 'pointer', padding: '0 4px', lineHeight: 1 },
  addBtn:         { background: 'none', border: '1px dashed #333', color: '#666', fontSize: 11, padding: '4px 8px', borderRadius: 4, cursor: 'pointer', marginTop: 2 },
  presetRow:      { display: 'flex', gap: 6, alignItems: 'center' },
  previewBtn:     { background: '#1a1a1a', border: '1px solid #2a2a2a', color: '#555', padding: '5px 9px', borderRadius: 4, fontSize: 13, cursor: 'pointer', flexShrink: 0, lineHeight: 1 },
  previewBtnActive:{ borderColor: '#c1440e', color: '#c1440e' },
  previewBox:     { background: '#111', border: '1px solid #1e1e1e', borderRadius: 4, padding: '8px 10px', display: 'flex', flexDirection: 'column', gap: 3 },
  advBtn:         { background: 'none', border: 'none', color: '#555', fontSize: 11, cursor: 'pointer', textAlign: 'left', padding: '4px 0', marginTop: 4 },
  advGrid:        { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px 8px', marginTop: 2 },
  runBtn:         { background: '#c1440e', border: 'none', color: '#fff', padding: '8px 0', borderRadius: 4, fontSize: 13, fontWeight: 700, cursor: 'pointer', marginTop: 8, fontFamily: 'inherit' },
}
