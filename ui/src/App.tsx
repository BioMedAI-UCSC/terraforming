import { useCallback, useEffect, useRef, useState } from 'react'
import { createRun, getRun, getRunFields, getRunNetcdf, importRunNetcdf, listRuns, stopRun, subscribeToRun, uploadAmes } from './api'
import { FieldMap } from './components/FieldMap'
import { RunForm } from './components/RunForm'
import { RunList } from './components/RunList'
import type { RunConfig, RunSummary } from './types'

type View = 'dmgcm' | 'benchmarks' | 'terraforming'
const FOLDER_DB = 'dmgcm-workbench'
const FOLDER_STORE = 'handles'
const FOLDER_KEY = 'run-output-directory'

function folderDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(FOLDER_DB, 1)
    request.onupgradeneeded = () => request.result.createObjectStore(FOLDER_STORE)
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error)
  })
}

async function cacheFolderHandle(handle:any) {
  const db = await folderDb()
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(FOLDER_STORE, 'readwrite')
    tx.objectStore(FOLDER_STORE).put(handle, FOLDER_KEY)
    tx.oncomplete = () => resolve(); tx.onerror = () => reject(tx.error)
  })
  db.close()
}

async function restoredFolderHandle():Promise<any|null> {
  const db = await folderDb()
  const result = await new Promise<any>((resolve, reject) => {
    const request = db.transaction(FOLDER_STORE).objectStore(FOLDER_STORE).get(FOLDER_KEY)
    request.onsuccess = () => resolve(request.result ?? null)
    request.onerror = () => reject(request.error)
  })
  db.close(); return result
}

export function App() {
  const [view, setView] = useState<View>('dmgcm')
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [saveFolder, setSaveFolder] = useState<any>(null)
  const [folderPermission, setFolderPermission] = useState<'granted'|'prompt'|'denied'|'unsupported'>('prompt')
  const [amesError, setAmesError] = useState<string | null>(null)
  const [savedOutputs, setSavedOutputs] = useState<any[]>([])
  const [importBusy, setImportBusy] = useState(false)
  const streams = useRef<Record<string, () => void>>({})

  const refresh = useCallback(() => listRuns().then(setRuns).catch(() => {}), [])
  useEffect(() => { refresh() }, [refresh])
  useEffect(() => {
    if (!('indexedDB' in window)) { setFolderPermission('unsupported'); return }
    restoredFolderHandle().then(async handle => {
      if (!handle) return
      setSaveFolder(handle)
      setFolderPermission(await handle.queryPermission({mode:'readwrite'}))
      scanSavedOutputs(handle)
    }).catch(() => {})
  }, [])
  useEffect(() => () => Object.values(streams.current).forEach(fn => fn()), [])

  async function saveCompleted(id: string) {
    if (!saveFolder) return
    const permission = await saveFolder.queryPermission({ mode:'readwrite' })
    setFolderPermission(permission)
    if (permission !== 'granted') throw new Error('Saved folder needs permission again; click Reconnect save folder.')
    const run = await getRun(id)
    const fields = await getRunFields(id)
    const safe = run.label.replace(/[^a-z0-9._-]+/gi, '_').slice(0, 80) || id
    const handle = await saveFolder.getFileHandle(`${safe}_${id}.json`, { create: true })
    const writable = await handle.createWritable()
    await writable.write(JSON.stringify({ run, fields }, null, 2))
    await writable.close()
    const ncHandle = await saveFolder.getFileHandle(`${safe}_${id}.nc`, { create: true })
    const ncWritable = await ncHandle.createWritable()
    await ncWritable.write(await getRunNetcdf(id))
    await ncWritable.close()
  }

  const watch = useCallback((id: string) => {
    streams.current[id]?.()
    streams.current[id] = subscribeToRun(id, () => {}, async status => {
      await refresh()
      if (status === 'done') saveCompleted(id).catch(e => setError(`Auto-save failed: ${e}`))
      delete streams.current[id]
    }, (progress, eta, completed, total) => {
      setRuns(old => old.map(run => run.id === id ? {
        ...run, progress, eta_seconds: eta, completed_steps: completed, total_steps: total,
      } : run))
    })
  }, [refresh, saveFolder])

  const start = useCallback(async (config: Partial<RunConfig>) => {
    try {
      const { run_id } = await createRun(config)
      const stub: RunSummary = {
        id: run_id, status:'running', progress:0, config:config as RunConfig,
        label:config.label || run_id, error:null, created_at:new Date().toISOString(), completed_at:null,
        eta_seconds:null, completed_steps:0, total_steps:0,
      }
      setRuns(old => [stub, ...old]); setSelectedId(run_id); setError(null); watch(run_id)
    } catch (e) { setError(`Simulation server unavailable: ${e instanceof Error ? e.message : e}`) }
  }, [watch])

  async function chooseFolder() {
    const picker = (window as any).showDirectoryPicker
    if (!picker) { setError('Folder auto-save requires a browser with the File System Access API.'); return }
    try {
      if (saveFolder) {
        const permission = await saveFolder.requestPermission({mode:'readwrite'})
        setFolderPermission(permission)
        if (permission === 'granted') { await scanSavedOutputs(saveFolder); setError(null); return }
      }
      const handle = await picker.call(window, { mode:'readwrite' })
      await cacheFolderHandle(handle)
      setSaveFolder(handle); setFolderPermission('granted'); setError(null)
      await scanSavedOutputs(handle)
    } catch {}
  }

  async function scanSavedOutputs(directory:any) {
    try {
      const outputs:any[] = []
      for await (const entry of directory.values()) {
        if (entry.kind === 'file' && /\.nc(?:4)?$/i.test(entry.name)) outputs.push(entry)
      }
      setSavedOutputs(outputs.sort((a,b) => a.name.localeCompare(b.name)))
    } catch { setSavedOutputs([]) }
  }

  async function importOutput(file:File) {
    setImportBusy(true)
    try {
      const result = await importRunNetcdf(file)
      await refresh(); setSelectedId(result.run_id); setError(null)
    } catch (e) { setError(`Could not import ${file.name}: ${e instanceof Error ? e.message : e}`) }
    finally { setImportBusy(false) }
  }

  async function upload(id: string, file: File) {
    try { await uploadAmes(id, file); setAmesError(null); setSelectedId(null); setTimeout(() => setSelectedId(id), 0) }
    catch (e) { setAmesError(e instanceof Error ? e.message : String(e)) }
  }

  const selected = runs.find(r => r.id === selectedId) ?? null
  const done = runs.filter(r => r.status === 'done').length
  const running = runs.filter(r => r.status === 'running').length

  return <div className="app-shell">
    <header className="topbar">
      <div className="brand">DMGCM <span>workbench</span></div>
      <button className="folder-button" onClick={chooseFolder} title={saveFolder ? `Cached folder: ${saveFolder.name}` : 'Choose where completed runs are saved'}>
        ▱ {saveFolder ? `${folderPermission === 'granted' ? '' : 'Reconnect '}${saveFolder.name}` : 'Select save folder'}
      </button>
      <nav>
        <button className={view === 'dmgcm' ? 'active' : ''} onClick={() => setView('dmgcm')}>DMGCM</button>
        <button className={view === 'benchmarks' ? 'active' : ''} onClick={() => setView('benchmarks')}>Benchmarks</button>
        <button disabled>Terraforming <small>(Upcoming)</small></button>
      </nav>
    </header>
    <div className="workspace">
      <aside className="sidebar">
        <RunForm onSubmit={start} benchmarkMode={view === 'benchmarks'} />
        {error && <div className="error-box">{error}</div>}
        <RunList runs={runs} selectedId={selectedId} onSelect={setSelectedId} />
      </aside>
      <main className="main-view">
        {view === 'terraforming' ? <div className="empty-state">Terraforming experiments are intentionally unavailable until the Mars physics stack is validated.</div> : <>
          {view === 'benchmarks' && <section className="import-panel"
            onDragOver={e => { e.preventDefault(); e.currentTarget.classList.add('dragging') }}
            onDragLeave={e => e.currentTarget.classList.remove('dragging')}
            onDrop={e => { e.preventDefault(); e.currentTarget.classList.remove('dragging'); const file=e.dataTransfer.files[0]; if(file) importOutput(file) }}>
            <div><b>Open existing DMGCM output</b><span>Drop a saved NetCDF here or choose one from the cached output folder.</span></div>
            <label className="upload-button">{importBusy ? 'Importing…' : 'Drop / choose .nc'}<input type="file" accept=".nc,.nc4" hidden disabled={importBusy} onChange={e=>e.target.files?.[0]&&importOutput(e.target.files[0])}/></label>
            {savedOutputs.length > 0 && <div className="saved-output-list">{savedOutputs.map(handle=><button key={handle.name} onClick={async()=>importOutput(await handle.getFile())}>{handle.name}</button>)}</div>}
            {saveFolder && savedOutputs.length === 0 && <small>No saved NetCDF outputs found in {saveFolder.name}.</small>}
          </section>}
          <section className="debug-header">
            <div><b>{runs.length}</b><span>runs</span></div><div><b>{running}</b><span>running</span></div><div><b>{done}</b><span>passed</span></div>
            <span className="debug-title">{view === 'benchmarks' ? 'Benchmark comparison' : 'Simulation debugger'}</span>
          </section>
          <section className="run-grid">
            {runs.map(run => <RunCard key={run.id} run={run} selected={run.id === selectedId}
              onOpen={() => setSelectedId(run.id)} onStop={() => stopRun(run.id).then(refresh)} />)}
          </section>
          {selected ? <section className="output-area">
            <div className="output-title"><div><b>{selected.label}</b><span>{selected.config.sols} sol · {selected.config.scale}</span></div>
              {view === 'benchmarks' && selected.status === 'done' && <label className="upload-button">Add AmesGCM .nc<input type="file" accept=".nc,.nc4,.cdf" hidden onChange={e => e.target.files?.[0] && upload(selected.id, e.target.files[0])} /></label>}
            </div>
            {amesError && <div className="error-box">{amesError}</div>}
            {selected.status === 'done' ? <FieldMap runId={selected.id} benchmarkMode={view === 'benchmarks'} exportLabel={selected.label} /> :
              <PendingOutputs status={selected.status} />}
          </section> : <div className="empty-state">Start or select a run to inspect every output.</div>}
        </>}
      </main>
    </div>
  </div>
}

function PendingOutputs({ status }: { status:RunSummary['status'] }) {
  return <div className="pending-grid">
    {['Temperature','Surface pressure','Zonal wind','Meridional wind','Wind speed','CO₂ ice caps','MOLA topography'].map(name =>
      <div className="pending-tile" key={name}><b>{name}</b><span>{status === 'running' ? 'Computing…' : 'No completed output'}</span></div>)}
  </div>
}

function RunCard({ run, selected, onOpen, onStop }: { run:RunSummary; selected:boolean; onOpen:()=>void; onStop:()=>void }) {
  const pct = Math.round(run.progress * 100)
  const eta = run.eta_seconds == null ? 'estimating…' : run.eta_seconds < 60 ? `${Math.ceil(run.eta_seconds)}s` : `${Math.ceil(run.eta_seconds / 60)}m`
  return <article className={`run-card ${selected ? 'selected' : ''}`} onClick={onOpen}>
    <div className="run-card-head"><b>{run.label}</b><span className={`status ${run.status}`}>{run.status}</span></div>
    <div className="progress"><i style={{ width:`${pct}%` }} /></div>
    <div className="run-stats"><span>{pct}%</span><span>{run.completed_steps || 0}/{run.total_steps || '—'} steps</span><span>ETA {run.status === 'running' ? eta : '—'}</span></div>
    {run.status === 'running' && <button className="stop-button" onClick={e => { e.stopPropagation(); onStop() }}>Stop</button>}
    {run.error && <div className="run-error">{run.error.split('\n')[0]}</div>}
  </article>
}
