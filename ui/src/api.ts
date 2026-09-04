import type { DataPoint, PresetValues, Run, RunConfig, RunFields, RunSummary } from './types'

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init)
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`)
  return res.json() as Promise<T>
}

export const createRun = (config: Partial<RunConfig>) =>
  req<{ run_id: string }>('/api/runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  })

export const listRuns  = () => req<RunSummary[]>('/api/runs')
export const getRun    = (id: string) => req<Run>(`/api/runs/${id}`)
export const getRunFields = (id: string, year?: number) =>
  req<RunFields>(`/api/runs/${id}/fields${year != null ? `?year=${year}` : ''}`)
export const getRunSnapshots = (id: string) =>
  req<{ years: number[] }>(`/api/runs/${id}/snapshots`)
export const getRunNetcdf = async (id:string) => {
  const res = await fetch(`/api/runs/${id}/netcdf`)
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`)
  return res.blob()
}
export const importRunNetcdf = async (file:File) => {
  const res = await fetch('/api/runs/import/netcdf', {
    method:'POST', headers:{'Content-Type':'application/x-netcdf','X-Filename':file.name}, body:file,
  })
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`)
  return res.json() as Promise<{run_id:string;label:string}>
}
export const getPresets       = () => req<string[]>('/api/presets')
export const getPresetConfig  = (name: string) => req<PresetValues>(`/api/presets/${name}`)
export const getCompounds     = () => req<string[]>('/api/compounds')
export const stopRun = (id: string) => req<{ status: string }>(`/api/runs/${id}/stop`, { method: 'POST' })
export const uploadAmes = async (id: string, file: File) => {
  const res = await fetch(`/api/runs/${id}/benchmarks/ames`, {
    method: 'POST', headers: { 'Content-Type': 'application/octet-stream', 'X-Filename': file.name }, body: file,
  })
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`)
  return res.json()
}
export const configureMcd = (id:string, config:{ls?:number;local_time?:number|null;dust:number;auto_match:boolean}) =>
  req<{ls_deg:number;dust_scenario:number;local_times_hours:number[];auto_matched:boolean}>(`/api/runs/${id}/benchmarks/mcd`, {
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(config),
  })

/** Subscribe to the SSE stream for a run.  Returns an unsubscribe function. */
export function subscribeToRun(
  id: string,
  onPoint: (pt: DataPoint) => void,
  onDone:  (status: string) => void,
  onProgress?: (progress: number, eta: number | null, completed: number, total: number) => void,
): () => void {
  const es = new EventSource(`/api/runs/${id}/events`)
  es.onmessage = (e) => {
    const msg = JSON.parse(e.data as string) as { type: string; data?: DataPoint; status?: string; progress?: number; eta_seconds?: number | null; completed_steps?: number; total_steps?: number }
    if (msg.type === 'point' && msg.data) {
      onPoint(msg.data)
    } else if (msg.type === 'progress') {
      onProgress?.(msg.progress ?? 0, msg.eta_seconds ?? null, msg.completed_steps ?? 0, msg.total_steps ?? 0)
    } else if (msg.type === 'done') {
      onDone(msg.status ?? 'done')
      es.close()
    }
  }
  es.onerror = () => {
    onDone('error')
    es.close()
  }
  return () => es.close()
}
