import type { DataPoint, PresetValues, Run, RunConfig, RunSummary } from './types'

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
export const getPresets       = () => req<string[]>('/api/presets')
export const getPresetConfig  = (name: string) => req<PresetValues>(`/api/presets/${name}`)
export const getCompounds     = () => req<string[]>('/api/compounds')

/** Subscribe to the SSE stream for a run.  Returns an unsubscribe function. */
export function subscribeToRun(
  id: string,
  onPoint: (pt: DataPoint) => void,
  onDone:  (status: string) => void,
): () => void {
  const es = new EventSource(`/api/runs/${id}/events`)
  es.onmessage = (e) => {
    const msg = JSON.parse(e.data as string) as { type: string; data?: DataPoint; status?: string }
    if (msg.type === 'point' && msg.data) {
      onPoint(msg.data)
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
