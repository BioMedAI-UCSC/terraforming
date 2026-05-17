import type { DataPoint, RunSummary } from '../types'

export interface AxisInfo {
  xKey:   string
  xLabel: string
}

/**
 * Generate discrete, human-readable tick positions for a numeric x-axis.
 *
 * Picks a step size from the sequence 1-2-5-10 (×10ⁿ) so the axis always
 * shows round numbers (e.g. 0,5,10,…,30 for 30 sols; 0,100,…,1000 for years).
 */
export function genTicks(data: DataPoint[], xKey: string, targetCount = 8): number[] {
  const vals = data
    .map(d => (d as unknown as Record<string, unknown>)[xKey] as number)
    .filter(v => typeof v === 'number' && isFinite(v))
  if (vals.length === 0) return []

  const maxVal = Math.max(...vals)
  if (maxVal <= 0) return [0]

  const raw   = maxVal / targetCount
  const exp   = Math.floor(Math.log10(raw))
  const base  = Math.pow(10, exp)
  const ratio = raw / base
  const step  = ratio <= 1 ? base
              : ratio <= 2 ? 2  * base
              : ratio <= 5 ? 5  * base
              :              10 * base

  const ticks: number[] = []
  // use rounding to avoid floating-point drift in the loop
  for (let i = 0; i * step <= maxVal + step * 0.01; i++) {
    ticks.push(parseFloat((i * step).toFixed(10)))
  }
  return ticks
}

/**
 * Derive the correct x-axis data key and label from a run's config.
 *
 * Rules:
 *   intervention          → 'year'     / 'Mars Year'
 *   sol                   → 'sol'      / 'Sol'
 *   year, 1 year          → 'sol'      / 'Sol'   (shows diurnal oscillation)
 *   year, N > 1 years     → 'mars_year'/ 'Mars Year'
 */
export function getAxisInfo(run: RunSummary): AxisInfo {
  const { exp_type, years } = run.config
  if (exp_type === 'intervention') return { xKey: 'year',      xLabel: 'Mars Year' }
  if (exp_type === 'sol')          return { xKey: 'sol',       xLabel: 'Sol'       }
  // year type
  const n = years ?? 1
  return n > 1
    ? { xKey: 'mars_year', xLabel: 'Mars Year' }
    : { xKey: 'sol',       xLabel: 'Sol'       }
}
