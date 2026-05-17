export interface PresetValues {
  planet: {
    surface_temperature: number
    surface_pressure:    number
    albedo:              number
    greenhouse_factor:   number
    ice_mass:            number
    latitude:            number
    longitude:           number
    elevation_m:         number
    initial_ls_deg:      number
  }
  engine: {
    dt:       number
    accuracy: string
  }
}

/** One data point from the simulation — shape varies by exp_type. */
export interface DataPoint {
  // intervention
  year?: number
  temperature_k: number
  temp_min_k?: number
  temp_max_k?: number
  pressure_pa: number
  ice_mass_kg: number
  delta_F?: number
  greenhouse_factor?: number
  // sol / year
  time_h?: number
  sol?: number
  mars_year?: number
  solar_flux?: number
}

export interface RunConfig {
  preset: string
  exp_type: string
  years: number
  sols: number
  accuracy: string
  dt: number
  lat: number | null
  lon: number | null
  elevation: number | null
  ls: number | null
  surface_temp: number | null
  surface_pressure: number | null
  albedo: number | null
  greenhouse_factor: number | null
  ice_mass: number | null
  inject: Record<string, number>
  label: string | null
}

export interface RunSummary {
  id: string
  status: 'running' | 'done' | 'error'
  progress: number
  config: RunConfig
  label: string
  error: string | null
  created_at: string
  completed_at: string | null
}

export interface Run extends RunSummary {
  data: DataPoint[]
}
