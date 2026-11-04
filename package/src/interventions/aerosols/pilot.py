"""A reproducible flat-Mars screening setup shared by command/server clients."""
import dataclasses
import hashlib
from pathlib import Path
from importlib.resources import files
import numpy as np
from src.framework.gcm._dinosaur import jnp, primitive_equations, scales
from src.framework.gcm.coordinates import coordinate_system
from src.framework.gcm.specs import physics_specs
from src.framework.physics.gcm import initial_column_state
from src.celestials.planets.mars import MARS_BODY_3D as BODY
from src.celestials.planets.mars import gcm as mars_gcm
from .specs import aluminium_rod,iron_rod,Injection
from .data import rod_table
from .optics import make_band_table
from .state import initialize
from .experiment import coupled_step,ExperimentOptions


def prepare(*,material='aluminium',mode='passive',dt_s=60.,n_layers=12,
            truncation='T21',sources=(),initial_burden_kg_m2=0.,
            surface_temperature_k=210.,surface_pressure_pa=610.,albedo=.25,
            diurnal=True,initial_ls_deg=0.,co2_lw_scale=1.,dust_lw_scale=1.,
            surface_exchange_multiplier=1.,coagulation=False,charge_e=0.,
            collection_rate_s=0.,collection_source=None,initial_climate=None):
    """Return state, step and context. Flat terrain/ε=1 are recorded assumptions.

    Use a spun-up compatible climate checkpoint for research, rather than
    interpreting this uniform initial state as equilibrated Mars climatology.
    """
    if material not in ('aluminium','iron'):
        raise ValueError('verified optics are available only for aluminium and iron rods')
    if n_layers<2: raise ValueError('at least two layers required')
    for x in (surface_temperature_k,surface_pressure_pa):
        if not np.isfinite(x) or x<=0: raise ValueError('positive finite temperature/pressure required')
    particle=aluminium_rod() if material=='aluminium' else iron_rod()
    sources=tuple(x if isinstance(x,Injection) else Injection(**x) for x in sources)
    options=ExperimentOptions(mode,dt_s,coagulation,charge_e,collection_rate_s,collection_source)
    coords=coordinate_system(truncation,n_layers);specs=physics_specs(BODY);grid=coords.horizontal
    zeros=jnp.zeros((n_layers,)+grid.modal_shape)
    pressure=float(specs.nondimensionalize(surface_pressure_pa*scales.units.pascal))
    dynamics=primitive_equations.State(vorticity=zeros,divergence=zeros,temperature_variation=zeros,
        log_surface_pressure=grid.to_modal(jnp.full((1,)+grid.nodal_shape,np.log(pressure))),sim_time=0.)
    forcing=dataclasses.replace(mars_gcm.radiative_forcing(albedo=albedo,diurnal=diurnal,
        init_orbital_angle_rad=np.radians(initial_ls_deg),co2_radiation_enabled=True),
        emissivity=1.,ames_co2_longwave_opacity_scale=co2_lw_scale,
        ames_dust_longwave_opacity_scale=dust_lw_scale,
        surface_exchange_multiplier=surface_exchange_multiplier)
    climate=initial_column_state(dynamics,coords,surface_temperature_k,specs)
    start_contract=dict(kind='aerosol-physical-start-v1',body=dataclasses.asdict(BODY),
        truncation=truncation,n_layers=n_layers,sigma_boundaries=np.asarray(coords.vertical.boundaries).tolist(),
        forcing=dataclasses.asdict(forcing),co2_forcing=dataclasses.asdict(mars_gcm.co2_forcing()),terrain='flat',
        units='Dinosaur physics_specs(body) nondimensional dynamics; nodal surface temperature in kelvin')
    start_hash=None
    if initial_climate is not None:
        from .starting import load_physical_start
        climate=load_physical_start(initial_climate,climate,start_contract)
        start_hash=hashlib.sha256(Path(initial_climate).read_bytes()).hexdigest()
    state=initialize(climate,coords,specs,BODY,particle,initial_burden_kg_m2)
    table=rod_table(particle);bands=make_band_table(table,particle) if mode=='active' else None
    step=coupled_step(coords,specs,BODY,forcing,mars_gcm.co2_forcing(),particle,sources,options,bands=bands)
    asset=files('src.terraforming.aerosols').joinpath('assets/manifest.json').read_bytes()
    configuration=dict(schema=2,body=dataclasses.asdict(BODY),particle=particle.as_dict(),
        options=dataclasses.asdict(options),sources=[dataclasses.asdict(s) for s in sources],
        truncation=truncation,n_layers=n_layers,forcing=dataclasses.asdict(forcing),
        initial_surface_temperature_k=surface_temperature_k if initial_climate is None else None,
        initial_surface_pressure_pa=surface_pressure_pa if initial_climate is None else None,
        start_contract=start_contract,initial_climate_sha256=start_hash,
        initial_absolute_sim_time_nd=float(climate.dynamics.sim_time),
        initial_burden_kg_m2=initial_burden_kg_m2,asset_manifest_sha256=hashlib.sha256(asset).hexdigest(),
        terrain='flat screening pilot',surface_emissivity=1.,baseline_readiness='unverified')
    return state,step,dict(coords=coords,specs=specs,body=BODY,particle=particle,configuration=configuration)
