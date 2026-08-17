"""gcm3d — a planet-agnostic differentiable 3-D GCM core on the NeuralGCM dycore.

A subpackage of the terraforming ``src`` package providing the mid-fidelity 3-D
tier of the model ladder. It is **not specific to any planet**: the core consumes
a :class:`~src.gcm3d.body.BodyConstants`, and concrete bodies supply an instance.
Mars's instance lives with the planet definition in
``src/celestials/planets/mars.py`` (``MARS_BODY_3D``), built from the existing
``MARS_*`` constants — so the core extends to other planets and moons by defining
a new ``BodyConstants``.

It depends on **JAX** (via ``dinosaur``), which the torch core does not — so
``dinosaur`` is an **optional extra**::

    pip install 'terraforming[gcm3d]'

``src.gcm3d.body`` (the ``BodyConstants`` abstraction) is pure Python and always
importable; the coordinate/equation builders require the extra. The two
frameworks meet only at the experiment layer via DLPack — never by cross-import.
See ``docs/ideas/dinosaur-mars-workplan.md`` for the phased plan.
"""

from __future__ import annotations

# Pure-Python body abstraction — always importable (no JAX / dinosaur).
from src.gcm3d.body import EARTH, BodyConstants

__all__ = ["BodyConstants", "EARTH", "__version__"]
__version__ = "0.0.1"

# Core builders need dinosaur (the gcm3d extra). Expose them when available;
# keep ``from src.gcm3d import BodyConstants`` working without the extra.
try:  # pragma: no cover - import-availability branch
    from src.gcm3d.coordinates import coordinate_system, grid
    from src.gcm3d.benchmarks import (
        DryDriftDiagnostics,
        TracerAdvectionDiagnostics,
        HeldSuarezDiagnostics,
        ConservedQuantities,
        ConservationDrift,
        dry_drift_diagnostics,
        run_resting_atmosphere,
        run_solid_body_tracer,
        run_balanced_jet,
        run_held_suarez,
        dry_conserved_quantities,
        conservation_drift,
        resting_convergence_matrix,
    )
    from src.gcm3d.dynamics import (
        integrate,
        primitive_equations,
        reference_temperature,
        stepper,
    )
    from src.gcm3d.maps import (
        MAP_SCALES,
        DEFAULT_SCALE,
        MarsMapFields,
        plot_maps,
        resolve_scale,
        run_maps,
        save_maps,
        save_netcdf,
    )
    from src.gcm3d.specs import nondimensionalization_scale, physics_specs
    from src.gcm3d.restart import (
        RESTART_FORMAT_VERSION,
        integrate_with_averaging,
        load_restart,
        save_restart,
    )
    from src.gcm3d.topography import (
        load_mola_meg,
        mola_modal_orography,
        regrid_to_nodal,
    )
    from src.gcm3d.physics import (
        CO2Forcing,
        ColumnPhysicsState,
        ColumnPhysicsTendencies,
        RadiativeForcing,
        SurfaceEnergyDiagnostics,
        RadiativeFluxDiagnostics,
        co2_frost_point_k,
        column_primitive_equations,
        cos_zenith_nodal,
        mean_anomaly_for_ls,
        orbital_distance,
        forced_co2_primitive_equations,
        forced_primitive_equations,
        initial_co2_state,
        positivity_preserving_co2_step,
        project_co2_reservoirs,
        initial_column_state,
        mars_co2_forcing,
        mars_radiative_forcing,
        radiative_heating_tendency,
        surface_energy_tendencies,
        two_stream_radiative_fluxes,
    )
    from src.gcm3d.terraforming_ode import (
        SeasonalForcing,
        SeasonalTrajectory,
        initial_seasonal_state,
        run_seasonal,
        seasonal_ode,
        seasonal_tendency,
        solar_flux,
        solar_longitude,
    )

    __all__ += [
        "coordinate_system",
        "DryDriftDiagnostics",
        "TracerAdvectionDiagnostics",
        "HeldSuarezDiagnostics",
        "ConservedQuantities",
        "ConservationDrift",
        "dry_drift_diagnostics",
        "run_resting_atmosphere",
        "run_solid_body_tracer",
        "run_balanced_jet",
        "run_held_suarez",
        "dry_conserved_quantities",
        "conservation_drift",
        "resting_convergence_matrix",
        "grid",
        "physics_specs",
        "nondimensionalization_scale",
        "RESTART_FORMAT_VERSION",
        "save_restart",
        "load_restart",
        "integrate_with_averaging",
        "primitive_equations",
        "reference_temperature",
        "stepper",
        "integrate",
        # 0-D terraforming ODE on the dinosaur substrate (seasonal / Ls outputs)
        "SeasonalForcing",
        "SeasonalTrajectory",
        "initial_seasonal_state",
        "run_seasonal",
        "seasonal_ode",
        "seasonal_tendency",
        "solar_flux",
        "solar_longitude",
        # 3-D column radiative forcing on the dycore (bridges 0-D physics -> 3-D)
        "RadiativeForcing",
        "ColumnPhysicsState",
        "ColumnPhysicsTendencies",
        "SurfaceEnergyDiagnostics",
        "RadiativeFluxDiagnostics",
        "mars_radiative_forcing",
        "initial_column_state",
        "column_primitive_equations",
        "forced_primitive_equations",
        "radiative_heating_tendency",
        "surface_energy_tendencies",
        "two_stream_radiative_fluxes",
        "cos_zenith_nodal",
        # 3-D CO2 condensation cycle (Leighton-Murray) on a tuple-wrapped state
        "CO2Forcing",
        "mars_co2_forcing",
        "co2_frost_point_k",
        "forced_co2_primitive_equations",
        "initial_co2_state",
        "positivity_preserving_co2_step",
        "project_co2_reservoirs",
        "mean_anomaly_for_ls",
        "orbital_distance",
        # 3-D dry-dynamics maps over MOLA terrain (Ames/LMD-comparable form)
        "load_mola_meg",
        "regrid_to_nodal",
        "mola_modal_orography",
        "MarsMapFields",
        "run_maps",
        "plot_maps",
        "save_netcdf",
        "save_maps",
        "MAP_SCALES",
        "DEFAULT_SCALE",
        "resolve_scale",
    ]
except ModuleNotFoundError:
    pass
