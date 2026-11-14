# Surface CO2 mass exchange on the sigma grid

The old surface cycle changed surface pressure and frost mass. It did not
transport gas between layers when their sigma masses changed. It also omitted
the exchanged sensible heat and momentum. Latent surface heating alone did
not close the gas/cloud/frost energy budget.

NASA Ames v3.2.1, commit `ba47cada5f9eb89c75c4f10d06dbae6e7f9ecb26`,
distributes the surface mass change over its lower layers in
[`mars_physics.F90`](https://github.com/nasa/AmesGCM/blob/ba47cada5f9eb89c75c4f10d06dbae6e7f9ecb26/atmos_param_mars/mars_physics.F90#L1066).
Its default CO2-only pressure-ratio threshold is 0.991. Its initialization
finds the first level below that threshold, and its mass exchange includes two
additional layers above that index. The implementation here uses the same
index rule on our sigma grid. This is a grid-dependent mass-distribution
choice, not a measured universal mixing depth.

For each layer, gas mass is `surface_pressure * delta_sigma / gravity`.
The gas source is placed in the Ames lower-layer region. The difference between
that local source and the sigma-layer mass change sets the vertical interface
flux. The upwind donor supplies enthalpy, cloud ice and horizontal momentum.
Condensed gas gives its momentum to the surface reservoir. Sublimated gas
enters at surface temperature and at rest relative to the ground. This last
boundary condition is explicit in our implementation; it is not a claim that
the external Ames FV3 core uses the same discretization.

The heat solve includes cloud sensible heat and its hydrostatic potential
energy. The surface solve includes slab plus frost heat capacity, ice enthalpy,
and the sensible/kinetic energy carried by the gas. A separate surface momentum
reservoir records the equal and opposite exchange. No wind or temperature cap
is used. The existing one-hour atmospheric condensation scheme is unchanged.
This correction does not implement Ames atmospheric saturation adjustment,
convective momentum mixing, or its full external dynamical core.

Independent derivative tests check total gas/cloud/frost mass, resolved
enthalpy/kinetic/cloud-potential/surface energy, and atmospheric plus ground
horizontal momentum. They cover deposition and sublimation, clear and cloudy
columns, a vertical wind/temperature gradient, and zero exchange. Long-run
dry-core truncation and projection errors are measured separately.

The reference files were checked against NASA's GitHub API at v3.2.1. Local
Git blob hashes match the published source:

| File | Git blob SHA-1 |
| --- | --- |
| `mars_physics.F90` | `ca0b7c0774b1b23cafaee219062c3bffa3d0c4cc` |
| `update_mars_atmos.F90` | `d0351317dfb6edd4c83c01c67d9ad3a39bff3b2d` |
| `co2micromod_mgcm.F90` | `aa566e12d6e01afa346162385055f41d8d2bc18f` |
