# Surface CO2 mass transport: matched annual controls

Run the same T21/L12, MOLA/TES/dust annual protocol as the archived source
d9247914c7bb71c8611b00921c9295bf83d48b24. Change only the surface gas-exchange
coupling: include sigma redistribution, transported enthalpy/cloud/momentum,
surface heat capacity, and ground momentum transfer. Compare 450, 300 and
150-second steps over 59,356,800 seconds. Initial fields and frozen input bytes
must match the archived baseline before comparing errors or extrema.

Stage the frozen reviewer inputs at `/work/reviewer-inputs` on an isolated PVC.
Launch with a full 40-character source commit. There are no job time limits.
Archive every diagnostic, including raw and projected mass changes, exchange
energy, momentum, located Mach/wind/CFL extrema, frost deficits and spectra.
Final NetCDF files permit local field comparisons; global means alone cannot
establish stable climate. Do not infer observed accuracy or equilibrium from
successful integration. The separate five-sol forecast campaign measures
held-out predictive error with these same corrected physics.

`frost-capacity-job.json` pins source
`361bff80d9101009eb8b12a0f29215ce58811135`, including the common surface/frost
heat-capacity correction. The earlier `1f8fc20e` campaign is an intermediate
control with surface mass transport but inconsistent radiative frost capacity.
It remains immutable and runs to completion. The new campaign has its own
volume and can use an available A100 in any region. The input transfer is
verified by SHA-256 before submission. Manifests omit deadlines and timed
deletion; cluster admission still injects post-completion Job cleanup.
