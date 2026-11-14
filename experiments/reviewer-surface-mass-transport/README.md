# Surface CO2 mass transport: matched annual controls

Run the same T21/L12, MOLA/TES/dust annual protocol as the archived source
d9247914c7bb71c8611b00921c9295bf83d48b24. Change only the surface gas-exchange
coupling: include sigma redistribution, transported enthalpy/cloud/momentum,
surface heat capacity, and ground momentum transfer. Compare 450, 300 and
150-second steps over 59,356,800 seconds. Initial fields and frozen input bytes
must match the archived baseline before comparing errors or extrema.

Use the existing reviewer-validation PVC, which contains `/work/reviewer-inputs`.
Launch with a full 40-character source commit. There are no job time limits.
Archive every diagnostic, including raw and projected mass changes, exchange
energy, momentum, located Mach/wind/CFL extrema, frost deficits and spectra.
Final NetCDF files permit local field comparisons; global means alone cannot
establish stable climate. Do not infer observed accuracy or equilibrium from
successful integration. The separate five-sol forecast campaign measures
held-out predictive error with these same corrected physics.
