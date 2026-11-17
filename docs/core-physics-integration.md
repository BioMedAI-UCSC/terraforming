# Core physics integration into dev

The core port is based on `dev` at
`c4d05f48b86e942f97b5a4097d34880346af44cc`. It consolidates the existing
CO2, calibration, gas-unit and CLI fixes from PRs #55, #115, #116, #119 and #120,
then adds the supported released GCM physics and module CI from PR #122.

The previous combined head, `7d54aa4c3adbdcdb52608bc081d84d67ab38a875`, is
preserved on
[`archive/physics-port-with-engineered-aerosols-20261116`](https://github.com/BioMedAI-UCSC/terraforming/tree/archive/physics-port-with-engineered-aerosols-20261116).
The `experimenal` branch also remains available. Existing experiment revisions
and saved outputs retain their original identity.

Engineered-particle work from PRs #111–#114 is deferred on its original
branches. The core branch excludes the engineered-particle package, optical
injection hooks, CLI/server/UI controls, experiment scripts and pilot workflow.
It retains the shared state, restart, tracer-conservation and energy-accounting
code required by the physics port.

Natural Mars dust, water clouds, CO2 clouds, settling and their radiative
effects remain in scope. These are required GCM physics.

The core branch must pass the complete selected regression suite before merge
into `dev`. Pushes to `dev` also run the complete suite. Native parity reports
record implementation hashes; reports affected by the separation are rerun.
The real-input T21/56-level smoke uses the pinned native MOLA16 file and checks
restart continuation. [CI groups and commands](ci.md) describe the test jobs.

Code integration does not establish whole-model equivalence, equilibrated
climate skill or a performance advantage. Those require the separate matched
experiments and budget analysis documented in [physics parity](physics-parity.md).
