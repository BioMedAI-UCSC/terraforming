# Mars dynamical and solar clocks

The 3-D body uses 88,642.663 s for sidereal rotation and Coriolis, and
88,775.244 s for the solar day. The legacy `MARS_ROTATION_PERIOD` and radiation
forcing's `rotation_period_s` retain their historical solar-day meaning.
`BodyConstants.rotation_period_s` is sidereal; `solar_day_seconds` supplies the
solar clock for sol counts, forcing windows and diagnostics. Generic bodies
without an explicit solar day retain their previous rotation-based clock.

Reference: [NASA Mars Fact Sheet](https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html)
distinguishes 24.6229 h sidereal rotation from a 24.6597 h solar day; the model
retains the more precise second values above. Regolith and diurnal illumination
continue using the solar forcing period.

Existing archived runs used the solar day for Coriolis. Their arrays remain
loadable, but resuming with the corrected body changes dynamics and is not a
bitwise continuation. Preserve the old body explicitly for reproduction, and
record both clocks and the source commit for new runs. State-only NPZ restarts
do not contain a body configuration; callers must retain their run manifest.
Comparison NetCDF exports record both clocks.
