# Reviewer validation on Nautilus

Use the existing `apps/nautalis/jobs.py` runtime. `job.json` records the submitted
A100 campaign with source pinned to `dc37c66`. It uses the staged, hashed current
inputs on PVC `mars-review-104-20261113-v1` in namespace `razvanlab`.

The job runs three bounded calibration starts with independent seasonal train,
validation, and test windows. It then runs the full default T170/L40 stability
case and one Mars orbit at each of 450, 300, and 150 seconds with full Ames
physics at T21/L12. Each timestep case uses the same starting state and physical
duration. The 300-second calibration guard remains in place.

The GPU is required. Hardware identity, source bytes, configuration, progress,
checkpoints, raw and corrected mass budgets, regional pressure corrections,
energy and momentum effects, extrema, and spectra are stored on the PVC.
Wind locations, local Mach number, and horizontal advective CFL use the existing
field-audit calculation. CFL is a diagnostic of horizontal transport, not a
stability bound for the full solver.
Interrupted stability runs resume only when their source and input hashes match.
Calibration retries keep separate traces rather than overwriting an attempt.

After all three timestep cases finish, `timestep-comparison.json` reports
time-weighted global-diagnostic differences at exact shared checkpoint times.
The comparison rejects changed inputs, initial states, source bytes, or other
numerical settings. It does not interpolate missing times or infer accuracy
thresholds from the observed differences. Local field errors still require a
separate analysis of the saved states.

Check the submitted campaign:

```sh
rtk kubectl --context nautilus -n razvanlab get pods -l job-name=mars-review-104-20261113-a100-v2 -o wide
rtk kubectl --context nautilus -n razvanlab logs job/mars-review-104-20261113-a100-v2 --tail=80
```

H100 quota is zero. A40 and V100 attempts could not schedule and were deleted
before computation. The active A100 job requests four CPUs and 32 GiB of host
memory. Pending capacity does not establish a GPU run or a successful result.
Keep the PVC when removing a Job. Download and analyse the saved results before
claiming convergence, timestep agreement, or physical accuracy.
