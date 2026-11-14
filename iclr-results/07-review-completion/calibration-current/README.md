# Two-hour calibration pilot

These are the original completed A100 results at source
`d9247914c7bb71c8611b00921c9295bf83d48b24`. The saved reports distinguish
stationary training endpoints from the nonstationary validation-selected
checkpoint. The 0.1985 K held-out temperature gain is a short-lead result and
does not establish a material climate improvement.

`source-snapshot.zip` preserves the 118 Python files recorded in
`provenance.json`, with their original relative paths and byte hashes. Keeping
them in an archive prevents historical test files from being collected as
current tests. Use the full pinned repository checkout for reruns; the Python
snapshot does not contain the separately tracked radiation data tables.

`../summarize_calibration.py` verifies all recorded Python source hashes,
gradient checks, convergence records, pairing, and the reported held-out gain.
The five-sol experiment is recorded separately under
`experiments/reviewer-multiday-calibration`.
