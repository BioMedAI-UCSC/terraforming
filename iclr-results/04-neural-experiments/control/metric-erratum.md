# Final-time baseline correction (#100)

The original `report.json` and `report.md` are preserved as historical outputs.
Their `initial_rmse_k` of 1.4718313341402984 K compares the initial atmosphere
against a target at the end of the forecast. It is not the no-control baseline.

The target construction is uncontrolled-final air temperature minus 1.5 K,
so the correct no-control baseline is exactly 1.5 K. The archived controlled
final RMSE is 1.4825477186397238 K, giving 0.0174522813602762 K improvement.
This correction follows algebraically from the target definition; it does not
claim a rerun of the dirty-tree archived source. Regularized objectives are
separate from RMSE and are left unchanged.

New reports save both initial-state and uncontrolled-final RMSE, their explicit
baseline definition, improvement and the uncontrolled-final field in the NPZ.
The archived 0.0002 K/s heating bound differs from the current 0.002 K/s default;
do not combine their performance claims.
