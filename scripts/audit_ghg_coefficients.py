#!/usr/bin/env python3
"""Verify coefficient provenance and quantify the published Mars gray-fit check."""
import argparse
import json
from pathlib import Path

from src.interventions.compounds import COMPOUNDS, COMPOUND_REFERENCE_DOMAIN


def audit(reference):
    table = reference["ipcc"]
    assert set(table["rows"]) == set(COMPOUNDS)
    for name, expected in table["rows"].items():
        compound = COMPOUNDS[name]
        actual = [compound.atmospheric_lifetime_yr, compound.rf_efficiency_W_m2_ppb, compound.gwp100]
        assert actual == expected, (name, actual, expected)
    mars = reference["marinova"]
    baseline = (mars["solar_flux_w_m2"] * (1. + .75 * mars["co2_gray_opacity"])
                / mars["stefan_boltzmann_w_m2_k4"])**.25
    rows = []
    for gas, (amplitude, exponent) in mars["gray_coefficients"].items():
        for pressure, expected in zip(mars["gas_partial_pressures_pa"], mars["temperature_increases_k"][gas], strict=True):
            opacity = mars["co2_gray_opacity"] + amplitude * pressure**exponent
            temperature = (mars["solar_flux_w_m2"] * (1. + .75 * opacity)
                           / mars["stefan_boltzmann_w_m2_k4"])**.25
            rows.append(dict(gas=gas, gas_partial_pressure_pa=pressure,
                table1_warming_k=expected, published_gray_fit_warming_k=temperature-baseline,
                gray_minus_table_k=temperature-baseline-expected,
                relative_warming_difference=(temperature-baseline)/expected-1.))
    return dict(registry_status="verified_against_primary_table", gases=len(COMPOUNDS),
        registry_domain=COMPOUND_REFERENCE_DOMAIN, references=reference,
        marinova_gray_fit_comparison=rows,
        mars_radiation_validation="not_established",
        limitations="Earth efficiencies are not Mars efficiencies. The independent calculation of the published gray fits does not reproduce every Table 1 warming within the stated fit errors. Full spectral reproduction is still required; no coefficient is tuned to hide that difference.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path,
        default=Path(__file__).resolve().parents[1] / "iclr-results/07-review-completion/compound-sources.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(json.loads(args.reference.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: result[key] for key in ("registry_status", "gases", "mars_radiation_validation")}))
