"""Host-side explanations of a rejected step; never changes validity limits."""
import math
import numpy as np


def json_safe(value):
    """Keep nonfinite failure data serializable, with reasons recorded separately."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def explain_arrays(air_k, pressure_pa, mass, number, gas, *, max_mass_loading,
                   lat_deg, lon_deg, sigma):
    """Describe the same particle-regime checks used by the online step.

    Arrays are [layer, longitude, latitude], as in the GCM. These checks explain
    rejected candidates; they do not reclassify them or bypass the ledger gate.
    """
    air_k, pressure_pa, mass, number, gas = map(np.asarray, (air_k, pressure_pa, mass, number, gas))
    reasons = []
    for name, values in (("air_temperature", air_k), ("layer_pressure", pressure_pa),
                         ("particle_mass", mass), ("particle_number", number), ("gas_mass", gas)):
        if not np.all(np.isfinite(values)):
            reasons.append(f"nonfinite_{name}")
    if np.any(air_k < 100.): reasons.append("air_temperature_below_100_k")
    if np.any(air_k > 350.): reasons.append("air_temperature_above_350_k")
    if np.any(pressure_pa < 1.): reasons.append("layer_pressure_below_1_pa")
    if np.any(pressure_pa > 2000.): reasons.append("layer_pressure_above_2000_pa")
    with np.errstate(divide="ignore", invalid="ignore"):
        loading = mass / gas
    if np.any(loading > max_mass_loading): reasons.append("particle_mass_loading_exceeded")
    if np.any(number < 0.): reasons.append("negative_particle_number")

    def extrema(values):
        finite = np.isfinite(values)
        if not finite.any():
            return dict(min=None, max=None, minimum_location=None, maximum_location=None)
        def location(index):
            level, lon, lat = np.unravel_index(index, values.shape)
            return dict(layer=int(level), sigma=float(sigma[level]),
                        longitude_deg=float(lon_deg[lon]), latitude_deg=float(lat_deg[lat]))
        minimum = int(np.argmin(np.where(finite, values, np.inf)))
        maximum = int(np.argmax(np.where(finite, values, -np.inf)))
        return dict(min=float(values.flat[minimum]), max=float(values.flat[maximum]),
                    minimum_location=location(minimum), maximum_location=location(maximum))

    return dict(reasons=reasons, air_temperature_k=extrema(air_k),
                layer_pressure_pa=extrema(pressure_pa), particle_mass_loading=extrema(loading),
                minimum_particle_number=float(np.min(number)) if np.all(np.isfinite(number)) else None,
                limits=dict(air_temperature_k=[100., 350.], layer_pressure_pa=[1., 2000.],
                            max_particle_mass_loading=max_mass_loading))
