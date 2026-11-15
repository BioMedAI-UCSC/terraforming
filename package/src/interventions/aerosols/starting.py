"""Physical climate starts are distinct from resuming particle experiments."""
import numpy as np
from src.framework.gcm._dinosaur import jax
from . import checkpoint


def save_physical_start(path,climate,contract):
    """Export a tracer-free GCM state with its body/grid/forcing contract.

    Obtain the contract from a matching pilot's configuration['start_contract'].
    The caller must export the actual source setup; this is metadata validation,
    not independent evidence that a climate has equilibrated or been validated.
    """
    if contract.get('kind')!='aerosol-physical-start-v1':
        raise ValueError('physical start requires a body/grid/forcing contract')
    if climate.dynamics.tracers:
        raise ValueError('physical starts currently require a tracer-free climate; use experiment resume for particles')
    if climate.dynamics.sim_time is None or not all(np.all(np.isfinite(np.asarray(x)))
            for x in jax.tree_util.tree_leaves(climate)):
        raise ValueError('physical start must have finite prognostic fields and an absolute clock')
    checkpoint.save(path,climate,contract)


def load_physical_start(path,template,contract):
    climate=checkpoint.load(path,template,contract)
    if climate.dynamics.tracers:
        raise ValueError('particle experiments must use experiment resume')
    if climate.dynamics.sim_time is None or not all(np.all(np.isfinite(np.asarray(x)))
            for x in jax.tree_util.tree_leaves(climate)):
        raise ValueError('physical start has nonfinite fields or no absolute clock')
    return climate
