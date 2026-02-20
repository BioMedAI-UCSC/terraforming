"""Project-wide constants and TensorFlow dtype configuration.

This module is the single source of truth for:
    • The project TensorFlow dtype (``TF_DTYPE = tf.float64``)
    • The ``_c()`` helper for creating typed scalar tensors
    • Universal physical constants (SI units, as ``tf.Tensor``)

Import example::

    from src.constants import TF_DTYPE, _c, STEFAN_BOLTZMANN, PI
"""

from __future__ import annotations

import tensorflow as tf

# ---------------------------------------------------------------------------
# TensorFlow dtype used throughout the project
# ---------------------------------------------------------------------------
TF_DTYPE = tf.float64


# ---------------------------------------------------------------------------
# Helper: convert a Python scalar to a tf.constant with the project dtype
# ---------------------------------------------------------------------------

def _c(value: float) -> tf.Tensor:
    """Convenience: scalar → ``tf.constant(value, dtype=TF_DTYPE)``."""
    return tf.constant(value, dtype=TF_DTYPE)


# ---------------------------------------------------------------------------
# Universal physical constants  (all tf.Tensor, float64, SI units)
# ---------------------------------------------------------------------------
STEFAN_BOLTZMANN: tf.Tensor   = _c(5.670374419e-8)     # W m⁻² K⁻⁴
BOLTZMANN_K: tf.Tensor        = _c(1.380649e-23)       # J K⁻¹
G_NEWTON: tf.Tensor           = _c(6.67430e-11)        # m³ kg⁻¹ s⁻²
AU_METRES: tf.Tensor          = _c(1.49597870700e11)   # 1 AU in metres
SOLAR_CONSTANT_1AU: tf.Tensor = _c(1361.0)             # W m⁻² at 1 AU
PI: tf.Tensor                 = _c(3.141592653589793)   # π
