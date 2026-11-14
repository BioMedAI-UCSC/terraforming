"""Reusable physical operators used by framework simulation engines.

Planet modules configure these operators with their own constants and datasets;
this package deliberately does not import from :mod:`src.celestials`.
"""

__all__ = ["ames_radiation", "gcm"]
