"""
Sky calculation functions for `~skymatch.skymatch.match`.

stsci_python/trunk/drizzlepac/lib/drizzlepac/sky.py

"""
# THIRD PARTY
from stsci.imagestats import ImageStats

__all__ = ['mean', 'median', 'mode']

def mean(data, **kwargs):
    """Return clipped mean of data as sky."""
    return ImageStats(data, **kwargs).mean

def median(data, **kwargs):
    """Return clipped median of data as sky."""
    return ImageStats(data, fields='midpt', **kwargs).midpt

def mode(data, **kwargs):
    """Return clipped mode of data as sky."""
    return ImageStats(data, fields='mode', **kwargs).mode
