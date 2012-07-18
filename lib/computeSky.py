"""Sky calculation functions for `~skymatch.skymatch.match`."""

# THIRD PARTY
from stsci.imagestats import ImageStats

__all__ = ['mode']

def mode(data, nclip=10):
    """
    Return clipped mode of data as sky.

    stsci_python/trunk/drizzlepac/lib/drizzlepac/sky.py

    Parameters
    ----------
    data : array_like

    nclip : int
        Number of clipping iterations.

    Returns
    -------
    Sky value in image data unit.
    
    """
    return ImageStats(data, fields='mode', nclip=nclip).mode
