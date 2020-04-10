"""
Sky statistics computation class for `~skymatch.skymatch` and
`~skymatch.skymatch._weighted_sky`.

:Authors: Mihai Cara

:License: :doc:`LICENSE`

"""
# THIRD PARTY
from stsci.imagestats import ImageStats
from copy import deepcopy

__all__ = ['SkyStats']
__taskname__ = 'skystatistics'


class SkyStats(object):
    """
    This is a superclass build on top of
    :py:class:`stsci.imagestats.ImageStats`. Compared to ``ImageStats``,
    ``SkyStats`` has "persistent settings" in the sense
    that object's parameters need to be set once and these settings
    will be applied to all subsequent computations on different data.

    """
    def __init__(self, skystat='mean', **kwargs):
        """ Initializes the SkyStats object.

        Parameters
        -----------
        skystat: str
            Sets the statistics that will be returned by the
            `~SkyStats.calc_sky`. The following statistics are supported:
            ``'mean'``, ``'mode'``, `'midpt'`, and ``'median'``.
            First three statistics have the same meaning as in
            `stsdas.toolbox.imgtools.gstatistics <http://stsdas.stsci.edu/
            cgi-bin/gethelp.cgi?gstatistics>`_
            while ``skystat='median'`` will compute the median of the
            distribution.

        kwargs: dict
            A dictionary of optional arguments to be passed to ``ImageStats``.

        """
        self.npix = None
        self.skyval = None

        self._fields = ','.join(['npix', skystat])

        self._kwargs = deepcopy(kwargs)
        if 'fields' in self._kwargs:
            del self._kwargs['fields']

        if 'image' in self._kwargs:
            del self._kwargs['image']

        self._skystat = {
            'mean': self._extract_mean,
            'mode': self._extract_mode,
            'median': self._extract_median,
            'midpt': self._extract_midpt
        }[skystat]

    def _extract_mean(self, imstat):
        return imstat.mean

    def _extract_median(self, imstat):
        return imstat.median

    def _extract_mode(self, imstat):
        return imstat.mode

    def _extract_midpt(self, imstat):
        return imstat.midpt

    def calc_sky(self, data):
        """ Computes statistics on data.

        Parameters
        -----------
        data: numpy.ndarray
            A numpy array of values for which the statistics needs to be
            computed.

        Returns
        --------
        statistics: tuple
            A tuple of two values: (``skyvalue``, ``npix``), where ``skyvalue``
            is the statistics specified by the `skystat` parameter during
            the initialization of the ``SkyStats`` object and ``npix`` is the
            number of pixels used in comuting the statistics reported in
            ``skyvalue``.

        """
        imstat = ImageStats(image=data, fields=self._fields, **(self._kwargs))
        self.skyval = self._skystat(imstat)
        self.npix = imstat.npix
        return (self.skyval, self.npix)
