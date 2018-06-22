"""
A module that provides functions for computing Pixel Area Maps (PAM) based
on distortion model contained in a FITS WCS.

:Authors: Mihai Cara (contact: help@stsci.edu)

:License: :doc:`LICENSE`

"""
from __future__ import division, print_function

import numpy as np
import astropy
from astropy.io import fits
from astropy import wcs as fitswcs
import stwcs
from distutils.version import LooseVersion

from . import __version__
from . import __vdate__


__author__ = 'Mihai Cara'

__all__ = ['pam_from_file', 'pam_from_wcs']


ASTROPY_VER_GE13 = LooseVersion(astropy.__version__) >= LooseVersion('1.3')


def _compute_pam_sd(wcs, shape=None, blc=(1, 1), idcscale=1.0, cdscale=1.0):
    """
    Computes Pixel Area Map (PAM) using the distortion model defined in WCS
    and described through Simple Image Polynomials (SIP) by computing
    the Jacobian of the distortion model.

    This function computes the Jacobian of the distortion model using
    *symbolic differentiation* of Simple Image Polynomials.

    Parameters
    ----------

    wcs : astropy.wcs.WCS, stwcs.wcsutil.HSTWCS
        A `WCS` object containing the distortion model.

    shape : tuple of int, None (Default = None)
        A tuple of two integers (ny, nx) indicating the size of the PAM image
        to be generated. When the default value is used (`None`), the size
        of the returned PAM array will be determined from `wcs._naxis1`
        and `wcs._naxis2` attributes of the supplied `WCS` object
        as (`wcs._naxis2`, `wcs._naxis1`).

    blc : tuple of int or float (Default = (1, 1))
        A tuple indicating the coordinates of the bottom-left pixel of the
        PAM array to be computed. These coordinates should be given
        in the image coordinate system defined by the input WCS (in which,
        for example, `WCS.crpix` is defined). The first element specifies
        the column (`"x"`-coordinate) and the second element specifies
        the row (`"y"`-coordinate).

    idcscale : float (Default = 1.0)
        A positive number indicating the pixel scale used in the
        "Instrument Distortion Correction" for HST instruments. For
        non-HST instruments this parameter may be set to be equal
        to `cdscale`.

    cdscale : float (Default = 1.0)
        A positive number indicating the pixel scale as computed from the
        CD matrix. HST instruments CD matrix includes linear distortion
        terms.

    Returns
    -------
    PAM : numpy.ndarray
        Pixel area map.

    """
    if shape is None:
        shape = (wcs._naxis2, wcs._naxis1)

    # rescale factor:
    rf = (cdscale / idcscale)**2

    # distortion does not exist or is linear:
    if wcs.sip is None or wcs.sip.a_order < 1 or wcs.sip.b_order < 1 or \
       (wcs.sip.a_order == 1 and wcs.sip.b_order == 1):
        return rf * np.ones(shape, dtype=np.float64)

    # prepare coordinates:
    x = np.arange(shape[1], dtype=np.float) - wcs.sip.crpix[0] + float(blc[0])
    y = np.arange(shape[0], dtype=np.float) - wcs.sip.crpix[1] + float(blc[1])

    ar = np.arange(wcs.sip.a_order + 1)
    br = np.arange(wcs.sip.b_order + 1)

    ones_a = np.ones(wcs.sip.a_order + 1)
    ones_b = np.ones(wcs.sip.b_order + 1)

    # "coordinate vectors" (e.g., (1, x, x**2, x**3, ...)) used in
    # distortion bilinear forms:
    ax = np.outer(x, ones_a)**ar
    ay = np.outer(y, ones_a)**ar
    bx = np.outer(x, ones_b)**br
    by = np.outer(y, ones_b)**br

    # derivatives of the "coordinate vectors" with regard to x & y:
    adx = np.roll(ax, 1, 1) * ar
    ady = np.roll(ay, 1, 1) * ar
    bdx = np.roll(bx, 1, 1) * br
    bdy = np.roll(by, 1, 1) * br

    # derivatives of the binomial forms:
    A = wcs.sip.a.T
    B = wcs.sip.b.T
    dadx = 1.0 + np.tensordot(ay.T, np.tensordot(A, adx, (1, 1)), (0, 0))
    dady = np.tensordot(ady.T, np.tensordot(A, ax, (1, 1)), (0, 0))
    dbdx = np.tensordot(by.T, np.tensordot(B, bdx, (1, 1)), (0, 0))
    dbdy = 1.0 + np.tensordot(bdy.T, np.tensordot(B, bx, (1, 1)), (0, 0))

    # compute rescaled Jacobian
    jacobian = rf * np.abs(dadx * dbdy - dady * dbdx)

    return jacobian


def pam_from_file(image, ext, output_pam, ignore_vacorr=False,
                  normalize_at_crpix=False):
    """
    Generate a **P**\ ixel **A**\ rea **M**\ ap (PAM) file from the ``FITS``
    ``WCS`` contained in an image extension of a calibrated ``HST`` image
    file specified by ``image``.

    .. note::
       PAM computation is performed using the distortion model defined in the
       ``WCS and described through Simple Image Polynomials (SIP).
       Non-polynomial distortions are ignored!

    Parameters
    ----------

    image : str
        File name of a ``FITS`` image that will provide a ``FITS`` ``WCS``.

    ext : int, str, tuple of (str, int)
        Extension specification. May be an integer extension number,
        a string extension name, or a tuple of extension name *and*
        extension version.

    output_pam : str
        Output file name to which PAM will be written.

        .. warning::
            If the output file already exists, it will be overwritten
            without warnings.


    ignore_vacorr : bool, optional
        When set to `True`, ``PAM`` will be generated _as if_ vellocity
        aberration has not applied to the ``WCS``.

        .. warning::
           This function does not know whether velocity aberration (VA)
           correction has been applied to the ``WCS`` or not. It is user's
           responsibility to check the appropriateness of settung this
           parameter to `True`. Setting ``ignore_vacorr`` to `True` when
           ``WCS`` was not VA-corrected will result in larger errors in
           computed ``PAM``. **Default value is highly recommended!**

    normalize_crpix : bool, optional
        Indicates whether to normalize computed ``PAM`` to 1 at *``CRPIX``
        position. Historically, ``WFC3`` pixel area maps have been normalized
        to 1 at ``CRPIX`` position.

    """
    with fits.open(image, mode='readonly') as h:
        data_shape = h[ext].data.shape
        try:
            wcs = stwcs.wcsutil.HSTWCS(h, ext)
        except:
            wcs = fitswcs.WCS(h[ext].header, h)

    pam = pam_from_wcs(wcs, shape=data_shape, ignore_vacorr=ignore_vacorr,
                       normalize_at_crpix=normalize_at_crpix)

    if ASTROPY_VER_GE13:
        fits.PrimaryHDU(pam).writeto(output_pam, overwrite=True)
    else:
        fits.PrimaryHDU(pam).writeto(output_pam, clobber=True)


def pam_from_wcs(wcs, shape=None, ignore_vacorr=False,
                 normalize_at_crpix=False):
    """
    Generate a **P**\ ixel **A**\ rea **M**\ ap (PAM) file from a ``FITS``
    ``WCS``.

    .. note::
       PAM computation is performed using the distortion model defined in the
       ``WCS and described through Simple Image Polynomials (SIP).
       Non-polynomial distortions are ignored!

    Parameters
    ----------

    wcs : astropy.wcs.WCS
        An `~astropy.wcs.WCS` object to be used for generating PAM file.

    shape : tuple of two int, None, optional
        Shape of the output image ``(ny, nx)``. If se to default `None`, this
        function will try to deduce the shape of the output image from the
        values of ``_naxis1`` and ``_naxis2`` attributes of the input ``wcs``
        object.

    ignore_vacorr : bool, optional
        When set to `True`, ``PAM`` will be generated _as if_ vellocity
        aberration has not applied to the ``WCS``.

        .. warning::
           This function does not know whether velocity aberration (VA)
           correction has been applied to the ``WCS`` or not. It is user's
           responsibility to check the appropriateness of settung this
           parameter to `True`. Setting ``ignore_vacorr`` to `True` when
           ``WCS`` was not VA-corrected will result in larger errors in
           computed ``PAM``. **Default value is highly recommended!**

    normalize_at_crpix : bool, optional
        Indicates whether to normalize computed ``PAM`` to 1 at *``CRPIX``
        position. Historically, ``WFC3`` pixel area maps have been normalized
        to 1 at ``CRPIX`` position.

    Returns
    -------

    pam : numpy.ndarray
        A 2D `numpy.ndarray` containing PAM.

    Raises
    ------

    ValueError
        When ``shape`` is `None` but the ``wcs`` object has no
        ``_naxis1`` and ``_naxis2`` attributes.

    """
    if shape is None:
        if hasattr(wcs, '_naxis'):
            shape = tuple(reversed(wcs._naxis))
        elif hasattr(wcs, '_naxis1') and hasattr(wcs, '_naxis2'):
            shape = (wcs._naxis2, wcs._naxis1)
        else:
            raise ValueError("When 'shape' is None, 'wcs' object must have "
                             "either attributes '_naxis1' and '_naxis2' or "
                             "attribute '_naxis'.")

    cdscale = np.sqrt(fitswcs.utils.proj_plane_pixel_area(wcs)) * 3600.0

    # Ignoring VAFACTOR helps get PAMs that are in better
    # agreement with the ones produced by R. Hook in 2004.
    if ignore_vacorr and hasattr(wcs, 'vafactor'):
        cdscale /= wcs.vafactor

    idcscale = wcs.idcscale if hasattr(wcs, 'idcscale') else cdscale
    pam = _compute_pam_sd(wcs=wcs, shape=shape, idcscale=idcscale,
                          cdscale=cdscale)

    if normalize_at_crpix:
        crpix1, crpix2 = [int(np.floor(v + 0.5)) - 1 if v >= 0.0 else
                          int(np.ceil(v - 0.5)) - 1 for v in wcs.wcs.crpix]
        pam /= pam[crpix2, crpix1]

    return pam
