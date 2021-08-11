"""
This module provides support for working with footprints
on the sky. Primary use case would use the following
generalized steps:

#. Initialize `SkyLine` objects for each input image.
   This object would be the union of all the input
   image's individual chips WCS footprints.

#. Determine overlap between all images. The
   determination would employ a recursive operation
   to return the extended list of all overlap values
   computed as [img1 vs [img2,img3,...,imgN],img2 vs
   [img3,...,imgN],...]

#. Select the pair with the largest overlap, or the
   pair which produces the largest overlap with the
   first input image. This defines the initial
   reference `SkyLine` object.

#. Perform some operation on the 2 images: for example,
   match sky in intersecting regions, or aligning
   second image with the first (reference) image.

#. Update the second image, either apply the sky value
   or correct the WCS, then generate a new `SkyLine`
   object for that image.

#. Create a new reference `SkyLine` object as the union
   of the initial reference object and the newly
   updated `SkyLine` object.

#. Repeat Steps 2-6 for all remaining input images.

This process will work reasonably fast as most operations
are performed using the `SkyLine` objects and WCS information
solely, not image data itself.

:Authors: Mihai Cara, Warren Hack, Pey-Lian Lim

:License: :doc:`LICENSE`

"""
# STDLIB
import sys
import os
from copy import copy, deepcopy
from os.path import basename
import numpy as np
import math

# THIRD-PARTY
from stwcs import wcsutil
from astropy import wcs as pywcs
from stwcs.distortion.utils import output_wcs
from spherical_geometry.polygon import SphericalPolygon
try:
    from stsci.tools.bitmask import bitfield_to_boolean_mask
except ImportError:
    from stsci.tools.bitmask import bitmask2mask as bitfield_to_boolean_mask

# LOCAL
from .utils import (is_countrate, ext2str, MultiFileLog, ImageRef,
                    file_name_components, temp_mask_file, in_memory_mask,
                    get_instrument_info)
from .parseat import FileExtMaskInfo, _Stat
from .hstinfo import supported_telescopes, supported_instruments, photcorr_kwd

# DEBUG
SKYLINE_DEBUG = False

__all__ = ['SkyLineMember', 'SkyLine']


class SkyLineMember(object):
    """
    Container for :py:class:`SkyLine` members that holds information about
    properties of a *single* extension (chip) in a FITS image such as:

    * WCS of the chip image;
    * bounding spherical polygon;
    * file name and extension from which the chip's image has originated;
    * information required for unit conversions (``EXPTIME``,
      ``PHOTFLAM``, ``BUNIT``, etc.);
    * user mask and DQ array associated with chip's image data.

    """
    def __init__(self, image, ext, dq_bits=0, dqimage=None, dqext=None,
                 usermask=None, usermask_ext=None):
        """
        Parameters
        ----------
        image: ImageRef
            An :py:class:`~stsci.skypac.utils.ImageRef` object that refers
            to an open FITS file

        ext: tuple, int, str
            Extension specification in the `image` the `SkyLineMember`
            object will be associated with.

            An int `ext` specifies extension number. A tuple in the form
            (str, int) specifies extension name and number. A string `ext`
            specifies extension name and the extension version is assumed
            to be 1. See documentation for `astropy.io.fits.getData`
            for examples.

        dq_bits: int, None (Default = 0)
            Integer sum of all the DQ bit values from the
            input `image`'s DQ array that should be considered "good"
            when building masks for sky computations. For example,
            if pixels in the DQ array can be combinations of 1, 2, 4,
            and 8 flags and one wants to consider DQ "defects" having
            flags 2 and 4 as being acceptable for sky computations,
            then `dq_bits` should be set to 2+4=6. Then a DQ pixel
            having values 2,4, or 6 will be considered a good pixel,
            while a DQ pixel with a value, e.g., 1+2=3, 4+8=12, etc.
            will be flagged as a "bad" pixel.

            | Default value (0) will make *all* non-zero
              pixels in the DQ mask to be considered "bad" pixels,
              and the corresponding image pixels will not be used
              for sky computations.

            | Set `dq_bits` to `None` to turn off the use of
              image's DQ array for sky computations.

            .. note::
                DQ masks (if used), *will be* combined with user masks
                specified by the `usermask` parameter.

        dqimage: ImageRef
            An :py:class:`~stsci.skypac.utils.ImageRef` object that refers
            to an open FITS file that has DQ data of the input `image`.

            .. note::
               When DQ data are located in the same FITS file as the
               science image data (e.g., HST/ACS, HST/WFC3, etc.),
               `dqimage` may point to the
               same :py:class:`~stsci.skypac.utils.ImageRef` object.
               In this case the reference count of the
               :py:class:`~stsci.skypac.utils.ImageRef` object must be
               increased adequately.

        dqext: tuple, int, str
            Extension specification of the `dqimage` that contains
            `image`'s DQ information. See help for `ext` for more
            details on acceptable formats for this parameter.

        usermask: ImageRef
            An :py:class:`~stsci.skypac.utils.ImageRef` object that refers
            to an open FITS file that has user mask data that indicate
            what pixels in the input `image` should be used for sky
            computations (``1``) and which pixels should **not** be used
            for sky computations (``0``).

        usermask_ext: tuple, int, str
            Extension specification of the `usermask` mask file that
            contains user's mask data that should be associated with
            the input `image` and `ext`. See help for `ext` for more
            details on acceptable formats for this parameter.

        """
        assert(hasattr(self.__class__, '_initialized') and
               self.__class__._initialized)
        self._reset()

        # check that input images and extensions are valid --
        # either integers or tuples of strings and integers, e.g., ('sci',1):
        _check_valid_imgext(image, 'image', ext, 'ext', can_img_be_None=False)
        if dq_bits is not None:
            if dqimage is None:
                dq_bits = 0
            else:
                _check_valid_imgext(dqimage, 'dqimage', dqext, 'dqext')
        _check_valid_imgext(usermask, 'usermask', usermask_ext, 'usermask_ext')

        # get telescope, instrument, and detector info:
        self.telescope, self.instrument, self.detector = get_instrument_info(
            image, ext)

        # check dq_bits:
        if dq_bits is not None and not isinstance(dq_bits, int):
            if image:
                dqimage.release()
            if usermask:
                usermask.release()
            if dqimage:
                dqimage.release()
            raise TypeError(
                "Argument 'dq_bits' must be either an integer or None."
            )

        # buld mask:
        self._buildMask(image.original_fname, ext, dq_bits,
                        dqimage, dqext, usermask, usermask_ext)
        if dqimage:
            dqimage.release()
        if usermask:
            usermask.release()

        # save file, user mask, and DQ extension info:
        self._fname = image.original_fname
        self._basefname = basename(self._fname)
        self._image = image
        self._ext = ext
        self._can_free_image = (image.can_reload_data and
                                self.optimize != 'speed')

        # check extension and create a string representation:
        try:
            extstr = ext2str(ext)
        except ValueError:
            raise ValueError("Unexpected extension type '{}' for file {}.".
                             format(ext, self._basefname))

        self._id = "{:s}[{:s}]".format(self._basefname, extstr)

        # extract WCS for bounding-box computation
        try:
            if hasattr(image.hdu[ext], 'wcs'):
                self._wcs = image.hdu[ext].wcs
            else:
                if self.telescope in supported_telescopes:
                    self._wcs = wcsutil.HSTWCS(image.hdu, ext)
                else:
                    self._wcs = pywcs.WCS(image.hdu[ext].header, image.hdu)
            if self._wcs is None:
                raise Exception("Invalid WCS.")

        except Exception as e:
            msg = "Unable to obtain WCS information for the file {:s}." \
                .format(self._id)
            self._ml.error(msg)
            self._ml.flush()
            self._release_all()
            raise e

        # determine pixel scale:
        self._get_pixel_scale()

        # see if image data are in counts or count-rate
        # and compute count(-rate) to flux (per arcsec^2) conversion factor:
        self._brightness_conv_from_hdu(image.hdu, self._idcscale)

        # process Sky user's keyword and its value:
        self._init_skyuser(image.hdu[ext].header)

        # Set polygon to be the bounding box of the chip:
        self._polygon = SphericalPolygon.from_wcs(self.wcs, steps=1)

    @classmethod
    def init_class(cls, skyuser_kwd='SKYUSER', units_kwd='BUNIT',
                   invsens_kwd=None, verbose=True, logfile=sys.stdout,
                   clobber=False, optimize='balanced'):
        cls._skyuser_header_keyword = skyuser_kwd
        cls._units_header_keyword = units_kwd
        cls._inv_sensitivity_kwd = invsens_kwd
        cls._verbose = verbose
        cls._clobber = clobber
        if isinstance(optimize, str):
            cls._optimize = optimize.lower()
        else:
            raise TypeError("The 'optimize' argument must be a string object.")

        if cls._optimize not in ['balanced', 'speed', 'inmemory']:
            raise ValueError("Currently supported values for the "
                             "'optimize' argument are: "
                             "'balanced', 'speed', or 'inmemory'.")

        # Set-up log files:
        if isinstance(logfile, MultiFileLog):
            cls._ml = logfile
        else:
            cls._ml = MultiFileLog(console=verbose)
            if logfile not in ('', None):
                cls._ml.add_logfile(logfile)
                cls._ml.skip(2)

        cls._initialized = True

    def _clean(self):
        if len(self._files2clean) > 0:
            for f in self._files2clean:
                try:
                    os.remove(f)
                except Exception:
                    self._ml.warning("Unable to remove temporary file "
                                     "'{:s}'.", f)

    def _reset(self):
        self._image = None
        self._ext = None
        self._mask = None
        self._maskext = None
        self._files2clean = []
        self._can_free_mask = False
        self._mask_is_imref = False
        self._id = ''
        self._pixscale = 1.0
        self._is_countrate = False
        self._skyuser = 0.0
        self._skyuser_delta = 0.0
        self._polygon = None
        self._wcs = None
        self._inv_sensitivity = None
        self._exptime = 1.0
        self._data2brightness_conv = 1.0

    def _release_all(self):
        if self._image is not None:
            self._image.release()
        if self._mask_is_imref:
            self._mask.release()

    def close(self, clean=True):
        self._release_all()
        if clean:
            self._clean()
        self._reset()

    def _buildMask(self, image_fname, ext, dq_bits, dq, dqext, msk, mskext):
        if dq_bits is None or dq is None:
            # we will use only the user mask:
            if msk is not None:
                if (self._optimize == 'balanced' and msk.can_reload_data) or \
                   self._optimize == 'speed' or self._optimize == 'inmemory':
                    # nothing to do: simply re-use the user mask:
                    self._mask = msk
                    self._maskext = mskext
                    self._mask.hold()
                else:
                    # self._optimize == 'balanced' but the mask cannot be freed
                    # so we will create a temporary fits file to hold mask
                    # data:
                    maskdata = (msk.hdu[mskext].data != 0).astype(np.uint8)
                    root, suffix, fext = file_name_components(image_fname)
                    mfname, self._mask = temp_mask_file(
                        maskdata, root, prefix='',
                        suffix='skymatch_mask',
                        ext=ext, randomize_prefix=False)
                    self._files2clean.append(mfname)
                    self._maskext = 0

                self._can_free_mask = self._mask.can_reload_data
                self._mask_is_imref = True

            else:
                # no mask will be used in sky computations:
                self._mask = None
                self._can_free_mask = False
                self._mask_is_imref = False

        else:
            # compute a new mask by:
            #    1. applying dq_bits to DQ array
            #    2. combining previous array with the user mask data
            #
            # If dq_bits show the "bad" bits then DQ mask should be computed
            # using:
            #
            # dqmskarr = np.logical_not(
            #     np.bitwise_and(dq.hdu[dqext].data,dq_bits)
            # )
            #
            # However, to keep the same convention with astrodrizzle, dq_bits
            # will show the "good" bits that should be removed from the DQ
            # array:
            dqmskarr = bitfield_to_boolean_mask(dq.hdu[dqext].data, dq_bits,
                                                dtype=np.bool_)

            # 2. combine with user mask:
            if msk is not None:
                maskdata = (msk.hdu[mskext].data != 0)
                dqmskarr = np.logical_and(dqmskarr, maskdata)
            # create a temporary file with the combined mask:
            (root, suffix, fext) = file_name_components(image_fname)
            if self._optimize == 'inmemory':
                self._mask = in_memory_mask(dqmskarr.astype(np.uint8))
                # strext = ext2str(ext, compact=True, default_extver=None)
                self._mask.original_fname = "{1:s}{0:s}{2:s}{0:s}{3:s}" \
                    .format('_', root, suffix, 'in-memory_skymatch_mask')
            else:
                mfname, self._mask = temp_mask_file(
                    dqmskarr.astype(np.uint8), root,
                    prefix='', suffix='skymatch_mask', ext=ext,
                    randomize_prefix=False
                )
                self._files2clean.append(mfname)

            self._maskext = 0
            self._can_free_mask = self._mask.can_reload_data
            self._mask_is_imref = True

    def _init_skyuser(self, image_header):
        skyuser_kwd = self.get_skyuser_kwd()

        if skyuser_kwd in image_header:
            self._skyuser = image_header[skyuser_kwd]
        else:
            self._skyuser = 0.0
        self._skyuser_delta = 0.0

    def reset_skyuser_from_header(self):
        self._init_skyuser(self._image.hdu[self._ext].header)

    def _get_pixel_scale(self):
        self._idcscale = None
        nominal_pscale = None

        if hasattr(self._wcs, 'idcscale') and self._wcs.idcscale is not None:
            self._idcscale = self._wcs.idcscale
            nominal_pscale = 'IDCSCALE'
        elif 'PAMSCALE' in self._image.hdu[self._ext].header:
            self._idcscale = float(self._image.hdu[self._ext]
                                   .header['PAMSCALE'])
            nominal_pscale = 'PAMSCALE'
        elif 'IDCSCALE' in self._image.hdu[self._ext].header:
            self._idcscale = float(self._image.hdu[self._ext]
                                   .header['IDCSCALE'])
            nominal_pscale = 'IDCSCALE'
        elif 'PAMSCALE' in self._image.hdu[0].header:
            self._idcscale = float(self._image.hdu[0].header['PAMSCALE'])
            nominal_pscale = 'PAMSCALE'
        elif 'IDCSCALE' in self._image.hdu[0].header:
            self._idcscale = float(self._image.hdu[0].header['IDCSCALE'])
            nominal_pscale = 'IDCSCALE'

        # try to compute "actual" pixel scale from the CD matrix.
        #TODO: Add support for more WCS representations (PC, CDELT, CROTA)
        self._pixscale = None
        if self._wcs.wcs.has_cd():
            self._pixscale = 3600.0 * math.sqrt(
                np.abs(np.linalg.det(self._wcs.wcs.cd))
            )
        else:
            if self._idcscale is not None:
                self._pixscale = self._idcscale
                self._ml.warning(
                    "Unable to compute \"actual\" pixel scale for image "
                    "{:s}[{:s}].\nWCS object does not have a CD matrix.\n"
                    "Using the value of '{:s}' pixel scale for actual "
                    "pixel scale: {:g}",
                    self._basefname, ext2str(self._ext), nominal_pscale,
                    self._idcscale)
            else:
                self._pixscale = 1.0
                self._idcscale = 1.0
                self._ml.warning(
                    "Unable to determine pixel scale of image in file "
                    "{:s}[{:s}].\nSetting pixel scale to 1.",
                    self._basefname, ext2str(self._ext))

        if self._idcscale is None:
            self._idcscale = self._pixscale
            self._ml.warning(
                "Unable to determine \"nominal\" pixel scale of image in "
                "file {:s}[{:s}].\nSetting pixel scale to \"actual\" pixel "
                "scale computed from CD matrix: {:g}",
                self._basefname, ext2str(self._ext), self._pixscale)

    def _brightness_conv_from_hdu(self, hdulist, pscale,
                                  primHDUname='PRIMARY'):
        #TODO: remove primHDUname from the argument list and
        # replace it with 0 once the bug in fileutil.getExtn() is fixed.
        # The bug causes imageObject[0] to return first image HDU instead
        # of the PRIMARY HDU.
        assert(pscale > 0.0)

        sci_header = hdulist[self.ext].header
        primary_header = hdulist[primHDUname].header

        inv_sensitivity_kwd = self.get_inv_sensitivity_kwd()

        # start with pixel-area scaling and add other conversion factors later
        self._data2brightness_conv = 1.0 / (pscale**2)

        # check if image data are in counts or count-rate
        self._is_countrate = is_countrate(
            hdulist, self.ext, units_kwd=self.get_units_kwd(),
            guess_if_missing=True, verbose=self.is_verbose(),
            flog=self._ml.unclose_copy()
        )

        # Check that all the necessary information for conversion
        # to flux (brightness) units is available:
        if self.is_countrate is None:
            self._ml.warning(
                "Unable to determine units of data in file {0}.\n"
                "*ASSUMING* image data are \"COUNT-RATE\".",
                self._basefname
            )

        # retrieve PHOTFLAM:

        # Initially assume that PHOTCORR was performed. For non-HST images
        # or unsupported instruments, we assume that PHOTCORR was performed,
        # and in the end we will check if PHOTFLAM (inverse sensitivity)
        # has a reasonable value (>0) if present.
        photcorr = inv_sensitivity_kwd is not None

        if 'TELESCOP' in primary_header:
            telescope = primary_header['TELESCOP'].strip().upper()
        else:
            telescope = None

        if telescope not in supported_telescopes:
            self._ml.warning("Skipping check of photometric correction "
                             "step for\nnon-HST file '{:s}': "
                             "Unsupported telescope.", self._basefname)
            telescope = None

        if telescope is not None:
            if 'INSTRUME' in primary_header:
                instrument = primary_header['INSTRUME'].strip().upper()
                if instrument not in supported_instruments:
                    self._ml.warning("Skipping check of photometric "
                                     "correction step for\nfile '{:s}': "
                                     "Unsupported instrument '{:s}'.",
                                     self._basefname, instrument)
                    instrument = None
            else:
                # For HST instruments we expect 'INSTRUME' to be present
                # in the primary header.
                errmsg = "Missing instrument information in '{:s}' data " \
                    "file '{:s}'.".format(telescope, self._basefname)
                self._ml.error(errmsg)
                self._ml.close()
                raise KeyError(errmsg)

        # see if photometric correction step was performed:
        if telescope is not None and instrument is not None and \
           inv_sensitivity_kwd is not None:
            phot_switch = photcorr_kwd[instrument][0]
            phot_done = photcorr_kwd[instrument][1]

            if phot_switch in primary_header:
                phot_switch_val = primary_header[phot_switch].strip()
                if phot_switch_val.upper() != phot_done:
                    photcorr = False
                    self._ml.warning(
                        "Photometric correction was not performed for data "
                        "file {:s}.\nVariations in detector sensitivity "
                        "WILL NOT be accounted for.", self._basefname
                    )
                    self._inv_sensitivity = None
            else:
                # For HST instruments we expect ~'PHOTCORR' to be present
                # in the primary header.
                errmsg = "Missing photometry switch keyword '{:s}' in " \
                         "{:s}-{:s} data file '{:s}'." \
                         .format(phot_switch, telescope, instrument,
                                 self._basefname)
                self._ml.error(errmsg)
                self._ml.close()
                raise KeyError(errmsg)

        if photcorr:
            if inv_sensitivity_kwd in sci_header:
                self._inv_sensitivity = sci_header[inv_sensitivity_kwd]
                if self._inv_sensitivity > 0.0:
                    self._data2brightness_conv *= self._inv_sensitivity
                else:
                    self._ml.warning(
                        "'{3:s}' value must be a *strictly* positive "
                        "number.\nFound: '{3:s}' = {0} in extension "
                        "{1:s} in data file '{2:s}'.\n'{3:s}' "
                        "value will be ignored.\nVariations in detector "
                        "sensitivity WILL NOT be accounted for.",
                        self._inv_sensitivity, ext2str(self.ext),
                        self._basefname, inv_sensitivity_kwd)
                    self._inv_sensitivity = None
            elif inv_sensitivity_kwd in primary_header:
                self._inv_sensitivity = primary_header[inv_sensitivity_kwd]
                if self._inv_sensitivity > 0.0:
                    self._data2brightness_conv *= self._inv_sensitivity
                else:
                    self._ml.warning(
                        "'{2:s}' value must be a *strictly* positive "
                        "number.\nFound: '{2:s}' = {0} in the primary "
                        "header in data file '{1:s}'.\n'{2:s}' "
                        "value will be ignored.\nVariations in detector "
                        "sensitivity WILL NOT be accounted for.",
                        self._inv_sensitivity, self._basefname,
                        inv_sensitivity_kwd)
                    self._inv_sensitivity = None
            else:
                self._inv_sensitivity = None
                self._ml.warning(
                    "Keyword '{2:s}' not found neither in the "
                    "Primary header nor\nin the header of "
                    "extension ({0:s}) in file {1:s}.\n"
                    "Variations in detector sensitivity WILL NOT "
                    "be accounted for.",
                    ext2str(self.ext), self._basefname, inv_sensitivity_kwd)

        # retrieve EXPTIME:
        if 'EXPTIME' in primary_header:
            self._exptime = primary_header['EXPTIME']
            if not self.is_countrate:
                if self._exptime > 0.0:
                    self._data2brightness_conv /= self._exptime
                else:
                    self._ml.warning(
                        "'EXPTIME' value must be a *strictly* positive "
                        "number.\nFound: 'EXPTIME' = {} in the primary "
                        "header in data file '{:s}'.\n'EXPTIME' "
                        "value will be ignored.\nVariations in exposure "
                        "time WILL NOT be accounted for.",
                        self._exptime, self._basefname
                    )
                    self._exptime = None
        elif 'EXPTIME' in sci_header:
            self._exptime = sci_header['EXPTIME']
            if not self.is_countrate:
                if self._exptime > 0.0:
                    self._data2brightness_conv /= self._exptime
                else:
                    self._ml.warning(
                        "'EXPTIME' value must be a *strictly* positive "
                        "number.\nFound: 'EXPTIME' = {} in extension "
                        "{:s} in data file '{:s}'.\n'EXPTIME' "
                        "value will be ignored.\nVariations in exposure "
                        "time WILL NOT be accounted for.",
                        self._exptime, ext2str(self.ext), self._basefname
                    )
                    self._exptime = None
        else:
            self._exptime = None
            if not self.is_countrate:
                self._ml.warning(
                    "Keyword 'EXPTIME' not found neither in the "
                    "Primary header nor\nin the header of "
                    "extension ({:s}) in file {:s}.\n"
                    "Variations in exposure time WILL NOT be accounted for.",
                    ext2str(self.ext), self._basefname
                )

    def __repr__(self):
        return '%s(%r, %r, %r, %r)' % (self.__class__.__name__, self.fname,
                                       self.ext, self.wcs, self.polygon)

    @classmethod
    def get_skyuser_kwd(cls):
        return cls._skyuser_header_keyword

    @classmethod
    def set_skyuser_kwd(cls, kwd):
        cls._skyuser_header_keyword = kwd

    @classmethod
    def get_units_kwd(cls):
        return cls._units_header_keyword

    @classmethod
    def set_units_kwd(cls, kwd):
        cls._units_header_keyword = kwd

    @classmethod
    def get_inv_sensitivity_kwd(cls):
        return cls._inv_sensitivity_kwd

    @classmethod
    def set_inv_sensitivity_kwd(cls, kwd):
        cls._inv_sensitivity_kwd = kwd

    @classmethod
    def is_verbose(cls):
        return cls._verbose

    @classmethod
    def set_verbose(cls, verbose):
        cls._verbose = verbose

    @classmethod
    def optimize(cls):
        return cls._optimize

    @property
    def fname(self):
        return self._fname

    @property
    def basefname(self):
        return self._basefname

    @property
    def image_hdulist(self):
        return self._image.hdu

    @property
    def image_header(self):
        return self._image.hdu[self._ext].header

    @property
    def image_data(self):
        data = self._image.hdu[self._ext].data
        return data

    @image_data.setter
    def image_data(self, newdata):
        self._image.hdu[self._ext].data = newdata

    def free_image_data(self):
        if self._can_free_image:
            del self._image.hdu[self._ext].data

    @property
    def ext(self):
        return self._ext

    @property
    def maskext(self):
        return self._maskext

    @property
    def mask_data(self):
        if self._mask_is_imref:
            data = self._mask.hdu[self._maskext].data
            return data
        else:
            return self._mask

    def free_mask_data(self):
        if self._mask_is_imref and self._can_free_mask:
            del self._mask.hdu[self._maskext].data

    @property
    def wcs(self):
        return self._wcs

    @property
    def polygon(self):
        return self._polygon

    @property
    def inv_sensitivity(self):
        return self._inv_sensitivity

    @property
    def exptime(self):
        return self._exptime

    @property
    def data2brightness_conv(self):
        return self._data2brightness_conv

    @property
    def id(self):
        return self._id

    @property
    def skyuser(self):
        return self._skyuser

    @property
    def skyuser_delta(self):
        return self._skyuser_delta

    @property
    def skyuser_total(self):
        return self._skyuser + self._skyuser_delta

    @property
    def is_countrate(self):
        return self._is_countrate

    def update_skyuser(self, delta_skyval=None):
        if delta_skyval is None:
            self._skyuser += self._skyuser_delta
        else:
            self._skyuser += delta_skyval
        self._skyuser_delta = 0.0

    def update_skydelta(self, delta_skyval):
        self._skyuser_delta += delta_skyval

    def set_skydelta(self, delta_skyval):
        self._skyuser_delta = delta_skyval

    def set_skyuser(self, skyval):
        self._skyuser = skyval
        self._skyuser_delta = 0.0

    def data2brightness(self, data):
        return (self.data2brightness_conv * data)

    def brightness2data(self, brightness):
        return (brightness / self.data2brightness_conv)


class SkyLine(object):
    """
    Manage outlines on the sky.

    Skylines are designed to capture and manipulate HST WCS image
    information as spherical polygons. They are represented by
    the :py:class:`~stsci.skypac.skyline.SkyLine` class, which is an
    extension of :py:class:`~spherical_geometry.polygon.SphericalPolygon`
    class.

    Each skyline has a list of members,
    `~stsci.skypac.skyline.SkyLine.members`, and a composite spherical polygon,
    `~stsci.skypac.skyline.SkyLine.polygon`, members. The polygon has all
    the functionalities of `~spherical_geometry.polygon.SphericalPolygon`.

    Each `SkyLine` has a list of `~SkyLine.members` and
    a composite `~SkyLine.polygon` with all the
    functionalities of `~spherical_geometry.polygon.SphericalPolygon`.

    Each member in `~stsci.skypac.skyline.SkyLine.members` belongs
    to the `~stsci.skyline.skyline.SkyLineMember` class, which contains
    image name (with path if given), science extension(s),
    and composite WCS and polygon of the extension(s). All skylines start
    out with a single member from a single image. When operations are used
    to find composite or intersecting skylines, the
    resulting skyline can have multiple members.

    For example, a skyline from an ACS/WFC full-frame image would give 1
    member, which is a composite of extensions 1 and 4. A skyline from the
    union of 2 such images would have 2 members, and so forth.

    """
    def __init__(self, mlist):
        """
        Parameters
        ----------
        fname: str
            FITS image. `None` to create empty `SkyLine`.

        ext: a list of tuples ('extname',extver).

        """
        if isinstance(mlist, FileExtMaskInfo):
            self.init_from_imextmask_info(mlist)
        else:
            self.members = mlist

        self._skydiff = None

    def init_from_imextmask_info(self, fi):
        if fi.fnamesOnly:
            fi.convert2ImageRef()
        if not fi.finalized:
            fi.finalize()

        n = fi.count
        if n < 1:
            raise ValueError("Input 'FileExtMaskInfo' object must contain "
                             "at least one valid extension specification.")

        im = fi.image
        if im.closed:
            raise ValueError("Input 'FileExtMaskInfo' object must contain "
                             "at an opened image object.")

        if fi.DQimage.closed:
            dq = None
            dqext = n * [None]
        else:
            dq = fi.DQimage
            dqext = fi.dqext

        slm = []

        for i in range(n):
            um = fi.mask_images[i]
            ume = fi.maskext[i]
            if um.closed:
                um = None
                ume = None

            im.hold()
            if dq is not None:
                dq.hold()
            if um is not None:
                um.hold()
            m = SkyLineMember(im, fi.fext[i], dq_bits=fi.dq_bits,
                              dqimage=dq, dqext=dqext[i],
                              usermask=um, usermask_ext=ume)
            if um is not None:
                um.release()
            slm.append(m)

        # SkyLine does not need im, dq anymore:
        im.release()
        if dq is not None:
            dq.release()

        self.members = slm

    def __getattr__(self, what):
        """
        Control attribute access to
        `~spherical_geometry.polygon.SphericalPolygon`.

        """
        if what in ('from_radec', 'from_cone', 'from_wcs',
                    'multi_union', 'multi_intersection',
                    '_find_new_inside',):
            raise AttributeError("'%s' object has no attribute '%s'"
                                 (self.__class__.__name__, what))
        else:
            return getattr(self.polygon, what)

    def __copy__(self):
        return deepcopy(self)

    def __repr__(self):
        return '%s(%r, %r)' % (self.__class__.__name__, self.polygon,
                               self.members)

    def close(self, clean=True):
        for m in self.members:
            m.close(clean=clean)

    @property
    def skyuser_hdr_keyword(self):
        return self._skyuser_hdr_keyword

    @property
    def is_mf_mosaic(self):
        """
        returns *True* if `SkyLine` members are from distinct image files
        (multi-file mosaic) and *False* otherwise.

        """
        return self._is_mf_mosaic

    @property
    def polygon(self):
        """
        `~spherical_geometry.polygon.SphericalPolygon` portion of `SkyLine`
        that contains the composite skyline from `members`
        belonging to *self*.

        """
        return self._polygon

    @polygon.setter
    def polygon(self, value):
        assert isinstance(value, SphericalPolygon)
        self._polygon = copy(value)  # Deep copy

    @property
    def skydiff(self):
        return self._skydiff

    @skydiff.setter
    def skydiff(self, sky):
        self._skydiff = sky

    @property
    def id(self):
        return self._id

    @property
    def members(self):
        """
        List of `SkyLineMember` objects that belong to *self*.
        Duplicate members are discarded. Members are kept in
        the order of their additions to *self*.

        """
        return self._members

    @members.setter
    def members(self, mlist):
        self.set_members(mlist=mlist, polygon=None)

    def set_members(self, mlist, polygon):
        if mlist is None:
            member_list = []
        elif isinstance(mlist, list):
            member_list = mlist
            if [1 for m in mlist if not isinstance(m, SkyLineMember)]:
                raise ValueError("The 'mlist' argument must be either "
                                 "a single 'SkyLineMember' object or a "
                                 "Python list of 'SkyLineMember' objects.")
        elif isinstance(mlist, SkyLineMember):
            member_list = [mlist]
        else:
            raise ValueError("The 'mlist' argument must be either "
                             "a single 'SkyLineMember' object or a "
                             "Python list of 'SkyLineMember' objects.")

        self._members = []

        # Not using set to preserve order
        n = len(member_list)
        if n == 0:
            if polygon is None:
                self.polygon = SphericalPolygon([])
            else:
                self.polygon = deepcopy(polygon)
            self._id = ''
            self._is_mf_mosaic = False

        elif n == 1:
            assert isinstance(member_list[0], SkyLineMember)
            if polygon is None:
                self.polygon = deepcopy(member_list[0].polygon)
            else:
                self.polygon = deepcopy(polygon)
            self._id = member_list[0].id
            self._members.append(member_list[0])
            self._is_mf_mosaic = False

        else:
            assert isinstance(member_list[0], SkyLineMember)
            if polygon is None:
                mpol = deepcopy(member_list[0].polygon)
            else:
                mpol = deepcopy(polygon)
            self._members.append(member_list[0])

            for m in member_list[1:]:
                # Report corrupted members list instead of skipping
                assert isinstance(m, SkyLineMember)

                if m not in self._members:
                    self._members.append(m)
                    if polygon is None:
                        mpol = mpol.union(m.polygon)

            self.polygon = mpol
            self._update_mosaic_flag_id()

    def _update_mosaic_flag_id(self, val=None):
        # updates _is_mf_mosaic flag and recomputes SkyLine's _id
        # val:
        #    True - set to mosaic
        #    False - set _is_mf_mosaic=False and recompute _id.
        #        NOTE: No check will be done to see if the SkyLine is a "true"
        #        mosaic. Therefore computed _id may be innacurate since it
        #        will be computed based on the first member's file name.
        #    None - autodetect status
        #
        if val is True:
            self._is_mf_mosaic = True
            self._id = 'mosaic'
            return

        elif val is False:
            self._is_mf_mosaic = False

            if len(self.members) == 0:
                self._id = ''
                return

            extlist = []
            for m in self.members:
                if m.ext not in extlist:
                    extlist.append(m.ext)
            self._id = self._id_from_fname_ext(self.members[0].basefname,
                                               extlist)
            return

        else:
            # autodetect status
            basefname = None
            fstats = []
            nfnames = 0
            extlist = []

            for m in self.members:
                mstat = _Stat(m.fname)
                if mstat not in fstats:
                    if basefname is None:
                        basefname = m.basefname  # store the first occurence
                    fstats.append(mstat)
                    nfnames += 1
                    if nfnames > 1:
                        break
                if m.ext not in extlist:
                    extlist.append(m.ext)

            if nfnames == 0:
                self._is_mf_mosaic = False
                self._id = ''
            elif nfnames == 1:
                self._is_mf_mosaic = False
                self._id = self._id_from_fname_ext(basefname, extlist)
            else:
                self._is_mf_mosaic = True
                self._id = "mosaic"

    def _indv_mem_wcslist(self):
        """List of original HSTWCS from each EXT in each member."""
        wcs_list = []

        for m in self.members:
            wcs_list.append(m.wcs)

        return wcs_list

    def to_wcs(self):
        """
        Combine `HSTWCS` objects from all `members` and return
        a new `HSTWCS` object. If no `members`, return `None`.

        .. warning:: This cannot return WCS of intersection.

        """
        wcs_list = self._indv_mem_wcslist()

        n = len(wcs_list)
        if n > 1:
            wcs = output_wcs(wcs_list)
        elif n == 1:
            wcs = wcs_list[0]
        else:
            wcs = None

        return wcs

    @staticmethod
    def _id_from_fname_ext(fname, ext_list):
        extdic = {}
        for ext in ext_list:
            if isinstance(ext, tuple):
                extname = ext[0].upper()
                extver = ext[1]
            elif isinstance(ext, int):
                extname = None
                extver = ext
            if extname in extdic:
                extdic[extname].append(extver)
            else:
                extdic.update({extname: [extver]})

        strext = []
        disctinct_ext = list(extdic.keys())
        disctinct_ext.sort()
        for extname in disctinct_ext:
            if extname is None:
                strext.append("{:s}".format(
                    ','.join(map(str, extdic[extname]))))
            else:
                strext.append("'{:s}',{:s}".format(extname,
                              ','.join(map(str, extdic[extname]))))

        fextid = "{:s}[{:s}]".format(fname, ';'.join(strext))

        return fextid

    def _draw_members(self, map, **kwargs):
        """
        Draw individual extensions in members.
        Useful for debugging.

        Parameters
        ----------
        map: Basemap axes object

        **kwargs: Any plot arguments to pass to basemap

        """
        wcs_list = self._indv_mem_wcslist()

        for wcs in wcs_list:
            poly = SphericalPolygon.from_wcs(wcs)
            poly.draw(map, **kwargs)

    def _find_members(self, given_members):
        """
        Find `SkyLineMember` in *given_members* that is in
        *self*. This is used for intersection.

        Parameters
        ----------
        self: obj
            `SkyLine` instance.

        given_members: list
            List of `SkyLineMember` to consider.

        Returns
        -------
        new_members: list
            List of `SkyLineMember` belonging to *self*.

        """
        if len(list(self.points)) > 3:
            out_mem = [m for m in given_members if
                       self.intersects_poly(m.polygon)]
        else:
            out_mem = []
        return out_mem

    def add_image(self, other):
        """
        Return a new `SkyLine` that is the union of *self*
        and *other*.

        .. warning::
            `SkyLine.union` only returns `polygon` without `members`.

        Parameters
        ----------
        other: `SkyLine` object

        Examples
        --------
        >>> s1 = SkyLine('image1.fits')
        >>> s2 = SkyLine('image2.fits')
        >>> s3 = s1.add_image(s2)

        """
        newcls = self.__class__(None)
        newcls.polygon = self.union(other)

        newcls._members = []
        for v in self.members:
            newcls._members.append(v)
        for v in other.members:
            if v not in newcls._members:
                newcls._members.append(v)

        if self.is_mf_mosaic or other.is_mf_mosaic:
            newcls._update_mosaic_flag_id(True)
        else:
            newcls._update_mosaic_flag_id(None)

        return newcls

    def find_intersection(self, other):
        """
        Return a new `SkyLine` that is the intersection of
        *self* and *other*.

        .. warning:: `SkyLine.intersection` only returns
            `polygon` without `members`.

        Parameters
        ----------
        other: `SkyLine` object

        Examples
        --------
        >>> s1 = SkyLine('image1.fits')
        >>> s2 = SkyLine('image2.fits')
        >>> s3 = s1.find_intersection(s2)

        """
        newcls = self.__class__(None)
        mlist = newcls._find_members(self.members + other.members)
        poly = self.intersection(other)
        newcls.set_members(mlist, poly)
        return newcls

    def find_max_overlap(self, skylines):
        """
        Find `SkyLine` from a list of *skylines* that overlaps
        the most with *self*.

        Parameters
        ----------
        skylines: list
            A list of `SkyLine` instances.

        Returns
        -------
        max_skyline: `SkyLine` instance or `None`
            `SkyLine` that overlaps the most or `None` if no
            overlap found. This is *not* a copy.

        max_overlap_area: float
            Area of intersection.

        """
        # from mpl_toolkits.basemap import Basemap
        # from matplotlib import pyplot as plt

        max_skyline = None
        max_overlap_area = 0.0

        for next_s in skylines:
            try:
                intersect_poly = self.intersection(next_s)
                overlap_area = intersect_poly.area()
            except (ValueError, AssertionError):

                # m = Basemap()
                # self.polygon.draw(m)
                # next_s.polygon.draw(m)

                if SKYLINE_DEBUG:
                    print('WARNING: Intersection failed for {0} and {1}. '
                          'Ignoring {1}...'.format(self.id, next_s.id))
                    overlap_area = 0.0
                else:
                    raise

            if overlap_area > max_overlap_area:
                max_overlap_area = overlap_area
                max_skyline = next_s

        return max_skyline, max_overlap_area

    @staticmethod
    def max_overlap_pair(skylines):
        """
        Find a pair of skylines with maximum overlap.

        Parameters
        ----------
        skylines: list
            A list of `SkyLine` instances.

        Returns
        -------
        max_pair: tuple
            Pair of `SkyLine` objects with max overlap
            among given *skylines*. If no overlap found,
            return `None`. These are *not* copies.

        """
        max_pair = None
        max_overlap_area = 0.0

        for i in range(len(skylines) - 1):
            curr_s = skylines[i]
            next_s, i_area = curr_s.find_max_overlap(skylines[i + 1:])

            if i_area > max_overlap_area:
                max_overlap_area = i_area
                max_pair = (curr_s, next_s)

        return max_pair


def _is_valid_ext(ext):
    if isinstance(ext, int):
        return True
    v = isinstance(ext, tuple) and len(ext) == 2 and \
        isinstance(ext[0], str) and isinstance(ext[1], int)
    return v


def _check_valid_imgext(image, imgargname, ext, extargname,
                        can_img_be_None=True):
    # check image object:
    if (image is None and not can_img_be_None) or \
       (image is not None and not isinstance(image, ImageRef)):
        raise ValueError("Input argument '{:s}' must be a valid "
                         "'ImageRef' object.".format(imgargname))
    # check extension:
    if image is not None and (ext is None or not _is_valid_ext(ext)):
        raise TypeError("Input argument '{:s}' must be either an "
                        "integer or a tuple of the form ('str', int)."
                        .format(extargname))
