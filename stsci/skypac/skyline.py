# -*- coding: utf-8 -*-

# Copyright (C) 2011 Association of Universities for Research in
# Astronomy (AURA)
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     1. Redistributions of source code must retain the above
#       copyright notice, this list of conditions and the following
#       disclaimer.
#
#     2. Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials
#       provided with the distribution.
#
#     3. The name of AURA and its representatives may not be used to
#       endorse or promote products derived from this software without
#       specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY AURA ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL AURA BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

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

"""
from __future__ import division, print_function, absolute_import

# STDLIB
import sys, os
from copy import copy, deepcopy
from os.path import basename, split, splitext
import numpy as np

# THIRD-PARTY
import pyfits
from stwcs import wcsutil
from stwcs.distortion.utils import output_wcs
from stsci.sphere.polygon import SphericalPolygon
#from stsci.tools.fileutil import openImage
#from drizzlepac.imageObject import imageObject

# LOCAL
from .utils import is_countrate, ext2str, MultiFileLog, ImageRef, \
     file_name_components, temp_mask_file
from .parseat import FileExtMaskInfo

# DEBUG
SKYLINE_DEBUG = True

__all__ = ['SkyLineMember', 'SkyLine']
__version__ = '0.6'
__vdate__ = '12-Dec-2013'





#Skylines
#--------

#Skylines are designed to capture and manipulate HST WCS image information as
#spherical polygons. They are represented by the `~stsci.sphere.skyline.SkyLine`
#class, which is an extension of `~stsci.sphere.polygon.SphericalPolygon` class.

#Representation
#``````````````
#Each skyline has a list of members, `~stsci.sphere.skyline.SkyLine.members`,
#and a composite spherical polygon, `~stsci.sphere.skyline.SkyLine.polygon`,
#members. The polygon has all the functionalities of
#defined by those `~stsci.sphere.polygon.SphericalPolygon`.

#What is a skyline member?
#^^^^^^^^^^^^^^^^^^^^^^^^^

#Each member in `~stsci.sphere.skyline.SkyLine.members` belongs to the
#`~stsci.sphere.skyline.SkyLineMember` class, which contains image name (with
#path if given), science extension(s), and composite WCS and polygon of the
#extension(s). All skylines start out with a single member from a single image.
#When operations are used to find composite or intersecting skylines, the
#resulting skyline can have multiple members.

#For example, a skyline from an ACS/WFC full-frame image would give 1 member,
#which is a composite of extensions 1 and 4. A skyline from the union of 2 such
#images would have 2 members, and so forth.

#Creating skylines
#`````````````````

#`~stsci.sphere.skyline.SkyLine` constructor takes an image name and an optional
#`extname` keyword, which defaults to "SCI". To create skyline from
#single-extension FITS, change `extname` to "PRIMARY".

#If `None` is given instead of image name, an empty skyline is created with no
#member and an empty spherical polygon.

#Operations on skylines
#``````````````````````

#`~stsci.sphere.skyline.SkyLine` has direct access to most of the
#`~stsci.sphere.polygon.SphericalPolygon` properties and methods *except* for the
#following (which are still accessible indirectly via
#`~stsci.sphere.skyline.SkyLine.polygon`):

  #- `~stsci.sphere.polygon.SphericalPolygon.from_radec`
  #- `~stsci.sphere.polygon.SphericalPolygon.from_cone`
  #- `~stsci.sphere.polygon.SphericalPolygon.from_wcs`
  #- `~stsci.sphere.polygon.SphericalPolygon.multi_union`
  #- `~stsci.sphere.polygon.SphericalPolygon.multi_intersection`

#In addition, `~stsci.sphere.skyline.SkyLine` also has these operations available:

  #- `~stsci.sphere.skyline.SkyLine.to_wcs`: Return a composite HST WCS object
    #defined by all the members. In a skyline resulting from intersection, this
    #does *not* return the WCS of the intersecting polygons.

  #- `~stsci.sphere.skyline.SkyLine.add_image`: Return a new skyline that is the
    #union of two skylines. This should be used, *not* `SkyLine.union` (which is
    #actually `~sphere.polygon.SphericalPolygon.union`) that will not include
    #members.

  #- `~stsci.sphere.skyline.SkyLine.find_intersection`: Return a new skyline
    #that is the intersection of two skylines. This should be used, *not*
    #`SkyLine.intersection` (which is actually
    #`~stsci.sphere.polygon.SphericalPolygon.intersection`) that will not include
    #members.

  #- `~stsci.sphere.skyline.SkyLine.find_max_overlap` and
    #`~stsci.sphere.skyline.SkyLine.max_overlap_pair`: Return a pair of skylines
    #that overlap the most from a given list of skylines.

  #- `~stsci.sphere.skyline.SkyLine.mosaic`: Return a new skyline that is a mosaic of
    #given skylines that overlap, a list of image names of the skylines used, and
    #a list of image names of the excluded skylines. A pair of skylines with the
    #most overlap is used as a starting point. Then a skyline that overlaps the
    #most with the mosaic is used, and so forth until no overlapping skyline is
    #found.



class SkyLineMember(object):
    """
    Container for `SkyLine` members with these attributes:

        * `fname`: Image name (with path if given)
        * `ext`: Tuple of extensions read
        * `wcs`: `HSTWCS` object the composite data
        * `polygon`: :py:class:`~stsci.sphere.polygon.SphericalPolygon`
          object of the composite data

    """
    def __init__(self, image, ext, DQFlags=0, dqimage=None, dqext=None,
                 usermask=None, usermask_ext=None):
        """
        Parameters
        ----------
        fname : str
            FITS image.

        extname : str
            EXTNAME to use. SCI is recommended for normal
            HST images. PRIMARY if image is single ext.

        """
        assert(hasattr(self.__class__, '_initialized') and \
               self.__class__._initialized)
        self._reset()

        # check that input images and extensions are valid --
        # either integers or tuples of strings and integers, e.g., ('sci',1):
        _check_valid_imgext(image, 'image', ext, 'ext', can_img_be_None=False)
        if DQFlags is not None:
            if dqimage is None:
                DQFlags = 0
            else:
                _check_valid_imgext(dqimage, 'dqimage', dqext, 'dqext')
        _check_valid_imgext(usermask, 'usermask', usermask_ext,'usermask_ext')

        # check DQFlags:
        if DQFlags is not None and not isinstance(DQFlags,int):
            if image:  dqimage.release()
            if usermask: usermask.release()
            if dqimage:  dqimage.release()
            raise TypeError("Argument 'DQFlags' must be either an integer or None.")

        # buld mask:
        self._buildMask(image.original_fname, ext, DQFlags,
                        dqimage, dqext, usermask, usermask_ext)
        if dqimage:  dqimage.release()
        if usermask: usermask.release()

        # save file, user mask, and DQ extension info:
        self._fname          = image.original_fname
        self._basefname      = basename(self._fname)
        self._image          = image
        self._ext            = ext
        self._can_free_image = image.can_reload_data and self.optimize != 'speed'

        # check extension and create a string representation:
        try:
            extstr = ext2str(ext)
        except ValueError:
            raise ValueError("Unexpected extension type \'{}\' for file {}.".\
                             format(ext,self._basefname))

        self._id    = "{:s}[{:s}]".format(self._basefname, extstr)

        # extract WCS for bounding-box computation
        try:
            if hasattr(image.hdu[ext], 'wcs'):
                self._wcs = image.hdu[ext].wcs
            else:
                self._wcs = wcsutil.HSTWCS(image.hdu, ext)
            if self._wcs is None:
                raise Exception("Invalid WCS.")
        except:
            msg = "Unable to obtain WCS information for the file {:s}.".format(self._id)
            self._ml.error(msg)
            self._ml.flush()
            self._release_all()
            raise
            #raise RuntimeError(msg)

        # determine pixel scale:
        self._get_pixel_scale()

        # see if image data are in counts or count-rate
        # and compute count(-rate) to flux (per arcsec^2) conversion factor:
        self._brightness_conv_from_hdu(image.hdu, self._pixscale)

        # process Sky user's keyword and its value:
        self._init_skyuser(image.hdu[ext].header)

        # Set polygon to be the bounding box of the chip:
        self._polygon = SphericalPolygon.from_wcs(self.wcs)

    # variable class member that holds FITS header keyword used to store
    # computed sky value:
    #_skyuser_header_keyword = 'SKYUSER'

    # variable class member that holds FITS header keyword for data units:
    #_units_header_keyword = 'BUNIT'

    # variable class member that holds 'verbose' preference:
    #_verbose = True

    # variable class member that holds MultiFileLog:
    #_ml = sys.stdout

    @classmethod
    def init_class(cls, skyuser_kwd='SKYUSER', units_kwd='BUNIT',
                   verbose=True, logfile=sys.stdout, clobber=False,
                   optimize='balanced'):
        cls._skyuser_header_keyword = skyuser_kwd
        cls._units_header_keyword   = units_kwd
        cls._verbose                = verbose
        cls._clobber                = clobber
        if isinstance(optimize,str):
            cls._optimize           = optimize.lower()
        else:
            raise TypeError("The 'optimize' argument must be a string object.")

        if not (cls._optimize == 'balanced' or cls._optimize == 'speed'):
            raise ValueError("Currently supported values for the " \
                             "'optimize' argument are: 'balanced' or 'speed'.")

        # Set-up log files:
        if isinstance(logfile, MultiFileLog):
            cls._ml = logfile
        else:
            cls._ml = MultiFileLog(console = verbose)
            if logfile not in ('', None):
                cls._ml.add_logfile(logfile)
                cls._ml.skip(2)

        cls._initialized = True

    def _clean(self):
        if len(self._files2clean) > 0:
            for f in self._files2clean:
                try:
                    os.remove(f)
                except:
                    self._ml.warning("Unable to remove temporary file " \
                                     "\'{:s}\'.", f)

    def _reset(self):
        self._image         = None
        self._ext           = None
        self._mask          = None
        self._maskext       = None
        self._files2clean   = []
        self._can_free_mask = False
        self._mask_is_imref = False
        self._id            = ''
        self._pixscale      = 1.0
        self._is_countrate  = False
        self._skyuser       = 0.0
        self._skyuser_delta = 0.0
        self._polygon       = None
        self._wcs           = None
        self._photflam      = 1.0
        self._exptime       = 1.0
        self._data2brightness_conv= 1.0

    def _release_all(self):
        if self._image is not None:
            self._image.release()
        if self._mask_is_imref:
            self._mask.release()

    def close(self, clean=True):
        self._release_all()
        if clean: self._clean()
        self._reset()

    def _buildMask(self, image_fname, ext, DQFlags, dq, dqext, msk, mskext):
        if DQFlags is None or dq is None:
            # we will use only the user mask:
            if msk is not None:
                if (self._optimize == 'balanced' and msk.can_reload_data) or \
                    self._optimize == 'speed':
                    # nothing to do: simply re-use the user mask:
                    self._mask     = msk
                    self._maskext  = mskext
                    self._mask.hold()
                else:
                    # self._optimize == 'balanced' but the mask cannot be freed
                    # so we will create a temporary fits file to hold mask data:
                    maskdata = (msk.hdu[mskext].data != 0).astype(np.uint8)
                    (root,suffix,fext)  = file_name_components(image_fname)
                    mfname, self._mask = temp_mask_file(root, 'skymatch_mask', \
                                                        ext, maskdata)
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
            #    1. applying DQFlags to DQ array
            #    2. combining previous array with the user mask data
            #
            # If DQFlags show the "bad" bits then DQ mask should be computed
            # using:
            #
            #dqmskarr = np.logical_not(np.bitwise_and(dq.hdu[dqext].data,DQFlags))
            #
            # However, to keep the same convention with astrodrizzle, DQFlags
            # will show the "good" bits that should be removed from the DQ array:
            dqmskarr = np.logical_not(np.bitwise_and(dq.hdu[dqext].data,~DQFlags))
            # 2. combine with user mask:
            if msk is not None:
                maskdata = (msk.hdu[mskext].data != 0)
                dqmskarr = np.logical_and(dqmskarr, maskdata)
            # create a temporary file with the combined mask:
            (root,suffix,fext)  = file_name_components(image_fname)
            mfname, self._mask = temp_mask_file(root, 'skymatch_mask', \
                                            ext, dqmskarr.astype(np.uint8))

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

    def _get_pixel_scale(self):
        if hasattr(self._wcs, 'idcscale') and self._wcs.idcscale is not None:
            #TODO: it is not clear why astrodrizzle uses "comanded" pixel scale
            # instead of the "distorted" scale for sky subraction. Talk to INS?
            self._pixscale = self._wcs.idcscale
        else:
            # try to compute from the CD matrix. This is better than
            # wcs.pscale and likely is more accurate than wcs.idcscale used
            # by astrodrizzle but this last part depends on how flat fields
            # are defined. For this reason (until we get a clarification from
            # INSwe will continue to use astrodrizzle's approach
            # (see "then" part of the "if" above).
            try:
                if not self._wcs.wcs.has_cd():
                    raise # CD matrix is not available
                self._pixscale = math.sqrt(
                    math.abs(np.linalg.det(self._wcs.wcs.cd)))*3600.0
                self._ml.warning(
                    "WCS object for file {1:s}[{2:s}] does not have "        \
                    "'pascale' attribute.{0}Using the value of pixel scale " \
                    "computed from the CD matrix: {3}",                      \
                    os.linesep, self._basefname, ext2str(self._ext), self._pixscale)
            except:
                self._pixscale = 1.0
                self._ml.warning(
                    "Unable to determine pixel scale of image in file "      \
                    "{1}[{2}].{0}Setting pixel scale to 1.",                 \
                    os.linesep, self._basefname, ext2str(self._ext))

    def _brightness_conv_from_hdu(self, hdulist, pscale, primHDUname='PRIMARY'):
        #TODO: remove primHDUname from the argument list and
        # replace it with 0 once the bug in fileutil.getExtn() is fixed.
        # The bug causes imageObject[0] to return first image HDU instead
        # of the PRIMARY HDU.

        # check if image data are in counts or count-rate
        self._is_countrate = is_countrate(hdulist, self.ext,
                units_kwd=self.get_units_kwd(),
                guess_if_missing=True,
                verbose=self.is_verbose(), flog=self._ml.unclose_copy())

        # Check that all the necessary information for conversion
        # to flux (brightness) units is available:
        missing_info = False

        if self.is_countrate == None:
            self._ml.warning(
                "Unable to determine units of data in file {0}.{1}"  \
                "No conversion to brightness units will be performed " \
                "(except for pixel area rescaling).". \
                self._basefname, os.linesep)
            missing_info = True

        # retrieve PHOTFLAM:
        if 'PHOTFLAM' in hdulist[self.ext].header:
            self._photflam = hdulist[self.ext].header['PHOTFLAM']
        elif 'PHOTFLAM' in hdulist[primHDUname].header:
            self._photflam = hdulist[primHDUname].header['PHOTFLAM']
        else:
            self._photflam = None
            if not missing_info:
                self._ml.warning(
                    "Keyword \'PHOTFLAM\' not found in the header of "   \
                    "extension ({0:s}) in file {1:s}.{2}"                \
                    "No conversion to brightness units will be performed.", \
                    ext2str(self._ext), self._basefname, os.linesep)
                missing_info = True

        # retrieve EXPTIME:
        if 'EXPTIME' in hdulist[primHDUname].header:
            self._exptime = hdulist[primHDUname].header['EXPTIME']
        elif 'EXPTIME' in hdulist[self.ext].header:
            self._exptime = hdulist[self.ext].header['EXPTIME']
        else:
            self._exptime = None
            if not missing_info and self.is_countrate == False:
                self._ml.warning(
                    "Keyword \'EXPTIME\' not found in the primary header " \
                    "of file {0:s}.{1}"                                    \
                    "No conversion to brightness units will be performed.".\
                    self._basefname, os.linesep)
                missing_info = True

        # compute conversion factor (data units -> brightness units):
        if missing_info:
            self._data2brightness_conv = 1.0/(pscale**2)
        else:
            self._data2brightness_conv = self._photflam/(pscale**2)
            if not self.is_countrate:
                self._data2brightness_conv /= self._exptime

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
    def photflam(self):
        return self._photflam

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
    def is_countrate(self):
        return self._is_countrate

    def update_skyuser(self, delta_skyval):
        self._skyuser_delta = delta_skyval
        self._skyuser += delta_skyval

    def data2brightness(self, data):
        return (self.data2brightness_conv * data)

    def brightness2data(self, brightness):
        return (brightness / self.data2brightness_conv)


class SkyLine(object):
    """
    Manage outlines on the sky.

    Each `SkyLine` has a list of `~SkyLine.members` and
    a composite `~SkyLine.polygon` with all the
    functionalities of `~stsci.sphere.polygon.SphericalPolygon`.

    """
    def __init__(self, mlist):
        """
        Parameters
        ----------
        fname : str
            FITS image. `None` to create empty `SkyLine`.

        ext : a list of tuples ('extname',extver).

        """
        if isinstance(mlist, FileExtMaskInfo):
            self.init_from_imextmask_info(mlist)
        else:
            self.members = mlist

    def init_from_imextmask_info(self, fi):
        if fi.fnamesOnly:
            fi.convert2ImageRef()
        if not fi.finalized:
            fi.finalize()

        n = fi.count
        if n < 1:
            raise ValueError("Input 'FileExtMaskInfo' object must contain " \
                             "at least one valid extension specification.")

        im = fi.image
        if im.closed:
            raise ValueError("Input 'FileExtMaskInfo' object must contain " \
                             "at an opened image object.")

        if fi.DQimage.closed:
            dq    = None
            dqext = n * [ None ]
        else:
            dq    = fi.DQimage
            dqext = fi.dqext

        slm = []

        for i in range(n):
            um  = fi.mask_images[i]
            ume = fi.maskext[i]
            if um.closed:
                um  = None
                ume = None

            im.hold()
            if dq is not None: dq.hold()
            if um is not None: um.hold()
            m = SkyLineMember(im, fi.fext[i], DQFlags=fi.DQFlags,
                              dqimage=dq, dqext=dqext[i],
                              usermask=um, usermask_ext=ume)
            if um is not None: um.release()
            slm.append(m)

        # SkyLine does not need im, dq anymore:
        im.release()
        if dq is not None: dq.release()

        self.members = slm

    def __getattr__(self, what):
        """Control attribute access to `~stsci.sphere.polygon.SphericalPolygon`."""
        if what in ('from_radec', 'from_cone', 'from_wcs',
                    'multi_union', 'multi_intersection',
                    '_find_new_inside',):
            raise AttributeError('\'%s\' object has no attribute \'%s\'' %
                                 (self.__class__.__name__, what))
        else:
            return getattr(self.polygon, what)

    def __copy__(self):
        return deepcopy(self)

    def __repr__(self):
        return '%s(%r, %r)' % (self.__class__.__name__,
                               self.polygon, self.members)

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
        `~stsci.sphere.polygon.SphericalPolygon` portion of `SkyLine`
        that contains the composite skyline from `members`
        belonging to *self*.

        """
        return self._polygon

    @polygon.setter
    def polygon(self, value):
        assert isinstance(value, SphericalPolygon)
        self._polygon = copy(value)  # Deep copy

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
        self.set_members(mlist = mlist, polygon = None)

    def set_members(self, mlist, polygon):
        if mlist is None:
            member_list = []
        elif isinstance(mlist, list):
            member_list = mlist
            if [1 for m in mlist if not isinstance(m,SkyLineMember)]:
                raise ValueError("The \'mlist\' argument must be either "\
                                 "a single \'SkyLineMember\' object or a " \
                                 "Python list of \'SkyLineMember\' objects.")
        elif isinstance(mlist, SkyLineMember):
            member_list = [ mlist ]
        else:
            raise ValueError("The \'mlist\' argument must be either "\
                             "a single \'SkyLineMember\' object or a " \
                             "Python list of \'SkyLineMember\' objects.")

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
            self._id = self._id_from_fname_ext(self.members[0].basefname, \
                                               extlist)
            return

        else: # autodetect status
            fnames  = []
            nfnames = 0
            extlist = []

            for m in self.members:
                if m.fname not in fnames:
                    fnames.append(m.basefname)
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
                self._id = self._id_from_fname_ext(fnames[0], extlist)
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
            if isinstance(ext,tuple):
                extname = ext[0].upper()
                extver  = ext[1]
            elif isinstance(ext,int):
                extname = None
                extver  = ext
            if extname in extdic.keys():
                extdic[extname].append(extver)
            else:
                extdic.update( {extname : [extver]} )

        strext = []
        disctinct_ext = extdic.keys()
        disctinct_ext.sort()
        for extname in disctinct_ext:
            if extname is None:
                strext.append("{:s}".format( \
                    ','.join(map(str,extdic[extname])) ) )
            else:
                strext.append("\'{:s}\',{:s}".format(extname, \
                              ','.join(map(str,extdic[extname])) ) )

        fextid = "{:s}[{:s}]".format(fname, ';'.join(strext))

        return fextid

    def _draw_members(self, map, **kwargs):
        """
        Draw individual extensions in members.
        Useful for debugging.

        Parameters
        ----------
        map : Basemap axes object

        **kwargs : Any plot arguments to pass to basemap

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
        self : obj
            `SkyLine` instance.

        given_members : list
            List of `SkyLineMember` to consider.

        Returns
        -------
        new_members : list
            List of `SkyLineMember` belonging to *self*.

        """
        if len(self.points) > 0:
            out_mem = [m for m in given_members if
                       self.intersects_poly(m.polygon)]
        else:
            out_mem = []
        return out_mem

    def add_image(self, other):
        """
        Return a new `SkyLine` that is the union of *self*
        and *other*.

        .. warning:: `SkyLine.union` only returns `polygon`
            without `members`.

        Parameters
        ----------
        other : `SkyLine` object

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
        other : `SkyLine` object

        Examples
        --------
        >>> s1 = SkyLine('image1.fits')
        >>> s2 = SkyLine('image2.fits')
        >>> s3 = s1.find_intersection(s2)

        """
        newcls = self.__class__(None)
        mlist = newcls._find_members(self.members + other.members)
        poly  = self.intersection(other)
        newcls.set_members(mlist, poly)
        return newcls

    def find_max_overlap(self, skylines):
        """
        Find `SkyLine` from a list of *skylines* that overlaps
        the most with *self*.

        Parameters
        ----------
        skylines : list
            A list of `SkyLine` instances.

        Returns
        -------
        max_skyline : `SkyLine` instance or `None`
            `SkyLine` that overlaps the most or `None` if no
            overlap found. This is *not* a copy.

        max_overlap_area : float
            Area of intersection.

        """
        from mpl_toolkits.basemap import Basemap
        from matplotlib import pyplot as plt


        max_skyline = None
        max_overlap_area = 0.0

        for next_s in skylines:
            try:
                overlap_area = self.intersection(next_s).area()
            except (ValueError, AssertionError):

                #m = Basemap()
                #self.polygon.draw(m)
                #next_s.polygon.draw(m)

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
        skylines : list
            A list of `SkyLine` instances.

        Returns
        -------
        max_pair : tuple
            Pair of `SkyLine` objects with max overlap
            among given *skylines*. If no overlap found,
            return `None`. These are *not* copies.

        """
        max_pair = None
        max_overlap_area = 0.0

        for i in xrange(len(skylines) - 1):
            curr_s = skylines[i]
            next_s, i_area = curr_s.find_max_overlap(skylines[i+1:])

            if i_area > max_overlap_area:
                max_overlap_area = i_area
                max_pair = (curr_s, next_s)

        return max_pair

    @classmethod
    def mosaic(cls, skylines, verbose=True):
        """
        Mosaic all overlapping *skylines*.

        A pair of skylines with the most overlap is used as
        a starting point. Then a skyline that overlaps the
        most with the mosaic is used, and so forth until no
        overlapping skyline is found.

        Parameters
        ----------
        skylines : list
            A list of `SkyLine` objects.

        verbose : bool
            Print info to screen.

        Returns
        -------
        mosaic : `SkyLine` instance or `None`
            Union of all overlapping *skylines*, or `None` if
            no overlap found.

        included : list
            List of image names added to mosaic in the order
            of addition.

        excluded : list
            List of image names excluded because they do not
            overlap with mosaic.

        """
        out_order = []
        excluded  = []

        if verbose:
            print('***** SKYLINE MOSAIC *****')

        starting_pair = cls.max_overlap_pair(skylines)
        if starting_pair is None:
            if verbose:
                print('    Cannot find any overlapping skylines. Aborting...')
            return starting_pair, out_order, excluded

        remaining = list(skylines)

        s1, s2 = starting_pair
        if verbose:
            print('    Starting pair: %s, %s' %
                  (s1.id, s2.id))

        mosaic = s1.add_image(s2)
        out_order = [s1.id, s2.id]
        remaining.remove(s1)
        remaining.remove(s2)

        while len(remaining) > 0:
            next_skyline, i_area = mosaic.find_max_overlap(remaining)

            if next_skyline is None:
                for r in remaining:
                    if verbose:
                        print('    No overlap: Excluding %s...' % r.id)
                    excluded.append(r.id)
                break

            try:
                new_mos = mosaic.add_image(next_skyline)
            except (ValueError, AssertionError):
                if SKYLINE_DEBUG:
                    print('WARNING: Cannot add %s to mosaic. Skipping it...' %
                          next_skyline.id)
                    excluded.append(next_skyline.id)
                else:
                    raise
            else:
                print('    Adding %s to mosaic...' % next_skyline.id)
                mosaic = new_mos
                out_order.append(next_skyline.id)
            finally:
                remaining.remove(next_skyline)

        return mosaic, out_order, excluded


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
        raise ValueError("Input argument \'{:s}\' must be a valid " \
                         "\'ImageRef\' object.".format(imgargname))
    # check extension:
    if image is not None and (ext is None or not _is_valid_ext(ext)):
        raise TypeError("Input argument \'{:s}\' must be either an " \
                         "integer or a tuple of the form (\'str\', int)."\
                         .format(extargname))
