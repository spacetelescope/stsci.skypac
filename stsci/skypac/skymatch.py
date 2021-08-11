"""
A module that provides functions for matching sky in overlapping images.

:Authors: Mihai Cara, Warren Hack, Pey-Lian Lim

:License: :doc:`LICENSE`

"""
# STDLIB
import os
from datetime import datetime
from os.path import basename
import copy

# THIRD PARTY
import numpy as np
from astropy.io import fits
try:
    from stsci.tools.bitmask import interpret_bit_flags
except ImportError:
    from stsci.tools.bitmask import interpret_bits_value as interpret_bit_flags

try:
    from stsci.tools import teal
except ImportError:
    teal = None

# LOCAL
from .skystatistics import SkyStats
from .utils import ext2str, MultiFileLog
from .parseat import FileExtMaskInfo, parse_cs_line, parse_at_file
from .skyline import SkyLineMember, SkyLine
from . import region
from . import __version__

__all__ = ['TEAL_SkyMatch', 'skymatch']
__taskname__ = 'skymatch'


# DEBUG
__local_debug__ = False


def TEAL_SkyMatch(input, skymethod='globalmin+match', match_down=True,
                  skystat='mode', lower=None, upper=None,
                  nclip=5, lsigma=4.0, usigma=4.0, binwidth=0.1,
                  skyuser_kwd='SKYUSER', units_kwd='BUNIT', invsens_kwd=None,
                  readonly=True, subtractsky=False, dq_bits=None,
                  optimize='balanced', clobber=False, clean=True,
                  verbose=True, logfile='skymatch.log'):
    """
    TEAL interface for :py:func:`skymatch`. Most parameters are identical
    to those of the :py:func:`skymatch`. Here we mention only the differences:

    Parameters
    ----------

    logfile: str, optional
        Store execution log in this file. Always openned in append mode.
        If not given (``logfile`` is `None`), print to screen instead.
        NOTE: Unlike :py:func:`skymatch`, ``logfile`` can *only* be either
        a string file name or `None`.

    """
    # Initialize logging:
    from .utils import MultiFileLog
    ml = MultiFileLog(console=verbose)
    if logfile not in ('', None):
        ml.add_logfile(logfile)
    mluncl = ml.unclose_copy()

    try:
        skymatch(input, skymethod=skymethod, match_down=match_down,
                 skystat=skystat, lower=lower, upper=upper,
                 nclip=nclip, lsigma=lsigma, usigma=usigma, binwidth=binwidth,
                 skyuser_kwd=skyuser_kwd, units_kwd=units_kwd,
                 invsens_kwd=invsens_kwd,
                 readonly=readonly, subtractsky=subtractsky,
                 dq_bits=dq_bits, optimize=optimize,
                 clobber=clobber, clean=clean, verbose=verbose, flog=mluncl)
        # sanity check:
        assert ml.count == mluncl.count
    except Exception as e:
        raise e
    finally:
        ml.close()


def skymatch(input, skymethod='globalmin+match', match_down=True,
             skystat='mode', lower=None, upper=None,
             nclip=5, lsigma=4.0, usigma=4.0, binwidth=0.1,
             skyuser_kwd='SKYUSER', units_kwd='BUNIT', invsens_kwd=None,
             readonly=True, subtractsky=False, dq_bits=None,
             optimize='balanced', clobber=False, clean=True,
             verbose=True, flog='skymatch.log',
             _taskname4history='SkyMatch'):
    r"""
    skymatch(input, skymethod='globalmin+match', \
skystat='mode', lower=None, upper=None, nclip=5, lsigma=4.0, usigma=4.0, \
binwidth=0.1, skyuser_kwd='SKYUSER', units_kwd='BUNIT', readonly=True, \
subtractsky=False, dq_bits=None, optimize='balanced', clobber=False, \
clean=True, verbose=True, flog='skymatch_log.txt')

    Standalone task to compute and/or "equalize" sky in input images.

    .. note::

       Sky matching ("equalization") is possible only for **overlapping**
       exposures.

    .. warning::

       When ``readonly`` is `False`, image headers will be modified
       and image data will be background-subtracted if ``subtractsky`` is
       `True`. Remember to back up original copies as desired.

    .. warning::

       Unlike previous sky subtraction algorithm used by
       `astrodrizzle <http://stsdas.stsci.edu/stsci_python_sphinxdocs_2.13/
       drizzlepac/astrodrizzle.html>`_, :py:func:`skymatch` accounts for
       differences in chip sensitivities by performing sky computations on
       data multiplied by inverse sensitivity (e.g., value of ``PHOTFLAM``
       in image headers -- see "Notes" section below).

    Parameters
    ----------
    input: str, list of FileExtMaskInfo
        A list of of :py:class:`~stsci.skypac.parseat.FileExtMaskInfo` objects
        or a string containing one of the following:

            * a comma-separated list of valid science image file names
              (see note below) and (optionally) extension specifications,
              e.g.: ``'j1234567q_flt.fits[1], j1234568q_flt.fits[sci,2]'``;

            * an @-file name, e.g., ``'@files_to_match.txt'``. See notes
              section for details on the format of the @-files.

        .. note::
            **Valid science image file names** are:

            * file names of existing FITS, GEIS, or WAIVER FITS files;

            * partial file names containing wildcard characters, e.g.,
              ``'*_flt.fits'``;

            * Association (ASN) tables (must have ``_asn``, or ``_asc``
              suffix), e.g., ``'j12345670_asn.fits'``.

        .. warning::
            @-file names **MAY NOT** be followed by an extension
            specification.

        .. warning::
            If an association table or a partial file name with wildcard
            characters is followed by an extension specification, it will be
            considered that this extension specification applies to **each**
            file name in the association table or **each** file name
            obtained after wildcard expansion of the partial file name.

    skymethod: {'localmin', 'globalmin+match', 'globalmin', 'match'}, optional

        Select the algorithm for sky computation:

        * **'localmin'**\ : compute a common sky for all members of
          *an exposure* (see "Notes" section below). For a typical use, it
          will compute sky values for each chip/image extension (marked for
          sky subtraction in the :py:obj:`input` parameter) in an input image,
          and it will subtract the previously found minimum sky value
          from all chips (marked for sky subtraction) in that image.
          This process is repeated for each input image.

          .. note::
            This setting is recommended when regions of overlap between images
            are dominated by "pure" sky (as opposite to extended, diffuse
            sources).

          .. note::
            This is similar to the "skysub" algorithm used in previous
            versions of astrodrizzle.

        * **'globalmin'**\ : compute a common sky value for all members of
          *all exposures* (see "Notes" section below). It will compute
          sky values for each chip/image extension (marked for sky
          subtraction in the :py:attr:`input` parameter) in **all** input
          images, find the minimum sky value, and then it will
          subtract the **same** minimum sky value from **all** chips
          (marked for sky subtraction) in **all** images. This method *may*
          useful when input images already have matched background values.

        * **'match'**\ : compute differences in sky values between images
          in common (pair-wise) sky regions. In this case computed sky values
          will be relative (delta) to the sky computed in one of the
          input images whose sky value will be set to (reported to be) 0.
          This setting will "equalize" sky values between the images in
          large mosaics. However, this method is not recommended when used
          in conjunction with `astrodrizzle <http://stsdas.stsci.edu/
          stsci_python_sphinxdocs_2.13/drizzlepac/astrodrizzle.html>`_ because
          it computes relative sky values while ``astrodrizzle`` needs
          "measured" sky values for median image generation and CR rejection.

        * **'globalmin+match'**\ : first find a minimum "global" sky value
          in all input images and then use ``'match'`` method to
          equalize sky values between images.

          .. note::
            This is the *recommended* setting for images
            containing diffuse sources (e.g., galaxies, nebulae)
            covering significant parts of the image.

    match_down: bool, optional
        Specifies whether the sky *differences* should be subtracted from
        images with higher sky values (``match_down`` is `True`) to match the
        image with the lowest sky or sky differences should be added to the
        images with lower sky values to match the sky of the image with the
        highest sky value (``match_down`` is `False`).

        .. note::
          This setting applies *only* when ``skymethod`` parameter is
          either ``'match'`` or ``'globalmin+match'``.

    skystat: {'mode', 'median', 'mode', 'midpt'}, optional
        Statistical method for determining the sky value from the image
        pixel values. See `~stsci.skypac.computeSky` for more detals.

    lower: float, None, optional
        Lower limit of usable pixel values for computing the sky.
        This value should be specified in the units of the input image(s).

    upper: float, None, optional
        Upper limit of usable pixel values for computing the sky.
        This value should be specified in the units of the input image(s).

    nclip: int, optional
        A non-negative number of clipping iterations to use when computing
        the sky value.

    lsigma: float, optional
        Lower clipping limit, in sigma, used when computing the sky value.

    usigma: float, optional
        Upper clipping limit, in sigma, used when computing the sky value.

    binwidth: float, optional
        Bin width, in sigma, used to sample the distribution of pixel
        brightness values in order to compute the sky background statistics.

    skyuser_kwd: str, optional
        Name of header keyword which records the sky value previously
        *subtracted* (if ``subtractsky`` is `True`) from the image data or
        the *computed* (if ``subtractsky`` is `False`) sky value.
        This keyword's value will be updated by :py:func:`skymatch` (if
        ``readonly`` is `False`).

        .. warning::
            When ``subtractsky`` is `True` then ``skyuser_kwd`` is treated as a
            **cummulative** value. That is, subtracted sky value will be
            **added** to the ``skyuser_kwd`` value and thus ``skyuser_kwd``
            represents *total* sky *subtracted* from the image by the user
            over the entire "history" of the image.
            If ``skyuser_kwd`` is missing in the input image,
            "previous" sky value will be considered to be 0.0.

            When ``subtractsky`` is `False` then ``skyuser_kwd`` represents
            **computed** sky value and it is **not** treated as a
            **cummulative** value. Any previous value of the ``skyuser_kwd``
            header keyword will be **overwritten** with the newly computed
            value.

            Because of different meanings of the value represented by the
            ``skyuser_kwd`` header keyword depending on the value of the
            ``subtractsky`` parameter, it is important to be consistent
            and not to mix the two modes when using :py:func:`skymatch`
            multiple times on the same images.

    units_kwd: str, optional
        Name of header keyword which records the units of the data in the
        image.

    invsens_kwd: str, None, optional
        Name of header keyword which records the inverse sensitivity of the
        detector used to acquire data. For ``HST`` detectors, ``'PHOTFLAM'`` is
        proportional to detector's inverse sensitivity. It is used
        to convert electron counts-like to photon counts by multiplying
        count-like data (or count-rates) by the value indicated by this
        keyword.

        By performing matching using photon counts ("flux units"), one can
        match images from heterogeneous instruments. Default value ``''``
        or `None` turns off use of inverse sensitivity.

    readonly: bool, optional
        Report the sky matching values but do not modify the input files.

    subtractsky: bool, optional
        Subtract computed sky value from image data and add this value to the
        existing value represented by ``skyuser_kwd`` (**subtracted sky**)
        or simply report the computed sky value in the header keyword
        specified by ``skyuser_kwd`` (**computed sky**).

        .. warning::
          Because ``subtractsky`` changes the *meaning* of the value of the
          header keyword ``skyuser_kwd`` it is important to be consistent
          in using ``subtractsky`` parameter: inconsistent use may lead
          to sky values reported in ``skyuser_kwd`` header keyword that do not
          reflect correct sky value *computed for* or *subtracted from*
          flat-fielded images. A possible workaround is to use different
          keywords for subtracted and computed sky, keeping in mind that the
          order of operation will affect reported *computed* sky values.

          Also see warning for ``skyuser_kwd`` parameter.

        .. note::
          When ``readonly`` is `True`, reported sky values will be consistent
          with the setting specified by ``subtractsky`` (as if ``readonly`` is
          `False`), however sky values will **NOT** be subtracted from
          the image data when ``subtractsky`` is `True`.

        .. note::

          `astrodrizzle <http://stsdas.stsci.edu/
          stsci_python_sphinxdocs_2.13/drizzlepac/astrodrizzle.html>`_
          does not subtract computed sky values from input
          flat-fielded images. Therefore, when using :py:func:`skymatch` on
          images that subsequently will be processed by
          `astrodrizzle <http://stsdas.stsci.edu/
          stsci_python_sphinxdocs_2.13/drizzlepac/astrodrizzle.html>`_
          it is *recommended* to use the following suggestions:

          * If one plans to turn on sky subtraction step in
            `astrodrizzle <http://stsdas.stsci.edu/
            stsci_python_sphinxdocs_2.13/drizzlepac/astrodrizzle.html>`_ that
            will involve additional sky computation (as opposite to using
            ``astrodrizzle``'s ``skyuser`` or ``skyfile`` parameters), then
            it is recommended to set ``subtractsky`` to `False` and set
            ``skyuser_kwd`` to the default value used by ``astrodrizzle``:
            ``MDRIZSKY``.

          * If one wants to effectively subtract the computed sky values
            from the flat-fielded image data, then it is recommended to
            set ``subtractsky`` to `True`, ``skyuser_kwd`` parameter to
            something different from ``MDRIZSKY``, (e.g., ``SKYUSER``),
            and set ``skyuser`` parameter in
            `astrodrizzle <http://stsdas.stsci.edu/
            stsci_python_sphinxdocs_2.13/drizzlepac/astrodrizzle.html>`_
            to the same value as the values of ``skyuser_kwd`` used
            in the call to :py:func:`skymatch`.

    dq_bits: int, str, None, optional
        Integer sum of all the DQ bit values from the input image's
        DQ array that should be considered "good" when building masks for
        sky computations. For example, if pixels in the DQ array can be
        combinations of 1, 2, 4, and 8 flags and one wants to consider DQ
        "defects" having flags 2 and 4 as being acceptable for sky
        computations, then ``dq_bits`` should be set to 2+4=6. Then a DQ pixel
        having values 2,4, or 6 will be considered a good pixel, while a
        DQ pixel with a value, e.g., 1+2=3, 4+8=12, etc. will be flagged as
        a "bad" pixel.

        Alternatively, one can enter a comma- or '+'-separated list
        of integer bit flags that should be added to obtain the
        final "good" bits. For example, both ``4,8`` and ``4+8``
        are equivalent to setting ``dq_bits`` to 12.

        | Default value (0) will make *all* non-zero
          pixels in the DQ mask to be considered "bad" pixels, and the
          corresponding image pixels will not be used for sky computations.

        | Set ``dq_bits`` to `None` to turn off the use of image's DQ array
          for sky computations.

        | In order to reverse the meaning of the ``dq_bits``
          parameter from indicating values of the "good" DQ flags
          to indicating the "bad" DQ flags, prepend '~' to the string
          value. For example, in order not to use pixels with
          DQ flags 4 and 8 for sky computations and to consider
          as "good" all other pixels (regardless of their DQ flag),
          set ``dq_bits`` to ``~4+8``, or ``~4,8``. To obtain the
          same effect with an `int` input value (except for 0),
          enter -(4+8+1)=-13. Following this convention,
          a ``dq_bits`` string value of ``'~0'`` would be equivalent to
          setting ``dq_bits=None``.

        .. note::
            DQ masks (if used), *will* *be* combined with user masks
            specified in the input @-file.

    optimize: {'balanced', 'speed', 'inmemory'}, optional
        Specifies whether to optimize execution for speed (maximum memory
        usage - loaded masks and images are not unloaded until the end
        of the execution) or use a balanced approach in which a minimal
        amount of image data is kept in memory and retrieved from
        disk as needed. The 'inmemory' option is similar to the 'speed'
        setting except that masks are never saved to the files on a
        physical disk but are created in memory. The default setting
        is recommended for most systems.

    clobber: bool, optional
        When a input image file is in GEIS or WAIVER FITS format it must be
        converted to simple/MEF FITS file format before it can be used by
        :py:func:`skymatch`. This setting specifies whether any existing
        simple/MEF files be overwritten during this conversion process. If
        ``clobber`` is `False`, existing simple/MEF FITS files will be opened.
        If ``clobber`` is `True`, input GEIS or WAIVER FITS will be first
        converted to simple FITS/MEF format overwritting (if necessary)
        existing files and then these newly created simple FITS/MEF files
        will be opened.

    clean: bool, optional
        Specifies whether to delete at the end of the execution any temporary
        files created by :py:func:`skymatch`.

    verbose: bool, optional
        Specifies whether to print warning messages.

    flog: str, file object, MultiFileLog, None, optional
        Log file to which messages shoul be written. It can be a file name,
        file object, or a MultiFileLog object. The later two allow the
        log to be written to an existing open output stream passed
        from the calling function such as
        `astrodrizzle <http://stsdas.stsci.edu/stsci_python_sphinxdocs_2.13/
        drizzlepac/astrodrizzle.html>`_. Log file is always openned in append
        mode. If not provided (`None`), print messages to screen only.

    Raises
    ------
    RuntimeError
        Could not add an image to mosaic. Possibly this SkyLine does
        not intersect the mosaic.

    TypeError
        The `input` argument must be either a Python list of
        :py:class:`~stsci.skypac.parseat.FileExtMaskInfo` objects, or a string
        either containing either a comma-separated list file names,
        or an @-file name.

    Notes
    -----

    :py:func:`skymatch` provides new algorithms for sky value computations
    and enhances previously available algorithms used by, e.g.,
    `astrodrizzle <http://stsdas.stsci.edu/
    stsci_python_sphinxdocs_2.13/drizzlepac/astrodrizzle.html>`_.

    First, the standard sky computation algorithm
    (see ``skymethod = 'localmin'``) was upgraded to be able to use
    DQ flags and user supplied masks to remove "bad" pixels from being
    used for sky statistics computations. Values different from zero in
    user-supplied masks indicate "good" data pixels.

    Second, two new methods have been introduced: ``'globalmin'`` and
    ``'match'``, as well as a combination of the two -- ``'globalmin+match'``.

    - The ``'globalmin'`` method computes the minimum sky value across *all*
      chips in *all* input images. That sky value is then considered to be
      the background in all input images.

    - The ``'match'`` algorithm is somewhat
      similar to the traditional sky subtraction method
      (``skymethod = 'localmin'``) in the sense that it measures the sky
      indipendently in input images (or detector chips). The major differences
      are that, unlike the traditional method,

        #. ``'match'`` algorithm computes *relative* sky values with regard
           to the sky in a reference image chosen from the input list
           of images; *and*

        #. Sky statistics is computed only in the part of the image
           that intersects other images.

      This makes ``'match'`` sky computation algorithm particularly useful
      for "equalizing" sky values in large mosaics in which one may have
      only (at least) pair-wise intersection of images without having
      a common intersection region (on the sky) in all images.

      The ``'match'`` method works in the following way: for each pair
      of intersecting images, an equation is written that
      requires that average surface brightness in the overlapping part of
      the sky be equal in both images. The final system of equations is then
      solved for unknown background levels.

      .. warning::

        Current algorithm is not capable of detecting cases when some groups of
        intersecting images (from the input list of images) do not intersect
        at all other groups of intersecting images (except for the simple
        case when *single* images do not intersect any other images). In these
        cases the algorithm will find equalizing sky values for each group.
        However since these groups of images do not intersect each other,
        sky will be matched only within each group and the "inter-group"
        sky mismatch could be significant.

        Users are responsible for detecting such cases and adjusting processing
        accordingly.

      .. warning::

        Because this method computes *relative sky values* compared to a
        reference image (which will have its sky value set to 0), the sky
        values computed with this method usually are smaller than the
        "absolute" sky values computed, e.g., with the ``'localmin'``
        algorithm. Since `astrodrizzle <http://stsdas.stsci.edu/
        stsci_python_sphinxdocs_2.13/drizzlepac/astrodrizzle.html>`_ expects
        "true" (as opposite to *relative*) sky values in order to
        correctly compute the median image or to perform cosmic-ray
        detection, this algorithm in not recommended to be used *alone*
        for sky computations to be used with ``astrodrizzle``.

        For the same reason, IVM weighting in ``astrodrizzle`` should **not**
        be used with ``'match'`` method: sky values reported in ``MDRIZSKY``
        header keyword will be relative sky values (sky offsets) and derived
        weights will be incorrect.

    - The ``'globalmin+match'`` algorithm combines ``'match'`` and
      ``'globalmin'`` methods in order to overcome the limitation of the
      ``'match'`` method described in the note above: it uses ``'globalmin'``
      algorithm to find a baseline sky value common to all input images
      and the ``'match'`` algorithm to "equalize" sky values in the mosaic.
      Thus, the sky value of the "reference" image will be equal to the
      baseline sky value (instead of 0 in ``'match'`` algorithm alone)
      making this method acceptable for use in conjunction with
      ``astrodrizzle``.

    **"Surface Brightness":**
      :py:func:`skymatch` converts "raw" sky values (in image data units)
      obtained directly from image data to "surface brightness"-like units and
      all computations are performed in these units. Computed sky surface
      brightness values are converted back to image data units before being
      subtracted from the image data and/or reported in the ``skyuser_kwd``
      in the image header.

      This conversion from image data units to "surface brightness"-like units
      is necessary in order to perform correct sky computations for data
      from various intsruments/detectors. It accounts for differences in
      exposure times (if image data are in "counts" units) in each input
      image, differences in pixel scales of different detector chips
      (instruments), and detector sensitivities.

      For images with data in "counts"-like units, the conversion from data
      units to surface brightness is given by:

      .. math::
          sky_{\mathrm{surface\: brightness}} = sky_{\mathrm{data\: units}} \
          \cdot \mathrm{PHOTFLAM} / (\mathrm{pixel\: Area}^2 \cdot \
          \mathrm{EXPTIME})

      and for image data in "count-rate"-like units, this conversion is given
      by:

      .. math::
          sky_{\mathrm{surface\: brightness}} = sky_{\mathrm{data\: units}} \
          \cdot \mathrm{PHOTFLAM} / \mathrm{pixel\: Area}^2 .

    **Important Header Keywords:**
      As discussed above, :py:func:`skymatch` uses values of various keywords
      in image headers to perform conversion of sky values to/from data units
      from/to surface brightness units. The most important keywords are:

      * ``BUNIT`` describes the units of the image data. The units of data are
        determined from the ``BUNIT`` header keyword by searching its value
        for the division sign ``'/'``. If the division sign is not found, then
        the units are assumed to be "counts". If the division sign is found
        in the ``BUNIT`` value and if the numerator is one of the following:
        ``'ELECTRONS'``, ``'COUNTS'``, or ``'DN'``, and denumerator is
        either ``'S'``, ``'SEC'``, or ``'SECOND'``, then
        the units are assumed to be count-rate.

        If ``BUNIT`` is missing then for non-``HST`` images the units will
        be assumed to be "count-rate", while for ``HST`` images
        (header keyword ``TELESCOP = 'HST'``) the ``'INSTRUME'``
        and ``'DETECTOR'`` keywords will be used to infer the units. For the
        ``NICMOS`` instrument, ``'UNITCORR'`` will be used to infer the units.
        If relevant keywords are missing, the units of image data will
        be assumed to be "count-rate". Check the log file for selected units.

      * ``EXPTIME`` -- total exposure time, assumed to be in seconds. While
        the units of ``EXPTIME`` are not important for sky computation, it is
        important that all input images to :py:func:`skymatch` use *the same*
        units. This keyword is used only when inferred units for image
        data are "count-rates". If ``EXPTIME`` is missing when image data
        units are counts, then variations in exposure time **WILL NOT** be
        accounted for. First, the primary header of the image file is
        searched for ``EXPTIME`` and if it is not found in the primary header,
        then image extension is searched for the presense of ``EXPTIME``
        keyword.

      * ``PHOTFLAM`` -- inverse sensitivity of the detector. At first
        :py:func:`skymatch` will try to detect ``PHOTFLAM`` in the image
        extension header and if not found, it will look for ``PHOTFLAM``
        in the primary header. If ``PHOTFLAM`` is not
        present at all, the variations in detector sensitivity **WILL NOT**
        be accounted for.

    **Glossary:**
      **Exposure** -- a *subset* of FITS image extensions in an input image
      that correspond to different chips in the detector used to acquire
      the image. The subset of image extensions that form an exposure
      is defined by specifying extensions to be used with input images
      (see parameter ``input``).

      See help for :py:func:`stsci.skypac.parseat.parse_at_line` for details
      on how to specify image extensions.

      **Footprint** -- the outline (edge) of the projection of a chip or
      of an exposure on the celestial sphere.

      .. note::

        * Footprints are managed by the
          :py:class:`~spherical_geometry.polygon.SphericalPolygon` class.

        * Both footprints *and* associated exposures (image data, WCS
          information, and other header information) are managed by the
          :py:class:`~stsci.skypac.skyline.SkyLine` class.

        * Each :py:class:`~stsci.skypac.skyline.SkyLine` object contains one
          or more :py:class:`~stsci.skypac.skyline.SkyLineMember` objects
          that manage both footprints *and* associated *chip* data that
          form an exposure.

    **Remarks:**
      * :py:func:`skymatch` works directly on *geometrically distorted*
        flat-fielded images thus avoiding the need to perform an additional
        drizzle step to perform distortion correction of input images.

        Initially, the footprint of a chip in an image is aproximated by a
        2D planar rectangle representing the borders of chip's distorted
        image. After applying distortion model to this rectangle and
        progecting it onto the celestial sphere, it is approximated by
        spherical polygons. Footprints of exposures and mosaics are
        computed as unions of such spherical polygons while overlaps
        of image pairs are found by intersecting these spherical polygons.

    **@-File Format:**
      A catalog file containing a science image file
      and extension specifications and optionally followed by a
      comma-separated list of mask files and extension specifications
      (or None).

      File names will be stripped of leading and trailing white spaces. If it
      is essential to keep these spaces, file names may be enclosed in single
      or double quotation marks. Quotation marks may also be required when
      file names contain special characters used to separate file names and
      extension specifications: ,[]{}

      Extension specifications must follow the file name and must be delimited
      by either square or curly brackets. Curly brackets allow specifying
      multiple comma-separated extensions: integer extension numbers and/or
      tuples ('ext name', ext version).

      Some possible ways of specifying extensions:
        [1] -- extension number

        ['sci',2] -- extension name and version

        {1,4,('sci',3)} -- multiple extension specifications, including tuples

        {('sci',*)} -- wildcard extension versions (i.e., all extensions with
        extension name 'sci')

        ['sci'] -- equivalent to ['sci',1]

        {'sci'} -- equivalent to {('sci',*)}

      For extensions in the science image for which no mask file is provided,
      the corresponding mask file names may be omitted (but a comma must still
      be used to show that no mask is provided in that position) or None can
      be used in place of the file name. NOTE: 'None' (in quotation marks)
      will be interpreted as a file named None.

      Some examples of possible user input:
        image1.fits{1,2,('sci',3)},mask1.fits,,mask3.fits[0]

        In this case:

        ``image1.fits``\ [1] is associated with ``mask1.fits``\ [0];

        ``image1.fits``\ [2] does not have an associated mask;

        ``image1.fits``\ ['sci',3] is associated with ``mask3.fits``\ [0].

        -- Assume ``image2.fits`` has 4 'SCI' extensions:

        image2.fits{'sci'},None,,mask3.fits

        In this case:

        ``image2.fits``\ ['sci',1] and ``image2.fits``\ ['sci',2] **and**
        ``image2.fits``\ ['sci',4] do not have an associated mask;

        ``image2.fits``\ ['sci',3] is associated with ``mask3.fits``\ [0]

      .. note::
        User mask data that indicate what pixels in the input `image` should
        be used for sky computations (``1``) and which pixels should **not**
        be used for sky computations (``0``).


    **Limitations and Discussions:**
      Primary reason for introducing "sky match" algorithm was to try to
      equalize the sky in large mosaics in which computation of the
      "absolute" sky is difficult due to the presence of large diffuse
      sources in the image. As discussed above, :py:func:`skymatch`
      accomplishes this by comparing "sky values" in a pair of images in the
      overlap region (that is common to both images). Quite obviously the
      quality of sky "matching" will depend on how well these "sky values"
      can be estimated. We use quotation marks around *sky values* because
      for some image "true" background may not be present at all and the
      measured sky may be the surface brightness of large galaxy, nebula, etc.

      Here is a brief list of possible limitations/factors that can affect
      the outcome of the matching (sky subtraction in general) algorithm:

      * Since sky subtraction is performed on *flat-fielded* but
        *not distortion corrected* images, it is important to keep in mind
        that flat-fielding is performed to obtain uniform surface brightness
        and not flux. This distinction is important for images that have
        not been distortion corrected. As a consequence, it is advisable that
        point-like sources be masked through the user-supplied mask files.
        Values different from zero in user-supplied masks indicate "good" data
        pixels. Alternatively, one can use ``upper`` parameter to limit the use
        of bright objects in sky computations.

      * Normally, distorted flat-fielded images contain cosmic rays. This
        algorithm does not perform CR cleaning. A possible way of minimizing
        the effect of the cosmic rays on sky computations is to use
        clipping (``nclip > 0``) and/or set ``upper`` parameter to a value
        larger than most of the sky background (or extended source) but
        lower than the values of most CR pixels.

      * In general, clipping is a good way of eliminating "bad" pixels:
        pixels affected by CR, hot/dead pixels, etc. However, for
        images with complicated backgrounds (extended galaxies, nebulae,
        etc.), affected by CR and noise, clipping process may mask different
        pixels in different images. If variations in the background are
        too strong, clipping may converge to different sky values in
        different images even when factoring in the "true" difference
        in the sky background between the two images.

      * In general images can have different "true" background values
        (we could measure it if images were not affected by large diffuse
        sources). However, arguments such as ``lower`` and ``upper`` will
        apply to all images regardless of the intrinsic differences
        in sky levels.

    Examples
    --------
    #. This task can be used to match skies of a set of ACS
       images simply with:

           >>> from stsci.skypac import skymatch
           >>> skymatch.skymatch('j*q_flt.fits')

    #. The TEAL GUI can be used to run this task using::

           >>> from stsci.skypac import skymatch
           >>> epar skymatch

       or from a general Python command line:

           >>> from stsci.skypac import skymatch
           >>> from stsci.tools import teal
           >>> teal.teal('skymatch')

    """
    # Time it
    runtime_begin = datetime.now()

    # Set-up log files:
    if isinstance(flog, MultiFileLog):
        ml = flog
    else:
        ml = MultiFileLog(console=verbose)
        if flog not in ('', None):
            ml.add_logfile(flog)
            ml.skip(2)
    mlcopy = ml.unclose_copy()

    #  BEGIN:
    ml.logentry("***** {0} started on {1}", __taskname__, runtime_begin)
    ml.logentry("      Version {0}", __version__, skip=1)

    if readonly:
        ml.logentry("'skymatch' task will be run in read-only mode.", skip=1)
        ml.logentry("NOTE: Computed sky values WILL NOT be subtracted from "
                    "image data ('readonly'=True).\n"
                    "Computed sky values will be reported in the specified "
                    "log file.", skip=1)
    else:
        ml.logentry("'skymatch' task will apply computed sky differences "
                    "to input image file(s).", skip=1)
        if subtractsky:
            ml.logentry("NOTE: Computed sky values WILL be subtracted from "
                        "image data ('subtractsky'=True).\n"
                        "'{:s}' header keyword will represent sky value "
                        "*subtracted* from data.", skyuser_kwd, skip=1)
        else:
            ml.logentry("NOTE: Computed sky values WILL NOT be subtracted "
                        "from image data ('subtractsky'=False).\n"
                        "'{:s}' header keyword will represent sky value "
                        "*computed* from data.", skyuser_kwd, skip=1)

    # Process inverse sensitivity parameter:
    if invsens_kwd is not None:
        invsens_kwd = invsens_kwd.strip()
        if invsens_kwd == '':
            invsens_kwd = None

    # Initialize SkyLineMember *class* with common to all objects settings:
    ml.logentry("-----  User specified keywords:  -----\n"
                "       Sky Value Keyword:  \'{:s}\'\n"
                "       Data Units Keyword: \'{:s}\'",
                skyuser_kwd.upper(), units_kwd.upper(), skip=1)

    SkyLineMember.init_class(skyuser_kwd=skyuser_kwd, units_kwd=units_kwd,
                             invsens_kwd=invsens_kwd, optimize=optimize,
                             verbose=verbose, logfile=mlcopy)

    # Parse input to get list of filenames to process
    errmsg = "The \'input\' argument must be either a Python list of " \
        "\'FileExtMaskInfo\' objects, or a string either containing either " \
        "a comma-separated list file names, or an @-file name."

    dq_bits = interpret_bit_flags(dq_bits)

    if isinstance(input, list):
        if [1 for i in input if not isinstance(i, FileExtMaskInfo)]:
            raise TypeError(errmsg)

        finfo = []
        for fi in input:
            cpfi = copy.copy(fi)
            if cpfi.fnamesOnly:
                cpfi.convert2ImageRef()
            cpfi.finalize()
            finfo.append(cpfi)

        if dq_bits is not None:
            for fi in finfo:
                fi.dq_bits = dq_bits

    elif isinstance(input, str):
        ml.skip()
        ml.logentry("-----  Parsing input image file lists:  -----")
        if input.lstrip()[0] == '@':
            finfo = parse_at_file(
                input.lstrip()[1:], default_ext=('SCI', '*'),
                clobber=clobber,
                fnamesOnly=False,
                doNotOpenDQ=dq_bits is None,
                im_fmode='readonly' if readonly else 'update',
                dq_fmode='readonly',
                msk_fmode='readonly',
                logfile=mlcopy,
                verbose=verbose
            )

        else:
            finfo = parse_cs_line(
                input,
                default_ext=('SCI', '*'),
                clobber=clobber,
                fnamesOnly=False,
                doNotOpenDQ=dq_bits is None,
                im_fmode='readonly' if readonly else 'update',
                dq_fmode='readonly',
                msk_fmode='readonly',
                logfile=mlcopy,
                verbose=verbose
            )

        for fi in finfo:
            fi.dq_bits = dq_bits

    else:
        raise ValueError(errmsg)

    # infiles, output = parseinput.parseinput(input)

    # print input file information:
    ml.skip()
    ml.logentry("-----  Input file list:  -----")

    for fi in finfo:
        ml.skip()
        # file name
        ml.logentry(
            "   **  Input image: \'{:s}\'",
            basename(fi.image.filename),
            skip=-1
        )
        if fi.image.filename != fi.image.original_fname:
            ml.logentry("  (original: \'{:s}\')",
                        basename(fi.image.original_fname), skip=-1)
        ml.skip()
        # DQ file name
        if fi.image.DQ_model.lower() == 'external' and not fi.DQimage.closed:
            ml.logentry("       DQ image: \'{:s}\'", basename(
                fi.DQimage.filename), skip=-1)
            if fi.DQimage.filename != fi.DQimage.original_fname:
                ml.logentry("  (original: \'{:s}\')",
                            basename(fi.DQimage.original_fname), skip=-1)
            ml.skip()
        # Extension information:
        for i in range(fi.count):
            ml.logentry("       EXT: {}", ext2str(fi.fext[i]), skip=-1)
            if not fi.DQimage.closed:
                ml.logentry(";\tDQ EXT: {}", ext2str(fi.dqext[i]), skip=-1)
            if fi.mask_images[i] is not None and not fi.mask_images[i].closed:
                ml.logentry(
                    ";\tMASK: {}[{}]",
                    fi.mask_images[i].original_fname,
                    ext2str(fi.maskext[i]),
                    skip=-1
                )
            ml.skip()

    ml.skip()

    if len(finfo) < 2 and 'match' in skymethod:
        ml.logentry('{0}: Need at least 2 images for sky matching. '
                    'Aborting...', __taskname__)
        ml.print_endlog_msg()
        ml.close()
        return

    # Sky statistics parameters:
    sky_stat = SkyStats(skystat=skystat, lower=lower, upper=upper, nclip=nclip,
                        lsig=lsigma, usig=usigma, binwidth=binwidth)

    ml.logentry("-----  Sky statistics parameters:  -----\n"
                "       statistics function: '{0}'\n"
                "       lower = {2}\n"
                "       upper = {3}\n"
                "       nclip = {4}\n"
                "       lsigma = {5}\n"
                "       usigma = {6}\n"
                "       binwidth = {7}",
                skystat, skystat, lower, upper, nclip, lsigma, usigma,
                binwidth, skip=1)

    # Initialize skylines
    skylines = []
    ml.logentry("-----  Data->Brightness conversion parameters "
                "for input files:  -----", skip=1)

    for fi in finfo:
        ml.logentry("   *   Image: {}", basename(fi.image.filename))

        sl = SkyLine(fi)

        # EXPTIME:
        if sl.members[0].exptime is None:
            exptime = "UNKNOWN"
        else:
            exptime = str(sl.members[0].exptime)

        for m in sl.members:
            # reset skyuser if necessary:
            if not subtractsky:
                m.set_skyuser(0.0)  # => forget existing header value if any...

            # Units *TYPE* (counts or rate?):
            if m.is_countrate is None:
                unittype = "UNKNOWN"
                invs_units = "flux units/data units"
            else:
                unittype = "COUNT-RATE" if m.is_countrate else "COUNTS"
                invs_units = "flux units/(data units/time)"

            # inverse sensitivity:
            if invsens_kwd is None:
                invs_str = ''
            else:
                if m.inv_sensitivity is None:
                    invs_str = "{:s}Inverse Sensitivity: UNAVAILABLE\n" \
                        .format(13 * ' ')
                else:
                    invskwd = m.get_inv_sensitivity_kwd()
                    # rate or counts? derive units for inverse sensitivity:
                    invs_str = "{:s}Inverse Sensitivity [{:s}]: {} [{:s}]\n" \
                        .format(13 * ' ', invskwd, m.inv_sensitivity,
                                invs_units)

            # exptime:
            if m.is_countrate:
                expt_str = ''
            else:
                expt_str = "{:s}EXPTIME: {} [s]\n".format(13 * ' ', exptime)

            # add a log entry with units and conversion factors:
            ml.logentry("{0:s}EXT = {2}\n"
                        "{1:s}Data units type: {3}\n"
                        "{4:s}{5:s}"
                        "{1:s}Conversion factor (data->brightness):  {6}",
                        7 * ' ', 13 * ' ', ext2str(m.ext), unittype,
                        invs_str, expt_str, m.data2brightness_conv)

        skylines.append(sl)
        ml.skip()

    # store old sky value and then reset sky values in SkyLineMember:
    orig_skyuser = {}
    for s in skylines:
        for m in s.members:
            m.reset_skyuser_from_header()
            orig_skyuser[m] = m.skyuser

    # -----------------------------------------------------------#
    #  1. Compute the minimum sky background value in each       #
    #     sky line member of a skyline and return.               #
    #     This is an improved (use of masks) replacement         #
    #     for the classical 'subtractsky' used by astrodrizzle.  #
    #                                                            #
    #     NOTE: incompatible with "match"-containing             #
    #           'skymethod' modes.                               #
    # -----------------------------------------------------------#
    if skymethod == 'localmin':
        ml.skip()
        ml.logentry("-----  Computing sky values requested image "
                    "extensions (detector chips):  -----", skip=1)
        for sl in skylines:
            sky = _minsky(sl, sky_stat, ml)

            ml.logentry(
                "   *   Image:   '{}'  --  SKY = {} (brightness units)\n"
                "       Sky change (data units):",
                sl.id, sky
            )

            if sky is None:
                sky = 0.0
            _update_sky_delta(sl, sky)

            for m in sl.members:
                ml.logentry(
                    "      - EXT = {0:s}   delta({1:s}) = {2:G}"
                    "   NEW {1:s} = {3:G}",
                    ext2str(m.ext), m.get_skyuser_kwd(), m.skyuser_delta,
                    m.skyuser_total if subtractsky else m.skyuser_delta
                )

            _apply_skyuser(sl, readonly, subtractsky, _taskname4history)
            sl.close(clean=clean)

        # Time it
        runtime_end = datetime.now()
        ml.logentry("***** {} ended on {}", __taskname__, runtime_end)
        ml.logentry("TOTAL RUN TIME: {}", runtime_end - runtime_begin)
        if ml.count == 0 and not verbose:
            print("TOTAL RUN TIME: {}".format(runtime_end - runtime_begin))

        # Finalize/close log files
        ml.print_endlog_msg()
        ml.close()

        return

    # ---------------------------------------------------------#
    #  2. Compute the minimum sky background value             #
    #     *across* *all* sky line members.                     #
    # ---------------------------------------------------------#
    minsky = None  # in flux/area units

    if skymethod == 'globalmin':
        ml.skip()
        ml.logentry("-----  Computing \"global\" sky on requested image "
                    "extensions (detector chips):  -----", skip=1)
        for sl in skylines:
            sky = _minsky(sl, sky_stat, ml)
            if minsky is None or sky < minsky:
                minsky = sky

        if minsky is None:
            minsky = 0.0

        ml.logentry("    \"Global\" sky value: {} (brightness units)",
                    minsky, skip=1)

        # update skyuser and return (no sky matching requested):
        if minsky is None:
            minsky = 0.0
        for sl in skylines:
            ml.logentry("    Computed sky change (data units) "
                        "for image {:s}:", sl.id)

            _update_sky_delta(sl, minsky)

            for m in sl.members:
                ml.logentry(
                    "      - EXT = {0:s}   delta({1:s}) = {2:G}"
                    "   NEW {1:s} = {3:G}",
                    ext2str(m.ext), m.get_skyuser_kwd(), m.skyuser_delta,
                    m.skyuser_total if subtractsky else m.skyuser_delta
                )

            _apply_skyuser(sl, readonly, subtractsky, _taskname4history)
            sl.close(clean=clean)

        # Time it
        runtime_end = datetime.now()
        ml.logentry("***** {} ended on {}", __taskname__, runtime_end)
        ml.logentry("TOTAL RUN TIME: {}", runtime_end - runtime_begin)
        if ml.count == 0 and not verbose:
            print("TOTAL RUN TIME: {}".format(runtime_end - runtime_begin))

        # Finalize/close log files
        ml.print_endlog_msg()
        ml.close()

        return

    # ---------------------------------------------------------#
    #  3. Find sky "deltas" that will match sky across all     #
    #     (intersecting) images.                               #
    # ---------------------------------------------------------#
    sky_deltas = _find_optimum_sky_deltas(skylines, sky_stat, ml)
    sky_good = np.isfinite(sky_deltas)

    # match sky "Up" or "Down":
    if match_down:
        refsky = np.amin(sky_deltas[sky_good], initial=0.0)
    else:
        refsky = np.amax(sky_deltas[sky_good], initial=0.0)
    sky_deltas[sky_good] -= refsky

    # apply sky differences and report computed sky values:
    ml.skip()
    ml.logentry("-----  Computing differences in sky values in "
                "overlapping regions:  -----")

    for sl, sd, gs in zip(skylines, list(sky_deltas), list(sky_good)):
        if gs:
            # update sky information in SkyLine-s:
            _update_sky_delta(sl, sd)

            # report computed delta values:
            ml.skip()
            ml.logentry(
                "   *   Image \'{0:s}\' SKY = {1:G} [brightness units]\n"
                "       Updating sky of image extension(s) "
                "[data units]:", sl.id, sd
            )
            for m in sl.members:
                if subtractsky:
                    ml.logentry("       - EXT = {0:s}"
                                "   delta({1:s}) = {2:G}"
                                "   NEW {1:s} = {3:G}",
                                ext2str(m.ext), m.get_skyuser_kwd(),
                                m.skyuser_delta, m.skyuser_total)
                else:
                    ml.logentry("       -  EXT = {0:s}"
                                "   delta({1:s}) = {2:G}",
                                ext2str(m.ext), m.get_skyuser_kwd(),
                                m.skyuser_delta)

            # apply computed sky values to image header (and data):
            if skymethod == 'match':
                _apply_skyuser(sl, readonly, subtractsky, _taskname4history)

        else:
            # report that sky delta could not be computed:
            ml.skip()
            ml.logentry(
                "   *   Image \'{0:s}\' SKY = None (undetermined)\n"
                "       Sky of image extension(s) [data units]:",
                sl.id
            )
            for m in sl.members:
                ml.logentry(
                    "       -  EXT = {0:s}   delta({1:s}) = None"
                    "   NEW {1:s} = OLD {1:s} = {2:G}",
                    ext2str(m.ext), m.get_skyuser_kwd(), m.skyuser_total
                )

    # --------------------------------------------------------#
    #  4. Compute the minimum sky background value            #
    #     *across* *all* sky line members.                    #
    # --------------------------------------------------------#
    if skymethod == 'globalmin+match':
        minsky = None
        ml.skip()
        ml.logentry("-----  Computing \"global\" sky on requested image "
                    "extensions (detector chips):  -----", skip=1)
        for sl in skylines:
            sky = _minsky(sl, sky_stat, ml)
            if minsky is None or sky < minsky:
                minsky = sky

        if minsky is None:
            minsky = 0.0

        ml.logentry("   \"Global\" sky value: {} (brightness units)",
                    minsky, skip=1)

        ml.logentry("    Final (match+global) sky [image units] for:")
        for s in skylines:
            # update image's header and data (if requested):
            _update_sky_delta(s, minsky)
            for m in s.members:
                ml.logentry("    # {}:   {}", m.id, m.skyuser_total
                            if subtractsky else m.skyuser_delta)
            _apply_skyuser(s, readonly, subtractsky, _taskname4history)

    ml.skip()

    for sl in skylines:
        sl.close(clean=clean)

    # Time it
    runtime_end = datetime.now()
    ml.logentry("***** {} ended on {}", __taskname__, runtime_end)
    ml.logentry("TOTAL RUN TIME: {}", runtime_end - runtime_begin)
    if ml.count == 0 and not verbose:
        print("TOTAL RUN TIME: {}".format(runtime_end - runtime_begin))

    # Finalize/close log files
    ml.print_endlog_msg()
    ml.close()


def _debug_write_region(fname, vert, multireg=False):
    fh = open(fname, 'w')
    fh.write('# Region file format: DS9 version 4.1\n')
    fh.write('global color=green dashlist=8 3 width=1 font="helvetica 10 '
             'normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 '
             'delete=1 include=1 source=1\n')
    fh.write('image\n')
    if multireg:
        for v in vert:
            fh.write('polygon({})\n'
                     .format(','.join(map(str, [x for e in v for x in e]))))
    else:
        fh.write('polygon({})\n'
                 .format(','.join(map(str, [x for e in vert for x in e]))))
    fh.close()


def _debug_write_region_fk5(fname, ivert, im1vert, im2vert, multireg=False):
    fh = open(fname, 'w')
    fh.write('# Region file format: DS9 version 4.1\n')
    fh.write('global color=green dashlist=8 3 width=2 font="helvetica 10 '
             'normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 '
             'delete=1 include=1 source=1\n')
    fh.write('fk5\n')

    if ivert:
        if multireg:
            for iv in ivert:
                fh.write(
                    'polygon({}) # color=red width=2\n'
                    .format(','.join(map(str, [x for e in iv for x in e])))
                )
        else:
            fh.write(
                'polygon({}) # color=red width=2\n'
                .format(','.join(map(str, [x for e in ivert for x in e])))
            )

    if im1vert:
        if multireg:
            for iv in im1vert:
                fh.write(
                    'polygon({}) # color=blue width=2 dash=1\n'
                    .format(','.join(map(str, [x for e in iv for x in e])))
                )
        else:
            fh.write(
                'polygon({}) # color=blue width=2 dash=1\n'
                .format(','.join(map(str, [x for e in im1vert for x in e])))
            )

    if im2vert:
        if multireg:
            for iv in im2vert:
                fh.write(
                    'polygon({}) # color=green width=2 dash=1\n'
                    .format(','.join(map(str, [x for e in iv for x in e])))
                )
        else:
            fh.write(
                'polygon({}) # color=green width=2 dash=1\n'
                .format(','.join(map(str, [x for e in im2vert for x in e])))
            )

    fh.close()


def _calc_sky(s1, s2, skystat):
    """
    Calculate the weighted average of sky from individual
    chips in the given skyline within a given RA and DEC
    of a polygon.

    Parameters
    ----------
    skyline: `SkyLine` object

    ra, dec: array_like
        RA and DEC of the polygon.

    skystat: A SkyStats object
        Class for sky calculations.

    Raises
    ------
    ValueError: Total weight is zero.

    Returns
    -------
    sky: float

    """
    t_wsky1 = 0.0
    t_wght1 = 0
    t_wsky2 = 0.0
    t_wght2 = 0
    area = 0.0

    for m1 in s1.members:
        for m2 in s2.members:
            pts1 = np.sort(list(m1.polygon.points)[0], axis=0)
            pts2 = np.sort(list(m2.polygon.points)[0], axis=0)
            if np.allclose(pts1, pts2, rtol=0, atol=5e-9):
                intersect_poly = m1.polygon.copy()
            else:
                intersect_poly = m1.polygon.intersection(m2.polygon)

            # Check if we even have regions that potentially may have an area.
            # This is done mostly to make sure tha code that follows does
            # not crash when no "good" polygons are available.
            # At the same time collect RA&DEC of valid polygons:
            radec = []
            valid_poly = False
            for p in intersect_poly.iter_polygons_flat():
                if p.points.shape[0] > 3:
                    valid_poly = True
                    radec.append(p.to_lonlat())
            if not valid_poly:
                continue

            area += intersect_poly.area()

            if __local_debug__:
                fn = (m1.id.split('_')[0] + '_' + ext2str(m1.ext, True) +
                      '_on_' + m2.basefname.split('_')[0] +
                      '_' + ext2str(m2.ext, True) + '.reg')
                ivert = []
                ivert1 = []
                ivert2 = []
                for rd in radec:
                    ivert.append(list(zip(*rd)))
                for m1poly in m1.polygon.iter_polygons_flat():
                    ivert1.append(list(zip(*m1poly.to_lonlat())))
                for m2poly in m2.polygon.iter_polygons_flat():
                    ivert2.append(list(zip(*m2poly.to_lonlat())))
                _debug_write_region_fk5(fn, ivert, ivert1, ivert2,
                                        multireg=True)

            sky1, wght1 = _member_sky(m1, radec, skystat, m2.id)
            sky2, wght2 = _member_sky(m2, radec, skystat, m1.id)
            if sky1 is not None and wght1 > 0.0:
                t_wsky1 += sky1 * wght1
                t_wght1 += wght1
            if sky2 is not None and wght2 > 0.0:
                t_wsky2 += sky2 * wght2
                t_wght2 += wght2

    if t_wght1 == 0.0 or t_wght2 == 0.0 or area == 0.0:
        return (None, None, None, None, area)
    else:
        return (t_wsky1 / t_wght1, t_wght1, t_wsky2 / t_wght2, t_wght2, area)


def _member_sky(member, radec, skystat, _dbg_name):
    wcs = member.wcs

    fill_mask = np.zeros(wcs.array_shape, dtype=bool)

    if __local_debug__:
        ivert1 = []

    # All pixels along intersection boundary for that chip
    for ra, dec in radec:
        sparse_x, sparse_y = wcs.all_world2pix(ra, dec, 0)
        ivert = list(
            zip(*[list(map(round, sparse_x)),
                  list(map(round, sparse_y))])
        )

        if __local_debug__:
            ivert1.append(
                list(zip(*[list(map(round, sparse_x + 1)),
                           list(map(round, sparse_y + 1))]))
            )

        pol = region.Polygon(True, ivert)
        fill_mask = pol.scan(fill_mask)

    if __local_debug__:
        fn = (_dbg_name.split('_')[0] + '_on_' +
              member.basefname.split('_')[0] + '_' +
              ext2str(member.ext, True) + '.reg')
        _debug_write_region(fn, ivert1, multireg=True)

    dqmask = member.mask_data  # may be a combination of DQ data and user mask
    if dqmask is not None:
        fill_mask = np.logical_and(fill_mask, dqmask)
        member.free_mask_data()
        del dqmask

    if __local_debug__:
        fn = (_dbg_name.split('_')[0] + '_on_' +
              member.basefname.split('_')[0] + '_' +
              ext2str(member.ext, True) + '.fits')
        if os.path.exists(fn):
            os.remove(fn)
        # write data to the "temporary" file
        hdu = fits.PrimaryHDU(fill_mask.astype(np.uint8))
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(fn)
        # clean-up
        hdulist.close()
        del hdu, hdulist

    # Calculate sky
    try:
        dat = member.image_data[fill_mask]
        sky, npix = skystat.calc_sky(dat)
    except ValueError:
        return (None, 0)

    sky = member.data2brightness(sky - member.skyuser_delta)

    return sky, npix


def _minsky(skyline, skystat, mlog):
    """
    Calculate the weighted average of sky from individual
    chips in the given skyline within a given RA and DEC
    of a polygon.

    Parameters
    ----------
    skyline: `SkyLine` object

    ra, dec: array_like
        RA and DEC of the polygon.

    skystat: A SkyStats object
        Class for sky calculations.

    Raises
    ------
    ValueError
        Total weight is zero.

    Returns
    -------
    sky: float
        Computed sky value minus any sky delta present.

    """
    minsky = None
    for member in skyline.members:
        # dqmask may be a combination of DQ data and user mask
        dqmask = member.mask_data
        if dqmask is None:
            dat = member.image_data
        else:
            dat = member.image_data[np.asarray(dqmask, dtype=bool)]
            member.free_mask_data()
            del dqmask

        if dat.size < 1:
            # we need at least 1 valid pixel to do statistics
            mlog.warning("Not enough data points to compute sky for \'{}\'.",
                         member.id)
            continue

        # Calculate sky
        try:
            sky, npix = skystat.calc_sky(dat)
        except ValueError:
            sky = 0
            npix = 0

        if __local_debug__:
            print(
                "_minsky({}) : raw sky stat : fields = \'{}\',  "
                "npix = {}, sky = {}"
                .format(member.id, skystat._fields, npix, sky)
            )
        if npix < 1:
            # we need at least 1 valid pixel to do statistics
            mlog.warning("Not enough data points to compute sky for "
                         "\'{}\' after clipping was applied.",
                         member.id)
            continue

        # convert to flux/pixel area units
        sky = member.data2brightness(sky - member.skyuser_delta)

        if minsky is None or minsky > sky:
            minsky = sky

    return minsky


def _update_sky_delta(skyline, skyval_brightness):
    """
    Updates `skyuser_delta` of each member of the
    input skyline. It *does not* modify image data or headers.

    .. note::
        This function does not update members of mosaiced skylines.

    Parameters
    ----------
    skyline: `SkyLine` object
        The `Skyline` of the image to update.

    skyval_brightness: float
        Value of sky (in units of brightness) to be added to
        `skyuser_delta` of each member of the input skyline.

    """
    if skyline.is_mf_mosaic or len(skyline.members) < 1:
        return

    for m in skyline.members:
        skyuser_delta = m.brightness2data(skyval_brightness)
        m.update_skydelta(skyuser_delta)
    return


def _apply_skyuser(skyline, readonly_mode, subtractsky, _taskname4history):
    """
    Set SKYUSER in SCI headers and/or subtract SKYUSER from
    SCI data.

    .. note:: ERR extensions are not modified even if sky
        subtraction could introduce additional errors.

    Parameters
    ----------
    skyline: `SkyLine` object
        Skyline of the image to update. Does not work if
        skyline is a product of union or intersection.

    """

    if skyline.is_mf_mosaic or len(skyline.members) < 1 or readonly_mode:
        return

    # if not readonly_mode => apply sky differences to data:
    hdr_keyword = skyline.members[0].get_skyuser_kwd()

    for m in skyline.members:
        if subtractsky:
            subsky = m.skyuser_delta
            m.image_hdulist[m.ext].data -= subsky
            m.update_skyuser()
            comment = 'Total sky value subtracted from data'
            m.image_header[hdr_keyword] = (m.skyuser_total, comment)
            if _taskname4history == 'SkyMatch' and subtractsky:
                m.image_header.add_history(
                    '{} {:E} subtracted from image by {:s}'
                    .format(hdr_keyword, subsky, _taskname4history)
                )
        else:
            comment = 'Sky value computed by {:s}'.format(_taskname4history)
            m.image_header[hdr_keyword] = (m.skyuser_delta, comment)

    if skyline.members and _taskname4history == 'SkyMatch':
        skyline.members[0].image_hdulist[0].header.add_history(
            '{} by {} {}'.format(hdr_keyword, __taskname__, __version__))


def _overlap_matrix(skylines, skystat):
    #TODO: to improve performance, the nested loops could be parallelized
    # since _calc_sky() here can be called independently from previous steps.
    ns = len(skylines)
    A = np.zeros((ns, ns), dtype=float)
    W = np.zeros((ns, ns), dtype=float)
    for i in range(ns):
        for j in range(i + 1, ns):
            s1, w1, s2, w2, area = _calc_sky(skylines[i], skylines[j], skystat)
            if area == 0.0 or s1 is None or s2 is None:
                continue
            A[j, i] = s1
            W[j, i] = w1
            A[i, j] = s2
            W[i, j] = w2
    return A, W


def _find_optimum_sky_deltas(skylines, skystat, mlog):
    ns = len(skylines)
    A, W = _overlap_matrix(skylines, skystat)

    def is_valid(i, j):
        return (W[i, j] > 0 and W[j, i] > 0)

    # We need to know how many "non-trivial" (at least for now... - we will
    # compute rank later) equations can be built so that we know the
    # shape of the arrays that need to be created...
    # NOTE: for now use only pairs that *both* have weights > 0 (but a
    # different scenario when only one image has a valid weight can be
    # considered):
    neq = 0
    for i in range(ns):
        for j in range(i + 1, ns):
            if is_valid(i, j):
                neq += 1

    # average weights:
    Wm = 0.5 * (W + W.T)

    # create arrays for coefficients and free terms:
    K = np.zeros((neq, ns), dtype=float)
    F = np.zeros(neq, dtype=float)
    invalid = (ns) * [True]

    # now process intersections between the rest of the images:
    ieq = 0
    for i in range(0, ns):
        for j in range(i + 1, ns):
            if is_valid(i, j):
                K[ieq, i] = Wm[i, j]
                K[ieq, j] = -Wm[i, j]
                F[ieq] = Wm[i, j] * (A[j, i] - A[i, j])
                invalid[i] = False
                invalid[j] = False
                ieq += 1

    try:
        rank = np.linalg.matrix_rank(K, 1.0e-12)
    except np.linalg.LinAlgError:
        mlog.warning("Unable to compute sky: No valid data in common "
                     "image areas")
        deltas = np.full(ns, np.nan, dtype=np.float)
        return deltas

    if rank < ns - 1:
        mlog.warning(
            "There are more unknown sky values ({}) to be solved for\nthan "
            "there are independent equations available (matrix rank={}).\n"
            "Sky matching (delta) values will be computed only for\n"
            "a subset (or more independent subsets) of input images."
            .format(ns, rank)
        )
    invK = np.linalg.pinv(K, rcond=1.0e-12)

    deltas = np.dot(invK, F)
    deltas[np.asarray(invalid, dtype=bool)] = np.nan
    return deltas


def _find_optimum_sky_deltas_wlead(skylines, skystat, mlog):
    # NOTE: This version sets the sky "delta" to zero for the skyline
    # with the most overlap. However, this is an unnecessary constraint
    # on the solution and therefore I do not recommend this
    # approach - Mihai Cara
    ns = len(skylines)
    A, W = _overlap_matrix(skylines, skystat)

    def is_valid(i, j):
        return (W[i, j] > 0 and W[j, i] > 0)

    # We need to know how many "non-trivial" (at least for now... - we will
    # compute rank later) equations can be build so that we know the
    # shape of the arrays that need to be created...
    # NOTE: for now use only pairs that *both* have weights > 0 (but a
    # different scenario when only one image has a valid weight can be
    # considered):
    neq = 0
    for i in range(ns):
        for j in range(i + 1, ns):
            if is_valid(i, j):
                neq += 1

    # average weights:
    Wm = 0.5 * (W + W.T)

    # create arrays for coefficients and free terms:
    K = np.zeros((neq, ns - 1), dtype=float)
    F = np.zeros(neq, dtype=float)
    invalid = (ns - 1) * [True]

    # Set the background of one of the skylines equal to zero, thus reducing
    # the number of unknowns to (ns-1). However, we want to make sure
    # this is a skyline that has most overlap area (weight) or most number
    # of overlaps with other images. This is simply because the "deltas"
    # of the sky of the rest of the images will be defined relative to the
    # sky of this image.
    nzcnt = np.array(list(map(np.count_nonzero, Wm)))
    maxcnt = np.amax(nzcnt)
    indxmcnt, = np.where(nzcnt == maxcnt)
    im = np.argmax(np.sum(Wm, axis=0)[indxmcnt])
    imax = indxmcnt[im]  # max number of intersections and then total weight
    # imax = np.argmax(np.sum(Wm,axis=0))  # max total weight
    invalid[imax] = False

    # Set the background of one of the skyline #imax equal to zero,
    # thus reducing the number of unknowns to (ns-1). Compute equations
    # for each intersection of 'imax' with the rest of the images.
    ieq = 0
    for j in range(0, ns):
        if j < imax:
            jk = j
        elif j > imax:
            jk = j - 1
        else:
            continue
        if is_valid(imax, j):
            K[ieq, jk] = -Wm[imax, j]
            F[ieq] = Wm[imax, j] * (A[j, imax] - A[imax, j])
            invalid[imax] = False
            ieq += 1

    # now process intersections between the rest of the images:
    for i in range(0, ns):
        if i < imax:
            ik = i
        elif i > imax:
            ik = i - 1
        else:
            continue

        for j in range(i + 1, ns):
            if j < imax:
                jk = j
            elif j > imax:
                jk = j - 1
            else:
                continue

            if is_valid(i, j):
                K[ieq, ik] = Wm[i, j]
                K[ieq, jk] = -Wm[i, j]
                F[ieq] = Wm[i, j] * (A[j, i] - A[i, j])
                invalid[ik] = False
                invalid[jk] = False
                ieq += 1

    try:
        rank = np.linalg.matrix_rank(K, 1.0e-12)
    except np.linalg.LinAlgError:
        mlog.warning("Unable to compute sky: No valid data in common "
                     "image areas")
        deltas = np.full(ns, np.nan, dtype=np.float)
        return deltas

    if rank < ns - 1:
        mlog.warning(
            "There are more unknown sky values ({}) to be solved for\nthan "
            "there are independent equations available (matrix rank={}).\n"
            "Sky values will be computed only for a subset of input images."
            .format(ns, rank)
        )

    invK = np.linalg.pinv(K, rcond=1.0e-12)

    deltas = np.dot(invK, F)
    deltas[np.asarray(invalid, dtype=bool)] = 0.0
    deltas = np.insert(deltas, imax, 0.0)
    return deltas


def _add_new_history(header, comment):
    #TODO: This function should be able to detect if the comment to be
    # added to the header is already in the header, and if so, it should
    # not add the same comment again.
    pass


#--------------------------
# TEAL Interface functions
#--------------------------
def run(configObj):
    TEAL_SkyMatch(
        input=configObj['input'],
        skymethod=configObj['skymethod'],
        match_down=configObj['match_down'],
        skystat=configObj['skystat'],
        lower=configObj['lower'],
        upper=configObj['upper'],
        nclip=configObj['nclip'],
        lsigma=configObj['lsigma'],
        usigma=configObj['usigma'],
        binwidth=configObj['binwidth'],
        skyuser_kwd=configObj['skyuser_kwd'],
        units_kwd=configObj['units_kwd'],
        invsens_kwd=configObj['invsens_kwd'],
        readonly=configObj['readonly'],
        subtractsky=configObj['subtractsky'],
        dq_bits=configObj['dq_bits'],
        optimize=configObj['optimize'],
        clobber=configObj['clobber'],
        clean=configObj['clean'],
        verbose=configObj['verbose'],
        logfile=configObj['logfile']
    )


def getHelpAsString(docstring=True):
    helpString = ''

    # if teal:
    #     helpString += teal.getHelpFileAsString(__taskname__, __file__)

    if helpString.strip() == '':
        helpString += __doc__ + '\n' + TEAL_SkyMatch.__doc__ + \
            '\n' + skymatch.__doc__

    return helpString


def help(file=None):
    """
    Print out syntax help for running skymatch

    Parameters
    ----------
    file: str, optional
        If given, write out help to the filename specified by this parameter
        Any previously existing file with this name will be deleted before
        writing out the help.
    """
    helpstr = getHelpAsString(docstring=True)
    if file is None:
        print(helpstr)
    else:
        if os.path.exists(file):
            os.remove(file)
        f = open(file, mode='w')
        f.write(helpstr)
        f.close()
