"""
This module provides utility functions for use
by :py:mod:`stsci.skypac` module.

:Authors: Mihai Cara

:License: :doc:`LICENSE`

"""
import sys
import os
import weakref
import tempfile
from os import path
from copy import copy, deepcopy

import numpy as np
from astropy.io import fits
from stsci.tools import fileutil, readgeis, convertwaiveredfits
from .hstinfo import (supported_telescopes, supported_instruments,
                      counts_only_instruments, rates_only_instruments)


__all__ = ['is_countrate', 'ext2str', 'MultiFileLog', 'ResourceRefCount',
           'ImageRef', 'openImageEx', 'count_extensions', 'get_ext_list',
           'get_extver_list', 'file_name_components', 'temp_mask_file',
           'get_instrument_info', 'almost_equal', 'skyval2txt']


def file_name_components(fname, detect_HST_FITS_suffix=True):
    """
    Splits base file name into a root, suffix, and extension.
    Given a full path, this function extracts the base name,
    and splits it into three components: root name, suffix,
    and file extension.

    Parameters
    ----------

    fname: str
        file name

    detect_HST_FITS_suffix: bool, optional
        If True, detects the suffix of most HST files by looking for the
        rightmost occurence of the underscore ('_') in the file name.

    Returns
    -------

    root: str
        Root name of the file. When ``detect_HST_FITS_suffix`` is `True`,
        this is the part of the file name *preceding* the rightmost suffix
        separator ('_'). Otherwise, it is the base file name without file
        extension.

    suffix: str
        If ``detect_HST_FITS_suffix`` is `True`, this field will contain the
        suffix of most HST files, i.e., the part of the file name contained
        between the rightmost suffix separator ('_') and file
        extension separator. This return value will be an empty string if
        If ``detect_HST_FITS_suffix`` is `False` or if the file name has no
        extension separator.

    fext: str
        File extension

    Examples
    --------

        >>> file_name_components('/data/m87/ua0x5001m_c0f.fits')
        ('ua0x5001m', 'c0f', '.fits')
        >>> file_name_components('/data/m87/ua0x5001m_c0f.fits',False)
        ('ua0x5001m_c0f', '', '.fits')

    """
    # get root name of the image file:
    (root, fname) = path.split(fname)
    (base, fext) = path.splitext(fname)

    if detect_HST_FITS_suffix and fext.lower() in ['.fits', '.fit']:
        ind = base.rfind('_')
        if ind >= 0:
            root = base[:ind]
            suffix = base[ind + 1:]
            #TODO: Additional check may be performed here to see if
            # the 'suffix' is in the list of registered HST suffixes,
            # e.g., '_raw', '_flt', '_c0f', etc.
            return (root, suffix, fext)

    root = base
    suffix = ''

    return (root, suffix, fext)


def in_memory_mask(data):
    """
    Creates an `ImageRef` object with that embeds an in memory
    (i.e., not attached to a file) ``HDUList`` containing the supplied mask
    data.


    Parameters
    ----------
    data: numpy.ndarray
        Data to be used to create an in-memory FITS file. Data will be
        written in the primary HDU.


    Returns
    -------
    ImageRef
        An open :py:class:`~skypac.utils.ImageRef` object of a temporary
        and in-memory (not attached to a file on disk) FITS file.
        Mask data will be in the Primary HDU.


    Examples
    --------
    >>> import numpy np
    >>> from stsci import skypac
    >>> skypac.utils.in_memory_mask(data=mask)
    <skypac.utils.ImageRef object at 0x101f5a3d0>

    """
    # create HDUList object
    hdu = fits.PrimaryHDU(data)
    hdulist = fits.HDUList([hdu])
    imageref = ImageRef(ResourceRefCount(hdulist))
    imageref.memmap = False
    imageref.can_reload_data = False
    return imageref


######################################################
## Returned mask file is a simple FITS and the data are in the
## primary HDU (ext=0).
def temp_mask_file(data, rootname, prefix='tmp', suffix='mask',
                   ext=('sci', 1), randomize_prefix=True,
                   sep='_', dir=os.path.curdir, fnameOnly=False):
    """
    temp_mask_file(rootname, suffix, ext, data, dir=os.path.curdir, \
    fnameOnly=False)
    Saves 2D data array to temporary simple FITS file.
    The name of the emporary file is generated based on the input parameters.

    Parameters
    ----------
    data: numpy array
        Data to be written to the temporary FITS file. Data will be
        written in the primary HDU.

    rootname: str
        Root name of the file.

    prefix: str, optional
        Prefix to be added in front of the root name. If ``randomize_prefix``
        is `True`, then a random string will be added to the right of the
        string specified by ``prefix`` (with no separator between them).
        Prefix (or the randomized prefix) will be separated from the
        root name by the string specified in ``sep``. If ``prefix`` is an empty
        string (``''``) then no prefix will be prepended to the root file
        name.

    suffix: str, optional
        Suffix to be added to the root name. Suffix will be separated from
        the root name by the string specified in ``sep``.

    ext: int, str, or tuple of the form (str, int), optional
        Extention to be added to the temporary file *after* the suffix.
        Extension name string will be separated from
        the suffix by the string specified in ``sep``.

    sep: str, optional
        Separator string to be inserted between (randomized) prefix
        and root name, root name and suffix, and suffix and extension.

    randomize_prefix: bool, optional
        Specifies whether to add (postpend) a random string to string
        specified by ``prefix``.

    dir: str, optional
        Directory to which the temporary file should be written. If directory
        ``dir`` is `None` then the file will be written to the default
        (for more details, see the explanation for argument ``dir`` to the
        `tempfile.mstemp <http://docs.python.org/2/library/
        tempfile.html#tempfile.mkstemp>`_ function).

    fnameOnly: bool, optional
        Specifies what should `temp_mask_file` return: file name of the
        created file (if ``fnameOnly`` is `True`), or a tuple with the file
        name of the created file and an open
        :py:class:`~skypac.utils.ImageRef` object of that file.

    Returns
    -------
    fname: str
        File name of the temporary file.

    mask: ImageRef
        An open :py:class:`~skypac.utils.ImageRef` object of the temporary
        FITS file. This is returned as a tuple together with the file name
        only when ``fnameOnly`` is `False`.

        .. note::
            Mask data will be in the Primary HDU.

    Raises
    ------
    TypeError
        Extension specifier must be either an integer, a string,
        or a tuple of the form (str, int).

    Examples
    --------
    >>> import numpy as np
    >>> from stsci import skypac
    >>> mask=np.ones((800,800),dtype=np.uint8)
    >>> skypac.utils.temp_mask_file(mask, 'ua0x5001m',
    ...     suffix='skymatch_mask', ext=('sci',4), dir='/data/m87',
    ...     fnameOnly=True)
    '/data/m87/tmp39gCpw_ua0x5001m_skymatch_mask_sci4.fits'
    >>> skypac.utils.temp_mask_file(mask, 'ua0x5001m',
    ...     suffix='skymatch_mask', ext=('sci',4), dir='.', fnameOnly=True)
    'tmpxl7LTO_ua0x5001m_skymatch_mask_sci4.fits'
    >>> skypac.utils.temp_mask_file(mask, 'ua0x5001m',
    ...     suffix='skymatch_mask', ext=('sci',4), dir='.',fnameOnly=False)
    ('tmpxMcL5g_ua0x5001m_skymatch_mask_sci4.fits', \
<skypac.utils.ImageRef object at 0x101f5a3d0>)

    """
    # convert extension to a string:
    try:
        strext = ext2str(ext, compact=True, default_extver=None)
    except Exception:
        raise TypeError("Extension specifier must be either an integer, "
                        "a string, or a tuple of the form (\'str\', int).")

    if prefix is None:
        prefix = ''

    if not (prefix.isupper() and rootname.isupper() and suffix.isupper()):
        strext = strext.lower()

    dir = os.path.expanduser(dir)

    if randomize_prefix:
        # new suffix for the temporary file:
        newbase = "{0:s}{1:s}{0:s}{2:s}{0:s}{3:s}{4:s}{5:s}" \
            .format(sep, rootname, suffix, strext, os.extsep, 'fits')
        # new temporary file handle:
        fh = tempfile.NamedTemporaryFile(prefix=prefix, suffix=newbase,
                                         dir=dir, delete=False, mode='wb')
    else:
        if prefix == '':
            fname = "{1:s}{0:s}{2:s}{0:s}{3:s}{4:s}{5:s}" \
                .format(sep, rootname, suffix, strext, os.extsep, 'fits')
        else:
            fname = "{1:s}{0:s}{2:s}{0:s}{3:s}{0:s}{4:s}{5:s}{6:s}" \
                .format(sep, prefix, rootname, suffix, strext, os.extsep,
                        'fits')
        full_name = os.path.join(dir, fname)
        fh = open(full_name, 'wb')

    # simplify path if 'dir' is current working directory:
    (pth, fname) = os.path.split(os.path.relpath(fh.name))
    if dir != os.path.curdir and not (pth == os.path.curdir or pth == ''):
        fname = fh.name

    # create HDUList object
    hdu = fits.PrimaryHDU(data)
    hdulist = fits.HDUList([hdu])

    # write data to the "temporary" file
    hdulist.writeto(fh)

    # clean-up
    hdulist.close()
    fh.close()
    del hdu, hdulist, fh

    if fnameOnly:
        return fname

    # open the "temporary" file and create a new ImageRef object:
    mask, dummy = openImageEx(
        fname, mode='readonly', memmap=False, saveAsMEF=False, clobber=False,
        imageOnly=True, openImageHDU=True, openDQHDU=False, preferMEF=True,
        verbose=False
    )

    return (fname, mask)


def get_extver_list(img, extname='SCI'):
    """
    Return a list of all extension versions with ``extname`` extension
    names. If ``extname`` is `None`, return extension **numbers** of all
    image-like extensions.

    .. note::
        If input image is a `~skypac.utils.ImageRef`, this function will
        **not** modify its reference count.

    Parameters
    ----------
    img: str, `astropy.io.fits.HDUList`, or `~skypac.utils.ImageRef`
        Input image object. If ``img`` is a string object (file name) then that
        file will be opened. If the file pointed to by the file name is a
        GEIS or WAIVER FITS file, it will be converted to a simple/MEF FITS
        format if ``clobber`` is `True`.

    extname: str, optional
        Indicates extension *name* for which all existing extension *versions*
        should be found. If ``extname`` is `None`, then
        `~skypac.utils.get_extver_list` will return a list of extension
        *numbers* of all image-like extensions.

    Returns
    -------
    extver: list
        List of extension versions corresponding to the input ``extname``.
        If ``extname`` is `None`, it will return a list of extension
        *numbers* of all image-like extensions.

    Raises
    ------
    IOError
        Unable to open input image file.

    TypeError
        Argument `img` must be either a file name (str),
        an `~.utils.ImageRef`, or a `astropy.io.fits.HDUList` object.

    TypeError
        Argument `extname` must be either a string or `None`.

    See Also
    --------
    get_ext_list, count_extensions

    Examples
    --------
    >>> get_extver_list('j9irw1rqq_flt.fits',extname='sci')
    [1, 2]
    >>> get_extver_list('j9irw1rqq_flt.fits',extname=None)
    [1, 2, 3, 4, 5, 6, 8, 9, 10, 11

    """
    doRelease = False
    if isinstance(img, fits.HDUList):
        hdulist = img
    elif isinstance(img, str):
        try:
            (img, dq) = openImageEx(
                img, mode='readonly', memmap=False, saveAsMEF=False,
                output_base_fitsname=None, clobber=False, imageOnly=True,
                openImageHDU=True, openDQHDU=False, preferMEF=False,
                verbose=False
            )
        except IOError:
            raise IOError("Unable to open file: \'{:s}\'".format(img))
        hdulist = img.hdu
        doRelease = True
    elif isinstance(img, ImageRef):
        assert(img.hdu is not None)
        img.hold()
        hdulist = img.hdu
        doRelease = True
    else:
        raise TypeError("Argument 'img' must be either a file name (string), "
                        "an ImageRef, or a `astropy.io.fits.HDUList` object.")

    # when extver is None - return the range of all 'image'-like FITS
    # extensions
    if extname is None:
        extn = []
        for i in range(len(hdulist)):
            hdr = hdulist[i].header
            if not ('NAXIS' in hdr and hdr['NAXIS'] == 2):
                continue
            if 'XTENSION' in hdr and \
               hdr['XTENSION'].upper().strip() == 'IMAGE':
                extn.append(i)
            elif 'SIMPLE' in hdr:
                extn.append(i)
        if doRelease:
            img.release()
        return extn

    if not isinstance(extname, str):
        if doRelease:
            img.release()
        raise TypeError(
            "Argument 'extname' must be either a string indicating the value"
            "of the 'EXTNAME' keyword of the extensions whose versions are to "
            "be returned or None to return extension numbers of all HDUs in "
            "the 'img' FITS file."
        )

    extname = extname.upper()

    extver = []
    for e in hdulist:
        #if not isinstance(e, fits.ImageHDU): continue
        #hkeys = map(str.upper, e.header.keys())
        if 'EXTNAME' in e.header and e.header['EXTNAME'].upper() == extname:
            extver.append(e.header['EXTVER'] if 'EXTVER' in e.header else 1)

    if doRelease:
        img.release()

    return extver


def get_ext_list(img, extname='SCI'):
    """
    Return a list of all extension versions of ``extname`` extensions.
    ``img`` can be either a file name or a `astropy.io.fits.HDUList` object.

    This function is similar to :py:func:`get_extver_list`, the main
    difference being that it returns a list of fully qualified extensions:
    either tuples of the form `(extname, extver)` or integer extension
    numbers (when ``extname`` is `None`).

    See Also
    --------
    get_extver_list, count_extensions

    Examples
    --------
    >>> get_ext_list('j9irw1rqq_flt.fits',extname='SCI')
    [('SCI', 1), ('SCI', 2)]
    >>> get_ext_list('j9irw1rqq_flt.fits',extname=None)
    [1, 2, 3, 4, 5, 6, 8, 9, 10, 11]

    """
    extver = get_extver_list(img=img, extname=extname)
    if extname is None:
        return extver

    extlist = [(extname, extv) for extv in extver]
    return extlist


def count_extensions(img, extname='SCI'):
    """
    Return the number of `extname` extensions in the input image ``img``.
    If ``extname`` is `None`, return the number of all image-like extensions.

    Input parameters are identical to those of :py:func:`get_extver_list`.

    See Also
    --------
    get_extver_list, get_ext_list

    Examples
    --------
    >>> count_extensions('j9irw1rqq_flt.fits',extname='SCI')
    2
    >>> count_extensions('j9irw1rqq_flt.fits',extname=None)
    10

    """
    return len(get_extver_list(img=img, extname=extname))


def ext2str(ext, compact=False, default_extver=1):
    """
    Return a string representation of an extension specification.

    Parameters
    ----------
    ext: tuple, int, str
        Extension specification can be a tuple of the form (str,int), e.g.,
        ('sci',1), an integer (extension number), or a string (extension
        name).

    compact: bool, optional
        If ``compact`` is `True` the returned string will have extension
        name quoted and separated by a comma from the extension number,
        e.g., ``"'sci',1"``.
        If ``compact`` is `False` the returned string will have extension
        version immediately follow the extension name, e.g., ``'sci1'``.

    default_extver: int, optional
        Specifies the extension version to be used when the ``ext`` parameter
        is a string (extension name).

    Returns
    -------
    strext: str
        String representation of extension specification ``ext``.

    Raises
    ------
    TypeError
        Unexpected extension type.

    Examples
    --------
    >>> ext2str('sci',compact=False,default_extver=6)
    "'sci',6"
    >>> ext2str(('sci',2))
    "'sci',2"
    >>> ext2str(4)
    '4'
    >>> ext2str('dq')
    "'dq',1"
    >>> ext2str('dq',default_extver=2)
    "'dq',2"
    >>> ext2str('sci',compact=True,default_extver=2)
    'sci2'

    """
    if isinstance(ext, tuple) and len(ext) == 2 and \
       isinstance(ext[0], str) and isinstance(ext[1], int):
        if compact:
            return "{:s}{:d}".format(ext[0], ext[1])
        else:
            return "\'{:s}\',{:d}".format(ext[0], ext[1])

    elif isinstance(ext, int):
        return "{:d}".format(ext)

    elif isinstance(ext, str):
        if default_extver is None:
            extver = ''
        else:
            extver = '{:d}'.format(default_extver)

        if compact:
            return "{:s}{:s}".format(ext, extver)
        else:
            return "\'{:s}\',{:s}".format(ext, extver)

    else:
        raise TypeError("Unexpected extension type.")


def is_countrate(hdulist, ext, units_kwd='BUNIT', guess_if_missing=True,
                 telescope=None, instrument=None, verbose=True, flog=None):
    """
    Infer the units of the data of the input image from the input image.
    Specifically, it tries to infer whether the units are counts (or
    count-like) or if the units are count-``rate``.

    The units of data are determined from the ``BUNIT`` header keyword by
    searching its value for the division sign ``'/'``. If the division sign is
    not found, then the units are assumed to be "counts". If the division
    sign is found in the ``BUNIT`` value and if the numerator is one of
    the following: 'ELECTRONS','COUNTS', or 'DN', and denumerator is either
    'S','SEC', or 'SECOND', then the units are assumed to be count-rate.

    Parameters
    ----------
    hdulist: `astropy.io.fits.HDUList`
        `astropy.io.fits.HDUList` of the image.

    ext: tuple, int, str
        Extension specification for whose data the units need to be inferred.
        An int ``ext`` specifies extension number. A tuple in the form
        (str, int) specifies extension name and number. A string ``ext``
        specifies extension name and the extension version is assumed to be
        1. See documentation for `astropy.io.fits.getData`
        for examples.

    units_kwd: str, optional
        FITS header keyword describing data units of the image. This keyword
        is assumed to be in the header of the extension specified by the
        ``ext`` parameter.

    guess_if_missing: bool, optional
        Instructs to try make best guess on image units when the keyword
        specified by ``units_kwd`` is not found in the image header. The first
        action will be to look for this keyword in the primary header, and
        if not found, infer the units based on the telescope, instrument,
        and detector information.

    telescope: str, None, optional
        Specifies the telescope from which the data came. If not specified,
        the value specified in the ``'TELESCOP'`` keyword in the primary
        header will be used.

    instrument: str, None, optional
        Specifies the instrument used for acquiring data. If not specified,
        the value specified in the ``'INSTRUME'`` keyword in the primary
        header will be used.

    verbose: bool, optional
        Specifies whether to print warning messages.

    flog: str, file, MultiFileLog, None, optional
        Specifies the log file to which the messages should be printed.
        It can be a file name, a file object, a `MultiFileLog` object, or
        `None`.

    Returns
    -------

    bool, None
        Returns `True` if the units of the input image for a given extension
        are count-rate-like and `False` if the units are count-like. Returns
        `None` if the units cannot be inferred from the header.

    """
    # Set-up log files:
    if isinstance(flog, MultiFileLog):
        ml = flog
    else:
        ml = MultiFileLog(console=verbose)
        if flog not in ('', None):
            ml.add_logfile(flog)
            ml.skip(2)

    data_units = ['ELECTRONS', 'COUNTS', 'DN']
    time_units = ['S', 'SEC', 'SECOND']

    fname = os.path.basename(hdulist.filename())

    sci_header = hdulist[ext].header
    primary_header = hdulist[0].header

    if units_kwd is None:
        guess_if_missing = True
        bunit = None
    else:
        if units_kwd in sci_header:
            bunit = sci_header[units_kwd]
        elif guess_if_missing and units_kwd in primary_header:
            # look for units in the primary header
            bunit = primary_header[units_kwd]
        else:
            bunit = None

    if bunit is not None:

        bunit_parts = list(map(str.upper,
                               list(map(str.strip, bunit.split('/')))))
        nbunit_parts = len(bunit_parts)

        if nbunit_parts == 1 and bunit_parts[0] in data_units:
            ml.close()
            return False

        if nbunit_parts == 2 and bunit_parts[0] in data_units and \
           bunit_parts[1] in time_units:
            ml.close()
            return True

        # else:
        ml.warning("Unable to infer units of data from the "
                   "\'{:s}\' keyword in file \'{:s}\'.", units_kwd, fname)
        ml.close()
        return None

    if not guess_if_missing:
        ml.warning("Unable to infer units of data from the "
                   "\'{:s}\' keyword in file \'{:s}\': Keyword not found.",
                   units_kwd, fname)
        ml.close()
        return None

    # Now, try to guess the units from the 'INSTRUME' (and 'DETECTOR',
    # if necessary) keyword values. Return the default for a specific
    # 'INSTRUME'/'DETECTOR' combination.
    if telescope is None:
        if 'TELESCOP' in primary_header:
            telescope = primary_header['TELESCOP'].strip().upper()
        else:
            ml.warning("Unable to infer units of data for file "
                       "\'{:s}\': Missing telescope information.", fname)
            ml.close()
            return None

    if telescope not in supported_telescopes:
        ml.warning("Unable to infer units of data for non-HST file "
                   "\'{:s}\': Unsupported telescope.", fname)
        ml.close()
        return None

    if instrument is None:
        if 'INSTRUME' in primary_header:
            instrument = primary_header['INSTRUME'].strip().upper()
        else:
            ml.warning("Unable to infer units of data for file "
                       "\'{:s}\': Missing instrument information.", fname)
            ml.close()
            return None

    if instrument not in supported_instruments:
        ml.warning("Unable to infer units of data for file "
                   "\'{:s}\': Unsupported instrument.", fname)
        return None

    if instrument in counts_only_instruments:
        ml.close()
        return False

    if instrument in rates_only_instruments:
        ml.close()
        return True

    if instrument == 'WFC3':
        if 'DETECTOR' in sci_header:
            detector = sci_header['DETECTOR'].strip().upper()
        elif 'DETECTOR' in primary_header:
            detector = primary_header['DETECTOR'].strip().upper()
        else:
            ml.warning("Unable to infer units of data for WFC3 file "
                       "\'{:s}\': Missing detector information.", fname)
            ml.close()
            return None
        if detector == 'UVIS':
            ml.close()
            return False
        elif detector == 'IR':
            ml.close()
            return True
        else:
            ml.warning("Unable to infer units of data for WFC3 file "
                       "\'{:s}\': Unrecognized detector.", fname)
            ml.close()
            return None

    ml.close()
    if instrument == 'NICMOS':
        if 'UNITCORR' in sci_header and \
           sci_header['UNITCORR'].strip().upper() == 'OMIT':
            return False
        else:
            # quietly assume units are in count-rate even when 'UNITCORR'
            # is not present:
            return True

    # Code should never reach this line
    raise RuntimeError("Logical error in \'is_countrate(...)\'.")


def get_instrument_info(image, ext):
    sci_header = image.hdu[ext].header
    primary_header = image.hdu[0].header

    telescope = None
    # check if ImageRef object has telescope info:
    if image.telescope is not None:
        telescope = image.telescope.strip().upper()
        if telescope in ['', 'UNKNOWN']:
            telescope = None

    # get telescope info from the primary header:
    if telescope is None and 'TELESCOP' in primary_header:
        telescope = primary_header['TELESCOP'].strip().upper()

    instrument = None
    # check if ImageRef object has instrument info:
    if image.instrument is not None:
        instrument = image.instrument.strip().upper()
        if instrument in ['', 'UNKNOWN']:
            instrument = None

    # get instrument info from the primary header:
    if instrument is None and 'INSTRUME' in primary_header:
        instrument = primary_header['INSTRUME'].strip().upper()

    # get detector info:
    detector = None
    if 'DETECTOR' in sci_header:
        if isinstance(sci_header['DETECTOR'], str):
            detector = sci_header['DETECTOR'].strip().upper()
        else:
            detector = sci_header['DETECTOR']
    elif 'DETECTOR' in primary_header:
        if isinstance(primary_header['DETECTOR'], str):
            detector = primary_header['DETECTOR'].strip().upper()
        else:
            detector = primary_header['DETECTOR']

    return (telescope, instrument, detector)


class MultiFileLog(object):
    """
    This is a class that facilitates writting to multiple files.
    `MultiFileLog` stores multiple file objects and can write the same
    log entry to all of them. It also facilitates controlling when a
    function can close a log file. Finally, it provides some utility
    functions that automate such things as appending EOL at the end of the
    log entry, flushing the files (to avoid losing log entries in case of
    uncaught exceptions), displaying ``WARNING``, ``ERROR``, etc. in bold
    on standard streams (`sys.stdout`, etc.)

    Parameters
    ----------
    console: bool, optional
        Enables writting to the standard output.

    enableBold: bool, optional
        Enable or disable writing bold text to console, e.g., ``'WARNING:'``
        ``'ERROR:'``, etc.

    flog: str, file, None, list of str or file objects, optional
        File name or file object to be added to `MultiFileLog` during the
        initialization. More files can be added lated with
        :py:meth:`add_logfile`.

    append: bool, optional
        Default open mode for the files that need to be opened (e.g., that
        are passed as file names). If ``append`` is `True`, new files added
        to a `MultiFileLog` object will be opened in the "append"
        mode: new log entries will be appended to existing files -- same as
        mode 'a' of standard function :py:func:`open`. When ``append`` is
        `False`, any existing files will be overwritten.

    autoflush: bool, optional
        Indicates whether or not to flush files after each log entry.

    appendEOL: bool, optional
        Indicates whether or not to add EOL at the end of each log entry.

    """
    def __init__(self, console=True, enableBold=True, flog=None, append=True,
                 autoflush=True, appendEOL=True):
        self._console = sys.stdout if console else None
        self._eol = '\n' if appendEOL else ''
        self._fmode = 'a' if append else 'w'
        self._logfh = []
        self._close = []
        self._autoflush = autoflush
        self._enableBold = enableBold
        if flog:
            self.add_logfile(flog)

    def unclose_copy(self):
        """
        Return a copy of the `MultiFileLog` object with all the attached files
        marked as "keep open", that is, the :py:meth:`close` will not close
        these files.

        This is useful before passing the `MultiFileLog` object to a function
        that may add its own log files, and then attempt to close the files
        with the :py:meth:`close` method. Thus, by passing an "unclose copy",
        one can be sure that the files opened at the top level will not be
        closed by other functions to which the `MultiFileLog` object may be
        passed.

        """
        newMultiFileLog = copy(self)
        newclose = self.count * [False]
        newMultiFileLog._close = newclose
        return newMultiFileLog

    @property
    def count(self):
        """
        Return the number of files attached to the `MultiFileLog` object
        excluding the `sys.stdout` file.

        """
        return len(self._logfh)

    def enable_console(self, enable=True, enableBold=True):
        """
        Enable output to the standard console -- `sys.stdout`.

        Parameters
        ----------
        enable: bool, optional
            Enable or disable output to `sys.stdout`.

        enableBold: bool, optional
            Enable or disable writing bold text to console, e.g.,
            ``'WARNING:'``, ``'ERROR:'``, etc.

        """
        if enable:
            if self._console is not None:
                return
            else:
                self._console = sys.stdout
        else:
            self._console = None
        self._enableBold = enableBold

    def add_logfile(self, flog, initial_skip=2, can_close=None, mode=None):
        """
        Add (and open, if necessary) a log file to the `MultiFileLog` object.

        Parameters
        ----------
        flog: str, file, None, list of str or file objects, optional
            File name or file object to be added.

        initial_skip: int, optional
            The number of blank line to be written to the file *if* *not*
            at the beginning of the file.

        can_close: bool, None, optional
            Indicates whether the file object can be closed by the
            :py:meth:`~skypac.utils.MultiFileLog.close` function:

            * ``can_close`` is `True` -- file can be closed by the
              :py:meth:`~skypac.utils.MultiFileLog.close` function;
            * ``can_close`` is `False` -- file will not be closed by the
              :py:meth:`~skypac.utils.MultiFileLog.close` function;
            * ``can_close`` is `None` -- automatic selection based on the type
              of the `flog` argument: `True` if flog is a string
              (e.g., file name), `False` otherwise.

        mode: str, None, optional
            File open mode: same meaning as the ``mode`` parameter of the
            Python's built-in :py:func:`open` function. If `None`, the mode
            will be inherited from file open mode set during initialization.

        Returns
        -------
        flog: file object
            File object of newly opened (or attached) file.

        """
        if isinstance(flog, list):
            fhlist = []
            for f in flog:
                fhlist.append(self.add_logfile(f))
            return fhlist

        if flog is None:
            return None

        if flog == sys.stdout:
            self.enable_console(True)
            return None

        if isinstance(flog, str):
            fmode = self._fmode if mode is None else mode
            fh = open(flog, fmode)
            self._logfh.append(fh)
            self._close.append(True if can_close is None else can_close)

            # skip 'initial_skip' number of lines
            # if not at the beginning of the file:
            if fh.tell() > 0:
                fh.write(initial_skip * '\n')
            return fh

        if flog not in self._logfh:
            self._logfh.append(flog)
            self._close.append(False if can_close is None else can_close)
            # skip 'initial_skip' number of lines
            # if not at the beginning of the file:
            # TODO: may not work well in Windows if fgets was used previously
            #       on the handle (see Python doc)
            if flog.tell() > 0:
                flog.write(initial_skip * '\n')
            return flog
        else:
            return None

    def set_close_flag(self, fh, can_close=True):
        """
        Modify the "can be closed" status of a given file.

        Parameters
        ----------
        fh: file object

        can_close: bool, optional
            Indicates if the file can be closed by the :py:meth:`close`
            function.

            .. note::
                File object ``fh`` **must** have been already added to
                the `MultiFileLog` object.

        """
        if fh is None or fh not in self._logfh:
            return
        index = self._logfh.index(fh)
        self._close[index] = can_close

    def close(self):
        """ Close all files *opened by* `~MultiFileLog`.

        It will close all files *opened by* `~MultiFileLog` - essentially,
        all files added as file name. It will *not* close files added
        as file handles.

        """
        n = len(self._logfh)
        todel = []
        assert(n == len(self._close))
        for i in range(n):
            if self._close[i]:
                self._logfh[i].close()
                todel.append(i)
        # delete closed files from the list of file handles:
        for i in todel:
            del self._logfh[i]
            del self._close[i]

    def flush(self):
        """
        Flush all files.

        """
        for fh in self._logfh:
            fh.flush()

    def print_endlog_msg(self):
        """
        Print a message ("Log written to...") indicating the end of the log.

        """
        if self._console:
            for fh in self._logfh:
                if hasattr(fh, 'name'):
                    self._console.write("Log written to \'{:s}\'.{}"
                                        .format(fh.name, self._eol))
                else:
                    self._console.write("Log written to an object of \'{}\'.{}"
                                        .format(type(fh), self._eol))

    def logentry(self, msg, *fmt, **skip):
        """
        Write a log entry to the log files.

        Parameters
        ----------
        msg: str
            String to be printed. Can contain replacement fields delimited
            by braces ``{}``.

        fmt
            Parameters to be passed to :py:meth:`str.format` method for
            formatting the string ``msg``.

        skip: int
            Number of empty lines that should follow the log message.

        Examples
        --------
        >>> logentry('Sky background for chip {} is {}', 'SCI1', 10.0, skip=2)
        will print:
        Sky background for chip SCI1 is 10.0
        followed by two blank lines.

        """
        if 'skip' in skip:
            nskip = skip['skip'] + 1
            nskip = max(nskip, 0)
        else:
            nskip = 1
        m = msg.format(*fmt) + nskip * self._eol
        # output to console
        if self._console:
            self._console.write(m)
        # output to disk log files
        for fh in self._logfh:
            fh.write(m)
            if self._autoflush:
                fh.flush()

    def skip(self, nlines=1):
        """
        Skip a specified number of blank lines.

        Parameters
        ----------
        nlines: int, optional

        """
        msg = nlines * '\n'
        if self._console:
            self._console.write(msg)
        for fh in self._logfh:
            fh.write(msg)

    def eol(self):
        # output to console
        if self._console:
            self._console.write(self._eol)
        # output to disk log files
        for fh in self._logfh:
            fh.write(self._eol)
            if self._autoflush:
                fh.flush()

    def bold(self, bmsg, msg, *fmt):
        m = msg.format(*fmt) + self._eol

        if self._console:
            if self._enableBold:
                bm = "\x1B[1m{:s}: \x1B[0m".format(bmsg)
            else:
                bm = "*{:s}:* ".format(bmsg)
            self._console.write(bm + m)

        bm = "*{:s}:* ".format(bmsg)
        m = bm + m
        for fh in self._logfh:
            fh.write(m)

    def warning(self, msg, *fmt):
        """
        Prints a warning message. The word 'WARNING:' will be printed as bold
        on the console and enclosed in asterisks (*) in a disk file.

        """
        self.bold("WARNING", msg, *fmt)

    def error(self, msg, *fmt):
        """
        Prints an error message. The word 'ERROR:' will be printed as bold
        on the console and enclosed in asterisks (*) in a disk file.

        """
        self.bold("ERROR", msg, *fmt)

    def important(self, msg, *fmt):
        """
        Prints an "important" message. The word 'IMPORTANT:' will be printed
        as bold on the console and enclosed in asterisks (*) in a disk file.

        """
        self.bold("IMPORTANT", msg, *fmt)


class ResourceRefCount(object):
    """
    A class that implements reference counting for various resources:
    file objects, FITS HDU lists, etc. It is intended to be used as a
    mechanism of controlling the "lifespan" of resources that can be used
    in different parts of the code "indipendently". The resource is kept
    open as long as the reference count is larger than zero. Once the
    reference count is decreased to 0, the resource is automatically closed.

    .. note::
        The reference count of the newly created `ResourceRefCount` object
        is set to 0. It is user's responsibility to increment the reference
        count of this object through the call to :py:meth:`hold` function.

    Parameters
    ----------
    resource
        An object who must be kept open or closed based on reference count.

    close_args: tuple
        Positional arguments to be passed to the
        :py:meth:`resource_close_fnct` function.

    resource_close_fnct: function, optional
        The function (usually a method of the attached resource), that can
        "close" the resource.

    close_kwargs: dict
        Keyword arguments to be passed to the :py:meth:`resource_close_fnct`
        function.

    """
    def __init__(self, resource, *close_args, resource_close_fnct=None,
                 **close_kwargs):
        self._resource = resource
        self._count = 0
        self._close_args = close_args
        self._close_kwargs = close_kwargs

        if resource_close_fnct is not None:
            self._res_close = resource_close_fnct
        elif hasattr(resource, 'close'):
            self._res_close = resource.close
        else:
            self._res_close = (lambda: None)
            self._close_args = ()
            self._close_kwargs = {}

        self._close_notify = weakref.WeakKeyDictionary()

    def subscribe_close_notify(self, obj, callback=None):
        """
        Set the object and its method that need to be called when the resource
        is about to be closed.

        """
        if not callback:
            self.unsubscribe_close_notify(obj)
            return
        self._close_notify[obj] = callback

    def unsubscribe_close_notify(self, obj):
        """
        Remove the object (and its method) from the list of callbacks that
        need to be notified of impending closing of the resource.

        """
        if obj in self._close_notify:
            del self._close_notify[obj]

    def is_subscribed_on_close(self, obj):
        """
        Check if the object is subscribed to on close notifications.

        """
        return obj in self._close_notify

    def _notify_subscribers(self):
        for n in self._close_notify:
            if not n:
                continue
            cb = self._close_notify[n]
            getattr(n, cb)()

    @property
    def resource(self):
        """
        Resource attached to a `ResourceRefCount` object.

        """
        return self._resource

    def hold(self):
        """
        Increment the reference count of the attached resource.

        """
        if self._resource is not None:
            self._count += 1

    def release(self):
        """
        Decrement reference count of the attached resource. If the reference
        count reaches zero, call all registered "on close" notify callbacks,
        and then call the "close" method on the resource which was set
        at initialization of the `ResourceRefCount` object. Finally,
        the `resource` property of the `ResourceRefCount` object will be
        set to `None`.

        """
        if self._count > 0:
            self._count -= 1
            if self._count == 0:
                self._notify_subscribers()
                if self._resource is not None:
                    self._res_close()
                self._res_close = None
                del self._resource
                self._resource = None
                self._close_notify = {}

    @property
    def refcount(self):
        """
        Reference count.

        """
        return self._count

    @property
    def closed(self):
        """
        Indicates if the resource is "closed".

        """
        return self._resource is None

    # Copying is not allowed for ResourceRefCount instances:
    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


class ImageRef(object):
    """
    A lightweight class that supports reference counting for FITS images
    and holds a :py:class:`ResourceRefCount` object. It provides several
    attributes that describe main characteristics of the image (file) and
    essential functions for manipulating reference count.

    Parameters
    ----------
    hdulist_refcnt: ResourceRefCount
        A :py:class:`ResourceRefCount` object holding a
        `astropy.io.fits.HDUList` object.

    Attributes
    ----------
    filename: str
        Name of the opened file. Can be `None` for in-memory created
        `astropy.io.fits.HDUList` objects.

    can_reload_data: bool
        `True` for files attached to a physical file, `False` for in-memory
        `astropy.io.fits.HDUList` objects.

    original_fname: str
        Original name of the file as requested by the user. *Note:* may be
        different from ``filename`` if the orininal file was in GEIS or
        WAIVER FITS format and subsequently was converted to a MEF FITS
        format. In that case this attribute will show the name of the
        original GEIS or WAIVER FITS file.

    original_ftype: str
        Type of the original file. Can take one of the following values:
        'MEF', 'SIMPLE', 'GEIS', 'WAIVER', or 'UNKNOWN'.

    original_exists: bool
        Indicates if the physical file exists. It is `False` for in-memory
        images.

    mef_fname: str, None
        Name of the MEF FITS file if exists, `None` otherwise.

    mef_exists: bool
        Indicates whether the MEF FITS file exists.

    DQ_model: str, None
        Type of the DQ model: 'external' for WFPC, WFPC2, and FOC
        instruments (or non-HST data if set so) that have DQ data in a
        separate (from image) file and 'intrinsic' for ACS, etc. images
        that have DQ extensions in the image file. It is `None` if
        the image does not have DQ data.

    telescope: str
        Telescope that acquired the image.

    instrument: str
        Instrument used to acquire data.

    fmode: str
        File mode used to open FITS file. See `astropy.io.fits.open` for
        more details.

    memmap: bool
        Is the `astropy.io.fits.HDUList` memory mapped?

    """
    def __init__(self, hdulist_refcnt=None):
        # loaded file name. This can be None for "pure" in-memory objects.
        # However, it will indicate the "original" file on the disk if the
        # "in-memory" HDUList was created with 'readgeis' or
        # 'convertwaiveredfits'
        self.filename = None
        # True for not-in-memory HDULists (e.g., loaded with fits.open()
        # from existing 'simple' or 'mef' FITS files)
        self.can_reload_data = False

        if hdulist_refcnt is not None:
            if hdulist_refcnt.closed:
                raise ValueError(
                    "Resource reference counter must not be closed."
                )
            hdulist_refcnt.hold()
            hdulist_refcnt.subscribe_close_notify(
                self, '_refcnt_about_to_close'
            )
        self._hdulist_refcnt = hdulist_refcnt

        # EXTNAME of the standard image extensions. When it is obtained
        # from openImageEx, this is valid only for HST products and
        # is computed based on file name suffix, file type, and instrument.
        self._img_extname = None

        self.original_fname = None  # input file name supplied to openImageEx

        # Possible values for original_ftype: 'MEF', 'SIMPLE', 'GEIS',
        # 'WAIVER', 'UNKNOWN'
        self.original_ftype = 'UNKNOWN'

        # original_exists can be False for DQ images that do not exist when the
        # science image is opened with openImageEx
        self.original_exists = False

        self.mef_fname = None
        self.mef_exists = False

        # Possible values for DQ_model:
        # - 'external' for WFPC, WFPC2, and FOC instruments or non-HST data
        #   if set so;
        # - 'intrinsic' for ACS, etc. that have DQ extensions in the image
        #   file;
        # - None if the image does not have DQ data.
        self.DQ_model = None

        self.telescope = 'UNKNOWN'
        self.instrument = 'UNKNOWN'

        self.fmode = 'readonly'
        self.memmap = True

    def __copy__(self):
        cp = ImageRef(self._hdulist_refcnt)
        cp.__dict__ = self.__dict__.copy()
        return cp

    def __deepcopy__(self, memo):
        cp = ImageRef(self._hdulist_refcnt)
        memo[id(self)] = cp
        for k, v in self.__dict__.items():
            setattr(cp, k, deepcopy(v, memo))
        return cp

    @property
    def extname(self):
        """ Extension name of the first extension. """
        return self._img_extname

    @extname.setter
    def extname(self, extn):
        self._img_extname = extn

    @property
    def refcount(self):
        """ Reference count of the attached `astropy.io.fits.HDUList`. """
        if self._hdulist_refcnt is not None:
            return self._hdulist_refcnt.refcount
        else:
            return None

    # callback for when the resource reference counter is about to be
    # closed
    def _refcnt_about_to_close(self):
        self.can_reload_data = False
        self.filename = None
        self._hdulist_refcnt = None

    def set_HDUList_RefCount(self, hdulist_refcnt=None):
        """
        Set (attach) a new :py:class:`ResourceRefCount` object that holds
        a `astropy.io.fits.HDUList` object. This is allowed only if the already
        attached :py:class:`ResourceRefCount` can be closed. The reference
        count of the :py:class:`ResourceRefCount` being attached will be
        incremented.

        Parameters
        ----------
        hdulist_refcnt: ResourceRefCount, None
            A :py:class:`ResourceRefCount` object containing a
            `astropy.io.fits.HDUList` object. If it is `None`,it will release
            and close the attached :py:class:`ResourceRefCount` object.

        Raises
        ------
        ValueError
            The (already) attached :py:class:`ResourceRefCount` must have
            reference count <= 1 so that it can be closed before being
            replaced with a new resource.

        ValueError
            The resource being attached must be in an open state.

        """
        # for now, allow setting a new RefCount object only if the
        # existing counter can be closed. see comments below...
        if self._hdulist_refcnt is not None and self._hdulist_refcnt.refcount:
            raise ValueError("set_HDUList_RefCount cannot replace the "
                             "assigned ResourceRefCount object when it "
                             "its reference count is larger than one.")

        if hdulist_refcnt is self._hdulist_refcnt:
            return

        if self._hdulist_refcnt is not None:
            if hdulist_refcnt.closed:
                raise ValueError("The new resource reference counter must "
                                 "not be closed.")

            #TODO: this will 'release' _hdulist_refcnt only once.
            #However there may be more copies of this instance of ImageRef
            # that "hold" the original (before replacement) _hdulist_refcnt.
            # Therefore, the original resource might still be open.
            #
            # Should we forbid replacing the resource counter if it is not
            # "closed"?
            # -- see RuntimeError below...
            #
            # In addition, wouldn't it be better to set callbacks to the
            # reference counter so that "subscribed" images could be "notified"
            # that the resource counter is about to be replaced?
            #
            # Also, there is the issue of dealing with multiple
            # copies/instances of this class: we should 'release' the refcount
            # as many times as it was held by a given instance of the ImageRef
            # class. At this moment this is not implemented.
            #
            self._hdulist_refcnt.release()

            if not self._hdulist_refcnt.closed:
                raise ValueError(
                    "An attempt was made to replace an open resource reference"
                    " counter. The existing resource counter must be closed "
                    "first before it can be replaced."
                )

        self._hdulist_refcnt = hdulist_refcnt
        if hdulist_refcnt is not None:
            self._hdulist_refcnt.hold()
            hdulist_refcnt.subscribe_close_notify(
                self, '_refcnt_about_to_close'
            )

    @property
    def closed(self):
        """
        Is the :py:class:`ImageRef` object closed?

        """
        # if self._hdulist_refcnt.resource is not None:
        #     return self._hdulist_refcnt.resource._HDUList__file.closed
        # else:
        #     return True
        return self._hdulist_refcnt is None or self._hdulist_refcnt.closed

    def release(self):
        """
        Decrement reference count of the attached resource. If the reference
        count reaches zero, the attached :py:class:`ResourceRefCount` object
        will be closed and set to `None`.

        """
        if self._hdulist_refcnt is not None:
            self._hdulist_refcnt.release()

            # check for self._hdulist_refcnt is not None below is
            # necessary as the callback (if set) may set
            # self._hdulist_refcnt to None
            if self._hdulist_refcnt is not None and \
               self._hdulist_refcnt.refcount <= 0:
                self.can_reload_data = False
                self.filename = None
                self._hdulist_refcnt = None

    def hold(self):
        """
        Increment the reference count of the attached resource.

        """
        if self._hdulist_refcnt is not None:
            self._hdulist_refcnt.hold()

    @property
    def hdu(self):
        """
        `astropy.io.fits.HDUList` of the attached image.

        """
        if self._hdulist_refcnt is not None:
            return self._hdulist_refcnt.resource
        else:
            return None

    def info(self, fh=sys.stdout):
        """
        info(self, fh=sys.stdout)
        Print a summary of the object attributes.

        """
        fh.write("Loaded file name:     '{}'\n".format(self.filename))
        fh.write("Original file name:   '{}'\n".format(self.original_fname))
        fh.write("Original file type:   '{}'\n".format(self.original_ftype))
        fh.write("Original file exists: '{}'\n".format(self.original_exists))
        fh.write("MEF file name:        '{}'\n".format(self.mef_fname))
        fh.write("MEF file exists:      '{}'\n".format(self.mef_exists))
        fh.write("Can reload data:      '{}'\n".format(self.can_reload_data))
        fh.write("EXTNAME:    '{}'\n".format(self.extname))
        fh.write("DQ model:   '{}'\n".format(self.DQ_model))
        fh.write("Telescope:  '{}'\n".format(self.telescope))
        fh.write("Instrument: '{}'\n".format(self.instrument))
        fh.write("File open mode:     '{}'\n".format(self.fmode))
        fh.write("File memory mapped: '{}'\n".format(self.memmap))
        fh.write("Reference counter object: '{}'\n"
                 .format(self._hdulist_refcnt))
        fh.write("Reference counter value:  '{}'\n".format(self.refcount))


def _extract_instr_info(header):
    tel = None
    instr = None
    ftype = None
    if 'TELESCOP' in header:
        tel = header['TELESCOP'].strip().upper()
    if 'INSTRUME' in header:
        instr = header['INSTRUME'].strip().upper()
    if 'FILETYPE' in header:
        ftype = header['FILETYPE'].strip().upper()
    return (tel, instr, ftype)


def basicGEIScheck(fname):
    """
    Perform a *quick* but *very basic* check if the input file
    possibly might be a GEIS file.

    Parameters
    ----------
    fname: str, file object

    Returns
    -------
    bool
        `True` if the file appears to be a GEIS file, and `False` otherwise.

    """
    if isinstance(fname, str):
        # if fname is a string - open the file
        fh = open(fname, 'rb')
        closefh = True
    else:
        # assume it is a file handle:
        fh = fname
        closefh = False

    cardLen = fits.Card.length
    readbytes = cardLen + 1

    try:
        # process first "card":
        card = fh.read(readbytes)
        if not isinstance(card, str):
            try:
                card = card.decode(encoding='ascii')
            except UnicodeDecodeError:
                if closefh:
                    fh.close()
                return False
        if len(card) != readbytes or card[-1] != '\n' or \
           card[:29].upper() != 'SIMPLE  =                    ':
            if closefh:
                fh.close()
            return False

        endcard_found = False
        while True:
            card = fh.read(readbytes)
            if not isinstance(card, str):
                try:
                    card = card.decode(encoding='ascii')
                except UnicodeDecodeError:
                    break
            cardn = len(card)
            if (cardn == cardLen or cardn == readbytes) and \
               card[:cardLen].upper() == 'END' + (cardLen - 3) * ' ':
                # we have reached the 'END' card
                endcard_found = True
                break
            if cardn != readbytes:
                # card does not have the standard length
                break
            if card[-1] != '\n':
                # card does not end with a new line character
                break
            # # EXTRA CHECK (these errors would make fits.verify to fail):
            # if card[0].isspace():
            #     if card[:8] != '        ':
            #         break
            # elif (card[:8].upper() not in ['COMMENT ', 'HISTORY '] and
            #       card[8:10] != '= '):
            #     break
    finally:
        if closefh:
            fh.close()

    return endcard_found


def basicFITScheck(fname):
    """
    Perform a *quick* but *very basic* check if the input file
    possibly might be a FITS file.

    Parameters
    ----------
    fname: str, file object

    Returns
    -------
    bool
        `True` if the file appears to be a FITS file, and `False` otherwise.

    """
    if isinstance(fname, str):
        # if fname is a string - open the file
        fh = open(fname, 'rb')
        closefh = True
    else:
        # assume it is a file handle:
        fh = fname
        closefh = False

    cardLen = fits.Card.length

    try:
        # process first "card":
        card = fh.read(cardLen)
        if not isinstance(card, str):
            try:
                card = card.decode(encoding='ascii')
            except UnicodeDecodeError:
                if closefh:
                    fh.close()
                return False

        if len(card) != cardLen or \
           card[:29].upper() != 'SIMPLE  =                    ':
            if closefh:
                fh.close()
            return False

        # process second "card":
        card = fh.read(cardLen)
        if len(card) != cardLen:
            if closefh:
                fh.close()
            return False
        # test whether card is a byte or a string
        # Required to support Python3 binary file operations vs
        #  Python2 string file results
        if not isinstance(card, str):
            try:
                card = card.decode(encoding='ascii')
            except UnicodeDecodeError:
                if closefh:
                    fh.close()
                return False
        if len(card) != cardLen or \
           card[:26].upper() != 'BITPIX  =                 ':
            if closefh:
                fh.close()
            return False

    except Exception:
        return False

    finally:
        if closefh:
            fh.close()

    return True


def _geis2FITSName(rootname, geisext):
    return rootname + '_' + geisext[1:] + os.extsep + 'fits'


# def _waiver2FITSName(rootname, waiverext):
#     if len(rootname) > 0:
#         ch = 'h' if rootname[-1].islower() else 'H'
#         return rootname[:-1] + ch + waiverext
#     else:
#         return rootname[:-1] + waiverext


def _getExtname(hdulist):
    if len(hdulist) < 2:
        return 'PRIMARY'
    if 'EXTNAME' in hdulist[1].header:
        return hdulist[1].header['EXTNAME']
    else:
        return None


def _openHDU(imageRef, doOpen, preferMEF, rc_orig, fmode, memmap):
    if doOpen:
        assert(rc_orig.refcount > 0)
        if imageRef.original_ftype in ['MEF', 'SIMPLE']:
            rc_new = rc_orig
            rc_new.hold()
            imageRef.extname = _getExtname(rc_orig.resource)
            imageRef.filename = imageRef.mef_fname
            imageRef.can_reload_data = True
            imageRef.memmap = memmap
            imageRef.fmode = fmode

        else:
            if preferMEF and imageRef.mef_exists:
                rc_new = ResourceRefCount(None)
                try:
                    hdulist = fits.open(imageRef.mef_fname, mode=fmode,
                                        memmap=memmap)
                    rc_new = ResourceRefCount(hdulist)
                    rc_new.hold()
                    imageRef.extname = _getExtname(hdulist)
                    imageRef.filename = imageRef.mef_fname
                    imageRef.can_reload_data = True
                    imageRef.memmap = memmap
                    imageRef.fmode = fmode

                except Exception:
                    imageRef.can_reload_data = False
                    if not rc_new.closed:
                        rc_new.release()
                    rc_new = rc_orig
                    rc_new.hold()
            else:
                rc_new = rc_orig
                rc_new.hold()
    else:
        rc_new = ResourceRefCount(None)
        imageRef.filename = None
        imageRef.extname = None
        imageRef.can_reload_data = False

    return rc_new


def _getDQHDUList(dqimage, dqfile, doOpen, ftype, basicChk, openOrigFn,
                  kw_openOrigFn, verbose):
    dqimage.original_exists = path.isfile(dqfile)
    dqimage.original_fname = dqfile
    dqimage.original_ftype = ftype
    dqimage.mef_exists = dqimage.mef_fname is not None \
        and path.isfile(dqimage.mef_fname)

    if doOpen and not dqimage.original_exists and verbose:
        print("Data quality file \'{}\' not found.".format(dqfile))
        dqimage.filename = None
        return None

    if dqimage.original_exists and doOpen:
        try:
            if basicChk(dqfile):
                dqhdulist = openOrigFn(dqfile, **kw_openOrigFn)
                dqhdr = dqhdulist[0].header

                dqimage.filename = dqfile
                dqimage.original_exists = True
                dqimage.extname = _getExtname(dqhdulist)

                if 'FILETYPE' in dqhdr and \
                   dqhdr['FILETYPE'].strip().upper() == 'SDQ':
                    return dqhdulist
                else:
                    dqhdulist.close()
                    if verbose:
                        print("Unsupported \'FILETYPE\' value in "
                              "data quality file \'{}\'".format(dqfile))
        except Exception:
            if verbose:
                print("Could not read data quality file "
                      "\'{}\'".format(dqfile))

        dqimage.original_exists = False
        dqimage.original_ftype = 'UNKNOWN'

    dqimage.filename = None
    return None


def openImageEx(filename, mode='readonly', dqmode='readonly', memmap=True,
                saveAsMEF=True, output_base_fitsname=None, clobber=True,
                imageOnly=False, openImageHDU=True, openDQHDU=False,
                preferMEF=True, verbose=False):
    """
    Open an image file and (if requested) the associated DQ file and return
    corresponding :py:class:`ImageRef` objects.

    This function is an enhanced version of
    :py:func:`stsci.tools.fileutil.openImage` function in that it can open
    both the image file and the associated DQ image. It also provides
    additional inormation about the opened files: file type, original file
    name, DQ model ("intrinsic", where DQ data are placed in the same file
    as the science data, or "extrinsic" when DQ data are in a separate file
    from the science data), etc. Because of the way it was implemented, it
    requires half of the number of calls to `astropy.io.fits.open` thus making
    it almost twice as fast as the :py:func:`~stsci.tools.fileutil.openImage`
    function.

    Parameters
    ----------
    filename: str
        File name of the file to be opened. The image file formats are
        recognized: simple/MEF FITS, HST GEIS format, or WAIVER FITS format.

    mode: str, optional
        File mode used to open main image FITS file. See `astropy.io.fits.open`
        for more details.

    dqmode: str, optional
        File mode used to open DQ image FITS file. See parameter ``mode``
        above for more details.

    memmap: bool, optional
        Should memory mapping to be used whe opening simple/MEF FITS?

    saveAsMEF: bool, optional
        Should an input GEIS or WAIVER FITS be converted to simple/MEF FITS?

    output_base_fitsname: str, None, optional
        The base name of the output simple/MEF FITS when ``saveAsMEF`` is
        `True`. If it is `None`, the file name of the converted file
        will be determined according to HST file naming conventions.

    clobber: bool, optional
        If ``saveAsMEF`` is `True`, should any existing output files be
        overwritten?

    imageOnly: bool, optional
        Should this function open the image file only? If `True`, then
        the DQ-related attributes will not be valid.

    openImageHDU: bool, optional
        Indicates whether the returned :py:class:`ImageRef` object
        corresponding to the science image file should be
        in an open or closed state.

    openDQHDU: bool, optional
        Indicates whether the returned :py:class:`ImageRef` object
        corresponding to the DQ image file should be in an open or closed
        state.

    preferMEF: bool, optional
        Should this function open an existing MEF file that complies with
        HST naming convention when the input file is in GEIS or WAIVER FITS
        format, even when ``saveAsMEF`` is `False` or ``clobber`` is `False`?

    verbose: bool, optional
        If `True`, some addition information will be printed out to
        standard output.

    Returns
    -------
    (img, dqimg): (ImageRef, ImageRef)
        A tuple of :py:class:`ImageRef` objects corresponding to the science
        image and to the DQ image. Each object in the returned tuple open
        or closed depending on the input arguments.

        .. note::
            If the returned object is *open*, then its reference count will
            be at least 1. The caller is responsible for "releasing" the
            object when it is no longer needed.

        .. note::
            If the DQ model of the opened file is "intrinsic", then the
            `dqimg` component of the returned tuple will hold a reference
            counter to the same image. Thus, for "intrinsic" DQ data models,
            the reference count of the returned objects may be 2 (if both
            science ``openImageHDU`` and ``openDQHDU`` are `True`).

    Raises
    ------
    ValueError
        If input file is neither a GEIS file nor a FITS file.

    ValueError
        Errors occured while accessing/reading the file possibly due to
        corrupted file, non-compliant file format, etc.s

    """
    # Initialize output dictionary:
    sci_image = ImageRef()
    dq_image = ImageRef()

    # Image file:
    hdulist = None

    # Data Quality (DQ) file (valid only if 'associated_DQ_file' is True):
    dqhdulist = None

    # Extra Info:
    telescope = 'UNKNOWN'
    instrument = 'UNKNOWN'

    # supported GEISS/WAIVER instruments (for DQ detection):
    supported_extern_DQ = ['WFPC2', 'WFPC', 'FOC', 'HRS']

    # Insure that the filename is always fully expanded
    # This will not affect filenames without paths or
    # filenames specified with extensions.
    filename = fileutil.osfn(filename)

    # Extract the rootname and extension specification
    # from input image name
    _fname, _iextn = fileutil.parseFilename(filename)
    sci_image.original_fname = _fname

    (_root, fext) = path.splitext(_fname)
    if output_base_fitsname is None:
        _outroot = _root
    else:
        fpath, oldbase = path.split(_root)
        _outroot = path.join(fpath, output_base_fitsname)

    # readgeis may hang on incorrect (non-GEIS) files. To avoid this,
    # do a basic GEIS file check:
    if len(fext) == 4 and fext[-1].lower() == 'h' and basicGEIScheck(_fname):
        # input file might be in GEIS format so try to read it:
        hdulist = readgeis.readgeis(_fname)
        ftype = 'GEIS'
        convert2MEF = saveAsMEF

        sci_image.original_exists = True
        sci_image.original_ftype = ftype
        sci_image.filename = _fname
        sci_image.mef_fname = _geis2FITSName(_outroot, fext)
        sci_image.mef_exists = path.isfile(sci_image.mef_fname)
        sci_image.extname = _getExtname(hdulist)
        sci_image.fmode = 'readonly'
        sci_image.memmap = False

        (telescope, instrument, hstftype) = _extract_instr_info(
            hdulist[0].header
        )

        has_DQ = (telescope == 'HST' and hstftype == 'SCI' and
                  instrument in supported_extern_DQ and fext[-2] == '0')

        if has_DQ:
            sci_image.DQ_model = 'external'

            _dqfext = fext[:2] + '1' + fext[-1]
            _dqfname = _root + _dqfext
            dq_image.mef_fname = _geis2FITSName(_outroot, _dqfext)

            dqhdulist = _getDQHDUList(
                dq_image, _dqfname, not imageOnly, ftype, basicGEIScheck,
                readgeis.readgeis, {}, verbose
            )
        else:
            sci_image.DQ_model = 'intrinsic'

    else:
        # Assume the input file is a FITS file
        if not basicFITScheck(_fname):
            raise ValueError(
                "Input file \'{}\' is neither a GEIS file nor a FITS file."
                .format(_fname)
            )

        hdulist = fits.open(_fname, mode=mode, memmap=memmap)

        hdr = hdulist[0].header
        (telescope, instrument, hstftype) = _extract_instr_info(hdr)

        #TODO: the check for existance of data in the primary header should be
        # replaced with a more efficient check of NAXIS, GCOUNT, etc. as it
        # is not very efficient to load data just to check that data exist.
        data0 = hdulist[0].data
        if data0 is None:
            if len(hdulist) > 1:
                ftype = 'MEF'
            else:
                raise ValueError(
                    "Input file \'{}\' has either an unsupported FITS format, "
                    "contains no image data, or is from an unsupported "
                    "instrument.".format(_fname)
                )
        else:
            if len(hdulist) > 1 and isinstance(hdulist[1], fits.TableHDU):
                ftype = 'WAIVER'
                if (not (telescope == 'HST' and
                         instrument in supported_extern_DQ) and verbose):
                    print("Input file \'{}\' appears to have \'WAIVER\'-FITS "
                          "like structure but originates from an unsupported "
                          "instrument.".format(_fname))
            else:
                ftype = 'SIMPLE'

        if ftype in ['MEF', 'SIMPLE']:

            # STANDARD SIMPLE OR MEF FITS FILE:
            convert2MEF = False

            sci_image.original_exists = True
            sci_image.original_ftype = ftype
            sci_image.filename = _fname
            sci_image.mef_fname = _fname
            sci_image.mef_exists = True
            sci_image.extname = _getExtname(hdulist)

            has_DQ = (telescope == 'HST' and
                      instrument in supported_extern_DQ and
                      hstftype == 'SCI' and len(_root) > 2 and
                      _root[-2] == '0')

            if has_DQ:
                sci_image.DQ_model = 'external'

                _dqfname = _root[:-2] + '1' + _root[-1] + fext
                dq_image.mef_fname = _dqfname
                dq_image.fmode = dqmode

                dqhdulist = _getDQHDUList(
                    dq_image,
                    _dqfname,
                    not imageOnly,
                    ftype,
                    (lambda x: True),
                    fits.open,
                    {'mode': dqmode, 'memmap': memmap},
                    verbose
                )

                if dqhdulist is None:
                    dq_image.mef_exists = False
            else:
                sci_image.DQ_model = 'intrinsic'

        else:
            # "WAIVER"-FITS FORMAT:
            convert2MEF = saveAsMEF

            if len(_root) == 0:
                ch = ''
            else:
                ch = 'h' if _root[-1].islower() else 'H'

            whdulist = hdulist

            try:
                hdulist = convertwaiveredfits.convertwaiveredfits(whdulist)

            except Exception as e:
                hdulist = None
                raise e

            finally:
                whdulist.close()
                del whdulist

            sci_image.original_exists = True
            sci_image.original_ftype = ftype
            sci_image.filename = _fname
            sci_image.mef_fname = _root[:-1] + ch + fext
            sci_image.mef_exists = path.isfile(sci_image.mef_fname)
            sci_image.extname = _getExtname(hdulist)
            sci_image.fmode = 'readonly'
            sci_image.memmap = False

            telescope, instrument, hstftype = _extract_instr_info(
                hdulist[0].header
            )

            has_DQ = (telescope == 'HST' and hstftype == 'SCI' and
                      instrument in supported_extern_DQ and
                      len(_root) > 2 and _root[-2] == '0')

            if has_DQ:
                sci_image.DQ_model = 'external'

                _dqfname = _root[:-2] + '1' + _root[-1] + fext
                dq_image.mef_fname = _outroot[:-2] + '1' + ch + fext

                dqhdulist = _getDQHDUList(
                    dq_image,
                    _dqfname,
                    not imageOnly,
                    ftype,
                    basicFITScheck,
                    convertwaiveredfits.convertwaiveredfits,
                    {},
                    verbose
                )

            else:
                sci_image.DQ_model = 'intrinsic'

    sci_image.telescope = telescope
    sci_image.instrument = instrument
    sci_image.ftype = hstftype

    dq_image.telescope = telescope
    dq_image.instrument = instrument
    dq_image.ftype = hstftype

    if convert2MEF:
        # convert GEIS or WAIVER to MEF:
        if (sci_image.mef_exists and clobber) or not sci_image.mef_exists:
            if verbose:
                print("Writing out {} as MEF to \'{}\'"
                      .format(ftype, sci_image.mef_fname))
            hdulist.writeto(sci_image.mef_fname, overwrite=clobber)
            sci_image.mef_exists = True

        if dq_image.original_exists and not imageOnly and \
           ((dq_image.mef_exists and clobber) or not dq_image.mef_exists):
            if verbose:
                print("Writing out DQ {} as MEF to \'{}\'"
                      .format(ftype, dq_image.mef_fname))
            dqhdulist.writeto(dq_image.mef_fname, overwrite=clobber)
            dq_image.mef_exists = True

    rcim_orig = ResourceRefCount(hdulist)
    rcim_orig.hold()
    if sci_image.DQ_model == 'intrinsic':
        rcdq_orig = rcim_orig
        rcdq_orig.hold()
        fmode = mode if openImageHDU else dqmode
        if openImageHDU or (openDQHDU and not imageOnly):
            rcim = _openHDU(sci_image, True, preferMEF, rcim_orig,
                            fmode, memmap)
            dq_image = copy(sci_image)
            if openDQHDU:
                rcdq = rcim
                rcdq.hold()
            else:
                rcdq = _openHDU(dq_image, False, preferMEF, rcdq_orig, fmode,
                                memmap)
            if not openImageHDU:
                rcim.release()
                rcim = _openHDU(sci_image, False, preferMEF, rcim_orig, mode,
                                memmap)
        else:
            dq_image = copy(sci_image)
            rcim = _openHDU(sci_image, False, preferMEF, rcim_orig, mode,
                            memmap)
            rcdq = _openHDU(dq_image, False, preferMEF, rcdq_orig, fmode,
                            memmap)

        # try to "guess" DQ extension name
        if count_extensions(hdulist, 'DQ') > 0:
            dq_image.extname = 'DQ'
        else:
            dq_image.extname = None

    else:
        rcdq_orig = ResourceRefCount(dqhdulist)
        rcdq_orig.hold()
        rcim = _openHDU(
            sci_image,
            openImageHDU,
            preferMEF,
            rcim_orig,
            mode,
            memmap
        )
        rcdq = _openHDU(
            dq_image,
            openDQHDU and not imageOnly and not rcdq_orig.closed,
            preferMEF,
            rcdq_orig,
            dqmode,
            memmap
        )
        dq_image.fmode = dqmode

    sci_image.set_HDUList_RefCount(None if rcim.closed else rcim)
    dq_image.set_HDUList_RefCount(None if rcdq.closed else rcdq)
    dq_image.DQ_model = None

    rcim_orig.release()
    rcdq_orig.release()
    rcim.release()
    rcdq.release()

    return (sci_image, dq_image)


def almost_equal(arr1, arr2, fp_accuracy=None, fp_precision=None):
    r"""
    Compares two values, or values of `numpy.ndarray` and verifies that these
    values are close to each other. For exact type (integers and boolean) the
    comparison is exact. For inexact types (`float`, `numpy.float32`, etc.)
    it checks that the values (or *all* values in a `numpy.ndarray`)
    satisfy the following inequality:

    .. math::
        |v1-v2| \leq a + 10^{-p} \cdot |v2|,

    where ``a`` is the accuracy ("absolute error") and ``p`` is precision
    ("relative error").

    Parameters
    ----------
    arr1: float, int, bool, str, numpy.ndarray, None, etc.
        First value or array of values to be compared.

    arr2: float, int, bool, str, numpy.ndarray, None, etc.
        Second value or array of values to be compared.

    fp_accuracy: int, float, None, optional
        Accuracy to withing values should be close. Default value will use
        twice the value of the machine accuracy (machine epsilon) for
        the input type. This parameter has effect only when the values
        to be compared are of inexact type (e.g., `float`).

    fp_precision: int, float, None, optional
        Accuracy to withing values should be close. Default value will use
        twice the value of the machine precision (resolution) for the
        input type. This parameter has effect only when the values
        to be compared are of inexact type (e.g., `float`).

    Returns
    -------
    almost_equal: bool
        Returns `True` if input values are close enough to each other
        or `False` otherwise.

    Raises
    ------
    ValueError
        If `fp_accuracy` is negative.

    TypeError
        If `fp_precision` is not `int`, `float`, or `None`.

    """
    if fp_accuracy is not None:
        if not (isinstance(fp_accuracy, float) or
                isinstance(fp_accuracy, int)) \
           or ((isinstance(fp_accuracy, float) or
                isinstance(fp_accuracy, int)) and fp_accuracy < 0):
            raise ValueError("Accuracy must be either a non-negative "
                             "number or None")
    if fp_precision is not None and not (isinstance(fp_precision, float) or
                                         isinstance(fp_precision, int)):
        raise ValueError("Precision must be either None, an integer, or "
                         "a floating-point number")

    a1 = np.asarray(arr1)
    a2 = np.asarray(arr2)

    if a1.shape != a2.shape:
        return False

    t1 = a1.dtype.type
    t2 = a2.dtype.type

    if issubclass(t1, np.inexact):
        fi = np.finfo(t1)
        acc = 2.0 * fi.eps if fp_accuracy is None else fp_accuracy
        prec = 2.0 * fi.resolution if fp_precision is None \
            else 10**(-fp_precision)
    elif issubclass(t1, np.integer) or issubclass(t1, np.bool_):
        acc = 0
        prec = 0
    else:
        acc = None
        prec = None

    if issubclass(t2, np.inexact):
        if acc is None:
            return False
        fi = np.finfo(t2)
        if fp_accuracy is None:
            acc = max(2.0 * fi.eps, acc)
        if fp_precision is None:
            prec = max(2.0 * fi.resolution, acc)
    elif (issubclass(t2, np.integer) or issubclass(t2, np.bool_)):
        if acc is None:
            return False
        acc = 0
        prec = 0

    if acc is None or (acc == 0 and prec == 0):
        return np.all(a1 == a2)
    else:
        return np.all(np.abs(a1 - a2) <= acc + prec * np.abs(a2))


def skyval2txt(files='*_flt.fits', skyfile='skyfile.txt', skykwd='SKYUSER',
               default_ext=('SCI', '*')):
    """
    A convenience function that allows retrieving computed sky background
    values from image headers and storing them in a text file that can be
    re-used by ``drizzlepac.astrodrizzle.AstroDrizzle()``. This is
    particularly useful when performing sky *matching* on a large number of
    images which takes considerable time. Saving computed sky values to a text
    file allows re-running ``AstroDrizzle()`` without re-computing sky values.

    .. warning::
       The file specified by ``skyfile`` is overwritten without warning.

    .. note::
       Images that do not have specified extensions will be ignored.

    Parameters
    ----------
    files: str
        File name(s), including extension specification if necessary,
        from which sky values should be retrieved:

            * a comma-separated list of valid science image file names
              (see note below) and (optionally) extension specifications,
              e.g.: ``'j1234567q_flt.fits[1], j1234568q_flt.fits[sci,2]'``;

            * an @-file name, e.g., ``'@fits_files_with_skyvals.txt'``.

        .. note::
            **Valid** **science** **image** **file** **names** are:

            * file names of existing FITS, GEIS, or WAIVER FITS files;

            * partial file names containing wildcard characters, e.g.,
              ``'*_flt.fits'``;

            * Association (ASN) tables (must have ``_asn``, or ``_asc``
              suffix), e.g., ``'j12345670_asn.fits'``.

        .. warning::
            @-file names **MAY** **NOT** be followed by an extension
            specification.

        .. warning::
            If an association table or a partial file name with wildcard
            characters is followed by an extension specification, it will be
            considered that this extension specification applies to **each**
            file name in the association table or **each** file name
            obtained after wildcard expansion of the partial file name.

    skyfile: str, optional
        Output "skyfile" to which sky values from the image headers should be
        written out.

    skykwd: str, optional
        Header keyword holding the value of the computed sky background.

    default_ext: int, tuple, optional
        Default extension to be used with image files that to not have
        an extension specified.

    """
    from .parseat import parse_cs_line

    files = parse_cs_line(files, doNotOpenDQ=True, im_fmode='readonly',
                          default_ext=default_ext)

    with open(skyfile, 'w') as f:
        for file in files:
            if len(file.fext) < 1:
                # ignore images that do not have required extension(s):
                file.release_all_images()
                continue

            f.write('{:s}'.format(file.image.filename))
            for ext in file.fext:
                skyval = file.image.hdu[ext].header.get(skykwd, 0.0)
                f.write('\t{:.16g}'.format(skyval))
            f.write('\n')

            file.release_all_images()
