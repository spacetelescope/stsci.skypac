"""
Module for parsing ``@-files`` or user input strings for use
by :py:mod:`stsci.skypac` module.

:Authors: Mihai Cara

:License: :doc:`LICENSE`

"""
import os
import sys
import glob
from copy import copy, deepcopy
from .utils import MultiFileLog, ImageRef, openImageEx, get_ext_list
from stsci.tools import fileutil, parseinput, asnutil


try:
    # added to provide interpretation of environment variables
    from stsci.tools.fileutil import osfn
except ImportError:
    osfn = None

__all__ = ['parse_at_file', 'parse_at_line', 'parse_cs_line',
           'FileExtMaskInfo']


# Because os.stat is not supported on all platforms, we use STDLIB
# implemenation on posix platforms and our own implementation (through simple
# file name comparison) of os.stat on other platforms.
if os.name == 'posix':
    _sameFile = os.path.samefile
    _sameStat = os.path.samestat

    # _Stat = os.stat

    def _Stat(fn):
        # class st(object):
        #     def __init__(self, ino, dev):
        #         self.st_ino = ino
        #         self.st_dev = dev
        if fn is None:
            # return st(None, None)
            return os.stat_result(10 * (None,))
        else:
            return os.stat(fn)

else:
    # TODO: not sure if this is needed for cygwin. check this at a later time!
    def _sameFile(fn1, fn2):
        # TODO: A better way would be to use GetFileInformationByHandle as
        #       shown here:
        # http://timgolden.me.uk/python/win32_how_do_i/see_if_two_files_are_the_same_file.html
        #       or as it is implemented in Python 3.3 though _getfinalpathname
        #       (still relying on WinAPI calls). Same applies to _Stat
        #       function below.
        return (fn1.st_ino == fn2.st_ino and fn1.st_dev == fn2.st_dev)

    def _sameStat(st1, st2):
        return st1 == st2

    def _Stat(fn):
        # class st(object):
        #     def __init__(self, ino, dev):
        #         self.st_ino = ino
        #         self.st_dev = dev
        if fn is None:
            # return st(None, None)
            return os.stat_result(10 * (None,))
        else:
            fname = os.path.abspath(os.path.expanduser(fn))
            # return st(fname, 0)
            return os.stat_result((0, fname) + 8 * (0,))


class CharAccumulator(object):
    def __init__(self, allowed_qmarks=['\'', '\"']):
        self.quotes = deepcopy(allowed_qmarks)
        self.dirty = False
        self.reset()

    def __str__(self):
        if self.quoted:
            return self.interpreted_str()
        else:
            return self._buff.rstrip()

    def interpreted_str(self):
        if not self.quoted:
            s = self._buff.rstrip()
            return s if len(s) > 0 else None
        elif self.closed:
            return self._buff
        else:
            raise SyntaxError("CharAccumulator object contains a "
                              "string with an opening quotation mark but "
                              "without a corresponding closing quotation "
                              "mark (EOL).")

    def reset(self):
        # if self.dirty:
        self._buff = ''
        self._dirty = False
        self._closed = False
        self._quoted = False
        self._qmark = ''

    @property
    def length(self):
        return len(self._buff)

    @property
    def dirty(self):
        return self._dirty

    @dirty.setter
    def dirty(self, dFlag):
        self._dirty = dFlag

    @property
    def qmark(self):
        return self._qmark

    @qmark.setter
    def qmark(self, quotation_mark):
        assert(quotation_mark in self.quotes)
        self._qmark = quotation_mark
        self.dirty = True
        self.quoted = True

    @property
    def closed(self):
        return self._closed

    def close(self):
        self._closed = True
        self._qmark = ''

    @property
    def quoted(self):
        return self._quoted

    @quoted.setter
    def quoted(self, qFlag):
        self._quoted = qFlag
        if not qFlag:
            self._qmark = ''

    def append(self, ch):
        assert(len(ch) > 0)

        # recursively add multiple characters:
        if len(ch) > 1:
            for c in ch:
                self.append(c)
            return

        # do not add characters (except for white spaces which will be skipped)
        # to the string buffer if the quotation marks have been closed
        # or if the accumulator was closed externally
        if self.closed:
            if ch.isspace():
                return
            raise BufferError("An attempt was made to add a character to "
                              "a closed CharAccumulator object.")

        # trim leading white spaces:
        if not self.dirty and ch.isspace():
            return

        if ch in self.quotes:
            if not (self.quoted or self.dirty):
                self.qmark = ch
                return
            elif self.quoted and ch == self.qmark:
                self.close()
                return

        self._buff = self._buff + ch
        self.dirty = True


class ExtSpec(object):
    def __init__(self, fname='Unknown'):
        self._extstr = CharAccumulator()
        self.reset(fname)

    def _set_sigle_ext_style(self):
        assert(self._ncomp <= 1)
        self._compound_ext_delim = [None, None]
        self._ext_style = 0
        self._nmin = 1
        self._nmax = 1

    def _set_square_ext_style(self):
        assert(self._ncomp <= 2)
        self._compound_ext_delim = ['[', ']']
        self._ext_style = 1
        self._nmin = 1
        self._nmax = 2

    def _set_tuple_ext_style(self):
        assert(self._ncomp <= 2)
        self._compound_ext_delim = ['(', ')']
        self._ext_style = 2
        self._nmin = 2
        self._nmax = 2

    def reset(self, fname=None):
        if fname is not None:
            self.fname = fname
        self._ext = None
        self._extcomp = []
        self._ncomp = 0
        self._cntbraket = 0
        self._dirty = False
        self._closed = False
        self._extstr.reset()
        self._set_sigle_ext_style()  # default

    @property
    def ext_style(self):
        return self._ext_style

    @property
    def count(self):
        return self._ncomp

    @property
    def dirty(self):
        return self._dirty

    def flag_as_dirty(self):
        self._dirty = True

    @property
    def ext(self):
        return self._ext

    @property
    def closed(self):
        return self._closed

    @property
    def can_close(self):
        if not self.dirty or self.closed:
            return True
        return (self.ext_style == 0 and (not self._extstr.quoted or
                                         self._extstr.closed))

    def close(self):
        if self.closed:
            return

        self._closed = True
        self._extstr.close()
        self._extcomp.append(self._extstr.interpreted_str())
        self._ncomp += 1

        # validate extension specification and convert to one of
        # the following formats: int, str, tuple
        if self._cntbraket != 0:
            raise ValueError("Unbalanced or nested brackets "
                             "have been detected.")

        if self.count > self._nmax:
            raise ValueError("Invalid extension specifier for file "
                             "'{:s}'.".format(self.fname))

        if self.ext_style == 0:
            if self.count < self._nmin or self._extcomp[0] is None:
                raise ValueError("Extension name (or number) cannot be empty.")

            try:
                if not self._extstr.quoted:
                    self._ext = int(self._extcomp[0])
                else:
                    self._ext = self._extcomp[0]
            except ValueError:
                self._ext = self._extcomp[0]

        elif self.ext_style == 1:
            # [extname, extver], [extname], or [extnumber]
            if self.count < self._nmin or \
               (self.count == 1 and self._extcomp[0] is None):
                raise ValueError("An extension specifier cannot be empty.")

            if self.count == 2:
                if self._extcomp[0] is None:
                    raise ValueError('Extension name cannot be empty.')
                elif self._extcomp[1] is None:
                    raise ValueError('Extension version cannot be empty.')

                # make sure first component is a string and second one is
                # an int or wildcard (*):
                if not isinstance(self._extcomp[0], str):
                    raise ValueError('Extension name must be a valid string.')

                if self._extcomp[1] == '*':
                    self._ext = (self._extcomp[0], '*')
                else:
                    try:
                        self._ext = (self._extcomp[0], int(self._extcomp[1]))
                    except ValueError:
                        raise ValueError("Extension version must be either "
                                         "an integer or a wild-card ('*').")

            elif self.count == 1:
                try:
                    self._ext = int(self._extcomp[0])  # is an integer?
                except ValueError:
                    self._ext = (self._extcomp[0], 1)  # [sci] == [sci,1]

            else:
                assert(False)  # we should not get here

        elif self.ext_style == 2:  # (extname, extver)
            if self.count < self._nmin:
                raise ValueError(
                    "A tuple extension specifier must contain precisely two "
                    "components: (extname, extver)."
                )

            if self._extcomp[0] is None:
                raise ValueError('Extension name cannot be empty.')

            elif self._extcomp[1] is None:
                raise ValueError('Extension version cannot be empty.')

            # make sure first component is a string and second one is
            # an int or wildcard (*):
            if not isinstance(self._extcomp[0], str):
                raise ValueError('Extension name must be a valid string.')

            if self._extcomp[1] == '*':
                self._ext = (self._extcomp[0], '*')

            else:
                try:
                    self._ext = (self._extcomp[0], int(self._extcomp[1]))
                except ValueError:
                    raise ValueError("Extension version must be either "
                                     "an integer or a wild-card ('*').")

    def append(self, ch):
        assert(len(ch) > 0)

        # recursively add multiple characters:
        if len(ch) > 1:
            for c in ch:
                self.append(c)
            return

        # do not add characters (except for white spaces which will be skipped)
        # to the string buffer if the quotation marks have been closed
        # or if the accumulator was closed externally
        if self.closed:
            if ch.isspace():
                return
            if ch in ['(', ')', '[', ']', '{', '}']:
                raise ValueError("Misplaced, unbalanced, or nested "
                                 "brackets have been detected.")
            raise ValueError("An attempt was made to add a character to "
                             "a closed ExtSpec object. Extension "
                             "specification cannot be followed by "
                             "non-white space characters.")

        if not self.dirty:
            if ch.isspace():
                # trim leading white spaces:
                return
            else:
                # set extension style:
                self.flag_as_dirty()
                if ch == '[':
                    self._cntbraket += 1
                    self._set_square_ext_style()
                    return
                elif ch == '(':
                    self._cntbraket += 1
                    self._set_tuple_ext_style()
                    return
                else:
                    self._cntbraket = 0
                    self._set_sigle_ext_style()

        if ch == ',' and (self._extstr.closed or not self._extstr.quoted):
            if self.ext_style == 0:
                self.close()
            else:
                self._extstr.close()
                self._extcomp.append(self._extstr.interpreted_str())
                self._ncomp += 1
                self._extstr.reset()
                if self.count > self._nmax:
                    raise ValueError("Invalid extension specifier for file "
                                     "'{:s}'.".format(self.fname))

            return

        if ch == self._compound_ext_delim[1] and \
           (self._extstr.closed or not self._extstr.quoted):
            self._cntbraket -= 1
            if self._cntbraket < 0:
                raise ValueError(
                    "Unbalanced or nested brackets have been detected."
                )
            self.close()
            return

        if not self._extstr.quoted and ch in ['(', ')', '[', ']', '{', '}']:
            raise ValueError("Misplaced, unbalanced, or nested brackets "
                             "have been detected.")

        self._extstr.append(ch)
        return


class MultiExtSpec(object):
    def __init__(self, fname='Unknown', default_strextv='*'):
        self._compound_ext_delim = ['{', '}']
        self._extspec = ExtSpec(fname)
        self._defextv = default_strextv
        self.reset()

    def reset(self, fname=None):
        if fname is not None:
            self.fname = fname
        self._extspec.reset()
        self._extlist = []
        self._next = 0
        self._cntbraket = 0
        self._dirty = False
        self._closed = False

    @property
    def count(self):
        return self._next

    @property
    def dirty(self):
        return self._dirty

    def flag_as_dirty(self):
        self._dirty = True

    @property
    def ext(self):
        return self._extlist

    @property
    def closed(self):
        return self._closed

    def close(self):
        if self.closed:
            return

        self._closed = True
        if self._extspec.dirty:
            self._extspec.close()

        if self._extspec.ext is not None:
            # for a string extension: leave a str extension == '*'
            # unchanged, otherwise create a tuple with
            # (str, default extension):
            if isinstance(self._extspec.ext, str) and self._defextv and \
               not self._extspec.ext == '*':
                self._extlist.append((self._extspec.ext, self._defextv))
            else:
                self._extlist.append(self._extspec.ext)
            self._next += 1

        # validate extension specification and convert to one of
        # the following formats: int, str, tuple
        if self._cntbraket != 0:
            raise ValueError("Unbalanced or nested brackets "
                             "have been detected.")

    def append(self, ch):
        assert(len(ch) > 0)

        # recursively add multiple characters:
        if len(ch) > 1:
            for c in ch:
                self.append(c)
            return

        # do not add characters (except for white spaces which will be skipped)
        # to the string buffer if the quotation marks have been closed
        # or if the accumulator was closed externally
        if self.closed:
            if ch.isspace():
                return
            if ch in ['(', ')', '[', ']', '{', '}']:
                raise ValueError("Misplaced, unbalanced, or nested "
                                 "brackets have been detected.")
            raise ValueError("An attempt was made to add a character to "
                             "a closed MultiExtSpec object. Extension "
                             "specification cannot be followed by "
                             "non-white space characters.")

        if not self.dirty:
            if ch.isspace():
                # trim leading white spaces:
                return
            else:
                # set extension style:
                self.flag_as_dirty()
                if ch == self._compound_ext_delim[0]:
                    self._cntbraket += 1
                    return
                else:
                    raise ValueError("Multi-extension specification must "
                                     "start with left curly ('{') bracket.")

        if ch == ',' and self._extspec.can_close:
            if self._extspec.dirty:
                self._extspec.close()

            if self._extspec.ext is not None:
                # for a string extension: leave a str extension == '*'
                # unchanged, otherwise create a tuple with
                # (str, default extension):
                if isinstance(self._extspec.ext, str) and self._defextv and \
                   not self._extspec.ext == '*':
                    self._extlist.append((self._extspec.ext, self._defextv))
                else:
                    self._extlist.append(self._extspec.ext)
                self._next += 1

            self._extspec.reset()
            return

        if ch == self._compound_ext_delim[1] and self._extspec.can_close:
            self._cntbraket -= 1
            if self._cntbraket < 0:
                raise ValueError("Unbalanced or nested brackets "
                                 "have been detected.")
            self.close()
            return

        self._extspec.append(ch)
        return


class FileExtMaskInfo(object):
    """
    A class that holds image, dq, user masks, and extensions to be used with
    these files. It is designed to facilitate keeping track of user input in
    catalog files.

    This class is intended to be used primarily for functions such as
    :py:func:`parse_at_line` and other related functions as a return value.
    It is also used to initialize :py:class:`skypac.skyline.SkyLine` objects.

    `FileExtMaskInfo` was designed to be used in a specific ordered
    workflow. Below is a typical use of this class:

    #. Initialize the object with the desired settings for default
       extensions to be used with the files (when a specific extension
       for a file is not provided) and the open modes for the files;

    #. Add image file using :py:meth:`image`;

    #. Add image's extension(s) using :py:meth:`append_ext`;

    #. [Optional; can be performed at any **later** stage] Add DQ file and
       extensions using :py:meth:`DQimage` and :py:meth:`dqext` methods;

    #. Append mask files and extensions using :py:meth:`append_mask`;

    #. [Optional] Finalize the :py:meth:`FileExtMaskInfo` object.

    Parameters
    ----------
    default_ext: int, tuple, optional
        Default extension to be used with image files that to not have
        an extension specified.

    default_mask_ext: int, tuple, optional
        Default extension to be used with image mask files that to not have
        an extension specified.

    clobber: bool, optional
        If a file being appended is in GEIS or WAIVER FITS format, should
        any existing MEF files be overwritten?

    doNotOpenDQ: bool, optional
        Should the DQ files be oppened when simultaneously with the image
        files?

    fnamesOnly: bool, optional
        Return file names only, or open the files and return
        :py:class:`~skypac.utils.ImageRef` objects?

    im_fmode: str, optional
        File mode to be used to open image FITS file.
        See `astropy.io.fits.open` for more details.

    dq_fmode: str, optional
        File mode to be used to open DQ FITS file. This is valid only if the
        DQ model of the image file is 'external' (see documentation for
        :py:class:`~skypac.utils.ImageRef` for more details). For 'intrinsic'
        DQ model the DQ files will use the same setting as for ``im_fmode``.

    msk_fmode: str, optional
        File mode to be used to open mask files.

    Attributes
    ----------
    clobber: bool
        If a file being appended is in GEIS or WAIVER FITS format, should
        any existing MEF files be overwritten?

    dq_bits: int
        Bitmask specifying what pixels in the mask should be removed
        (or kept) with the precise interpretation being left to the user.
        This flag is not used by this class but was designed to be
        passed to other functions that will use :py:class:`FileExtMaskInfo`.

    """
    def __init__(self, default_ext=('SCI', '*'), default_mask_ext=0,
                 clobber=False, doNotOpenDQ=False, fnamesOnly=False,
                 im_fmode='update', dq_fmode='readonly', msk_fmode='readonly'):
        if default_ext is not None:
            self._verify_ext(default_ext)
        self._verify_ext(default_mask_ext)
        if fnamesOnly:
            self._im = None
            self._dq = None
        else:
            self._im = ImageRef(None)
            self._dq = ImageRef(None)
        self._defext = default_ext
        self._defmext = default_mask_ext
        self._fnamesOnly = fnamesOnly
        self._dontopenDQ = doNotOpenDQ
        self.clobber = clobber
        self.dq_bits = None  # can be set from "outside" if needed
        self._im_fmode = im_fmode
        self._dq_fmode = dq_fmode
        self._msk_fmode = msk_fmode
        self.clear_ext()

    def __copy__(self):
        return deepcopy(self)

    def clear_ext(self):
        # a list to hold file extensions associated with the "main" file:
        self._fext = []

        # a list to hold file extensions associated with the "DQ" file:
        self._dqext = []

        # a list to hold file extensions associated with the mask file:
        self._maskext = []

        #self._maskfname = [] # a list to hold mask file names

        # ImageRef list for mask files
        self._maskimg = []

        # file signatures (store to avoid duplicate ImageRef objects), e.g.,
        # file 'stat'
        self._filesig = []

        # number of extensions (keys in the dictonary) associated with the
        # "main" file:
        self._nfext = 0

        # number of mask files (and extensions) associated with the "main"
        # file. _nmask <= _nfext !!!
        self._nmask = 0

        self._finalized = False

    def clear_masks(self):
        """
        Remove all attached mask files and extensions.

        """
        if not self._fnamesOnly:
            for im in self._maskimg:
                if im is not None:
                    im.release()

        self._maskext = []
        self._maskimg = []
        self._filesig = self._filesig[:2]
        self._nmask = 0
        self._finalized = False

    @property
    def count(self):
        """
        Number of extensions associated with the image file.

        """
        return self._nfext

    @property
    def finalized(self):
        """
        Is the `FileExtMaskInfo` object finalized?

        """
        return self._finalized

    @property
    def fnamesOnly(self):
        """
        Was the `FileExtMaskInfo` initialized to return file names or the
        :py:class:`~skypac.utils.ImageRef` objects?

        """
        return self._fnamesOnly

    @property
    def imfstat(self):
        #"""
        #`stat` of the image file.
        #
        #"""
        if len(self._filesig) > 0 and self._filesig[0] is not None:
            return self._filesig[0]
        return None

    @property
    def original_image_fname(self):
        return 'UNKNOWN' if self._im is None else self._im.original_fname

    @property
    def image(self):
        """
        Image file name or the associated :py:class:`~skypac.utils.ImageRef`
        object (depending on the ``fnamesOnly`` value).

        :getter: Get the :py:class:`~skypac.utils.ImageRef` image object.

        :setter: Set the image file.

        :type: str, ImageRef, None

        .. note::
            Setting the image will re-initialize ``FileExtMaskInfo``. All
            previous settings will be lost and previously attached files
            will be released/deleted.

        """
        return self._im

    @image.setter
    def image(self, img):
        """
        Set the image file.

        Parameters
        ----------
        img: ImageRef, str, None
            New image file to be attached to ``FileExtMaskInfo`` object.

        .. note::
            Setting the image will re-initialize ``FileExtMaskInfo``. All
            previous settings will be lost and previously attached files
            will be released/deleted.


        """
        # NOTE: if image is None this will effectively
        # initialize the object anew

        filesig = []

        if img is None:
            if self._fnamesOnly:
                self._im = None
                self._dq = None
            else:
                self._im.release()
                self._dq.release()
                self._im = ImageRef()
                self._dq = ImageRef()

        elif isinstance(img, str):
            self._verify_fname(img)
            if self._fnamesOnly:
                self._im = img
                self._dq = None
                if os.path.isfile(img):
                    filesig.append(_Stat(img))
                else:
                    filesig.append(None)
                filesig.append(None)  # <- DQ image
            else:
                (im, dq) = openImageEx(
                    img,
                    mode=self._im_fmode,
                    dqmode=self._dq_fmode,
                    memmap=False, saveAsMEF=True,
                    output_base_fitsname=None,
                    clobber=self.clobber,
                    imageOnly=self._dontopenDQ,
                    openImageHDU=True,
                    openDQHDU=not self._dontopenDQ,
                    preferMEF=True, verbose=True
                )

                self._im.release()
                self._dq.release()
                self._im = im
                self._dq = dq

                filesig.append(_Stat(im.original_fname))
                if dq.original_fname is None or not dq.original_exists:
                    filesig.append(None)
                else:
                    filesig.append(_Stat(dq.original_fname))

        elif isinstance(img, ImageRef):
            # This is designed to hold imageObjects from astrodrizzle
            # and DQ mask will be combined with static masks and user masks
            # in astrodrizzle. This is not designed to work as a stand-alone
            # app.
            assert(not img.closed and not self._fnamesOnly)
            self._im.release()
            self._dq.release()
            self._im = img
            self._im.hold()

            # as it has been mentioned above, DQ will be dealt with in
            # astrodrizzle. However, if DQ is needed here, then look for
            # the DQ file name self._im.hdu[1].dqfile. For now,
            # provide a dummy DQ image:
            self._dq = ImageRef(None)

            filesig.append(_Stat(img.original_fname))
            filesig.append(None)  # <- DQ image

        else:
            raise TypeError('Image can be None, a valid file name (string), '
                            'or an *open* ImageRef object.')

        self.clear_ext()
        self._filesig = filesig

    @property
    def DQimage(self):
        """
        DQ image (file or :py:class:`~skypac.utils.ImageRef`
        object depending on the ``fnamesOnly`` value).

        :getter: Get the :py:class:`~skypac.utils.ImageRef` DQ image object.

        :setter: Set the DQ file.

        :type: str, ImageRef

        """
        return self._dq

    @DQimage.setter
    def DQimage(self, dq):
        """
        Set DQ image.

        """
        if self._im is None:
            raise AssertionError("DQ image cannot be set if science image "
                                 "has not been previously set.")
        if (self._fnamesOnly and isinstance(dq, str)):
            assert(isinstance(self._dq, str))  # DEBUG
            self._dq = dq
            self._dqext = []
        elif (not self._fnamesOnly and isinstance(dq, ImageRef)):
            assert(isinstance(self._dq, ImageRef))  # DEBUG
            if self._im.DQ_model is None:
                raise ValueError("Cannot set DQ image when DQ model of the "
                                 "science image is 'None'.")
            self._dq.release()
            self._dq = dq
            self._dq.hold()
            dqext = self._find_DQ_extensions()
            self._dqext = dqext
            dq_stat = _Stat(self._dq.original_fname)
            if len(self._filesig) > 1:
                self._filesig[1] = dq_stat
            elif len(self._filesig) == 1:
                self._filesig.append(dq_stat)
            else:
                raise RuntimeError("File stat list has an unexpected length."
                                   "Suspected a logical error in the code.")

        else:
            raise TypeError("Type of the DQ image does not match the "
                            "value of the 'fnamesOnly' parameter.")

    @property
    def fext(self):
        """
        FITS extensions associated with the image file.

        """
        return self._fext

    @property
    def dqext(self):
        """
        FITS extensions associated with the DQ file.

        """
        return self._dqext

    @dqext.setter
    def dqext(self, dqext):
        """
        Set DQ extensions.

        Parameters
        ----------
        dqext: list
            A list of extension specifications to be used with DQ image.
            Must be of the same length as the number of extensions
            set for the image file (see :py:attr:`count`).

        """
        if len(dqext) != self._nfext:
            raise ValueError("Length of the DQ extension list must be "
                             "equal to the length of the science image "
                             "extension list")

        # theck that there are no wildcard characters:
        for ext in dqext:
            if isinstance(ext, str) or \
               (isinstance(ext, tuple) and not isinstance(ext[1], int)):
                raise ValueError("DQ extension list may not contain "
                                 "wildcard type extensions.")

        self._dqext = deepcopy(dqext)
        return

    @property
    def maskext(self):
        """
        FITS extensions associated with the mask files.

        """
        return self._maskext

    @property
    def mask_images(self):
        """
        Mask image file names or :py:class:`~skypac.utils.ImageRef`
        object depending on the ``fnamesOnly`` value.

        """
        return self._maskimg

    @staticmethod
    def _exnumber2extnv(hdulist, extnum):
        try:
            hdr = hdulist[extnum].header
            extname = hdr['EXTNAME']
            extver = hdr['EXTVER']
            return (extname, extver)
        except Exception:
            return None

    def _find_DQ_extensions(self):
        assert(self._im is not None and self._dq is not None)
        if self.fnamesOnly or self._im.DQ_model is None:
            return []
        if self._im.extname is None or self._dq.extname is None:
            return []

        dqext = []

        for ext in self.fext:
            if isinstance(ext, int):
                if self._im.DQ_model == 'intrinsic':
                    # step 1: convert integer extensions to tuples
                    #         (extname, extver)
                    #         (necessary for DQ_model='intrinsic')
                    tuple_ext = self._exnumber2extnv(self._im.hdu, ext)
                    if tuple_ext is None:
                        raise RuntimeError("Unable to compute DQ extensions.")

                    # step 2: Replace 'SCI' extension names in tuples with 'DQ'
                    #         extension names:
                    dqext.append((self._dq.extname, tuple_ext[1]))

                else:
                    # nothing to do when ext is 'ini' and DQ_model=='extrinsic'
                    dqext.append(ext)

            else:
                # ext is a tuple:
                # Replace 'SCI' extension names in tuples with 'DQ'
                # extension names:
                dqext.append((self._dq.extname, ext[1]))

        return dqext

    @staticmethod
    def _verify_fname(fname):
        if not (fname is None or isinstance(fname, str)):
            raise ValueError("File name must be a valid string.")

    # _verify_ext returns True if an extension has wildcards and False if it
    # is already "well" defined. It will raise ValueError if the extension
    # does not have the correct format: int, or (str, int), or '*'
    @classmethod
    def _verify_ext(cls, ext):
        if isinstance(ext, int):
            return False

        if isinstance(ext, str) and ext == '*':
            return True

        if isinstance(ext, tuple) and len(ext) == 2 and \
           isinstance(ext[0], str):
            if isinstance(ext[1], int):
                return False
            elif isinstance(ext[1], str) and ext[1] == '*':
                return True

        raise ValueError("Invalid extension specification: {}.".format(ext))

    def _expand_wildcard_ext(self, hdulist, ext):
        if isinstance(ext, tuple) and ext[1] == '*':
            expext = get_ext_list(hdulist, extname=ext[0])
        elif isinstance(ext, str) and ext == '*':
            # any "image-like" ext when default_extname=None:
            expext = get_ext_list(hdulist, extname=None)
        else:
            expext = [ext]
        return expext

    def append_ext(self, ext):
        """
        *Append* extensions to the list of "selected" extensions for the
        image file.

        .. note::
            This function *appends* the extensions. If it is desired
            to *set* the extensions, use :py:meth:`replace_ext` instead.

        Parameters
        ----------
        ext: int, tuple, None, list
            Extension specification: None, an integer extension *number*,
            a tuple (*extension number*, *extension version*)
            where extension version can be ``'*'`` which will be replaced with
            the extension versions of all extensions having given extension
            name. If ext is None, it will be replaced with the default
            extension specification set during the initialization of the
            :py:class:`FileExtMaskInfo` object.

        """
        assert(not self.finalized)
        try:
            if (ext is not None and
                ((not self._fnamesOnly and self._im.hdu is None) or
                 (self._fnamesOnly and self._im is None))):
                raise RuntimeError(
                    "Cannot add extensions to a FileExtMaskInfo object that "
                    "has not been assigned a file name."
                )

            if isinstance(ext, list):
                for e in ext:
                    self.append_ext(e)
                return

            if self._fnamesOnly:
                if ext is None:
                    extlist = [self._defext]
                else:
                    self._verify_ext(ext)
                    extlist = [ext]
            else:
                if ext is None:
                    extlist = self._expand_wildcard_ext(
                        self._im.hdu, self._defext
                    )
                else:
                    self._verify_ext(ext)
                    extlist = self._expand_wildcard_ext(self._im.hdu, ext)

            self._fext += extlist
            self._nfext += len(extlist)

        except Exception as e:
            self._im.release()
            self._dq.release()
            raise e

    def replace_fext(self, ext):
        """
        Replace/set image file extension list.

        See Also
        --------
        append_ext

        """
        assert(not self.finalized)
        self._nfext = 0
        self._fext = []
        self.append_ext(ext=ext)

        if self._im.DQ_model is not None and self._dq is not None and \
           not self._dq.closed and not self._fnamesOnly:
            self._dqext = self._find_DQ_extensions()

    def append_mask(self, mask, ext, mask_stat=None):
        """
        Append a mask image and its extension(s).

        .. note::
            Mask files and extensions are kept in ordered lists and their
            order is significant: the first mask file-extension pair is
            associated with the first extension of the science image file
            set with :py:meth:`append_ext` and so on.

        Parameters
        ----------
        mask: str, ImageRef, None
            Mask image file. Can be a string file name,
            an :py:class:`~skypac.utils.ImageRef` object (*only* if
            :py:attr:`fnamesOnly` is `False`), or `None` (to act as a place
            holder in the ordered list of extensions).

        ext: int, tuple, None, list
            Extension specification: None, an integer extension *number*,
            a tuple (*extension number*, *extension version*)
            where extension version can be ``'*'`` which will be replaced with
            the extension versions of all extensions having given extension
            name. If ext is None, it will be replaced with the default
            extension specification for mask images set during
            the initialization of the :py:class:`FileExtMaskInfo` object.

        mask_stat: `os.stat_result`, optional
            An `os.stat_result` structure for the input ``mask`` file.
            If `None`, then :py:meth:`append_mask` will compute `stat` for
            the input ``mask`` file.

        Raises
        ------
        RuntimeError
            Raised if attempting to add masks when the science image
            was not yet set.

        AssertionError
            Raised if :py:attr:`finalized` is `True`.

        TypeError
            Raised if `mask` is an :py:class:`~skypac.utils.ImageRef`
            object but :py:attr:`fnamesOnly` is `True` or if ``mask``
            argument is of incorrect type.

        ValueError
            If ``mask`` is an :py:class:`~skypac.utils.ImageRef`, it must
            *not* be closed.

        """
        assert(not self.finalized)
        if (not self._fnamesOnly and self._im.hdu is None) or \
           (self._fnamesOnly and not self._im):
            raise RuntimeError("Cannot add mask and extensions to a "
                               "FileExtMaskInfo object that has not "
                               "been assigned a file name.")

        if isinstance(ext, list):
            for e in ext:
                self.append_mask(mask, e)
            return

        if mask is None or mask == '':
            self._maskimg.append(None if self._fnamesOnly else ImageRef(None))
            self._filesig.append(None)
            self._maskext.append(None)
            self._nmask += 1
            return

        if isinstance(mask, str):
            if self._fnamesOnly:
                # identification of identical files does not make sense for
                # file names only.
                self._maskimg.append(mask)
                if ext is None:
                    self._maskext.append(self._defmext)
                else:
                    # next statement raise exception if non-compliant extension
                    self._verify_ext(ext)
                    self._maskext.append(ext)
                self._nmask += 1
                return

            # However, if we are going to return
            # ImageRef objects, it will be more efficient to return
            # the same ImageRef object with accordingly increased
            # reference count. Thus we will avoid opening (a time-consuming
            # operation) the same FITS file multiple times.

            if mask_stat is None:
                stat = _Stat(mask)
            else:
                stat = deepcopy(mask_stat)
            findex = None
            for i in range(len(self._filesig)):
                if self._filesig[i] is None:
                    continue
                if _sameStat(self._filesig[i], stat):
                    findex = i
                    break
            if findex is not None:
                #findex = self._filesig.index(stat)
                if findex == 0:
                    mask = self._im
                elif findex == 1:
                    mask = self._dq
                else:
                    mask = self._maskimg[findex - 2]

                mask.hold()
                self._filesig.append(stat)
            else:
                try:
                    mask, dq = openImageEx(
                        mask, mode=self._msk_fmode, memmap=False,
                        saveAsMEF=True, clobber=self.clobber,
                        imageOnly=True, openImageHDU=True, openDQHDU=False,
                        preferMEF=True, verbose=False
                    )
                    mask.DQ_model = None
                    self._filesig.append(stat)

                except IOError:
                    raise IOError("Unable to open file: '{:s}'".format(mask))

        elif isinstance(mask, ImageRef):
            if self._fnamesOnly:
                raise TypeError(
                    "Cannot set an ImageRef mask when "
                    "FileExtMaskInfo was initialized with fnamesOnly=True."
                )

            if mask.closed or mask.original_fname is None:
                raise ValueError(
                    "ImageRef mask must not be closed and "
                    "have a valid 'original_fname' attribute."
                )

            mask.hold()
            if mask_stat is None:
                stat = _Stat(mask.original_fname)
            else:
                stat = deepcopy(mask_stat)
            self._filesig.append(stat)

        else:
            raise TypeError("Argument 'mask' can be a string file name, "
                            "ImageRef type object, None, or a list "
                            "with elements of string type or None.")

        if self._fnamesOnly:
            if ext is None:
                extlist = [self._defmext]
            else:
                self._verify_ext(ext)
                extlist = [ext]
        else:
            if ext is None:
                extlist = self._expand_wildcard_ext(mask.hdu, self._defmext)
            else:
                self._verify_ext(ext)
                extlist = self._expand_wildcard_ext(mask.hdu, ext)

        nskip = len(extlist)
        self._maskimg += nskip * [mask]
        for i in range(nskip - 1):
            mask.hold()
        self._maskext += extlist
        self._nmask += nskip

    def finalize(self, toImageRef=False):
        """
        Finalize the object by trimming or extending mask image lists to
        match the number of science image extensions.

        In principle, the number of mask files and their extensions need not
        be equal to the number of extensions specified for the science image.
        If the number of masks/extensions is smaller than the number of
        science extensions, the list of mask extensions will be appended with
        None (if :py:attr:`fnamesOnly` is `True`) or dummy
        :py:class:`~skypac.utils.ImageRef` (if :py:attr:`fnamesOnly` is
        `False`) until the number of mask extensions is equal to the number
        of science image extensions. If the number of mask extensions is
        larger than the number of science image extensions, the list
        of mask extensions will be trimmed to match the number of science
        image extensions. The trimmed out mask files (if represented by
        :py:class:`~skypac.utils.ImageRef`) will be "released".

        """
        if self.finalized:
            return

        if self._fnamesOnly:
            if toImageRef:
                self.convert2ImageRef()
            else:
                # nothing to do if only file names are requested: we cannot
                # verify or expand wildcard extensions if we do not open
                # FITS images.
                return

        # make mask (fname & ext) of the same length as the "main" image
        # extension list:
        delta = self._nfext - self._nmask
        if delta > 0:
            elem = delta * [None]
            imgelem = delta * [ImageRef()]
            self._maskimg += imgelem
            self._maskext += elem
        elif delta < 0:
            for m in self._maskimg[self._nfext:]:
                m.release()
            self._maskimg = self._maskimg[:self._nfext]
            self._maskext = self._maskext[:self._nfext]

        if self._im.DQ_model is not None and self._dq is not None and \
           not self._dq.closed and not self._fnamesOnly:
            self._dqext = self._find_DQ_extensions()

        self._nmask = self._nfext
        self._finalized = True

    def convert2ImageRef(self):
        """
        Replace any existing file names with opened
        :py:class:`~skypac.utils.ImageRef`
        objects and change the :py:attr:`fnamesOnly` property to `False`.

        .. note::
            The :py:attr:`finalized` property will not be modified.

        .. warning::
            The :py:class:`FileExtMaskInfo` must not have been finalized
            (:py:attr:`finalized` is `False`) and must contain file
            names only (:py:attr:`fnamesOnly` is `True`).

        Raises
        ------
        AssertionError
            Raised if :py:attr:`finalized` is `True` or
            :py:attr:`fnamesOnly` is `False`.

        See Also
        --------
        release_all_images

        """
        assert(not self.finalized and self._fnamesOnly)
        # create a new FileExtMaskInfo object with fnamesOnly=False
        # (this should allow it to take care of opening files and identifying
        #  identical files):
        newfi = FileExtMaskInfo(
            default_ext=self._defext,
            default_mask_ext=self._defmext,
            clobber=self.clobber,
            doNotOpenDQ=self._dontopenDQ,
            fnamesOnly=False,
            im_fmode=self._im_fmode,
            dq_fmode=self._dq_fmode,
            msk_fmode=self._msk_fmode
        )

        # populate it's file and extension lists:
        newfi.image = self._im
        newfi.append_ext(self._fext)

        for i in range(self._nmask):
            newfi.append_mask(mask=self._maskimg[i], ext=self._maskext[i])

        # copy attributes of the new ofject to self:
        self.__dict__ = copy(newfi.__dict__)
        del newfi

    def release_all_images(self):
        """
        Release all images if ``fnamesOnly`` is `False` and replace any
        existing `~skypac.utils.ImageRef` with their *original* file names.

        .. note::
            This will set the :py:attr:`fnamesOnly` property to `True` and
            the :py:attr:`finalized` property to `False.`

        See Also
        --------
        convert2ImageRef

        """
        if self._fnamesOnly:
            return

        if isinstance(self._im, ImageRef):
            self._im.release()
            self._im = self._im.original_fname

        if isinstance(self._dq, ImageRef):
            self._dq.release()
            self._dq = self._dq.original_fname

        newmasklist = []
        for m in self._maskimg:
            if isinstance(m, ImageRef):
                m.release()
                newmasklist.append(m.original_fname)
            else:
                newmasklist.append(m)
        self._maskimg = newmasklist
        self._fnamesOnly = True
        self._finalized = False

    def info(self):
        """
        Print information about the state of the object.

        """
        print("\n---  FileExtMaskInfo:  ---")
        if self._fnamesOnly:
            print("Image File name: {}".format(self._im))
            print("Image File extensions: {}".format(self.fext))
            print("DQ File name:    {}".format(self._dq))
            for i in range(self._nmask):
                print("Mask File: {}  \tExtension: {}"
                      .format(self._maskimg[i], self._maskext[i]))
        else:
            print("Image File name: {}".format(self._im.original_fname))
            print("Image File extensions: {}".format(self.fext))
            print("DQ File name:    {}".format(self._dq.original_fname))
            for i in range(self._nmask):
                print(
                    "Mask File: {}  \tExtension: {}".format(
                        self._maskimg[i].original_fname, self._maskext[i]
                    )
                )


def parse_at_line(fstring, default_ext=('SCI', '*'), default_mask_ext=0,
                  clobber=False, fnamesOnly=False, doNotOpenDQ=False,
                  im_fmode='update', dq_fmode='readonly', msk_fmode='readonly',
                  verbose=False, _main_with_nchar=False, _external_flist=None):
    r"""
    parse_at_line(fstring, default_ext=('SCI', '*'), default_mask_ext=0, \
clobber=False, fnamesOnly=False, doNotOpenDQ=False,\
im_fmode='update', dq_fmode='readonly', msk_fmode='readonly',verbose=False)

    Parse a line from a catalog file containing a science image file
    and extension specifications and optionally followed by a
    comma-separated list of mask files and extension specifications
    (or `None`).

    File names will be stripped of leading and trailing white spaces. If it
    is essential to keep these spaces, file names may be enclosed in single
    or double quotation marks. Quotation marks may also be required when file
    names contain special characters used to separate file names and
    extension specifications: ``,[]{}``

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

        ['sci'] -- equivalent to ['sci', 1]

        {'sci'} -- equivalent to {('sci',*)}

    For extensions in the science image for which no mask file is provided,
    the corresponding mask file names may be omitted (but a comma must still
    be used to show that no mask is provided in that position) or ``None`` can
    be used in place of the file name. NOTE: ``'None'`` (in quotation marks)
    will be interpreted as a file named ``'None'``.

    Some examples of possible user input:
        ``image1.fits{1,2,('sci',3)},mask1.fits,,mask3.fits[0]``

        In this case:

        ``image1.fits[1]`` is associated with ``mask1.fits[0]``;

        ``image1.fits[2]`` does not have an associated mask;

        ``image1.fits['sci',3]`` is associated with ``mask3.fits[0]``.

        -- Assume ``image2.fits`` has 4 'SCI' extensions:

        ``image2.fits{'sci'},None,,mask3.fits``

        In this case:

        ``image2.fits['sci', 1]`` and ``image2.fits['sci', 2]`` **and**
        ``image2.fits['sci', 4]`` do not have an associated mask;
        ``image2.fits['sci', 3]`` is associated with ``mask3.fits[0]``

    .. note::
        Wildcard extension version in extension specification can be
        expanded *only* when ``fnamesOnly`` is `False`.

    Parameters
    ----------
    fstring: str
        A comma-separated string describing the image file name
        and (optionally) followed by the extension specifier
        (e.g., [sci,1,2], or [sci]). The image file name may be followed
        (comma-separated) by optional mask file names (and their extension
        specifier).

        File and extension names may NOT contain leading and/or trailing
        spaces, commas, and/or square or curly brakets.

    default_ext: int, tuple, optional
        Default extension to be used with image files that to not have
        an extension specified.

    default_mask_ext: int, tuple, optional
        Default extension to be used with image mask files that to not have
        an extension specified.

    fnamesOnly: bool, optional
        Return file names only, or open the files and return
        :py:class:`~skypac.utils.ImageRef` objects?

    doNotOpenDQ: bool, optional
        Should the DQ files be oppened when simultaneously with the image
        files?

    im_fmode: str, optional
        File mode to be used to open image FITS file.
        See `astropy.io.fits.open` for more details.

    dq_fmode: str, optional
        File mode to be used to open DQ FITS file. This is valid only if the
        DQ model of the image file is 'external' (see documentation for
        :py:class:`~skypac.utils.ImageRef` for more details). For 'intrinsic'
        DQ model the DQ files will use the same setting as for ``im_fmode``.

    msk_fmode: str, optional
        File mode to be used to open mask files.

    verbose: bool, optional
        Specifies whether to print warning messages.

    Raises
    ------
    ValueError
        * Input argument 'fstring' must be a Python string.
        * Input argument 'fstring' contains either unbalanced
          or nested square brackets.
        * Extension specification must be preceeded by a valid image file name.

    Returns
    -------
    FileExtMaskInfo
        A :py:class:`FileExtMaskInfo` object.

    """
    if not isinstance(fstring, str):
        raise ValueError("Input argument 'fstring' must be a Python string.")

    # 0 - main file, 1 - main file extension
    # 2 - mask file, 3 - mask file extension:
    current_item = 0

    fname = CharAccumulator()
    sext = ExtSpec()
    mext = MultiExtSpec()
    finfo = FileExtMaskInfo(
        default_ext=default_ext, default_mask_ext=default_mask_ext,
        clobber=clobber, fnamesOnly=fnamesOnly, doNotOpenDQ=doNotOpenDQ,
        im_fmode=im_fmode, dq_fmode=dq_fmode, msk_fmode=msk_fmode
    )

    closing_bracket = ''  # noqa: F841
    current_fname = 'Unknown'
    current_ext = None
    nchar = 0
    # this will be set only if a previous image was already openned:
    dq_image = None

    avoid_duplicate_open_files = _external_flist is not None and \
        len(_external_flist) > 0 and not fnamesOnly

    for ch in fstring:
        nchar += 1

        if current_item in [0, 2]:
            if fname.closed or not fname.quoted:
                if ch == ',':
                    # done parsing a file name without extension specifiers:
                    fname.close()

                    # check if file is a 'None' (or not provided):
                    strfname = str(fname)
                    if not fname.quoted:
                        strfname = strfname.upper()
                    if strfname == 'NONE' or strfname == '':
                        strfname = None

                    if (not fname.dirty or strfname is None) and \
                       current_item == 0:
                        # "main" image cannot be None!
                        finfo.release_all_images()
                        raise ValueError("Image file name must be a vaild "
                                         "string.")
                    closing_bracket = ''  # noqa: F841

                    if current_item == 0:
                        sfname = str(fname)
                        found = None
                        if avoid_duplicate_open_files:
                            # look through the external list of FileExtMaskInfo
                            # objects to see if this file was already opened:
                            stat_fname = _Stat(sfname)
                            for f in _external_flist:
                                if f.fnamesOnly:
                                    continue
                                if _sameStat(f.imfstat, stat_fname):
                                    found = f
                                    break
                        if found is None:
                            finfo.image = sfname
                        else:
                            finfo.image = f.image
                            # we cannot set dq image yet as we do not have
                            # extensions parsed. Set it delayed.
                            dq_image = f.DQimage
                        current_fname = sfname
                        finfo.append_ext(None)
                        if _main_with_nchar:
                            fname.reset()
                            break
                        current_item = 2
                    else:
                        if strfname is None:
                            current_fname = None
                        else:
                            current_fname = str(fname)
                        finfo.append_mask(current_fname, None)

                    fname.reset()

                    # else current_item will be unchanged (<-parsing masks)
                    continue

                elif ch == '[':
                    if not fname.dirty:
                        msg = "Extension specification must be preceeded " \
                              "by a valid image file name."
                        if current_item == 2:
                            msg += " Use curly brackets to enclose " \
                                   "multiple extension specifiers."
                        finfo.release_all_images()
                        raise ValueError(msg)
                    fname.close()

                    current_fname = str(fname)

                    if current_item == 0:
                        found = None
                        if avoid_duplicate_open_files:
                            # look through the external list of FileExtMaskInfo
                            # objects to see if this file was already opened:
                            stat_fname = _Stat(current_fname)
                            for f in _external_flist:
                                if f.fnamesOnly:
                                    continue
                                if _sameStat(f.imfstat, stat_fname):
                                    found = f
                                    break
                        if found is None:
                            finfo.image = current_fname
                        else:
                            finfo.image = f.image
                            # we cannot set dq image yet as we do not have
                            # extensions parsed. Set it delayed.
                            dq_image = f.DQimage

                    closing_bracket = ']'  # noqa: F841
                    current_item = 1 if current_item == 0 else 3
                    current_ext = sext
                    current_ext.reset(fname=current_fname)
                    current_ext.append(ch)
                    continue

                elif ch == '{':
                    if not fname.dirty:
                        msg = "Extension specification must be preceeded " \
                              "by a valid image file name."
                        if current_item == 2:
                            msg += " Use curly brackets to enclose " \
                                   "multiple extension specifiers."
                        finfo.release_all_images()
                        raise ValueError(msg)
                    fname.close()
                    current_fname = str(fname)
                    if current_item == 0:
                        found = None
                        if avoid_duplicate_open_files:
                            # look through the external list of FileExtMaskInfo
                            # objects to see if this file was already opened:
                            stat_fname = _Stat(current_fname)
                            for f in _external_flist:
                                if f.fnamesOnly:
                                    continue
                                if _sameStat(f.imfstat, stat_fname):
                                    found = f
                                    break
                        if found is None:
                            finfo.image = current_fname
                        else:
                            finfo.image = f.image
                            # we cannot set dq image yet as we do not have
                            # extensions parsed. Set it delayed:
                            dq_image = f.DQimage

                    closing_bracket = '}'  # noqa: F841
                    current_item = 1 if current_item == 0 else 3
                    current_ext = mext
                    current_ext.reset(fname=current_fname)
                    current_ext.append(ch)
                    continue

                elif ch in [']', '}']:
                    finfo.release_all_images()
                    raise ValueError("Misplaced, unbalanced, or nested "
                                     "brackets have been detected.")

            fname.append(ch)

        else:
            if ch == ',' and current_ext.closed:
                # done parsing an extension specifier:
                if current_item == 1:
                    finfo.append_ext(current_ext.ext)
                    if _main_with_nchar:
                        current_ext.reset(fname='Unknown')
                        fname.reset()
                        break
                else:
                    finfo.append_mask(current_fname, current_ext.ext)

                closing_bracket = ''  # noqa: F841
                current_fname = ''
                current_item = 2
                current_ext.reset(fname='Unknown')
                fname.reset()
                continue

            current_ext.append(ch)

    if current_item == 0:
        if fname.dirty:
            fname.close()
            sfname = str(fname)
            found = None
            if avoid_duplicate_open_files:
                # look through the external list of FileExtMaskInfo
                # objects to see if this file was already opened:
                stat_fname = _Stat(sfname)
                for f in _external_flist:
                    if f.fnamesOnly:
                        continue
                    if _sameStat(f.imfstat, stat_fname):
                        found = f
                        break
            if found is None:
                finfo.image = sfname
            else:
                finfo.image = f.image
                # we cannot set dq image yet as we do not have extensions
                # parsed. Set it delayed.
                dq_image = f.DQimage
            finfo.append_ext(None)
    elif current_item == 1:
        if current_ext.dirty:
            current_ext.close()
            finfo.append_ext(current_ext.ext)
    elif current_item == 2:
        if fname.dirty:
            fname.close()
            finfo.append_mask(str(fname), None)
    elif current_item == 3:
        if current_ext.dirty:
            current_ext.close()
            finfo.append_mask(current_fname, current_ext.ext)

    if dq_image is not None:
        finfo.DQimage = dq_image
    finfo.finalize()

    if _main_with_nchar:
        return(finfo, nchar)
    else:
        return finfo


# This is a simple version of the comma-separated file list parser
# that does not allow ASN tables or wild-card characters.
# USE parse_cs_line below instead.
#
#TODO: remove after the new version is extensively verified to work as expected
#
def _parse_cs_line(csline, default_ext=('SCI', '*'), clobber=False,
                   fnamesOnly=False, doNotOpenDQ=False, im_fmode='update',
                   dq_fmode='readonly', msk_fmode='readonly', logfile=None,
                   verbose=False):
    """
    This function is similar to :py:func:`parse_at_line`, the main
    difference being the content of the input string: a list of
    comma-separated *science* image file names. No masks can be specified and
    file names must be valid (i.e., None is not allowed). Extension
    specifications are allowed and must folow the same sintax as
    described for :py:func:`parse_at_line`.

    Below we describe only differences bewtween this function and
    :py:func:`parse_at_line`.

    Parameters
    ----------
    csline: str
        Comma-separated list of valid science image file names and
        extension specifications.

    logfile: str, file, MultiFileLog, None, optional
        Specifies the log file to which the messages should be printed.
        It can be a file name, a file object, a ``MultiFileLog`` object, or
        `None`.

    Returns
    -------
    list
        Returns a list of filenames if ``fnamesOnly`` is `True` or a list of
        :py:class:`FileExtMaskInfo` objects if ``fnamesOnly`` is `False`.

    """
    # Set-up log files:
    if isinstance(logfile, MultiFileLog):
        ml = logfile
    else:
        ml = MultiFileLog(console=verbose)
        if logfile not in ('', None, sys.stdout):
            ml.add_logfile(logfile)

    csl = csline.strip()
    fi = []

    try:
        while len(csl) > 0:
            f, nch = parse_at_line(
                csl, default_ext=default_ext, default_mask_ext=0,
                clobber=clobber, fnamesOnly=fnamesOnly,
                doNotOpenDQ=doNotOpenDQ, im_fmode=im_fmode, dq_fmode=dq_fmode,
                msk_fmode=msk_fmode, verbose=verbose, _main_with_nchar=True
            )
            fi.append(f)
            csl = csl[nch:]

    except Exception as e:
        ml.error("Unable to parse input comma-separated file name list "
                 "'{0}...'\n"
                 "       Reported error: \"{1}\".\n",
                 csl[:20], sys.exc_info()[1])
        raise e

    finally:
        ml.close()

    return fi


def parse_cs_line(csline, default_ext=('SCI', '*'), clobber=False,
                  fnamesOnly=False, doNotOpenDQ=False, im_fmode='update',
                  dq_fmode='readonly', msk_fmode='readonly', logfile=None,
                  verbose=False):
    """
    This function is similar to :py:func:`parse_at_line`, the main
    difference being the content of the input string: a list of
    comma-separated *science* image file names. No masks can be specified and
    file names must be valid (i.e., `None` is not allowed). Extension
    specifications are allowed and must folow the same sintax as
    described for :py:func:`parse_at_line`.

    Below we describe only differences bewtween this function and
    :py:func:`parse_at_line`.

    Parameters
    ----------
    csline: str
        User input string that needs to be parsed containing one of the
        following:

            * a comma-separated list of valid science image file names
              (see note below) and (optionally) extension specifications,
              e.g.: ``'j1234567q_flt.fits[1], j1234568q_flt.fits[sci,2]'``;

            * an @-file name, e.g., ``'@files_to_match.txt'``.

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

    logfile: str, file, MultiFileLog, None, optional
        Specifies the log file to which the messages should be printed.
        It can be a file name, a file object, a MultiFileLog object, or None.

    Returns
    -------
    list
        Returns a list of filenames if ``fnamesOnly`` is `True` or a list of
        :py:class:`FileExtMaskInfo` objects if ``fnamesOnly`` is `False`.

    """
    # Set-up log files:
    if isinstance(logfile, MultiFileLog):
        ml = logfile
    else:
        ml = MultiFileLog(console=verbose)
        if logfile not in ('', None, sys.stdout):
            ml.add_logfile(logfile)

    csl = csline.strip()

    # First, check if user specified an @-file:
    if csl[0] == '@':
        # read the @-file and return:
        ml.logentry("Parsing input @-file '{}'.", csl[1:])
        return parse_at_file(
            csl[1:], default_ext=default_ext,
            default_mask_ext=0,
            clobber=clobber,
            fnamesOnly=fnamesOnly,
            doNotOpenDQ=doNotOpenDQ,
            match2Images=None,
            im_fmode=im_fmode,
            dq_fmode=dq_fmode,
            msk_fmode=msk_fmode,
            logfile=logfile,
            verbose=verbose
        )

    # we expect a comma separated list of files:
    ml.logentry("Parsing comma-separated list of input file names.")

    # Now, parse input line generating a new FileExtMaskInfo for each
    # comma-separated entry. This needs to be performed at first
    # with fnamesOnly=True so that we later can expand ASN tables
    # or wildcard characters.
    fi_list = []

    try:
        while len(csl) > 0:
            f, nch = parse_at_line(
                csl, default_ext=default_ext, default_mask_ext=0,
                clobber=clobber, fnamesOnly=True, doNotOpenDQ=doNotOpenDQ,
                im_fmode=im_fmode, dq_fmode=dq_fmode, msk_fmode=msk_fmode,
                verbose=verbose, _main_with_nchar=True
            )
            fi_list.append(f)
            csl = csl[nch:]

    except Exception as e:
        ml.error("Unable to parse input comma-separated file name list "
                 "'{0}...'\n       Reported error: \"{1}\".\n",
                 csl[:20], sys.exc_info()[1])
        ml.close()
        raise e

    # Now walk trough each FileExtMaskInfo object and see if the
    # file name is an ASN table or has wildcard characters and expand them:

    fullfi = []

    for fi in fi_list:

        # check if it is an ASN table:
        if parseinput.checkASN(fi.image):

            # The input is an association table
            try:
                # Open the association table
                assocdict = asnutil.readASNTable(
                    fi.image, None, prodonly=False
                )
            except Exception:
                for f in fullfi:
                    f.release_all_images()
                raise ValueError("Unable to read Association file '{}'."
                                 .format(fi.image))

            for fname in assocdict['order']:
                newfi = copy(fi)
                newfi.image = fileutil.buildRootname(fname)
                # next line can be removed if we  not want to allow extension
                # specifications for ASN tables.
                newfi.append_ext(fi.fext)
                fullfi.append(newfi)

        else:
            # expand wildcard characters (if any)
            fname = fi.image if osfn is None else osfn(fi.image)
            flist = glob.glob(fname)

            if flist:
                for fname in flist:
                    newfi = copy(fi)
                    newfi.image = fname
                    newfi.append_ext(fi.fext)
                    fullfi.append(newfi)
            else:
                # if glob did not return any files, likely 'fi' will not
                # be able to open the file. However, we will let it try
                # and throw an error if it fails:
                fullfi.append(fi)

    # finally, convert file *name*-based FileExtMaskInfo objects to
    # ImageRef-based:
    try:
        for fi in fullfi:
            if not fnamesOnly:
                fi.finalize(toImageRef=True)
    except Exception as e:
        for f in fullfi:
            f.release_all_images()
        raise e

    return fullfi


def parse_at_file(fname, default_ext=('SCI', '*'), default_mask_ext=0,
                  clobber=False, fnamesOnly=False,
                  doNotOpenDQ=False, match2Images=None,
                  im_fmode='update', dq_fmode='readonly', msk_fmode='readonly',
                  logfile=None, verbose=False):
    """
    This function is similar to :py:func:`parse_at_line`, the main
    difference being that is can parse multiple (EOL terminated) lines
    of the format described in the documentation for
    :py:func:`parse_at_line`.

    Below we describe only differences bewtween this function and
    :py:func:`parse_at_line`.

    Parameters
    ----------
    fname: str
        File name of the catalog file.

    match2Images: list of str, list of ImageRef, None, optional
        List of file names or ImageRef objects whose mask specifications
        are to be read from the catalog file. Mask specifications
        for other files in the catalog that do not match the files in the
        ``match2Images`` list will be ignored. If ``match2Images`` is `None`,
        then all files from the catalog will be read.

    logfile: str, file, MultiFileLog, None, optional
        Specifies the log file to which the messages should be printed.
        It can be a file name, a file object, a ``MultiFileLog`` object, or
        `None`.

    Returns
    -------
    list
        Returns a list of filenames if ``fnamesOnly`` is `True` or a list of
        :py:class:`FileExtMaskInfo` objects if ``fnamesOnly`` is `False`.

    """
    # Set-up log files:
    if isinstance(logfile, MultiFileLog):
        ml = logfile
    else:
        ml = MultiFileLog(console=verbose)
        if logfile not in ('', None, sys.stdout):
            ml.add_logfile(logfile)

    # compute file 'stat' if necessary for file matching
    # (we compute it here so that 'parse_at_line' will not have to do
    # this for each line)
    doMatching = match2Images is not None and isinstance(match2Images, list)

    if doMatching:
        mstat = []
        match2ImagesLen = len(match2Images)
        for m in match2Images:
            if isinstance(m, ImageRef):
                if m.closed:
                    ml.close()
                    raise TypeError(
                        "The ImageRef elements of the 'match2Images' list "
                        "must not be 'closed'."
                    )
                mstat.append(_Stat(m.original_fname))
            elif isinstance(m, str):
                mstat.append(_Stat(m))
            else:
                ml.close()
                raise TypeError("Each element of 'match2Images' argument "
                                "must be either a string file name or "
                                "an ImageRef object.")

    fi = []
    try:
        fh = open(fname)
    except IOError:
        ml.close()
        raise IOError("Unable to open \"at\" file '{}'.".format(fname))

    lines = fh.readlines()

    nl = 0
    for l in lines:
        nl += 1

        # skip empty and comment lines:
        line = l.strip()
        if not line or line[0] == '#':
            continue

        try:
            # create a FileExtMaskInfo for each entry in the @file:
            f = parse_at_line(
                line, default_ext=default_ext,
                default_mask_ext=default_mask_ext, clobber=clobber,
                fnamesOnly=fnamesOnly, doNotOpenDQ=doNotOpenDQ,
                im_fmode=im_fmode, dq_fmode=dq_fmode, msk_fmode=msk_fmode,
                verbose=verbose, _main_with_nchar=False, _external_flist=fi
            )

            if doMatching:
                indx = None
                # find a matching image:
                for i in range(match2ImagesLen):
                    if _sameStat(f.imfstat, mstat[i]):
                        indx = i
                        break

                if indx is None:
                    # no matching image found... ignore
                    if f.fnamesOnly:
                        img_fname = f.image
                    else:
                        img_fname = f.image.original_fname
                        f.release_all_images()
                    ml.logentry("No matching image was found for the catalog "
                                "entry '{}'. This entry will be ignored.",
                                img_fname)
                    f.release_all_images()
                else:
                    # a match was found. append a tuple of the file info &
                    # index of the found image
                    fi.append((f, indx))
            else:
                fi.append(f)

        except Exception:
            ml.error("Unable to parse line #{:d}: '{}'...\n"
                     "       Reported error: \"{}\".\n"
                     "       This line will be ignored.",
                     nl, l[:20], sys.exc_info()[1])

    fh.close()
    ml.close()

    return fi
