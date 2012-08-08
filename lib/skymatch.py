"""Sky matching module."""
from __future__ import division, print_function

# STDLIB
import os
import sys
from datetime import datetime

# THIRD PARTY
import numpy
import pyfits
from sphere.skyline import SkyLine
from stsci.tools import parseinput

try:
    from stsci.tools import teal
except ImportError:
    teal = None

# LOCAL
from . import bresenham
from . import computeSky

__all__ = ['match4teal', 'match']
__taskname__ = 'skymatch'
__version__ = '0.5b'
__vdate__ = '08-Aug-2012'

# DEBUG - Can remove this when sphere is stable
__local_debug__ = True

def match4teal(input, skyfunc='mode', nclip=3, logfile='skymatch_log.txt'):
    """
    Teal interface for `match` because Teal cannot take
    output stream object.

    Parameters
    ----------
    input, skyfunc, nclip : See `match`

    logfile : str
        Store execution log in this file. Always append.
        If not given, print to screen instead.

    """
    # For logging
    if logfile in ('', None):
        flog = sys.stdout          # print to screen
    else:
        flog = open(logfile, 'a')  # always append

    match(input, skyfunc, nclip, flog)

def match(input, skyfunc='mode', nclip=3, flog=sys.stdout):
    """
    Standalone task to match sky in overlapping exposures.
    
    This task has these basic steps:

        #. Find the two exposures with the greatest overlap,
           with each exposure defined by the combined
           footprint of all its chips.
        #. Compute the sky for both exposures, ideally this
           would only need to be done in the region of overlap.
        #. Compute the difference in the sky values.
        #. Record that difference in the header of the exposure
           with the highest sky value as the SKYUSER keyword
           in the SCI headers. Also subtract from SCI data.
        #. Generate a footprint for that pair of exposures.
        #. Repeat the above steps for all remaining exposures
           using the newly created combined footprint as one of
           the exposures and using the sky value for this newly
           created footprint as one of the values (no need to
           recompute its sky value).

    The computation of the sky is done using the weighted mean
    of the clipped modes from all chips in the exposure.
    Alternately, user can use clipped mean or median to compute
    the sky values.

    These images could then be combined using AstroDrizzle with
    the 'skyuser' parameter set to 'SKYUSER' to generate a
    mosaic with a uniform background. Alternately, this task
    can be called from within AstroDrizzle to replace the
    current sky subtraction algorithm.

    .. warning:: Image headers are modified and data will be
        background-subtracted. Remember to back up original
        copies as desired.

    Parameters
    ----------
    input : str or list of str
        Name of FLT image(s) to be matched. The name(s) can be
        specified either as:

            * a single filename ('j1234567q_flt.fits')
            * a Python list of filenames
            * a partial filename with wildcards ('\*flt.fits')
            * filename of an ASN table ('j12345670_asn.fits')
            * an at-file ('@input')

    skyfunc : {'mean', 'median', 'mode'}
        Function for sky calculation.
        See `~skypac.computeSky`.

    nclip : int
        Number of clipping iterations for `skyfunc`.

    flog : output stream
        Can be file or stdout stream. This is designed such
        that log can be written to existing output stream
        from another Python program such as `astrodrizzle`.

    Examples
    --------
    #. This task can be used to match skies of a set of ACS
       images simply with:
   
           >>> from skypac import skymatch
           >>> skymatch.match('j*q_flt.fits')

    #. The TEAL GUI can be used to run this task using::

           --> from skypac import skymatch
           --> epar skymatch

       or from a general Python command line:

           >>> from skypac import skymatch
           >>> from stsci.tools import teal
           >>> teal.teal('skymatch')

    """
    # Time it
    runtime_begin = datetime.now()
    flog.write('***** {0} started on {1}{4}'
               'Version {2} ({3}){4}'.format(
        __taskname__, runtime_begin, __version__, __vdate__, os.linesep))

    # Parse input to get list of filenames to process
    infiles, output = parseinput.parseinput(input)
    if len(infiles) < 2:
        _print_and_close('{}: Need at least 2 images. Aborting...'.format(
            __taskname__), flog)
        return

    # Check sky function
    func2use = _pick_skyfunc(skyfunc)
    flog.write('    Using sky function {0} in {1}{3}'
               '    NCLIP = {2}{3}{3}'.format(
        func2use.__name__, func2use.__module__, nclip, os.linesep))

    # Extract skylines
    skylines = []
    for file in infiles:
        skylines.append( SkyLine(file) )

    remaining = list(skylines)

    #---------------------------------------------------------#
    # 1. Finding the two exposures with the greatest overlap, #
    #    with each exposure defined by the combined footprint #
    #    of all its chips.                                    #
    #---------------------------------------------------------#
    
    starting_pair = SkyLine.max_overlap_pair(skylines)
    if starting_pair is None:
        _print_and_close('{}: No overlapping pair. Aborting...'.format(
            __taskname__), flog)
        return

    #---------------------------------------------------------#
    # 2. Compute the sky for both exposures, ideally this     #
    #    would only need to be done in the region of overlap. #
    #---------------------------------------------------------#

    s1, s2 = starting_pair

    remaining.remove(s1)
    remaining.remove(s2)

    flog.write('    Starting pair: {}, {}{}'.format(
        s1._rough_id(), s2._rough_id(), os.linesep))

    #---------------------------------------------------------#
    # 3. Compute the difference in the sky values.            #
    #---------------------------------------------------------#

    sky1, sky2 = _calc_sky(s1, s2, func2use, nclip)
    diff_sky = numpy.abs(sky1 - sky2)

    #---------------------------------------------------------#
    # 4. Record that difference in the header of the exposure #
    #    with the highest sky value as the SKYUSER keyword in #
    #    the SCI headers. Also subtract from SCI data.        #
    #---------------------------------------------------------#

    if sky1 > sky2:
        skyline2update = s1
        skyline2zero = s2
    else:
        skyline2update = s2
        skyline2zero = s1

    _set_skyuser(skyline2update, diff_sky)
    _set_skyuser(skyline2zero, 0.0)  # Avoid Astrodrizzle crash

    flog.write('    Image 1: {0}{6}        SKY = {1:E}{6}'
               '    Image 2: {2}{6}        SKY = {3:E}{6}'
               '    Updating {4}{6}        SKYUSER = {5:E}{6}{6}'.format(
        s1._rough_id(), sky1, s2._rough_id(), sky2,
        skyline2update._rough_id(), diff_sky, os.linesep))

    #---------------------------------------------------------#
    # 5. Generate a footprint for that pair of exposures.     #
    #---------------------------------------------------------#

    mosaic = s1.add_image(s2)

    #---------------------------------------------------------#
    # 6. Repeat Steps 1-5 for all remaining exposures using   #
    #    the newly created combined footprint as one of the   #
    #    exposures and using the sky value for this newly     #
    #    created footprint as one of the values (no need to   #
    #    recompute its sky value).                            #
    #---------------------------------------------------------#

    while len(remaining) > 0:
        next_skyline, i_area = mosaic.find_max_overlap(remaining)

        if next_skyline is None:
            for r in remaining:
                flog.write('    No overlap: Excluding {}{}'.format(
                    r._rough_id(), os.linesep))
            break

        sky1, sky2 = _calc_sky(mosaic, next_skyline, func2use, nclip)
        diff_sky = sky2 - sky1

        _set_skyuser(next_skyline, diff_sky)

        flog.write('    Mosaic{4}        SKY = {0:E}{4}'
                   '    Updating {1}{4}        SKY = {2:E}{4}'
                   '        SKYUSER = {3:E}{4}'.format(
            sky1, next_skyline._rough_id(), sky2, diff_sky, os.linesep))

        try:
            new_mos = mosaic.add_image(next_skyline)
        except:
            if __local_debug__:
                flog.write('WARNING: Cannot add {} to mosaic.{}'.format(next_skyline._rough_id(), os.linesep))
            else:
                raise
        else:
            mosaic = new_mos
            flog.write('        Added to mosaic{}'.format(os.linesep))
        finally:
            remaining.remove(next_skyline)
            flog.write(os.linesep)

    # Time it
    runtime_end = datetime.now()
    flog.write('***** {} ended on {}{}'.format(
        __taskname__, runtime_end, os.linesep))
    _print_and_close('TOTAL RUN TIME: {}'.format(
        runtime_end - runtime_begin), flog)


def _print_and_close(emsg, flog):
    """Print error message and close log file."""
    flog.write(emsg + os.linesep)
    if flog is not sys.stdout:
        print(emsg)
        print('{} written'.format(flog.name))
        flog.close()

def _pick_skyfunc(choice):
    """   
    Use default unless a valid choice is provided.

    Parameters
    ----------
    choice : see `match.skyfunc`

    Returns
    -------
    Function to use for sky calculations.

    """
    if choice == 'mean':
        return computeSky.mean
    elif choice == 'median':
        return computeSky.median
    else:  # default
        return computeSky.mode

def _pixelate(arr, min, max):
    """
    Return valid pixel values within given limits.
    
    Values are rounded to nearest integer and clipped
    to given limits.

    Parameters
    ----------
    arr : float array
        Coordinates in a single dimension.

    min, max : int
        Allowed pixel limits, inclusive.

    Returns
    -------
    Rounded and clipped pixel values.

    """
    return numpy.clip(numpy.round(arr).astype('int'), min, max)

def _weighted_sky(skyline, ra, dec, skyfunc, nclip):
    """
    Calculate the weighted average of sky from individual
    chips in the given skyline within a given RA and DEC
    of a polygon.

    Parameters
    ----------
    skyline : `SkyLine` object

    ra, dec : array_like
        RA and DEC of the polygon.

    skyfunc : function
        Function for sky calculations.

    nclip : int
        Number of clipping iterations.

    Raises
    ------
    ValueError : Total weight is zero.

    Returns
    -------
    sky : float

    """
    w_sky = 0.0
    w_tot = 0.0

    for wcs in skyline._indv_mem_wcslist():
        # All pixels along intersection boundary for that chip
        sparse_x, sparse_y = wcs.all_sky2pix(ra, dec, 0)
        x, y = zip(*bresenham.lines(*zip(_pixelate(sparse_x, 0, wcs.naxis1-1),
                                         _pixelate(sparse_y, 0, wcs.naxis2-1))))
        x = numpy.array(x)
        y = numpy.array(y)

        fill_mask = numpy.zeros((wcs.naxis2, wcs.naxis1), dtype='bool')

        # Process each row in intersection boundary
        for cur_y in numpy.unique(y):
            idx = numpy.where(y == cur_y)

            # Can have odd or even number of matches near top or bottom.
            # In those cases, aliasing might occur.
            x1 = x[idx][::2]
            x2 = x[idx][1::2]
            nx = min([x1.size, x2.size])

            # Flag pixels within intersection
            for ix in xrange(nx):
                fill_mask[cur_y, x1[ix]:x2[ix]+1] = True

        # Calculate sky
        dat = pyfits.getdata(wcs.filename,
                             ext=wcs.extname)[numpy.where(fill_mask)]
        npix = dat.size
        if npix > 1:
            sky = skyfunc(dat, nclip=nclip)
            w_sky += npix * sky
            w_tot += npix

    if w_tot == 0:
        raise ValueError('_weighted_sky has invalid weight for '
                         '({}, {}, {}, {})'.format(skyline, ra, dec, nclip))
    else:
        return w_sky / w_tot

def _calc_sky(s1, s2, skyfunc, nclip):
    """
    Calculate weighted sky values and their difference in
    overlapping regions of given skylines.

    Parameters
    ----------
    s1, s2 : `SkyLine` objects

    skyfunc : function
        Function to use for sky calculations.

    nclip : int
        Number of clipping iterations.

    Returns
    -------
    sky1, sky2 : float
        Weighted sky values for `s1` and `s2`.

    """
    intersect = s1.find_intersection(s2)
    intersect_ra, intersect_dec = intersect.to_radec()

    sky1 = _weighted_sky(s1, intersect_ra, intersect_dec, skyfunc, nclip)
    sky2 = _weighted_sky(s2, intersect_ra, intersect_dec, skyfunc, nclip)

    return sky1, sky2

def _set_skyuser(skyline, value):
    """
    Set SKYUSER in SCI headers and subtract SKYUSER from
    SCI data.

    .. note:: ERR extensions are not modified even if sky
        subtraction could introduce additional errors.

    Parameters
    ----------
    skyline : `SkyLine` object
        Skyline of the image to update. Does not work if
        skyline is a product of union or intersection.

    value : float
        SKYUSER value for the image. It is the same for
        all extensions.

    """
    hdr_keyword = 'SKYUSER'
    im_name = skyline._rough_id()
    
    with pyfits.open(im_name, mode='update') as pf:
        for ext in skyline.members[0].ext:
            pf[ext].data -= value
            pf[ext].header.update(hdr_keyword, value)
            pf[ext].header.add_history('{} {:E} subtracted from image'.format(
                hdr_keyword, value))

        pf['PRIMARY'].header.add_history('{} by {} {} ({})'.format(
            hdr_keyword, __taskname__, __version__, __vdate__))


#--------------------------
# TEAL Interface functions
#--------------------------
def run(configObj):
    match4teal(configObj['input'], skyfunc=configObj['skyfunc'],
          nclip=configObj['nclip'], logfile=configObj['logfile'])

def getHelpAsString():   
    helpString = ''

    if teal:
        helpString += teal.getHelpFileAsString(__taskname__,__file__)

    if helpString.strip() == '':
        helpString += __doc__ + os.linesep + match4teal.__doc__ + os.linesep + match.__doc__

    return helpString
