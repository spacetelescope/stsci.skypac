"""Sky matching module."""
from __future__ import division, print_function

# STDLIB
import os
import sys
from datetime import datetime
from collections import defaultdict

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
from . import computeSky

__all__ = ['match']
__taskname__ = 'skymatch'
__version__ = '0.3b'
__vdate__ = '27-Jul-2012'

# DEBUG - Can remove this when sphere is stable
SKYMATCH_DEBUG = True

# Function to use for sky calculations
SKYFUNC = computeSky.mode
SKY_NCLIP = 0

# Track SKYUSER assignments to reduce FITS header I/O
SKYUSER = defaultdict(float)

def match(input, skyfunc='mode', nclip=3, logfile='skymatch_log.txt'):
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
           in the SCI headers.
        #. Generate a footprint for that pair of exposures.
        #. Repeat the above steps for all remaining exposures
           using the newly created combined footprint as one of
           the exposures and using the sky value for this newly
           created footprint as one of the values (no need to
           recompute its sky value).

    The computation of the sky is done using the same basic
    algorithm in AstroDrizzle; namely, the minimum value of the
    clipped modes from all chips in the exposure. Alternately,
    user can provide custom function to compute the sky values.

    These images could then be combined using AstroDrizzle with
    the 'skyuser' parameter set to 'SKYUSER' to generate a
    mosaic with a uniform background. Alternately, this task
    can be called from within AstroDrizzle to replace the
    current sky subtraction algorithm.

    .. warning:: Image headers are modified. Remember to back
        up original copies as desired.

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

    logfile : str
        Store execution log in this file. Always append.
        If not given, print to screen instead.

    Examples
    --------
    #. This task can be used to match skies of a set of ACS
       images simply with:
   
           >>> from skypac import skymatch
           >>> skymatch.match('j*q_flt.fits')

    #. The TEAL GUI can be used to run this task using::

           --> import skypac
           --> epar skypac.skymatch

       or from a general Python command line:

           >>> import skypac
           >>> from stsci.tools import teal
           >>> teal.teal('skypac.skymatch')

    """
    global SKY_NCLIP

    # For logging
    if logfile in ('', None):
        flog = sys.stdout          # print to screen
    else:
        flog = open(logfile, 'a')  # always append

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
    _pick_skyfunc(skyfunc)
    SKY_NCLIP = nclip
    flog.write('    Using sky function {0} in {1}{3}'
               '    NCLIP = {2}{3}{3}'.format(
        SKYFUNC.__name__, SKYFUNC.__module__, SKY_NCLIP, os.linesep))

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

    sky1, sky2 = _calc_sky(s1, s2)
    diff_sky = numpy.abs(sky1 - sky2)

    #---------------------------------------------------------#
    # 4. Record that difference in the header of the exposure #
    #    with the highest sky value as the SKYUSER keyword in #
    #    the SCI headers.                                     #
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

        sky1, sky2 = _calc_sky(mosaic, next_skyline)
        diff_sky = sky2 - sky1

        _set_skyuser(next_skyline, diff_sky)

        flog.write('    Mosaic{4}        SKY = {0:E}{4}'
                   '    Updating {1}{4}        SKY = {2:E}{4}'
                   '        SKYUSER = {3:E}{4}'.format(
            sky1, next_skyline._rough_id(), sky2, diff_sky, os.linesep))

        try:
            new_mos = mosaic.add_image(next_skyline)
        except (ValueError, AssertionError):
            if SKYMATCH_DEBUG:
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

    """
    global SKYFUNC
    if choice == 'mean':
        SKYFUNC = computeSky.mean
    elif choice == 'median':
        SKYFUNC = computeSky.median
    # else mode is default

def _get_min_max(arr, in_min, in_max):
    """
    Return minimum and maximum of given array bound by
    [min, max]. Values are rounded to the closest
    integer for index look-up.
    
    """
    out_min = int(round(max([arr.min(), in_min])))
    out_max = int(round(min([arr.max(), in_max])))
    return out_min, out_max

def _overlap_xy(wcs, ra, dec):
    """
    Find pixel coordinates of original image for a
    given HSTWCS that fall within polygon bound by
    given RA and DEC.

    For simplicity, RA and DEC are only used to
    determine min/max rows and columns. So limits
    returned are the minimum bounding box on the
    image of the polygon, not the actual polygon.
    If RA and DEC fall outside the image, X and Y are
    clipped to image edges.

    .. note:: A more accurate but slower algorithm
        could be implemented in the future to return
        pixels belonging to the exact polygon.

    Parameters
    ----------
    wcs : HSTWCS object
        WCS of the science extension from which to
        extract coordinates.

    ra, dec : array_like
        RA and DEC of the polygon.

    Returns
    -------
    min_x, max_x : int
        X limits, rounded to nearest integer, in Python
        format. Minimum is inclusive, maximum is exclusive.

    min_y, max_y : int
        Similar limits for Y.
    
    """
    x, y = wcs.all_sky2pix(ra, dec, 0)

    min_x, max_x = _get_min_max(x, 0, wcs.naxis1 - 1)
    min_y, max_y = _get_min_max(y, 0, wcs.naxis2 - 1)

    return min_x, max_x+1, min_y, max_y+1

def _weighted_sky(skyline, ra, dec):
    """
    Calculate the average sky weighted by the number of
    overlapped pixels with a polygon defined by the
    given RA and DEC from individual chips in a given
    skyline.

    If skyline is already assigned a SKYUSER value, this
    value is subtracted in the calculations.

    Parameters
    ----------
    skyline : `SkyLine` object

    ra, dec : array_like
        RA and DEC of the polygon.

    """
    w_sky = 0.0
    w_tot = 0.0
    
    for wcs in skyline._indv_mem_wcslist():
        min_x, max_x, min_y, max_y = _overlap_xy(wcs, ra, dec)
 
        with pyfits.open(wcs.filename) as pf:
            dat = pf[wcs.extname].data[min_y:max_y, min_x:max_x]
            sky = SKYFUNC(dat - SKYUSER[wcs.filename], nclip=SKY_NCLIP)

            npix = dat.size
            w_sky += npix * sky
            w_tot += npix

    if w_tot == 0:
        raise ValueError('_weighted_sky has invalid weight for '
                         '({}, {}, {})'.format(skyline, ra, dec))
    else:
        return w_sky / w_tot

def _calc_sky(s1, s2):
    """
    Calculate weighted sky values and their difference in
    overlapping regions of given skylines.

    Parameters
    ----------
    s1, s2 : `SkyLine` objects

    Returns
    -------
    sky1, sky2 : float
        Weighted sky values for `s1` and `s2`.
    
    """
    intersect = s1.find_intersection(s2)
    intersect_ra, intersect_dec = intersect.to_radec()

    sky1 = _weighted_sky(s1, intersect_ra, intersect_dec)
    sky2 = _weighted_sky(s2, intersect_ra, intersect_dec)

    return sky1, sky2

def _set_skyuser(skyline, value):
    """
    Set SKYUSER in image SCI headers and global dictionary.

    Parameters
    ----------
    skyline : `SkyLine` object
        Skyline of the image to update. Does not work if
        skyline is a product of union or intersection.

    value : float
        SKYUSER value for the image. It is the same for
        all extensions.

    """
    global SKYUSER
    im_name = skyline._rough_id()
    
    with pyfits.open(im_name, mode='update') as pf:
        for ext in skyline.members[0].ext:
            pf[ext].header.update('SKYUSER', value)

        pf['PRIMARY'].header.add_history('SKYUSER by {} {} ({})'.format(
            __taskname__, __version__, __vdate__))

    SKYUSER[im_name] = value


#--------------------------
# TEAL Interface functions
#--------------------------
def run(configObj):
    match(configObj['input'], skyfunc=configObj['skyfunc'],
          nclip=configObj['nclip'], logfile=configObj['logfile'])

def getHelpAsString():   
    helpString = ''

    if teal:
        helpString += teal.getHelpFileAsString(__taskname__,__file__)

    if helpString.strip() == '':
        helpString += __doc__ + '\n' + match.__doc__

    return helpString
