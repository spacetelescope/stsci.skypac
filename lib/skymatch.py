"""Sky matching module."""
from __future__ import division, print_function

# STDLIB
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
__version__ = '0.1b'
__vdate__ = '18-Jul-2012'

# DEBUG
SKYMATCH_DEBUG = True

# Function to use for sky calculations
SKYFUNC = computeSky.mode

# Track SKYUSER assignments to reduce FITS header I/O
SKYUSER = defaultdict(float)

def match(input, skyfunc='mode', verbose=True):
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

    skyfunc : {'mode'}
        Function for sky calculation.
        See `~skymatch.computeSky`.

    verbose : bool
        Print info to screen.

    Examples
    --------
    #. This task can be used to match skies of a set of ACS
       images simply with:
   
           >>> from skymatch import skymatch
           >>> skymatch.match('j*q_flt.fits')

    #. The TEAL GUI can be used to run this task using::

           --> import skymatch
           --> epar skymatch.skymatch

       or from a general Python command line:

           >>> import skymatch
           >>> from stsci.tools import teal
           >>> teal.teal('skymatch.skymatch')

    """
    # Time it
    if verbose:
        runtime_begin = datetime.now()
        print('***** SKYMATCH started on {}'.format(runtime_begin))
        print('Version {} ({})'.format(__version__, __vdate__))

    # Parse input to get list of filenames to process
    infiles, output = parseinput.parseinput(input)
    assert len(infiles) > 1, \
           '%s: Need at least 2 images. Aborting...' % __taskname__

    # Check sky function
    _pick_skyfunc(skyfunc, infiles[0])
    if verbose:
        print('    Using sky function {}'.format(SKYFUNC))
        print()

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
    assert starting_pair is not None, \
           '%s: No overlapping pair. Aborting.' % __taskname__

    #---------------------------------------------------------#
    # 2. Compute the sky for both exposures, ideally this     #
    #    would only need to be done in the region of overlap. #
    #---------------------------------------------------------#

    s1, s2 = starting_pair

    remaining.remove(s1)
    remaining.remove(s2)

    if verbose:
        print('    Starting pair: {}, {}'.format(s1._rough_id(),
                                                 s2._rough_id()))

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
    else:
        skyline2update = s2

    _set_skyuser(skyline2update, diff_sky)

    if verbose:
        print('    Image 1: {}'.format(s1._rough_id()))
        print('        SKY = {:E}'.format(sky1))
        print('    Image 2: {}'.format(s2._rough_id()))
        print('        SKY = {:E}'.format(sky2))
        print('    Updating {}'.format(skyline2update._rough_id()))
        print('        SKYUSER = {:E}'.format(diff_sky))
        print()

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
            if verbose:
                for r in remaining:
                    print('    No overlap: Excluding {}'.format(r._rough_id()))
            break

        sky1, sky2 = _calc_sky(mosaic, next_skyline)
        diff_sky = sky2 - sky1

        _set_skyuser(next_skyline, diff_sky)

        if verbose:
            print('    Mosaic')
            print('        SKY = {:E}'.format(sky1))
            print('    Updating {}'.format(next_skyline._rough_id()))
            print('        SKY = {:E}'.format(sky2))
            print('        SKYUSER = {:E}'.format(diff_sky))

        try:
            new_mos = mosaic.add_image(next_skyline)
        except (ValueError, AssertionError):
            if SKYMATCH_DEBUG:
                print('WARNING: Cannot add {} to mosaic.'.format(next_skyline._rough_id()))
            else:
                raise
        else:
            mosaic = new_mos
            if verbose:
                print('        Added to mosaic')
        finally:
            remaining.remove(next_skyline)
            print()

    # Time it
    if verbose:
        runtime_end = datetime.now()
        print('    TOTAL RUN TIME: {}'.format(runtime_end - runtime_begin))
        print('***** SKYMATCH ended on {}'.format(runtime_end))


def _pick_skyfunc(choice, im):
    """
    *FUTURE WORK*
    
    Use default unless a valid choice is provided.

    """
    #global SKYFUNC
    #if choice == 'new_choice':
    #    SKYFUNC = some_function
    pass

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
            sky = SKYFUNC(dat - SKYUSER[wcs.filename])

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
          verbose=configObj['verbose'])

def getHelpAsString():   
    helpString = ''

    if teal:
        helpString += teal.getHelpFileAsString(__taskname__,__file__)

    if helpString.strip() == '':
        helpString += __doc__ + '\n' + match.__doc__

    return helpString
