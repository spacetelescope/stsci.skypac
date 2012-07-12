"""Sky matching module."""
from __future__ import division, print_function

# STDLIB
from datetime import datetime
from inspect import isfunction
from collections import defaultdict

# THIRD PARTY
import numpy
import pyfits
from sphere.skyline import SkyLine
from stsci.tools import parseinput
from stsci.imagestats import ImageStats

try:
    from stsci.tools import teal
except ImportError:
    teal = None

__all__ = ['skymatch']
__taskname__ = 'skymatch'
__version__ = '0.1b'
__vdate__ = '12-Jul-2012'

# Function to use for sky calculations
SKYFUNC = computeSky

# Track SKYUSER assignments to reduce FITS header I/O
SKYUSER = defaultdict(float)

def skymatch(input, skyfunc=None, verbose=True):
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

    skyfunc : function
        Function for sky calculation. See `computeSky`.

    verbose : bool
        Print info to screen.

    Examples
    --------
    #. This task can be used to match skies of a set of ACS
       images simply with:
   
           >>> from skymatch import skymatch
           >>> skymatch('j*q_flt.fits')

    #. One could provide a custom function for sky
       calculation:

           >>> def simple_mean(image):
           >>>     return image.data.mean()
           >>>
           >>> skymatch('j*q_flt.fits', skyfunc=simple_mean)

    #. The TEAL GUI can be used to run this task using:

           >>> epar skymatch

       or from a general Python command line:

           >>> from stsci.tools import teal
           >>> teal.teal('skymatch')

    """
    # Time it
    if verbose:
        runtime_begin = datetime.now()
        print('***** SKYMATCH started on {}'.format(runtime_begin))

    # Parse input to get list of filenames to process
    infiles, output = parseinput.parseinput(input)
    assert len(infiles) > 1, '%s: Need at least 2 images. Aborting...' % __taskname__

    # Check sky function
    _pick_skyfunc(skyfunc, infiles[0])
    if verbose:
        print('    Using sky function {}'.format(SKYFUNC))

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
    assert starting_pair is not None, '%s: No overlapping pair. Aborting.' % __taskname__

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
        print('    {}'.format(skyline2update._rough_id()))
        print('        SKYUSER = {:%E} (abs({:%E} - {:%E}))'.format(
            diff_sky, sky1, sky2))

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
            print('    {}'.format(next_skyline._rough_id()))
            print('        SKYUSER = {:%E} ({:%E} - {:%E})'.format(
                diff_sky, sky2, sky1))

        new_mos = mosaic.add_image(next_skyline)
        mosaic = new_mos
        remaining.remove(next_skyline)

        if verbose:
            print('    Added {} to mosaic'.format(next_skyline._rough_id()))

    # Time it
    if verbose:
        runtime_end = datetime.now()
        print('    TOTAL RUN TIME: {}'.format(runtime_end - runtime_begin))
        print('***** SKYMATCH ended on {}'.format(runtime_end))


def computeSky(data, **kwargs):
    """
    Return clipped mode of data as sky.

    Parameters
    ----------
    data : array_like

    **kwargs : `ImageStats` keywords

    Returns
    -------
    Sky value in image data unit.

    See Also
    --------
    stsci_python/trunk/drizzlepac/lib/drizzlepac/sky.py
    
    """
    return ImageStats(data, **kwargs).mode

def _pick_skyfunc(f, im):
    """Use `computeSky` unless a valid function is provided."""
    global SKYFUNC
    
    if f is None:
        return

    with pyfits.open(im) as pf:
        try:
            sky = f(pf['SCI',1])
        except Exception:
            print('%s: skyfunc failed. Using default.' % __taskname__)
            return

    if not isinstance(sky, (int, long, float)):
        print('%s: skyfunc does not return a number. Using default.' %
              __taskname__)
        return

    SKYFUNC = f

def _overlap_xy(wcs, ra, dec):
    """
    Find pixel coordinates of original image for a
    given HSTWCS that fall within polygon bound by
    given RA and DEC.

    For simplicity, RA and DEC are only used to
    determine min/max rows and columns. So pixels
    returned are within the minimum bounding box on
    the image of the polygon, not the actual polygon.
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
    idx : Numpy indices
        Indices of pixel coordinates in Python format.
    
    """   
    x, y = wcs.all_sky2pix(ra, dec, 0)

    bound_x = numpy.clip(x, 0, wcs.naxis1 - 1)
    bound_y = numpy.clip(y, 0, wcs.naxis2 - 1)

    min_x, max_x = bound_x.min(), bound_x.max()
    min_y, max_y = bound_y.min(), bound_y.max()

    return numpy.where((x >= min_x) && (x <= max_x) &&
                       (y >= min_y) &&(y <= max_y))

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
        idx = _overlap_xy(wcs, ra, dec)
        npix = len(idx[0])
        
        with pyfits.open(wcs.filename) as pf:
            sky = SKYFUNC(pf[wcs.extname].data[idx]) - SKYUSER[wcs.filename]

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

    SKYUSER[im_name] = value


#--------------------------
# TEAL Interface functions
#--------------------------
def run(configObj):
    skymatch(configObj['input'])

def getHelpAsString():
    helpString = ''
    if teal:
        helpString += teal.getHelpFileAsString(__taskname__,__file__)

    if helpString.strip() == '':
        helpString += __doc__ + '\n' + skymatch.__doc__

    return helpString
