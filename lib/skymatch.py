"""
Sky matching module.

:Author: Pey Lian Lim

:Organization: Space Telescope Science Institute

:History:
    * 2012-06-27 PLL started this module.

"""
from __future__ import division, print_function

# STDLIB
from inspect import isfunction

# THIRD PARTY
import numpy
import pyfits
from stsci.tools import parseinput
from stsci.imagestats import ImageStats
from sphere.skyline import SkyLine

try:
    from stsci.tools import teal
except ImportError:
    teal = None

__all__ = ['skymatch']
__taskname__ = 'skymatch'
__version__ = '0.1b'
__vdate__ = '27-Jun-2012'

def skymatch(input, skyfunc=None):
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
        Function for sky calculation. Function can have only
        one argument that is the PyFITS pointer to a FITS
        extension (see Examples). If `None`, `computeSky`
        will be used.

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
    # Parse input to get list of filenames to process
    infiles, output = parseinput.parseinput(input)
    assert len(infiles) > 1, '%s: Need at least 2 images. Aborting.' % __taskname__

# need to put this into separate func to test skyfunc. no test if skyfunc is None. then skyfunc=computeSky

    # Check sky function
    assert isfunction(skyfunc), '%s: skyfunc is not a function' % __taskname__

    pf = pyfits.open(infiles[0])
    try:
        sky = skyfunc(pf['SCI',1])
    except Exception as err:
        print('%s: skyfunc failed. Aborting.' % __taskname__)
        raise
    finally:
        pf.close()

    assert isinstance(sky, (int, long, float)), '%s: skyfunc does not return a number. Aborting.' % __taskname__

    # Extract skylines
    skylines = []
    for file in infiles:
        skylines.append( SkyLine(file) )

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
    intersect = s1.find_intersection(s2)

    #wcs = intersect.to_wcs() --> so something with this? see Astrodrizzle?
    # how to extract image.data of overlapping region?
    
    #sky1 = skyfunc(intersect_s1)
    #sky2 = skyfunc(intersect_s2)

    #---------------------------------------------------------#
    # 3. Compute the difference in the sky values.            #
    #---------------------------------------------------------#

    #sky_diff = numpy.abs(sky1 - sky2)

    #---------------------------------------------------------#
    # 4. Record that difference in the header of the exposure #
    #    with the highest sky value as the SKYUSER keyword in #
    #    the SCI headers.                                     #
    #---------------------------------------------------------#

    # need to open pyfits with mode='update'
    # pyfits.header.update('SKYUSER', sky_diff)

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

    #see SkyLine.mosaic()

#The computation of the sky can be done using the same basic algorithm already used by AstroDrizzle; namely, the minimum value of the clipped modes from all chips in the exposure. We should be able to expand the task later to use additional methods for computing the sky values.

#These images could then be combined using AstroDrizzle with the 'skyuser' parameter set to 'SKYUSER' to generate a mosaic with a uniform background or even called from within AstroDrizzle to replace the current sky subtraction algorithm.  This task should be callable interactively from Python (naturally), but also eventually have a TEAL interface as well.
   

def computeSky(image, **kwargs):
    """
    Return clipped mode of data as sky.

    This is modeled after `drizzlepac.sky._computeSky`.

    Parameters
    ----------
    image : `pyfits` pointer to data extension

    **kwargs : `ImageStats` keywords

    Returns
    -------
    sky : float
        Sky value in image data unit.

    """
    # https://www.stsci.edu/trac/ssb/stsci_python/browser/stsci_python/trunk/drizzlepac/lib/drizzlepac/sky.py
    sky = ImageStats(image.data, **kwargs).mode
    return sky


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
