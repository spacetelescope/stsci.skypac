"""skymatch"""

# These lines allow TEAL to print out the names of TEAL-enabled tasks
# upon importing this package.
import os
from . import utils
from . import parseat
from . import region
from . import skystatistics
from . import skyline
from . import skymatch

from stsci.tools import teal
teal.print_tasknames(__name__, os.path.dirname(__file__))


__taskname__ = 'skymatch'
__version__ = '0.7'
__vdate__ = '20-May-2014'
__author__ = 'Mihai Cara'


def help():
    msg = \
""" The SkyPac package contains the following tasks that allow users perform sky level matching on user images.

skypac:
       skymatch - primary task for performing sky level matching on user images
"""
    print msg
