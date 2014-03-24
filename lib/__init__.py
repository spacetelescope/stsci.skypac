"""skymatch"""

if False :
    __version__ = ''

    __svn_version__ = 'Unable to determine SVN revision'
    __full_svn_info__ = ''
    __setup_datetime__ = None

    try:
        __version__ = __import__('pkg_resources').\
                            get_distribution('skypac').version
    except:
        pass

else :
    __version__ = '0.6'

import skypac

try:
    from skypac.svninfo import (__svn_version__, __full_svn_info__,
                                  __setup_datetime__)
except ImportError:
    pass

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


def help():
    msg = \
""" The SkyPac package contains the following tasks that allow users perform sky level matching on user images.

skypac:
       skymatch - primary task for performing sky level matching on user images
"""
    print msg
