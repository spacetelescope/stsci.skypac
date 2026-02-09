"""skymatch"""
import os

from .version import __version__

# from .version import version as __version__

__author__ = 'Mihai Cara'
__docformat__ = 'restructuredtext en'
__taskname__ = 'skymatch'


from . import parseat  # noqa: F401
from . import utils  # noqa: F401
from . import pamutils  # noqa: F401
from . import region  # noqa: F401
from . import skystatistics  # noqa: F401
from . import skyline  # noqa: F401
from . import skymatch  # noqa: F401

from stsci.tools import teal
teal.print_tasknames(__name__, os.path.dirname(__file__))


def help():
    msg = """The SkyPac package contains the following tasks that allow users
    perform sky level matching on user images.

    skypac:
        skymatch - primary task for performing sky level matching on
                   user images
    """
    print(msg)
