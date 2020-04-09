#!/usr/bin/env python
import os
import pkgutil
import sys
from setuptools import setup, find_packages
from subprocess import check_call, CalledProcessError

try:
    from distutils.config import ConfigParser
except ImportError:
    from configparser import ConfigParser

conf = ConfigParser()
conf.read(['setup.cfg'])

# Get some config values
metadata = dict(conf.items('metadata'))
PACKAGENAME = metadata.get('package_name', 'stsci.skypac')
DESCRIPTION = metadata.get('description', 'Sky matching on image mosaic')
LONG_DESCRIPTION = metadata.get('long_description', 'README.md')
LONG_DESCRIPTION_CONTENT_TYPE = metadata.get('long_description_content_type',
                                             'text/markdown')
AUTHOR = metadata.get('author', 'Mihai Cara, Warren Hack, Pey Lian Lim')
AUTHOR_EMAIL = metadata.get('author_email', 'help@stsci.edu')
URL = metadata.get('url', 'https://github.com/spacetelescope/stsci.skypac')
LICENSE = metadata.get('license', 'BSD-3-Clause')

# load long description
this_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_dir, LONG_DESCRIPTION), encoding='utf-8') as f:
    long_description = f.read()

PACKAGE_DATA = {
    '': [
        'README.rst',
        'LICENSE.txt',
        'CHANGELOG.rst',
        '*.fits',
        '*.txt',
        '*.inc',
        '*.cfg',
        '*.csv',
        '*.yaml',
        '*.json'
        ],

    PACKAGENAME: [
        '*.help',
        'pars/*',
    ],
}

INSTALL_REQUIRES = [
    'astropy>=3.1',
    'numpy',
    'spherical_geometry>=1.2.2',
    'stsci.imagestats',
    'stsci.tools',
    'stwcs',
]

SETUP_REQUIRES = [
    'setuptools_scm',
    'pytest-runner',
]

TESTS_REQUIRE = [
    'pytest',
    'pytest-cov',
    'pytest-doctestplus',
    'codecov',
]

DOCS_REQUIRE = [
    'numpydoc',
    'graphviz',
    'sphinx<=1.8.5',
    'sphinx_rtd_theme',
    'stsci_rtd_theme',
    'sphinx_automodapi',
]

setup(
    name=PACKAGENAME,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    url=URL,
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Development Status :: 5 - Production/Stable',
    ],
    use_scm_version={'write_to': 'stsci/skypac/version.py'},
    setup_requires=SETUP_REQUIRES,
    python_requires='>=3.5',
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
    packages=find_packages(),
    package_data=PACKAGE_DATA,
    ext_modules=[],
    extras_require={
        'docs': DOCS_REQUIRE,
        'test': TESTS_REQUIRE,
        'all': DOCS_REQUIRE + TESTS_REQUIRE,
    },
    project_urls={
        'Bug Reports': 'https://github.com/spacetelescope/stsci.skypac/issues/',
        'Source': 'https://github.com/spacetelescope/stsci.skypac/',
        'Help': 'https://hsthelp.stsci.edu/',
    },
)
