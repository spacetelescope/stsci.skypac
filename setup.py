#!/usr/bin/env python
import relic.release
from glob import glob
from numpy import get_include as np_include
from setuptools import setup, find_packages, Extension


version = relic.release.get_info()
relic.release.write_template(version, 'stsci/skypac')

setup(
    name = 'stsci.skypac',
    version = version.pep386,
    author = 'Mihai Cara, Warren Hack, Pey Lian Lim',
    author_email = 'help@stsci.edu',
    description = 'Sky matching on image mosaic',
    url = 'https://github.com/spacetelescope/stsci.skypac',
    classifiers = [
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    install_requires = [
        'astropy',
        'nose',
        'numpy',
        'sphinx',
        'stsci.sphere',
        'stsci.sphinxext',
        'stsci.tools',
        'stwcs'
    ],
    packages = find_packages(),
    package_data = {
        '': ['LICENSE.txt'],
        'stsci/skypac': [
            '*.help',
            'pars/*',
        ]
    },
)
