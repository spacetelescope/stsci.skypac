#!/usr/bin/env python
import os
import pkgutil
import sys
from setuptools import setup, find_packages
from subprocess import check_call, CalledProcessError


if not pkgutil.find_loader('relic'):
    relic_local = os.path.exists('relic')
    relic_submodule = (relic_local and
                       os.path.exists('.gitmodules') and
                       not os.listdir('relic'))
    try:
        if relic_submodule:
            check_call(['git', 'submodule', 'update', '--init', '--recursive'])
        elif not relic_local:
            check_call(['git', 'clone', 'https://github.com/spacetelescope/relic.git'])

        sys.path.insert(1, 'relic')
    except CalledProcessError as e:
        print(e)
        exit(1)

import relic.release

NAME_ROOT = 'stsci'
NAME_BASE = 'skypac'
NAME = '.'.join([NAME_ROOT, NAME_BASE])
NS_PATH = os.path.join(NAME_ROOT, NAME_BASE)

version = relic.release.get_info()
relic.release.write_template(version, NS_PATH)

setup(
    name=NAME,
    version=version.pep386,
    author='Mihai Cara, Warren Hack, Pey Lian Lim',
    author_email='help@stsci.edu',
    description='Sky matching on image mosaic',
    url='https://github.com/spacetelescope/stsci.skypac',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    install_requires=[
        'astropy',
        'numpy',
        'sphinx',
        'spherical_geometry>=1.2.2',
        'stsci.imagestats',
        'stsci.tools',
        'stwcs'
    ],
    packages=find_packages(),
    package_data={
        '': ['LICENSE.txt'],
        NAME: [
            '*.help',
            'pars/*',
        ]
    },
)
