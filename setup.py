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
VERSION_DATE = metadata.get('version-date', '')
VERSION = metadata.get('version', '')
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

# release version control:
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

version = relic.release.get_info()
if not version.date:
    default_version = metadata.get('version', VERSION)
    default_version_date = metadata.get('version-date', VERSION_DATE)
    version = relic.git.GitVersion(
        pep386=default_version,
        short=default_version,
        long=default_version,
        date=default_version_date,
        dirty=True,
        commit='',
        post='-1'
    )
relic.release.write_template(version, os.path.join(*PACKAGENAME.split('.')))

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


setup(
    name=PACKAGENAME,
    version=version.pep386,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
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
    python_requires='>=3.5',
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(),
    package_data=PACKAGE_DATA,
    project_urls={
        'Bug Reports': 'https://github.com/spacetelescope/stsci.skypac/issues/',
        'Source': 'https://github.com/spacetelescope/stsci.skypac/',
        'Help': 'https://hsthelp.stsci.edu/',
    },
)
