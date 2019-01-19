#!/usr/bin/env python
import os
import pkgutil
import sys
import importlib
import inspect
import shutil
from configparser import ConfigParser
from setuptools import setup, find_packages
from subprocess import check_call, CalledProcessError
from setuptools import setup, find_packages, Extension, _install_setup_requires
from setuptools.command.install import install

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

# Due to overriding `install` and `build_sphinx` we need to download
# setup_requires dependencies before reaching `setup()`. This allows
# `sphinx` to exist before the `BuildSphinx` class is injected.
SETUP_REQUIRES = [
    'sphinx',
]

_install_setup_requires(dict(setup_requires=SETUP_REQUIRES))
for dep_pkg in SETUP_REQUIRES:
    try:
        importlib.import_module(dep_pkg)
    except ImportError:
        print("{0} is required in order to install '{1}'.\n"
              "Please install {0} first.".format(dep_pkg, PACKAGENAME),
              file=sys.stderr)
        exit(1)

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

INSTALL_REQUIRES=[
    'astropy>=3.1',
    'numpy',
    'spherical_geometry>=1.2.2',
    'stsci.imagestats',
    'stsci.tools',
    'stwcs',
    'shpinx'
]

# Distribute compiled documentation alongside the installed package
docs_compiled_src = os.path.normpath('build/sphinx/html')
docs_compiled_dest = os.path.normpath(
    '{0}/htmlhelp'.format(os.path.join(*PACKAGENAME.split('.')))
)

class InstallCommand(install):
    """ Inform users to build (if desired) html help locally. """
    def run(self):
        super().run()

        if not os.path.exists(docs_compiled_dest):
            print('\nwarning: Sphinx "htmlhelp" documentation was NOT bundled!\n'
                  '         Execute the following then reinstall:\n\n'
                  '         $ python setup.py build_sphinx\n\n',
                  file=sys.stderr)

from sphinx.cmd.build import build_main
from sphinx.setup_command import BuildDoc

class BuildSphinx(BuildDoc):
    """Build Sphinx documentation after compiling C extensions"""

    description = 'Build Sphinx documentation'

    def initialize_options(self):
        BuildDoc.initialize_options(self)

    def finalize_options(self):
        BuildDoc.finalize_options(self)

    def run(self):
        build_cmd = self.reinitialize_command('build_ext')
        build_cmd.inplace = 1
        self.run_command('build_ext')
        build_main(['-b', 'html', 'docs/source', 'build/sphinx/html'])

        # Bundle documentation inside of drizzlepac
        if os.path.exists(docs_compiled_src):
            if os.path.exists(docs_compiled_dest):
                shutil.rmtree(docs_compiled_dest)

            shutil.copytree(docs_compiled_src, docs_compiled_dest)

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
    cmdclass={
        'install': InstallCommand,
        'build_sphinx': BuildSphinx,
    },
    project_urls={
        'Bug Reports': 'https://github.com/spacetelescope/stsci.skypac/issues/',
        'Source': 'https://github.com/spacetelescope/stsci.skypac/',
        'Help': 'https://hsthelp.stsci.edu/',
        },
)
