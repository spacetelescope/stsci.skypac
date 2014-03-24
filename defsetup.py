from __future__ import division # confidence high

#import distutils.core
#import distutils.sysconfig

#try:
#    import numpy
#except:
#    raise ImportError('NUMPY was not found. It may not be installed or it may not be on your PYTHONPATH')

#pythoninc = distutils.sysconfig.get_python_inc()
#numpyinc = numpy.get_include()

pkg =  "skypac"

#ext = [ distutils.core.Extension('', [], include_dirs = [pythoninc,numpyinc]) ]

setupargs = {
    'version' : "0.6",
    'description' : "Sky matching on image mosaic",
    'author' : "Mihai Cara, Pey Lian Lim",
    'author_email' : "help@stsci.edu",
    'license' : "http://www.stsci.edu/resources/software_hardware/pyraf/LICENSE",
    'data_files' : [(pkg+"/pars", ['lib/pars/*']),
                    (pkg, ['lib/*.help']),
                    (pkg, ['LICENSE.txt'])],
    'scripts' : [ ] ,
    'platforms' : ["Linux","Solaris","Mac OS X","Win"],
    'ext_modules' : [],
    'package_dir' : { 'skypac':'lib', },
}
