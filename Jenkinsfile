// Obtain files from source control system.
if (utils.scm_checkout()) return

def test_bin(env_name, bin) {
    def result = "with_env -n ${env_name} ${bin}"
    return result
}

def test_import(env_name, module) {
    def result = "with_env -n ${env_name} python -c 'import ${module}'"
    return result
}

// Globals
PIP_INST = "pip install"
def CONDA_CHANNEL = "http://ssb.stsci.edu/astroconda"
def DEPS = ['astropy', 'graphviz', 'numpy', 'numpydoc',
        'spherical-geometry',  'sphinx',  'sphinx_rtd_theme',
        'stsci_rtd_theme', 'stsci.imagestats', 'stwcs', 'setuptools']


matrix_python = ["3.6", "3.7"]
matrix_astropy = ["3"]
matrix_numpy = ["latest"]
matrix = []


// RUN ONCE:
//    "sdist" is agnostic enough to work without any big dependencies
def sdist = new BuildConfig()
sdist.nodetype = "linux"
sdist.name = "sdist"
sdist.conda_channels = [CONDA_CHANNEL]
sdist.conda_packages = ['astropy', 'numpy']
sdist.build_cmds = ["python setup.py sdist"]
matrix += sdist


// RUN ONCE:
//    "build_sphinx" with default python
def docs = new BuildConfig()
docs.nodetype = "linux"
docs.name = "docs"
docs.conda_channels = [CONDA_CHANNEL]
docs.conda_packages = DEPS
docs.build_cmds = ["pip install -e.",
                   "python setup.py build_sphinx",]
matrix += docs


// Generate installation compatibility matrix
for (python_ver in matrix_python) {
    for (astropy_ver in matrix_astropy) {
        for (numpy_ver in matrix_numpy) {
            // Astropy >=3.0 no longer supports Python 2.7
            if (python_ver == "2.7" && astropy_ver == "3") {
                continue
            }
            def install = utils.copy(docs)
            install.name = "install-py=${python_ver},np=${numpy_ver},ap=${astropy_ver}"
            install.conda_packages.add("python=${python_ver}")
            if (astropy_ver != "latest") {
                install.conda_packages += "astropy=${astropy_ver}"
            }
            if (numpy_ver != "latest") {
                install.conda_packages += "numpy=${numpy_ver}"
            }
            install.build_cmds = ["python setup.py egg_info",
                                  "python setup.py install",
                                  test_import(python_ver, 'stsci.skypac'),
                                  test_import(python_ver, 'stsci.skypac.hstinfo'),
                                  test_import(python_ver, 'stsci.skypac.pamutils'),
                                  test_import(python_ver, 'stsci.skypac.parseat'),
                                  test_import(python_ver, 'stsci.skypac.region'),
                                  test_import(python_ver, 'stsci.skypac.skyline'),
                                  test_import(python_ver, 'stsci.skypac.skymatch'),
                                  test_import(python_ver, 'stsci.skypac.skystatistics'),
                                  test_import(python_ver, 'stsci.skypac.utils'),]
            matrix += install
        }
    }
}

// Iterate over configurations that define the (distibuted) build matrix.
// Spawn a host of the given nodetype for each combination and run in parallel.
utils.run(matrix)
