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
CONDA_CHANNEL = "http://ssb.stsci.edu/astroconda"
CONDA_CREATE = "conda create -y -q -c ${CONDA_CHANNEL}"
CONDA_INST = "conda install -y -q -c ${CONDA_CHANNEL}"
PY_SETUP = "python setup.py"
PYTEST_ARGS = "tests --basetemp=tests_output --junitxml results.xml"
DEPS = "astropy graphviz numpy numpydoc \
        spherical-geometry sphinx sphinx_rtd_theme \
        stsci_rtd_theme stsci.imagestats stwcs setuptools"

matrix_python = ["2.7", "3.5", "3.6"]
matrix_astropy = ["2", "3"]
matrix_numpy = ["latest"]
matrix = []


// RUN ONCE:
//    "sdist" is agnostic enough to work without any big dependencies
sdist = new BuildConfig()
sdist.nodetype = "linux"
sdist.name = "sdist"
sdist.build_cmds = ["${CONDA_CREATE} -n dist astropy numpy",
                    "with_env -n dist ${PY_SETUP} sdist"]
matrix += sdist


// RUN ONCE:
//    "build_sphinx" with default python
docs = new BuildConfig()
docs.nodetype = "linux"
docs.name = "docs"
docs.build_cmds = ["${CONDA_CREATE} -n docs ${DEPS}",
                   "with_env -n docs ${PY_SETUP} build_sphinx"]
matrix += docs


// Generate installation compatibility matrix
for (python_ver in matrix_python) {
    for (astropy_ver in matrix_astropy) {
        for (numpy_ver in matrix_numpy) {
            // Astropy >=3.0 no longer supports Python 2.7
            if (python_ver == "2.7" && astropy_ver == "3") {
                continue
            }

            DEPS_INST = "python=${python_ver} "

            if (astropy_ver != "latest") {
                DEPS_INST += "astropy=${astropy_ver} "
            }

            if (numpy_ver != "latest") {
                DEPS_INST += "numpy=${numpy_ver} "
            }

            DEPS_INST += DEPS

            install = new BuildConfig()
            install.nodetype = "linux"
            install.name = "install-py=${python_ver},np=${numpy_ver},ap=${astropy_ver}"
            install.build_cmds = ["${CONDA_CREATE} -n ${python_ver} ${DEPS_INST}",
                                  "with_env -n ${python_ver} ${PY_SETUP} egg_info",
                                  "with_env -n ${python_ver} ${PY_SETUP} install",

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
