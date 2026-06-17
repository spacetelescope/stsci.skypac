# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import datetime
import sphinx
from pathlib import Path
from distutils.version import Version
import importlib

if sys.version_info < (3, 11):
    import tomli as tomllib
else:
    import tomllib

on_rtd = os.environ.get("READTHEDOCS", None) == "True"


def check_sphinx_version(expected_version):
    sphinx_version = Version(sphinx.__version__)
    expected_version = Version(expected_version)
    if sphinx_version < expected_version:
        raise RuntimeError(
            f"At least Sphinx version {expected_version} is required to build this "
            f"documentation.  Found {sphinx_version}."
        )


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

with open(Path(__file__).parent.parent / "pyproject.toml", "rb") as metadata_file:
    metadata = tomllib.load(metadata_file)["project"]

project = metadata["name"]
author = "Space Telescope Science Institute (`STScI <https://stsci.edu>`_)"
copyright = f"{datetime.datetime.today().year}, Association of Universities for Research in Astronomy (`AURA <https://www.aura-astronomy.org>`_)"

package = importlib.import_module(metadata["name"])
try:
    version = package.__version__.split("-", 1)[0]
    # The full version, including alpha/beta/rc tags.
    release = package.__version__
except AttributeError:
    version = "dev"
    release = "dev"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "numfig",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "numpydoc",
    "sphinx.ext.napoleon",
    "sphinx_automodapi.automodapi",
    "sphinx_automodapi.automodsumm",
    "sphinx_automodapi.autodoc_enhancements",
    "sphinx_automodapi.smart_resolver",
]

if on_rtd:
    extensions.append("sphinx.ext.mathjax")
elif Version(sphinx.__version__) < Version("1.4"):
    extensions.append("sphinx.ext.pngmath")
else:
    extensions.append("sphinx.ext.imgmath")

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# reST default role used for single backticks (`text`)
default_role = "obj"


# -- HTML output configuration ----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "collapse_navigation": True,
    "sticky_navigation": False,
    "style_external_links": True,
}
html_context = {
    "css_files": [
        "_static/css/skypac.css",
    ],
}
html_logo = "_static/stsci_pri_combo_mark_dark_bkgd.png"
html_last_updated_fmt = "%b %d, %Y"
html_sidebars = {"**": ["globaltoc.html", "relations.html", "searchbox.html"]}
html_domain_indices = True
html_use_index = True
html_show_sourcelink = False
htmlhelp_basename = "skypacdoc"

# -- EPUB output configuration -----------------------------------------------

epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright
epub_show_urls = "footnote"
epub_exclude_files = ["search.html"]

# -- LaTeX output configuration ----------------------------------------------

latex_elements = {
    "papersize": "letterpaper",  #'letterpaper' or 'a4paper'
    "pointsize": "11pt",  #'10pt', '11pt' or '12pt'
    "preamble": r"""\usepackage{enumitem} \setlistdepth{99}""",
}
latex_documents = [
    (
        "index",  # source start file
        f"{project}.tex",  # target name
        "skypac Documentation",  # title
        author,  # author
        "manual",  # documentclass [howto, manual, or own class]
    ),
]
latex_show_urls = "True"
latex_domain_indices = True
latex_logo = "_static/stsci_pri_combo_mark_white.png"

# -- Texinfo output configuration -------------------------------------------

texinfo_documents = [
    (
        "index",  # source start file
        f"{project}.tex",  # target name
        "SkyPac Documentation",  # title
        author,  # author
        project,  # dir menu entry
        "skypac Documentation",  # description
        "Miscellaneous",  # category
    ),
]
texinfo_domain_indices = True
texinfo_show_urls = "inline"  # 'footnote', 'no', or 'inline'

# If true, do not generate a @detailmenu in the "Top" node's menu.
# texinfo_no_detailmenu = False

# -- manpage output configuration ---------------------------------------

man_pages = [
    (
        "index",  # source start file
        project,  # name
        "skypac Documentation",  # description
        [author],  # authors
        1,  # manual section
    )
]
man_show_urls = True

# -- linkcheck configuration -------------------------------------------------

linkcheck_retry = 5
linkcheck_ignore = [
    "http://stsci.edu/schemas/fits-schema/",  # Old schema from CHANGES.rst
    "https://outerspace.stsci.edu",  # CI blocked by service provider
    "https://jira.stsci.edu/",  # Internal access only
    r"https://.*\.readthedocs\.io",  # 429 Client Error: Too Many Requests
    "https://doi.org",  # CI blocked by service provider (timeout)
]
linkcheck_timeout = 180
linkcheck_anchors = False
linkcheck_report_timeouts_as_broken = True
linkcheck_allow_unauthorized = False

# Enable nitpicky mode - which ensures that all references in the docs resolve.
nitpicky = True

# -- numpydoc configuration --------------------------------------------------

# Don't show summaries of the members in each class along with the class' docstring
numpydoc_show_class_members = False

# -- sphinx-automodapi configuration --------------------------------------------
# https://sphinx-automodapi.readthedocs.io/en/latest/automodapi.html

sys.path.insert(0, os.path.abspath("source/"))
sys.path.insert(0, os.path.abspath("exts/"))


def find_mod_objs_patched(*args, **kwargs):
    from sphinx_automodapi.utils import find_mod_objs

    return find_mod_objs(args[0], onlylocals=True)


def patch_automodapi(app):
    """Monkey-patch the automodapi extension to exclude imported members"""
    from sphinx_automodapi import automodsumm

    automodsumm.find_mod_objs = find_mod_objs_patched


def setup(app):
    try:
        app.add_css_file("css/skypac.css")
    except AttributeError:
        app.add_stylesheet("css/skypac.css")

    app.connect("builder-inited", patch_automodapi)


pygments_style = "sphinx"
autosummary_generate = True
automodapi_toctreedirnm = "api"
autoclass_content = "both"
graphviz_output_format = "svg"
graphviz_dot_args = [
    "-Nfontsize=10",
    "-Nfontname=Helvetica Neue, Helvetica, Arial, sans-serif",
    "-Efontsize=10",
    "-Efontname=Helvetica Neue, Helvetica, Arial, sans-serif",
    "-Gfontsize=10",
    "-Gfontname=Helvetica Neue, Helvetica, Arial, sans-serif",
]

# -- sphinx.ext.intersphinx configuration ------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#configuration

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
}
