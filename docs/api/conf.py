"""Sphinx configuration for the hybrid_aif API reference."""

import os

import hybrid_aif

project = "hybrid-aif"
author = "Noumenal Labs"
copyright = "Noumenal Labs"
release = hybrid_aif.__version__
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
]

# The docstrings are plain reStructuredText, so napoleon is not loaded.
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "README.md"]
templates_path = ["_templates"]

# Stubs are generated into docs/api/_generated (ignored by git) at build time.
autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_imported_members = True

autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_typehints_format = "short"
python_maximum_signature_line_length = 88
autodoc_class_signature = "separated"
autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "show-inheritance": True}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://docs.jax.dev/en/latest", None),
    "equinox": ("https://docs.kidger.site/equinox", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}
intersphinx_timeout = 10

html_theme = "sphinx_book_theme"
html_title = f"hybrid-aif {release}"
# The project page sits one level above the reference when both are published together; the logo
# links back to it. HYBRID_AIF_SITE_URL overrides the address (the default is the server's root).
html_logo = "../site/static/h-aif.svg"
html_favicon = "../site/static/favicon.png"
html_theme_options = {
    "logo": {"link": os.environ.get("HYBRID_AIF_SITE_URL", "/"), "alt_text": "hybrid-aif"},
    "repository_url": "https://github.com/noumenal-labs/hybrid-aif",
    "use_repository_button": False,
    "show_toc_level": 2,
    "show_navbar_depth": 0,
}
