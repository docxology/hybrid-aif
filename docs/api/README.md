# API reference (Sphinx source)

The source of the API reference for `hybrid_aif`: a landing page and one page per group of public
names, each with an autosummary table. The prose in each page is a first draft.

## Build

```bash
uv venv .venv-docs
uv pip install --python .venv-docs/bin/python -e ".[docs]"
.venv-docs/bin/sphinx-build -b html docs/api docs/api/_build/html
```

Open `docs/api/_build/html/index.html`. Any environment with the package and the `docs` extra
installed works; the build imports `hybrid_aif`, so it must be the checkout being documented.
Intersphinx fetches the Python, JAX, equinox and numpy inventories and needs network access;
without it the build still succeeds, with warnings.

## Layout

- `conf.py`: extensions (autodoc, autosummary, intersphinx, viewcode, MyST), theme
  (`sphinx-book-theme`).
- `index.rst`: landing page and navigation.
- `system.rst`, `factors.rst`, `distributions.rst`, `composition.rst`, `running.rst`,
  `inference.rst`, `objectives.rst`, `planning.rst`, `learning.rst`, `description.rst`,
  `environments.rst`: one group each. To add a public name, add it to the table of its group.
- `_templates/autosummary/`: the page templates for classes and other objects.
- `_generated/`: one stub per name, written by autosummary at build time. Generated, ignored by git
  (see `.gitignore`), not edited by hand. `_build/` is ignored too.
