"""Shared helpers for the analysis, plotting and measurement scripts.

Deliberately small: only helpers that were duplicated across three or more
scripts live here. Import as ``from _common import ...`` — the scripts are
run from the repository root, so ``scripts/`` is on ``sys.path``.
"""

from __future__ import annotations

import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS_IMG = REPO_ROOT / "docs" / "img"
DOCS_PARETO = REPO_ROOT / "docs" / "pareto"
FIGURES = REPO_ROOT / "figures"


def setup_mpl():
    """Select the headless Agg backend and return ``matplotlib.pyplot``.

    pyplot must be imported *after* ``use("Agg")`` for the scripts to work on
    a headless machine; centralising the sequence removes the repeated
    import-after-use dance (and the ``# noqa: E402`` it required).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def savefig(fig, path, dpi=150):
    """Write ``fig`` to ``path`` (creating parent directories) and close it."""
    import matplotlib.pyplot as plt

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"wrote {path}")
