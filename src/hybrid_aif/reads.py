"""The declaration of an emission: an output of a System realised from the state it carries.

An :class:`Emission` is a variable a System generates. Its parents are state factors (and control
nodes) of the System, its kernel is what turns them into a value, and its realisation says which
value: a draw, the most probable value, or the statistics. A System realises its emissions from
whatever it carries: a point (a world's state) or a belief (a planner's posterior).

An emission is also how a System reads its own belief. A modality of the System may observe an
emission of it (``System(observes=...)``), and the value realised after one round is the
observation of the next: the modality is explained by its own parents, so a factor can emit what
the model just believed. A belief read across components (a ``Link`` of
:mod:`hybrid_aif.composition.merge` with a ``realise``) declares the same pair on the source and
the reader.

Design: docs/design/nesting-and-coupling.md N3, N5; docs/design/engine-consolidation.md.
"""

from typing import NamedTuple

from hybrid_aif.cells import Grid

#: The realisations an emission takes: a draw, the most probable value (the argmax of a
#: categorical, the mean of a continuous belief), or the sufficient statistics (the probability
#: vector of a categorical; the mean and covariance of a Gaussian kernel's predictive).
REALISATIONS = ("sample", "mode", "stats")

#: The tag a step's key is folded with before its emissions are realised from it, so an
#: emission's draw is a function of the key it is given and independent of the step's other uses
#: of that key.
EMIT_TAG = 0x454D4954

#: The static kernel of an out-port of a black box (:attr:`System.black_box`): the emission has no
#: parents and no kernel the System states; its value is what the black box's step returns, and its
#: type is the one ``System.black_box`` declares.
BLACK_BOX = "black_box"


class Emission(NamedTuple):
    """The declaration of an emission: an output of a System realised from its state factors.

    ``parents`` names the state factors and control nodes the emission is conditioned on, in the
    order of its kernel's parent axes. ``realise`` is ``"sample"`` (a draw from the kernel),
    ``"mode"`` (its most probable level, or its mean) or ``"stats"`` (its probability vector, or
    its mean and covariance). ``kernel`` is the emission's static kernel:

    - ``None``: the kernel is ``SystemParams.emission[name]``, a likelihood of any family, which
      must be there;
    - ``"identity"``: the emission is its one state factor. A categorical factor gives a level
      port (its level, or under ``stats`` its probability vector), and a continuous one a float
      port of the factor's width (its location under ``mode``, a draw under ``sample``;
      ``stats`` is not defined);
    - a :class:`~hybrid_aif.cells.Grid`: an indicator likelihood over the cells of the grid of
      one continuous state factor, a level port with as many levels as the grid has cells: from a
      point the cell the point falls in, from a belief the belief's mass in each cell
      (:func:`~hybrid_aif.cells.cell_mass`), of which ``mode`` is the largest cell, ``stats``
      the vector and ``sample`` a draw;
    - :data:`BLACK_BOX` (``"black_box"``): an out-port of a black box, with no parents; nothing
      in the library realises it, the black box's step returns its value, and its type is its
      entry in ``System.black_box``.

    Declared in ``System.emissions`` as ``name -> parents`` (a draw through the params' kernel)
    or ``name -> Emission(parents, realise, kernel)``. Design: docs/design/nesting-and-coupling.md
    N5; docs/design/engine-consolidation.md.
    """

    parents: tuple[str, ...]
    realise: str = "sample"
    kernel: "Grid | None" = None
