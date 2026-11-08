"""The declaration of a belief read and the function that turns a belief into its datum.

A modality may take its datum from a belief of the System: the source's posterior of the previous
round, pushed through a kernel at the source and realised as a value. :class:`Read` declares
one; :func:`read_datum` evaluates it as a pure function of the belief, the declaration and a key.
``System(lagged=...)`` holds reads, and a merge's
:class:`~hybrid_aif.composition.merge.Link` with a ``realise`` makes one across components.

Design: docs/design/nesting-and-coupling.md N3.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from hybrid_aif.cells import Grid, cell_mass
from hybrid_aif.distributions import Categorical, Gaussian, MixtureGaussian
from hybrid_aif.inference.sample import collapse

#: The realisations a declared read takes: a draw, the argmax (or the mean of a continuous
#: belief), or the belief's sufficient statistics (the probability vector of a categorical).
REALISATIONS = ("sample", "mode", "stats")

#: The realisation a ``(modality, source)`` pair stands for: ``stats`` for a categorical source,
#: ``mode`` (the mean) for a continuous one, the level for an action node. Resolved when the
#: datum is taken, from the source's kind.
LEGACY = "legacy"

#: The kernel that leaves the source's belief as it is.
IDENTITY = "identity"

#: The tag a key is folded with before the reads draw from it, so a read's draw is a function
#: of the key it is given and independent of the face's other uses of that key.
READ_TAG = 0x52454144


class _ReadFields(NamedTuple):
    """One modality's datum read from a belief: ``Read(modality, source, kernel, realise)``.

    ``source`` names a state factor (of this System, or of another component of a merge, under
    its namespaced name) or an action node of this System. The datum is the source's posterior
    of the previous round (its belief after that round's inference, before it is pushed forward
    to this round), pushed through ``kernel`` and realised:

    - ``kernel`` is ``"identity"`` (the belief itself) or a :class:`~hybrid_aif.cells.Grid`, the
      indicator of the grid's cells over a continuous source, whose pushed belief is
      :func:`~hybrid_aif.cells.cell_mass`, a probability vector over the cells;
    - ``realise`` is ``"sample"`` (a draw from the pushed belief; the default), ``"mode"`` (the
      argmax of a categorical belief, the mean of a continuous one) or ``"stats"`` (the
      probability vector of a categorical belief, consumed by the filter as a soft observation;
      not defined for a Gaussian belief). A level is the datum of ``sample`` and ``mode`` of a
      categorical belief.

    The planner treats the modality as any other: its information gain is the estimators' score
    of a hard outcome, which for a ``stats`` datum is an upper bound on what the soft observation
    delivers (the Jensen gap of the soft observation).

    A ``(modality, source)`` pair in ``System.lagged`` is the identity kernel with the pair's own
    realisation, which is not a declared one: ``stats`` for a categorical source, ``mode`` for a
    Gaussian one, the level for an action node.

    Design: docs/design/nesting-and-coupling.md N3.
    """

    modality: str
    source: str
    kernel: "str | Grid" = IDENTITY
    realise: str = "sample"


class Read(_ReadFields):
    __doc__ = _ReadFields.__doc__
    __slots__ = ()

    def __new__(cls, modality, source, kernel=IDENTITY, realise="sample"):
        if realise not in REALISATIONS:
            raise ValueError(
                f"read of {modality!r}: realise must be one of {list(REALISATIONS)}, got "
                f"{realise!r}"
            )
        return super().__new__(cls, modality, source, kernel, realise)


class Emit(NamedTuple):
    """The declaration of an emission: an output of a System drawn from its state factors.

    ``parents`` names the state factors and control nodes the emission's kernel
    (``SystemParams.emission``) is conditioned on, in the order of the kernel's parent axes. An
    emission is realised once per round, by ``realise``: ``"sample"`` (a draw from the kernel),
    ``"mode"`` (its most probable level, or its mean) or ``"stats"`` (its probability vector, or
    its mean and covariance). Declared in ``System.emissions`` as ``name -> parents`` (a draw) or
    ``name -> Emit(parents, realise)``. Design: docs/design/nesting-and-coupling.md N5.
    """

    parents: tuple[str, ...]
    realise: str = "sample"


def as_read(entry: "tuple | Read") -> Read:
    """``entry`` (a ``(modality, source)`` pair or a :class:`Read`) as a :class:`Read`."""
    if isinstance(entry, Read):
        return entry
    modality, source = entry
    return Read._make((modality, source, IDENTITY, LEGACY))  # the pair's form, bypassing __new__


def check_declaration(read: Read) -> None:
    """Refuse a read whose kernel or realisation is not one of the declared forms."""
    if not (isinstance(read.kernel, Grid) or read.kernel == IDENTITY):
        raise ValueError(
            f"read of {read.modality!r}: kernel must be 'identity' or a Grid, got {read.kernel!r}"
        )
    if read.realise not in (*REALISATIONS, LEGACY):  # a pair stands for LEGACY
        raise ValueError(
            f"read of {read.modality!r}: realise must be one of {list(REALISATIONS)}, got "
            f"{read.realise!r}"
        )


def modality_key(key: Array, modality_index: int) -> Array:
    """The key modality ``modality_index``'s read draws with, from the ``read_key`` a face was
    given."""
    return jax.random.fold_in(jax.random.fold_in(key, READ_TAG), modality_index)


def _draw(probs: Array, key: Array | None, what: str) -> Array:
    if key is None:
        raise ValueError(
            f"{what} is realised by 'sample', which draws from the belief and needs a key; pass "
            "read_key=, or declare realise='mode' or 'stats'"
        )
    return jax.random.categorical(key, jnp.log(jnp.maximum(probs, 1e-30))).astype(jnp.int32)


def read_datum(
    belief: object,
    kernel: "str | Grid" = IDENTITY,
    realise: str = "sample",
    key: Array | None = None,
    what: str = "a read",
) -> Array:
    """The datum of a read: ``belief`` pushed through ``kernel`` and realised.

    Under the identity kernel a categorical belief gives a level (``sample``, ``mode``) or its
    probability vector (``stats``), and a Gaussian or mixture belief its mean (``mode``) or a
    draw's value (``sample``); ``"legacy"`` is ``belief.mean()``, which is ``stats`` for a
    categorical and ``mode`` for a Gaussian. Under a :class:`~hybrid_aif.cells.Grid` the belief
    must be a Gaussian or mixture, and the datum is a cell index (``sample``, ``mode``) or the
    cell mass (``stats``). ``key`` is needed for ``sample`` only. Pure and jit-friendly.
    """
    if isinstance(kernel, Grid):
        if not isinstance(belief, Gaussian | MixtureGaussian):
            raise ValueError(
                f"{what}: an indicator kernel reads a Gaussian or mixture belief, got "
                f"{type(belief).__name__}"
            )
        probs = cell_mass(belief, kernel)
        if realise == "sample":
            return _draw(probs, key, what)
        if realise == "mode":
            return jnp.argmax(probs).astype(jnp.int32)
        if realise == "stats":
            return probs
        raise ValueError(f"{what}: realise {realise!r} is not one of {list(REALISATIONS)}")
    if realise == LEGACY:
        return belief.mean()
    if isinstance(belief, Categorical):
        if realise == "sample":
            return _draw(belief.probs, key, what)
        if realise == "mode":
            return jnp.argmax(belief.probs).astype(jnp.int32)
        if realise == "stats":
            return belief.probs
        raise ValueError(f"{what}: realise {realise!r} is not one of {list(REALISATIONS)}")
    if realise == "mode":
        return belief.mean()
    if realise == "sample":
        if key is None:
            _draw(jnp.ones(1), None, what)  # raises: a draw needs a key
        return collapse(belief, key).mean()
    raise ValueError(
        f"{what}: realise {realise!r} of a Gaussian belief is not built; its sufficient "
        "statistics are deferred (design: nesting-and-coupling.md, Deferred), so declare "
        "'sample' or 'mode'"
    )
