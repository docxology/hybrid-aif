"""A neural conditional read at a Gaussian belief as a linear-Gaussian observation model.

A neural likelihood ``o = m(x, parents) + eps``, ``eps ~ N(0, diag sigma(x, parents)^2)`` (an
:class:`~hybrid_aif.distributions.MLPConditional`), over a Gaussian belief ``N(loc, chol chol^T)``
on its continuous parent ``x``, is read as the affine model ``o = H x + c + eps'``,
``eps' ~ N(0, R)`` whose moments match the belief's push through ``m`` under ``propagation``:

- ``"first_order"``: ``H`` is the Jacobian of ``m`` at ``loc``, ``c = m(loc) - H loc`` and
  ``R = diag sigma(loc)^2``; the extended Kalman reading, exact for ``m`` affine in ``x``.
- ``"unscented"``: the ``2d + 1`` sigma points of the belief through ``m``; ``H`` is the
  statistical linearisation ``P_xy^T Sigma^-1``, ``c = y_bar - H loc`` and ``R`` the weighted
  covariance of the fit's residuals at the points plus the averaged noise
  ``sum_i w_i diag sigma(x_i)^2`` (equal to ``P_yy - H Sigma H^T``, formed without the
  subtraction), so ``H loc + c`` and ``H Sigma H^T + R`` are the sigma points' mean and
  covariance of ``o``; exact for affine ``m``, moment-matching to degree three otherwise.

The other parents are held fixed (a categorical parent as its one-hot or probability vector).
:func:`linear_model` returns the noise as a
:class:`~hybrid_aif.inference.linear_gaussian.DiagonalNoise` (the averaged noise's standard
deviations and, under ``"unscented"``, the residuals' square-root factor), the form in which a
belief is conditioned in state space at ``O(d_obs d²)``; :func:`linearise` returns it as a dense
Cholesky factor. Design: docs/design/recognition-beside-analytic.md, amendment L1.
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax import Array

from hybrid_aif.distributions.gaussian import sigma_points
from hybrid_aif.inference.linear_gaussian import DiagonalNoise

PROPAGATIONS = ("first_order", "unscented")

#: The sigma points' scaling, the transitions' rule (objectives/hybrid.py): ``kappa = 0`` puts
#: the points at ``loc ± sqrt(d) · chol[:, i]`` with weight ``1/(2d)`` each and none on the centre.
UNSCENTED_KAPPA = 0.0

#: Added to the unscented noise's variances, against round-off in the residual covariance; far
#: below any declared noise floor.
_JITTER = 1e-9


def linear_model(
    decl,
    loc: Array,
    chol: Array,
    parents: Sequence[Array],
    index: int,
    propagation: str = "first_order",
) -> tuple[Array, Array, DiagonalNoise]:
    """``(H, c, noise)`` of ``decl`` read at ``N(loc, chol chol^T)`` on its parent ``index``, the
    noise in the form the state-space update reads (:class:`~hybrid_aif.inference.linear_gaussian.
    DiagonalNoise`).

    ``parents`` is the full parent tuple in the declaration's order; its entry at ``index`` is
    replaced by ``x`` and the others are held fixed. ``H`` is ``(d_obs, d)`` and ``c`` is
    ``(d_obs,)``. Under ``"first_order"`` the noise is ``DiagonalNoise(sigma(loc))``. Under
    ``"unscented"`` it is ``DiagonalNoise(sqrt(sum_i w_i sigma(x_i)^2 + jitter), low_rank)`` with
    ``low_rank`` ``(d_obs, 2d + 1)`` the fit's residuals at the points scaled by ``sqrt(w_i)``,
    so ``low_rank low_rank^T`` is their weighted covariance (rank at most ``2d``). With
    :data:`UNSCENTED_KAPPA` ``= 0`` the centre's weight is zero, so the first column is zero and
    the update runs at dimension ``3d + 1`` where ``3d`` would do; the square roots need ``kappa
    >= 0`` (a negative centre weight would give NaN). Batch over components with ``jax.vmap`` on
    ``loc`` and ``chol``.
    """

    def head(x: Array) -> tuple[Array, Array]:
        ps = list(parents)
        ps[index] = x
        return decl._head(tuple(ps))

    if propagation == "first_order":
        mean, scale = head(loc)
        jac = jax.jacfwd(lambda x: head(x)[0])(loc)
        return jac, mean - jac @ loc, DiagonalNoise(scale)
    if propagation == "unscented":
        points, weights = sigma_points(loc, chol, kappa=UNSCENTED_KAPPA)
        means, scales = jax.vmap(head)(points)
        y_bar = weights @ means
        dx, dy = points - loc, means - y_bar
        P_xy = (dx * weights[:, None]).T @ dy
        H = jax.scipy.linalg.cho_solve((chol, True), P_xy).T
        c = y_bar - H @ loc
        # the residual covariance from the residuals of the affine fit at the points, not as
        # P_yy - H Sigma H^T: the same matrix in exact arithmetic (the points reproduce Sigma),
        # positive semidefinite by construction, and the noise is added outside the subtraction
        resid = dy - dx @ H.T
        low_rank = (resid * jnp.sqrt(weights)[:, None]).T
        scale = jnp.sqrt(weights @ scales**2 + _JITTER)
        return H, c, DiagonalNoise(scale, low_rank)
    raise ValueError(
        f"propagation must be one of {list(PROPAGATIONS)}, got {propagation!r} (design: "
        "recognition-beside-analytic.md L1)"
    )


def linearise(
    decl,
    loc: Array,
    chol: Array,
    parents: Sequence[Array],
    index: int,
    propagation: str = "first_order",
) -> tuple[Array, Array, Array]:
    """``(H, c, R_chol)`` of ``decl`` read at ``N(loc, chol chol^T)`` on its parent ``index``:
    :func:`linear_model` with the noise as its dense lower-triangular Cholesky factor,
    ``(d_obs, d_obs)`` (:meth:`~hybrid_aif.inference.linear_gaussian.DiagonalNoise.chol`), for the
    readers that take a dense noise. Batch over components with ``jax.vmap`` on ``loc`` and
    ``chol``.
    """
    H, c, noise = linear_model(decl, loc, chol, parents, index, propagation)
    return H, c, noise.chol()
