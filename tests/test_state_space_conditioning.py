"""The state-space update of a Gaussian belief on readings of independent noise.

``condition`` with a :class:`~hybrid_aif.inference.linear_gaussian.DiagonalNoise` takes the
update in the square-root information form (matrices of the state's dimension), and with a
Cholesky factor in observation space (the innovation covariance, ``d_obs`` by ``d_obs``). The two
are the same mathematics: the posterior mean and covariance and the log evidence agree, and so do
their gradients. The dense form is the reference here.

Tolerances. Under x64, on well-conditioned problems, the two forms agree to ``1e-9`` (mean),
``1e-12`` (covariance, relative to its largest entry) and ``1e-10`` (log evidence, relative). On
ill-conditioned ones (noise ``1e-3`` against a prior of scale 10, so ``cond(H Σ Hᵀ + R)`` reaches
about ``1e13``) the dense form is the less accurate of the two, off a 50-digit reference by up to
``4e-6`` in the mean and ``8e2`` in a log evidence of ``-1e9`` where the state-space form is off by
``2e-15`` and ``1e-5`` (measured), so the comparison there is at ``2e-5`` (mean) and ``5e-6``
(relative log evidence). In float32 (the library's dtype) the state-space form is compared with the
x64 dense reference: ``3e-5`` (mean, relative to ``max(1, |mean|)``), ``1e-5`` (covariance) and
``1e-5`` (relative log evidence); the dense float32 form does not meet these on the ill-conditioned
problems (it returns NaN at noise ``1e-3``).
"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import Gaussian
from hybrid_aif.distributions import MixtureGaussian, MLPConditional
from hybrid_aif.inference import DiagonalNoise, condition
from hybrid_aif.inference.linear_gaussian import condition_on_expected
from hybrid_aif.inference.linearise import linear_model, linearise
from hybrid_aif.inference.mixture_filter import (
    condition_mixture_components,
    condition_mixture_switching_components,
)
from tests.test_neural_likelihood_inference import X0, gaussian_system
from tests.test_recognition_analytic import OBS, PRIORS, build

WELL = ((1.0, 1.0), (0.04, 1.0))  # (noise scale, prior scale)
ILL = ((1e-3, 10.0),)
SHAPES = ((1, 1), (1, 6), (7, 3), (7, 10), (64, 6), (300, 10))  # (d_obs, d): some d_obs < d
RANKS = (0, 5)  # the noise's correlated part: none, or rank 5


def problem(seed, p, d, rscale, pscale, k=0):
    """A random belief ``N(m, L Lᵀ)``, head, offset, noise and reading, as float64 numpy."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((d, d))
    P = a @ a.T * pscale**2 / d + 1e-2 * pscale**2 * np.eye(d)
    return dict(
        m=rng.standard_normal(d),
        L=np.linalg.cholesky(P),
        H=rng.standard_normal((p, d)),
        c=rng.standard_normal(p),
        r=rscale * np.exp(0.5 * rng.standard_normal(p)),
        V=None if k == 0 else 0.1 * rng.standard_normal((p, k)),
        y=2.0 * rng.standard_normal(p),
    )


def arrays(pr, dtype):
    f = lambda x: None if x is None else jnp.asarray(x, dtype)  # noqa: E731
    g = Gaussian(loc=f(pr["m"]), chol=f(pr["L"]))
    return g, f(pr["H"]), f(pr["c"]), DiagonalNoise(f(pr["r"]), f(pr["V"])), f(pr["y"])


def update(pr, dtype, state_space):
    """``(mean, cov, log evidence)`` as float64 numpy, from one form of :func:`condition`."""
    g, H, c, noise, y = arrays(pr, dtype)
    post, ll = condition(g, H, c, noise if state_space else noise.chol(), y)
    return np.asarray(post.loc, np.float64), np.asarray(post.cov, np.float64), float(ll)


def errors(got, want):
    """``(mean, covariance, log evidence)`` errors: mean absolute relative to ``max(1, |mean|)``,
    covariance absolute relative to its largest entry, log evidence relative."""
    (m1, P1, l1), (m2, P2, l2) = got, want
    return (
        np.max(np.abs(m1 - m2)) / max(1.0, np.max(np.abs(m2))),
        np.max(np.abs(P1 - P2)) / np.max(np.abs(P2)),
        abs(l1 - l2) / max(1.0, abs(l2)),
    )


@pytest.mark.parametrize("k", RANKS)
@pytest.mark.parametrize("p,d", SHAPES)
@pytest.mark.parametrize("rscale,pscale", WELL + ILL)
def test_the_state_space_update_is_the_dense_one_under_x64(p, d, rscale, pscale, k):
    tol = (1e-9, 1e-12, 1e-10) if (rscale, pscale) in WELL else (2e-5, 1e-8, 5e-6)
    for seed in range(2):
        pr = problem(seed, p, d, rscale, pscale, k)
        with jax.enable_x64(True):
            got, want = update(pr, jnp.float64, True), update(pr, jnp.float64, False)
        for e, t, what in zip(errors(got, want), tol, ("mean", "cov", "log evidence"), strict=True):
            assert e <= t, (what, e, t)


@pytest.mark.parametrize("k", RANKS)
@pytest.mark.parametrize("p,d", SHAPES)
@pytest.mark.parametrize("rscale,pscale", WELL + ILL)
def test_the_state_space_update_in_float32_is_close_to_the_float64_reference(
    p, d, rscale, pscale, k
):
    for seed in range(2):
        pr = problem(seed, p, d, rscale, pscale, k)
        with jax.enable_x64(True):
            want = update(pr, jnp.float64, False)
        got = update(pr, jnp.float32, True)
        tol = (3e-5, 1e-5, 1e-5)
        for e, t, what in zip(errors(got, want), tol, ("mean", "cov", "ll"), strict=True):
            assert e <= t, (what, e, t)


def test_the_posterior_factor_is_lower_triangular_with_a_positive_diagonal():
    for k in RANKS:
        g, H, c, noise, y = arrays(problem(3, 50, 6, 0.04, 1.0, k), jnp.float32)
        chol = np.asarray(condition(g, H, c, noise, y)[0].chol)
        assert np.array_equal(chol, np.tril(chol))
        assert np.all(np.diagonal(chol) > 0)


def test_the_expected_reading_update_is_the_dense_one():
    with jax.enable_x64(True):
        for k in RANKS:
            g, H, _, noise, _ = arrays(problem(4, 40, 5, 0.1, 1.0, k), jnp.float64)
            got = condition_on_expected(g, H, noise)
            want = condition_on_expected(g, H, noise.chol())
            np.testing.assert_array_equal(np.asarray(got.loc), np.asarray(g.loc))
            np.testing.assert_allclose(np.asarray(got.cov), np.asarray(want.cov), atol=1e-13)


# ------------------------------------------------------------------ no d_obs by d_obs array


def _shapes(jaxpr):
    """Every intermediate array shape of a closed jaxpr, sub-jaxprs included."""
    out = []
    for eqn in jaxpr.eqns:
        out += [v.aval.shape for v in eqn.outvars if hasattr(v.aval, "shape")]
        for param in eqn.params.values():
            subs = param if isinstance(param, tuple | list) else (param,)
            for sub in subs:
                if hasattr(sub, "jaxpr"):
                    out += _shapes(sub.jaxpr if hasattr(sub.jaxpr, "eqns") else sub.jaxpr.jaxpr)
                elif hasattr(sub, "eqns"):
                    out += _shapes(sub)
    return out


def _no_square(fn, *args, p):
    shapes = _shapes(jax.make_jaxpr(fn)(*args).jaxpr)
    assert shapes
    big = [s for s in shapes if sum(n >= p for n in s) >= 2]
    assert not big, big


def test_no_d_obs_by_d_obs_array_is_traced():
    p = 4096
    for k in RANKS:
        g, H, c, noise, y = arrays(problem(5, p, 6, 0.04, 1.0, k), jnp.float32)
        _no_square(lambda g, H, c, n, y: condition(g, H, c, n, y), g, H, c, noise, y, p=p)
        _no_square(lambda g, H, n: condition_on_expected(g, H, n), g, H, noise, p=p)
    # the dense form does trace one, so the check can see it
    with pytest.raises(AssertionError):
        _no_square(lambda g, H, c, n, y: condition(g, H, c, n.chol(), y), g, H, c, noise, y, p=p)


def test_a_reading_too_large_for_a_dense_update_is_conditioned_on():
    """``2**16`` readings: the dense innovation covariance alone would be 16 GiB in float32; the
    state-space update holds ``O(d_obs d)`` numbers."""
    p, d = 2**16, 6
    g, H, c, noise, y = arrays(problem(6, p, d, 0.04, 1.0), jnp.float32)
    post, ll = jax.jit(condition)(g, H, c, noise, y)
    assert np.all(np.isfinite(np.asarray(post.cov))) and np.isfinite(float(ll))
    # every reading informs the state: the posterior is far tighter than the prior
    assert float(jnp.max(jnp.diagonal(post.cov))) < 1e-3 * float(jnp.min(jnp.diagonal(g.cov)))


INFER_P = 512  # readings of the neural likelihood the infer-face traces read


def _infer_free_energy(system_kind, propagation, p):
    """``(free_energy(weights, y), weights)``: the free energy ``infer_states`` reports for a
    reading ``y`` through a neural likelihood of ``p`` readings, as a function of the network's
    weights (what ``learn`` differentiates). ``system_kind``: ``"gaussian"`` (a Gaussian-carried
    state factor), ``"mixture"`` (a mixture-carried one, no mode parent) or ``"mode parent"`` (a
    mixture-carried one read through a categorical state parent, the switching update)."""
    key = jax.random.PRNGKey(5)
    if system_kind == "gaussian":
        decl = MLPConditional.from_shapes(
            (2,), p, key=key, width=8, depth=1, propagation=propagation
        )
        system, params = gaussian_system(decl)
        obs, priors, name = {}, {"x": X0}, "o"
    else:
        deps = (1,) if system_kind == "mixture" else (1, 0)
        decl = MLPConditional.from_shapes(
            (2, 3) if len(deps) == 2 else (2,),
            p,
            key=key,
            width=8,
            depth=1,
            propagation=propagation,
        )
        system, params = build()
        system = dataclasses.replace(system, A_deps=(*system.A_deps[:2], deps))
        obs, priors, name = OBS, PRIORS, "img"
    dyn, static = eqx.partition(decl, eqx.is_inexact_array)

    def free_energy(dyn, y):
        prm = dataclasses.replace(
            params, likelihood={**params.likelihood, name: eqx.combine(dyn, static)}
        )
        _, info = system.infer_states(prm, {**obs, name: y}, priors=priors, detail=True)
        return info.free_energy

    return free_energy, dyn


@pytest.mark.parametrize("pass_", ("forward", "gradient"))
@pytest.mark.parametrize("propagation", ("first_order", "unscented"))
@pytest.mark.parametrize("system_kind", ("gaussian", "mixture", "mode parent"))
def test_the_infer_face_traces_no_d_obs_by_d_obs_array_for_a_neural_likelihood(
    system_kind, propagation, pass_
):
    free_energy, dyn = _infer_free_energy(system_kind, propagation, INFER_P)
    fn = free_energy if pass_ == "forward" else jax.grad(free_energy)
    _no_square(fn, dyn, jnp.zeros(INFER_P), p=INFER_P)


def test_the_infer_face_detector_fires_when_the_dense_noise_is_forced(monkeypatch):
    """The positive control of the test above: with the neural noise handed to the update as its
    dense Cholesky factor, the same trace holds a ``d_obs`` by ``d_obs`` array. Traced fresh: a
    reading length no other trace here uses, and the caches cleared, so no earlier trace is
    reused."""
    import hybrid_aif.system.faces.infer as infer_face

    def dense(*args, **kwargs):
        H, c, noise = linear_model(*args, **kwargs)
        return H, c, noise.chol()

    p = INFER_P + 7
    jax.clear_caches()
    monkeypatch.setattr(infer_face, "linear_model", dense)
    free_energy, dyn = _infer_free_energy("gaussian", "first_order", p)
    with pytest.raises(AssertionError):
        _no_square(free_energy, dyn, jnp.zeros(p), p=p)
    jax.clear_caches()


# ------------------------------------------------------------------ gradients, jit, vmap


def _scalar(state_space):
    def f(loc, chol, H, scale, y):
        noise = DiagonalNoise(scale)
        post, ll = condition(
            Gaussian(loc=loc, chol=chol),
            H,
            0.1 * H[:, 0],
            noise if state_space else noise.chol(),
            y,
        )
        return ll + jnp.sum(post.loc**2) + jnp.sum(jnp.sin(post.cov))

    return f


@pytest.mark.parametrize("rscale", (1.0, 0.04))
def test_gradients_agree_with_the_dense_form(rscale):
    with jax.enable_x64(True):
        g, H, _, noise, y = arrays(problem(7, 30, 4, rscale, 1.0), jnp.float64)
        args = (g.loc, g.chol, H, noise.scale, y)
        got = jax.grad(_scalar(True), argnums=range(5))(*args)
        want = jax.grad(_scalar(False), argnums=range(5))(*args)
    for a, b in zip(got, want, strict=True):
        a, b = np.asarray(a), np.asarray(b)
        if b.ndim == 2 and b.shape[0] == b.shape[1]:  # a triangular factor: its lower part
            a, b = np.tril(a), np.tril(b)
        np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-8 * np.max(np.abs(b)))


@pytest.mark.parametrize("propagation", ("first_order", "unscented"))
def test_gradients_in_a_decoder_s_weights_agree_through_its_linearisation(propagation):
    """What ``learn`` differentiates: the log evidence of a reading through a neural likelihood,
    in the network's weights, through :func:`linear_model` and the update."""
    with jax.enable_x64(True):
        decl = MLPConditional.from_shapes(
            (3,), 40, key=jax.random.PRNGKey(1), width=8, depth=1, propagation=propagation
        )
        decl = jax.tree.map(lambda a: jnp.asarray(a, jnp.float64), decl)
        g = Gaussian(loc=jnp.asarray([0.2, -0.1, 0.3]), chol=0.3 * jnp.eye(3))
        y = jnp.asarray(np.random.default_rng(8).standard_normal(40))

        def evidence(decl, state_space):
            H, c, noise = linear_model(decl, g.loc, g.chol, (g.loc,), 0, propagation)
            post, ll = condition(g, H, c, noise if state_space else noise.chol(), y)
            return ll + jnp.sum(post.cov)

        got = jax.grad(lambda d: evidence(d, True))(decl)
        want = jax.grad(lambda d: evidence(d, False))(decl)
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-7, atol=1e-9)


def test_jit_and_vmap_give_the_eager_update():
    probs = [arrays(problem(s, 20, 3, 0.1, 1.0, 2), jnp.float32) for s in range(4)]
    stacked = jax.tree.map(lambda *a: jnp.stack(a), *probs)
    batched = jax.vmap(condition)(*stacked)
    jitted = jax.jit(condition)(*probs[2])
    for i, args in enumerate(probs):
        eager = condition(*args)
        for a, b in zip(jax.tree.leaves(eager), jax.tree.leaves(batched), strict=True):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b[i]), rtol=1e-5, atol=1e-6)
    for a, b in zip(jax.tree.leaves(condition(*probs[2])), jax.tree.leaves(jitted), strict=True):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-6)


def test_batch_axes_broadcast_as_in_the_dense_form():
    with jax.enable_x64(True):
        g, H, c, noise, y = arrays(problem(9, 12, 3, 0.2, 1.0), jnp.float64)
        g2 = Gaussian(
            loc=jnp.stack([g.loc, -g.loc])[:, None], chol=jnp.stack([g.chol] * 2)[:, None]
        )
        H3 = jnp.stack([H, 2 * H, 0.5 * H])  # (3, p, d): broadcasts against (2, 1)
        n3 = DiagonalNoise(jnp.stack([noise.scale, 2 * noise.scale, noise.scale]))
        got, ll = condition(g2, H3, c, n3, y)
        want, ll_d = condition(g2, H3, c, n3.chol(), y)
        assert got.loc.shape == (2, 3, 3) and ll.shape == (2, 3)
        np.testing.assert_allclose(np.asarray(got.cov), np.asarray(want.cov), atol=1e-12)
        np.testing.assert_allclose(np.asarray(ll), np.asarray(ll_d), rtol=1e-11)


# ------------------------------------------------------------------ mixtures


def _mixture(rng, k, d):
    a = rng.standard_normal((k, d, d))
    covs = a @ np.swapaxes(a, -1, -2) / d + 0.1 * np.eye(d)
    return MixtureGaussian(
        log_weights=jnp.asarray(np.log(rng.dirichlet(np.ones(k)))),
        means=jnp.asarray(rng.standard_normal((k, d))),
        chols=jnp.asarray(np.linalg.cholesky(covs)),
    )


def test_a_mixture_update_with_a_head_per_component_is_the_dense_one():
    with jax.enable_x64(True):
        rng = np.random.default_rng(10)
        k, p, d = 3, 25, 4
        m = _mixture(rng, k, d)
        H = jnp.asarray(rng.standard_normal((k, p, d)))
        c = jnp.asarray(rng.standard_normal((k, p)))
        noise = DiagonalNoise(jnp.asarray(0.1 + rng.random((k, p))))
        y = jnp.asarray(rng.standard_normal(p))
        got = condition_mixture_components(m, H, c, noise, y)
        want = condition_mixture_components(m, H, c, noise.chol(), y)
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-9, atol=1e-11)


def test_a_switching_update_with_a_head_per_component_and_mode_is_the_dense_one():
    with jax.enable_x64(True):
        rng = np.random.default_rng(11)
        k, modes, p, d = 2, 3, 25, 4
        m = _mixture(rng, k, d)
        H = jnp.asarray(rng.standard_normal((k, modes, p, d)))
        c = jnp.asarray(rng.standard_normal((k, modes, p)))
        noise = DiagonalNoise(jnp.asarray(0.1 + rng.random((k, modes, p))))
        y = jnp.asarray(rng.standard_normal(p))
        log_q = jnp.log(jnp.asarray([0.2, 0.5, 0.3]))
        got = condition_mixture_switching_components(m, H, c, noise, y, log_q, per_component=True)
        want = condition_mixture_switching_components(
            m, H, c, noise.chol(), y, log_q, per_component=True
        )
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want), strict=True):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-9, atol=1e-11)


# ------------------------------------------------------------------ the unscented reading


def test_the_unscented_noise_is_the_dense_reading_s_noise():
    """:func:`linear_model`'s noise, ``diag(scale²) + low_rank low_rankᵀ``, is the covariance whose
    Cholesky factor :func:`linearise` returns, and its rank is at most ``2d``."""
    with jax.enable_x64(True):
        decl = MLPConditional.from_shapes(
            (3,), 30, key=jax.random.PRNGKey(2), width=8, depth=1, propagation="unscented"
        )
        decl = jax.tree.map(lambda a: jnp.asarray(a, jnp.float64), decl)
        loc, chol = jnp.asarray([0.1, 0.4, -0.2]), 0.5 * jnp.eye(3)
        _, _, noise = linear_model(decl, loc, chol, (loc,), 0, "unscented")
        _, _, R_chol = linearise(decl, loc, chol, (loc,), 0, "unscented")
        cov, dense = np.asarray(noise.cov()), np.asarray(R_chol @ R_chol.T)
    np.testing.assert_allclose(cov, dense, atol=1e-14)
    assert noise.low_rank.shape == (30, 7)
    assert np.linalg.matrix_rank(np.asarray(noise.low_rank), tol=1e-12) <= 6


# ------------------------------------------------------------------ the planner's chain


def test_a_neural_modality_selected_by_an_action_node_conditions_the_chain_in_state_space():
    """The selecting scalar model of ``test_action_node_modalities`` with its reading declared as
    an affine network over ``(x, one-hot level)``, whose scale is the level's: the epistemic
    chain conditions on the network's diagonal noise at each step's level (the chain's seed holds
    that noise's shape before a level is read), and each step's gain is the hand Kalman chain's."""
    from hybrid_aif.distributions.neural import _inverse_softplus
    from tests.test_action_node_modalities import (
        LEVEL_SD,
        _hand_chain,
        _infogain_by_step,
        scalar_model,
    )

    system, params = scalar_model(belief="posterior")
    decl = MLPConditional.from_shapes((1, 3), 1, key=jax.random.PRNGKey(3), depth=0)
    raw = [_inverse_softplus(sd - decl.min_scale) for sd in LEVEL_SD]
    weight = jnp.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, *raw]])
    decl = eqx.tree_at(
        lambda d: (d.layers[0].weight, d.layers[0].bias), decl, (weight, jnp.zeros(2))
    )
    decl = dataclasses.replace(decl, frozen=True)
    params = dataclasses.replace(params, likelihood={"y": decl})
    for levels in ((0, 1, 2), (1, 1, 0)):
        np.testing.assert_allclose(
            _infogain_by_step(system, params, levels), _hand_chain(levels), rtol=2e-5
        )


def test_a_diagonal_noise_of_the_wrong_shape_is_refused_with_a_sentence():
    g, H, c, noise, y = arrays(problem(12, 9, 3, 0.1, 1.0, 2), jnp.float32)
    with pytest.raises(ValueError, match="not a\\s+Cholesky factor"):
        condition(g, H, c, DiagonalNoise(jnp.diag(noise.scale)), y)
    with pytest.raises(ValueError, match="must be equal"):
        condition(g, H, c, DiagonalNoise(noise.scale[:-1]), y)
    with pytest.raises(ValueError, match="must be equal"):
        condition(g, H, c, noise, y[:-1])
    with pytest.raises(ValueError, match="low_rank has shape"):
        condition(g, H, c, DiagonalNoise(noise.scale, noise.low_rank[:-1]), y)
    with pytest.raises(ValueError, match="not a\\s+Cholesky factor"):
        condition_on_expected(g, H, DiagonalNoise(jnp.diag(noise.scale)))
