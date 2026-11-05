"""The information gain of a diagonal-noise neural modality in low-rank form.

A predictive component's covariance is ``diag(D) + U U^T`` with ``U`` of the state's few columns,
so the Kolchinsky-Tracey entropy pair needs only small factorisations
(:class:`~hybrid_aif.distributions.LowRankMixture`). Pinned: its entropy interval, gradients and
pairwise distances equal the dense mixture's; the neural modality's information-gain leaf equals
the dense path's; the cost is linear in the observation dimension.

Design: docs/design/recognition-beside-analytic.md L9.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_aif import Gaussian, LinearGaussianObservation
from hybrid_aif.distributions import Categorical, LowRankMixture, MixtureGaussian
from hybrid_aif.objectives import hybrid as hybrid_module
from hybrid_aif.objectives.continuous import (
    ComponentHead,
    DiagonalNoiseObservation,
    _predictive_lowrank,
    _predictive_obs_switching,
    conditional_obs_entropy,
    predictive_entropy,
)
from tests.test_neural_likelihood_planning import _mixture_params, _mixture_system, curved

P, D, K, M = 36, 4, 3, 3


def _head_and_belief(key, p=P):
    ks = jax.random.split(key, 6)
    H = jax.random.normal(ks[0], (K, M, p, D)) / jnp.sqrt(D)
    c = jax.random.normal(ks[1], (K, M, p))
    scale = 0.3 + jax.random.uniform(ks[2], (K, M, p))
    q_chol = jnp.tril(0.5 * jax.random.normal(ks[3], (K, D, D))) + jnp.eye(D)
    q = MixtureGaussian(
        log_weights=jax.random.normal(ks[4], (K,)),
        means=jax.random.normal(ks[5], (K, D)),
        chols=q_chol,
    )
    mode_logq = jnp.log(jnp.asarray([0.2, 0.5, 0.3]))
    head = ComponentHead(H, c, jax.vmap(jax.vmap(jnp.diag))(scale), scale)
    return q, head, mode_logq


def _both(q, head, mode_logq):
    dense = _predictive_obs_switching(q, head.H, head.c, head.R_chol, mode_logq, per_component=True)
    low = _predictive_lowrank(q, head, mode_logq)
    return dense, low


def test_the_entropy_interval_and_its_gradients_match_the_dense_mixture():
    q, head, mode_logq = _head_and_belief(jax.random.PRNGKey(0))
    dense, low = _both(q, head, mode_logq)
    a, b = dense.entropy(), low.entropy()
    np.testing.assert_allclose(b.lower, a.lower, rtol=1e-5)
    np.testing.assert_allclose(b.upper, a.upper, rtol=1e-5)

    def via(kind, which):
        def f(means, H):
            qq = MixtureGaussian(q.log_weights, means, q.chols)
            hh = head._replace(H=H)
            mix = _both(qq, hh, mode_logq)[kind]
            iv = mix.entropy()
            return getattr(iv, which)

        return f

    for which in ("lower", "upper"):
        gd = jax.grad(via(0, which), argnums=(0, 1))(q.means, head.H)
        gl = jax.grad(via(1, which), argnums=(0, 1))(q.means, head.H)
        for x, y in zip(gd, gl, strict=True):
            np.testing.assert_allclose(y, x, rtol=1e-4, atol=1e-5)


def test_the_pairwise_kl_and_bhattacharyya_match_the_dense_mixture():
    q, head, mode_logq = _head_and_belief(jax.random.PRNGKey(1))
    dense, low = _both(q, head, mode_logq)
    np.testing.assert_allclose(low.pairwise_kl(), dense._pairwise_kl(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        low.pairwise_bhattacharyya(), dense._pairwise_bhattacharyya(), rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        low.component_entropies(), dense.components.entropy().lower, rtol=1e-5
    )


def test_a_batched_mixture_equals_the_dense_mixture_slice_by_slice():
    batch, n, p, r = (2, 3), 5, 30, 3
    ks = jax.random.split(jax.random.PRNGKey(11), 4)
    lw = jax.random.normal(ks[0], batch + (n,))
    mu = 0.3 * jax.random.normal(ks[1], batch + (n, p))
    dg = jnp.exp(jax.random.uniform(ks[2], batch + (n, p), minval=-2.0, maxval=1.0))
    u = jax.random.normal(ks[3], batch + (n, p, r))
    low = LowRankMixture(lw, mu, dg, u).entropy()
    cov = jnp.einsum("...p,pq->...pq", dg, jnp.eye(p)) + u @ jnp.swapaxes(u, -1, -2)
    dense = MixtureGaussian(lw, mu, jnp.linalg.cholesky(cov)).entropy()
    np.testing.assert_allclose(low.lower, dense.lower, rtol=1e-4)
    np.testing.assert_allclose(low.upper, dense.upper, rtol=1e-4)
    assert float(jnp.ptp(low.upper)) > 0.0  # the slices differ


def test_a_batched_belief_with_a_shared_head_matches_the_dense_path():
    q, head, _ = _head_and_belief(jax.random.PRNGKey(6))
    head = jax.tree_util.tree_map(lambda a: a, head._replace(H=head.H[:, :2], c=head.c[:, :2]))
    head = head._replace(R_chol=head.R_chol[:, :2], R_diag=head.R_diag[:, :2])
    qb = MixtureGaussian(
        log_weights=jax.random.normal(jax.random.PRNGKey(8), (4, K)),
        means=jnp.broadcast_to(q.means, (4, K, D)) + jnp.arange(4.0)[:, None, None] * 0.2,
        chols=jnp.broadcast_to(q.chols, (4, K, D, D)),
    )
    modes = (
        Categorical.from_probs(jax.nn.softmax(jax.random.normal(jax.random.PRNGKey(9), (4, 2)))),
    )
    low = predictive_entropy(qb, head, modes)
    dense = predictive_entropy(qb, head._replace(R_diag=None), modes)
    assert low.upper.shape == (4,)
    np.testing.assert_allclose(low.lower, dense.lower, rtol=1e-4)
    np.testing.assert_allclose(low.upper, dense.upper, rtol=1e-4)


def test_a_gaussian_belief_through_a_diagonal_neural_head_is_exact_and_low_rank():
    q, head, _ = _head_and_belief(jax.random.PRNGKey(10))
    g = Gaussian(loc=q.means[0], chol=q.chols[0])
    H, c, R = head.H[0, 0], head.c[0, 0], head.R_chol[0, 0]
    diag = DiagonalNoiseObservation(H=H, c=c, noise=R, scale=head.R_diag[0, 0])
    plain = LinearGaussianObservation(H=H, c=c, noise=R)
    a, b = predictive_entropy(g, diag), predictive_entropy(g, plain)
    np.testing.assert_allclose(a.lower, b.lower, rtol=1e-5)
    assert float(a.lower) == float(a.upper)


def test_single_precision_at_a_thousand_pixels_keeps_the_interval_ordered():
    p = 1024
    ks = jax.random.split(jax.random.PRNGKey(0), 3)
    h = (jax.random.normal(ks[0], (1, 1, p, D)) / 2).astype(jnp.float32)
    scale = jnp.exp(
        jax.random.uniform(ks[1], (1, 1, p), minval=np.log(1e-3**0.5), maxval=np.log(10**0.5))
    ).astype(jnp.float32)
    eye = jnp.eye(D, dtype=jnp.float32)
    for eps in (0.0, 1e-3, 1e-1):
        mix = LowRankMixture(
            log_weights=jnp.zeros(2, jnp.float32),
            means=jnp.stack([jnp.zeros(p), jnp.full(p, 0.0).at[:D].set(eps)]).astype(jnp.float32),
            diag=jnp.broadcast_to(scale[0] ** 2, (2, p)),
            factors=jnp.broadcast_to(h[0, 0] @ eye, (2, p, D)),
        )
        assert float(jnp.max(jnp.abs(jnp.diagonal(mix.pairwise_kl())))) == 0.0
        iv = mix.entropy()
        assert float(iv.lower) <= float(iv.upper) + 1e-6
    # one component: the interval collapses to the exact entropy
    one = LowRankMixture(
        jnp.zeros(1, jnp.float32), jnp.zeros((1, p), jnp.float32), scale[0] ** 2, h[0] * 1.0
    )
    iv = one.entropy()
    assert float(iv.upper) == float(iv.lower)


def test_predictive_entropy_routes_a_diagonal_head_low_rank_and_a_dense_one_dense():
    q, head, mode_logq = _head_and_belief(jax.random.PRNGKey(3))
    mode = (Categorical.from_probs(jnp.exp(mode_logq)),)
    low = predictive_entropy(q, head, mode)
    dense = predictive_entropy(q, head._replace(R_diag=None), mode)
    np.testing.assert_allclose(low.lower, dense.lower, rtol=1e-5)
    np.testing.assert_allclose(low.upper, dense.upper, rtol=1e-5)
    np.testing.assert_allclose(
        conditional_obs_entropy(head, mode, q=q),
        conditional_obs_entropy(head._replace(R_diag=None), mode, q=q),
        rtol=1e-5,
    )


def test_the_information_gain_leaf_of_a_neural_modality_equals_the_dense_value(monkeypatch):
    decl = curved(parent_dims=(2, 3))
    params = _mixture_params(decl, target=False)

    def leaf():
        jax.clear_caches()
        system = _mixture_system(decomposition="utility_infogain")
        _, tree = system.infer_policies(params, system.initial_priors(params))
        return tree["infogain"]["img"]

    low = leaf()
    original = hybrid_module.neural_model

    def dense_model(head, q):
        m = original(head, q)
        return m._replace(R_diag=None) if isinstance(m, ComponentHead) else m

    monkeypatch.setattr(hybrid_module, "neural_model", dense_model)
    dense = leaf()
    monkeypatch.undo()
    jax.clear_caches()
    assert np.all(np.isfinite(np.asarray(low.midpoint)))
    np.testing.assert_allclose(low.lower, dense.lower, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(low.upper, dense.upper, rtol=1e-4, atol=1e-5)


def _timed_entropy(p, r=4, n=9):
    ks = jax.random.split(jax.random.PRNGKey(4), 4)
    mix = LowRankMixture(
        log_weights=jax.random.normal(ks[0], (n,)),
        means=jax.random.normal(ks[1], (n, p)),
        diag=0.5 + jax.random.uniform(ks[2], (n, p)),
        factors=jax.random.normal(ks[3], (n, p, r)) / jnp.sqrt(r),
    )
    f = jax.jit(lambda m: m.entropy())
    jax.block_until_ready(f(mix))
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        out = jax.block_until_ready(f(mix))
        best = min(best, time.perf_counter() - t0)
    assert np.isfinite(float(out.lower)) and float(out.lower) <= float(out.upper)
    return best


def test_the_entropy_at_four_thousand_pixels_runs_in_under_a_second_and_scales_linearly():
    big, small = _timed_entropy(4096), _timed_entropy(1024)
    assert big < 1.0, big
    assert big < 6.0 * small, (big, small)
