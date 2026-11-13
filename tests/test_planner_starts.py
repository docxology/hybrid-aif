"""Warm starts of the mixed and gradient planners: several deterministic continuous starts per
discrete policy, each (policy, start) pair a policy of its own.

Design: docs/design/mixed-control-planning.md, "Warm starts".
"""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.controls import Box
from hybrid_aif.distributions import Gaussian, MixtureGaussian
from hybrid_aif.factors import ConditionalGaussian, LinearGaussianTransition
from hybrid_aif.planning import (
    gradient_plan,
    make_mixed_planner,
    mixed_plan,
    policy_posterior,
)
from hybrid_aif.planning.gradient import spread_starts
from hybrid_aif.system import System, SystemParams
from tests.test_mixed_planner import STEPS, fix_levels, planned, toy

MODES = (-1.0, 1.6)


def bimodal():
    """A one-dimensional state that the continuous control sets, with a two-component Gaussian
    mixture as its target: ``G`` over the control has a minimum at each mode."""
    system = System(
        B_deps=((1, 2), (), ()),
        controls=(1, 2),
        action_names=("d", "c"),
        state_names=("x",),
        control_names=("u_d", "u_c"),
        policy_len=1,
        epistemic=False,
        planner="mixed",
    )
    transition = LinearGaussianTransition(
        F=jnp.zeros((1, 1)),
        innovations=ConditionalGaussian(
            loc=jnp.zeros((1, 1)), chol=0.1 * jnp.ones((1, 1, 1)), parent_ndims=1
        ),
        G=jnp.ones((1, 1)),
    )
    target = MixtureGaussian(
        log_weights=jnp.log(jnp.asarray([0.3, 0.7])),
        means=jnp.asarray([[m] for m in MODES]),
        chols=0.3 * jnp.ones((2, 1, 1)),
    )
    params = SystemParams(
        transition={"x": transition},
        initial_state_prior={"x": Gaussian(loc=jnp.zeros(1), chol=0.2 * jnp.eye(1))},
        state_targets={"x": target},
        control={"u_c": Box(-2 * jnp.ones(1), 2 * jnp.ones(1))},
    )
    return system, params


def test_one_start_is_the_single_start_result_exactly():
    ctx = planned(*toy(2))
    base = mixed_plan(ctx, steps=STEPS, init=jnp.asarray([0.5]))
    one = mixed_plan(ctx, steps=STEPS, init=jnp.asarray([0.5]), starts=1)
    stacked = mixed_plan(ctx, steps=STEPS, init=jnp.full((1, 2, 1), 0.5))
    for other in (one, stacked):
        for name in ("plan", "discrete", "trace", "gradient_trace", "best_step"):
            np.testing.assert_array_equal(
                np.asarray(getattr(base, name)), np.asarray(getattr(other, name))
            )
        np.testing.assert_array_equal(np.asarray(base.G.lower), np.asarray(other.G.lower))
        np.testing.assert_array_equal(np.asarray(base.q_pi.logits), np.asarray(other.q_pi.logits))
    assert base.plan.shape == (9, 2, 1)


def test_several_starts_keep_the_better_mode_and_one_start_keeps_the_worse():
    ctx = planned(*bimodal())
    single = mixed_plan(ctx)
    several = mixed_plan(ctx, starts=4)
    assert single.plan.shape == several.plan.shape == (1, 1, 1)
    assert float(single.plan[0, 0, 0]) == pytest.approx(MODES[0], abs=0.05)
    for starts in (3, 4):
        kept = mixed_plan(ctx, starts=starts)
        assert float(kept.plan[0, 0, 0]) == pytest.approx(MODES[1], abs=0.05)
    assert float(several.G.midpoint[0]) < float(single.G.midpoint[0]) - 0.1
    # every start's outcome is kept; the plan is the best of them
    assert several.start_plans.shape == (1, 4, 1, 1) and several.start_scores.shape == (1, 4)
    found = np.asarray(several.start_plans).ravel()
    for mode in MODES:
        assert np.any(np.abs(found - mode) < 0.05), (mode, found)
    assert float(several.start_scores.min()) == pytest.approx(float(several.start_scores[0].min()))
    np.testing.assert_allclose(np.asarray(several.q_pi.probs).sum(), 1.0, rtol=1e-5)
    assert single.start_scores.shape == (1, 1)


def test_spread_starts_are_a_latin_hypercube_inside_the_box():
    for starts in (2, 3, 4, 8):
        count = starts - 1
        for horizon, width in ((3, 2), (5, 2), (1, 1)):
            for lo, hi in ((-2.0, 2.0), (3.0, 5.0), (-9.0, -7.0)):
                low, high = jnp.full(width, lo), jnp.full(width, hi)
                pts = np.asarray(spread_starts(count, horizon, low, high), np.float64)
                assert pts.shape == (count, horizon, width)
                flat = (pts.reshape(count, -1) - lo) / (hi - lo)
                assert np.all(flat > 0.0) and np.all(flat < 1.0)
                shift = 0.75 if count == 1 else 0.5
                want = (np.arange(count) + shift) / count
                np.testing.assert_allclose(
                    np.sort(flat, axis=0), want[:, None] + 0 * flat, atol=1e-6
                )
                again = np.asarray(spread_starts(count, horizon, low, high), np.float64)
                np.testing.assert_array_equal(pts, again)


def test_the_posterior_is_over_the_discrete_policies_and_the_prior_is_untouched():
    system, params = toy(1)
    prior = jnp.log(jnp.asarray([0.2, 0.3, 0.5]))
    ctx = planned(system, dataclasses.replace(params, policy_prior=prior))
    result = mixed_plan(ctx, steps=STEPS, starts=3)
    assert result.plan.shape == (3, 1, 1) and result.q_pi.probs.shape == (3,)
    np.testing.assert_array_equal(np.asarray(result.discrete[:, 0, 0]), np.arange(3))
    np.testing.assert_allclose(np.asarray(result.q_pi.probs).sum(), 1.0, rtol=1e-5)
    want = policy_posterior(result.G, ctx.gamma, prior)
    np.testing.assert_allclose(np.asarray(result.q_pi.logits), np.asarray(want.logits), rtol=1e-6)
    # each policy's G is the best of its starts' scores
    np.testing.assert_allclose(
        np.asarray(result.G.midpoint),
        np.asarray(result.start_scores.min(axis=1)),
        rtol=1e-5,
        atol=1e-5,
    )


def test_explicit_init_with_a_leading_starts_axis_is_honoured():
    ctx = planned(*toy(2))
    given = jnp.asarray([[[0.5], [0.5]], [[-1.0], [1.0]], [[1.5], [-1.5]]])
    held = mixed_plan(ctx, steps=0, init=given)
    assert held.plan.shape == (9, 2, 1) and held.start_plans.shape == (9, 3, 2, 1)
    np.testing.assert_array_equal(
        np.asarray(held.start_plans), np.broadcast_to(np.asarray(given), (9, 3, 2, 1))
    )
    assert mixed_plan(ctx, steps=0, init=given, starts=3).start_plans.shape == (9, 3, 2, 1)
    with pytest.raises(ValueError, match="leading axis"):
        mixed_plan(ctx, steps=0, init=given, starts=4)


def test_the_planner_face_passes_starts_through():
    ctx = planned(*toy(1))
    q_pi, _, policies = make_mixed_planner(steps=STEPS, starts=2)(ctx)
    assert q_pi.probs.shape == (3,)
    assert policies.entries[0].shape == (3, 1) and policies.entries[1].shape == (3, 1, 1)


def test_the_gradient_planner_returns_the_best_of_its_starts_and_the_spread():
    system, params = bimodal()
    ctx = planned(system, params)
    # a continuous-only problem: the bimodal control with its discrete node held at level 0
    sub = fix_levels(ctx, jnp.zeros((1, 1), jnp.int32))
    single = gradient_plan(sub, steps=STEPS)
    assert single.start_scores.shape == (1,)
    assert float(single.plan[0, 0]) == pytest.approx(MODES[0], abs=0.05)
    several = gradient_plan(sub, steps=STEPS, starts=4)
    assert several.start_scores.shape == (4,)
    assert float(several.plan[0, 0]) == pytest.approx(MODES[1], abs=0.05)
    assert float(several.G.midpoint[0]) < float(single.G.midpoint[0]) - 0.1
    assert float(jnp.min(several.start_scores)) == pytest.approx(float(several.G.midpoint[0]))
    # the first start is the single descent's own
    assert float(several.start_scores[0]) == pytest.approx(float(single.start_scores[0]))
    assert single.trace.ndim == 1 and single.plan.ndim == 2
