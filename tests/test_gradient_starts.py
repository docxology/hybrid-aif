"""The gradient planner's face returns every warm start's plan, with q_pi a point mass on the best.

Design: docs/design/mixed-control-planning.md, "Warm starts".
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif.objectives import total
from hybrid_aif.planning import gradient_plan, make_gradient_planner, make_mixed_planner
from tests.test_lq_control import lq
from tests.test_mixed_planner import planned, toy

STARTS = 4
STEPS = 40


@pytest.fixture(scope="module")
def ctx():
    system, params = lq.make_lq()
    beliefs = system.initial_priors(params)
    return system, params, beliefs, system.planner_context(params, beliefs)


def test_every_starts_plan_is_returned_beside_the_picked_index_and_scores(ctx):
    *_, c = ctx
    result = gradient_plan(c, steps=STEPS, starts=STARTS)
    assert result.G.midpoint.shape == (1,) and result.start_G.midpoint.shape == (STARTS,)
    np.testing.assert_allclose(
        float(result.G.midpoint[0]), float(result.start_G.midpoint[result.pick]), rtol=1e-6
    )
    horizon, dim = result.plan.shape
    assert result.start_plans.shape == (STARTS, horizon, dim)
    assert result.start_scores.shape == (STARTS,)
    assert int(result.pick) == int(jnp.argmin(result.start_scores))
    np.testing.assert_array_equal(
        np.asarray(result.plan), np.asarray(result.start_plans[result.pick])
    )
    one = gradient_plan(c, steps=STEPS, starts=1)
    assert one.start_plans.shape == (1, horizon, dim) and int(one.pick) == 0


def test_the_picked_plan_is_the_one_the_best_start_gives_run_alone(ctx):
    *_, c = ctx
    result = gradient_plan(c, steps=STEPS, starts=STARTS)
    from hybrid_aif.planning.gradient import warm_starts

    low, high = c.continuous_controls.bounds
    inits = warm_starts(None, STARTS, c.policy_len, low, high)
    alone = [gradient_plan(c, steps=STEPS, init=inits[i]) for i in range(STARTS)]
    scores = np.asarray([float(a.start_scores[0]) for a in alone])
    best = alone[int(np.argmin(scores))]
    np.testing.assert_allclose(np.asarray(result.plan), np.asarray(best.plan), atol=1e-5)
    for i in range(STARTS):
        np.testing.assert_allclose(
            np.asarray(result.start_plans[i]), np.asarray(alone[i].plan), atol=1e-5
        )


def test_q_pi_is_a_point_mass_on_the_picked_start_and_the_tree_spans_the_starts(ctx):
    system, params, beliefs, _ = ctx
    planner = make_gradient_planner(steps=STEPS, starts=STARTS)
    q_pi, tree, policies = planner(system.planner_context(params, beliefs))
    result = gradient_plan(system.planner_context(params, beliefs), steps=STEPS, starts=STARTS)
    pick = int(result.pick)
    assert policies.shape == (STARTS, *result.plan.shape)
    probs = np.asarray(q_pi.probs)
    assert probs.shape == (STARTS,)
    np.testing.assert_array_equal(probs, np.eye(STARTS)[pick])
    np.testing.assert_array_equal(np.asarray(policies[pick]), np.asarray(result.plan))
    totals = np.asarray(total(tree).midpoint)
    assert totals.shape == (STARTS,)
    np.testing.assert_allclose(totals[pick], float(result.start_scores[pick]), rtol=1e-4, atol=1e-4)
    for leaves in tree.values():
        for leaf in leaves.values():
            assert leaf.lower.shape[0] == STARTS


def test_one_start_is_unchanged_a_one_member_set_and_a_unit_q_pi(ctx):
    system, params, beliefs, c = ctx
    q_pi, tree, policies = make_gradient_planner(steps=STEPS)(c)
    assert policies.shape == (1, c.policy_len, c.continuous_controls.action_dim)
    np.testing.assert_array_equal(np.asarray(q_pi.probs), [1.0])
    assert total(tree).midpoint.shape == (1,)


def test_act_takes_the_picked_starts_first_step_under_a_laplace_posterior(ctx):
    system, params, beliefs, c = ctx
    planner = make_gradient_planner(steps=STEPS, starts=STARTS, posterior="laplace")
    result = gradient_plan(c, steps=STEPS, starts=STARTS, posterior="laplace")
    det = system.act(
        jax.random.PRNGKey(0), params, beliefs, planner=planner, selection="deterministic"
    )[0]["u"]
    np.testing.assert_allclose(np.asarray(det), np.asarray(result.plan[0]), atol=1e-5)
    plain = make_gradient_planner(steps=STEPS, starts=STARTS)
    again = system.act(
        jax.random.PRNGKey(0), params, beliefs, planner=plain, selection="deterministic"
    )[0]["u"]
    np.testing.assert_allclose(np.asarray(again), np.asarray(result.plan[0]), atol=1e-5)


def test_the_mixed_planners_policy_set_is_still_one_plan_per_discrete_policy():
    mctx = planned(*toy(1))
    q_pi, tree, policies = make_mixed_planner(steps=20, starts=3)(mctx)
    assert q_pi.probs.shape == (3,)
    assert policies.entries[0].shape[0] == 3
    assert total(tree).midpoint.shape == (3,)
