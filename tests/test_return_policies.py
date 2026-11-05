"""``infer_policies(return_policies=True)`` hands back the planner's clamped policy set."""

import jax
import numpy as np

from hybrid_aif.planning.mixed import make_mixed_planner
from tests.test_mixed_planner import toy


def test_the_policy_set_rides_beside_the_posterior_eagerly_and_compiled():
    system, params = toy(2)
    beliefs = system.initial_priors(params)
    planner = make_mixed_planner(steps=5, starts=2)
    q, tree, policies = system.infer_policies(
        params, beliefs, planner=planner, return_policies=True
    )
    assert len(policies.entries) == 2
    levels, controls = policies.entries
    assert levels.shape == (q.probs.shape[0], 2) and controls.shape == (q.probs.shape[0], 2, 1)
    assert q.probs.shape[0] == 9  # one policy per discrete policy, the best of its starts
    q2, tree2 = system.infer_policies(params, beliefs, planner=planner)
    np.testing.assert_allclose(np.asarray(q2.probs), np.asarray(q.probs), rtol=1e-6)
    compiled = system.compiled()
    qc, treec, pc = compiled.infer_policies(params, beliefs, planner=planner, return_policies=True)
    jax.block_until_ready(qc.probs)
    np.testing.assert_allclose(np.asarray(qc.probs), np.asarray(q.probs), rtol=1e-4)
    np.testing.assert_array_equal(np.asarray(pc.entries[0]), np.asarray(levels))
    np.testing.assert_allclose(np.asarray(pc.entries[1]), np.asarray(controls), rtol=1e-4)
    assert compiled.trace_counts["infer_policies"] == 1
