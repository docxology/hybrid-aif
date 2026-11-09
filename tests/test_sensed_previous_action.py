"""A sensed action node's prior at ``infer_states`` is the entry ``priors`` carries, else uniform.

The caller supplies the previous realised action as a point prior (the one-hot at its level); a
node it leaves out is uniform over its levels, which a modality that selects its head and noise at
the node refuses.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import Categorical
from tests.test_action_node_modalities import scalar_model

Y = {"y": jnp.asarray([0.55])}


def _run(system, params, priors=None):
    return system.infer_states(params, Y, priors, detail=True)


def _point(level, levels=3):
    return Categorical.from_probs(jax.nn.one_hot(level, levels))


def test_a_point_prior_selects_the_reading_at_that_level():
    system, params = scalar_model()
    beliefs = system.initial_priors(params)
    posteriors = [_run(system, params, {**beliefs, "a": _point(level)})[0] for level in range(3)]
    # the reading's noise sd is the level's, so the posterior of x differs by the node's level
    locs = [np.asarray(q["x"].loc) for q in posteriors]
    assert not np.allclose(locs[0], locs[1]) and not np.allclose(locs[1], locs[2])
    for q, level in zip(posteriors, range(3), strict=True):
        np.testing.assert_allclose(np.asarray(q["a"].probs), np.eye(3)[level])


def test_with_no_entry_the_point_prior_refusal_fires():
    system, params = scalar_model()
    beliefs = system.initial_priors(params)
    pattern = r"selects its head and noise by action node 'a'.*is a point.*MixtureGaussian"
    with pytest.raises(ValueError, match=pattern):
        _run(system, params, beliefs)


def test_a_point_prior_works_under_jit():
    system, params = scalar_model()
    beliefs = system.initial_priors(params)
    run = jax.jit(
        lambda level: (
            system.infer_states(
                params, Y, {**beliefs, "a": Categorical.from_probs(jax.nn.one_hot(level, 3))}
            )[0]["x"].loc
        )
    )
    for level in range(3):
        ref = system.infer_states(params, Y, {**beliefs, "a": _point(level)})[0]["x"].loc
        np.testing.assert_allclose(run(jnp.asarray(level)), ref, rtol=1e-5)


def test_an_unnamed_node_falls_to_uniform_not_to_level_zero():
    from tests.test_noise_map_mixture import _levelled, _showcase

    system, params = _showcase(_levelled())
    beliefs = system.initial_priors(params)
    y = {"fix": jnp.array([0.2, 0.1])}
    pattern = r"noise map by action node 'sensor'.*belief is a point"
    with pytest.raises(ValueError, match=pattern):
        system.infer_states(params, y, beliefs)


def test_a_continuous_node_entry_rides_beside_a_named_discrete_node():
    from tests.test_noise_map_mixture import _levelled, _sensor, _showcase

    system, params = _showcase(_levelled())
    beliefs = system.initial_priors(params)
    y = {"fix": jnp.array([0.2, 0.1])}
    got = system.infer_states(params, y, {**beliefs, "sensor": _sensor(1)})[0]
    assert got["x"].means.shape[0] >= 1
    np.testing.assert_allclose(got["sensor"].probs, [0.0, 1.0])


def test_a_prior_over_the_wrong_number_of_levels_is_refused():
    system, params = scalar_model()
    beliefs = system.initial_priors(params)
    with pytest.raises(ValueError, match=r"belief over action node 'a'.*3 level\(s\)"):
        _run(system, params, {**beliefs, "a": _point(1, levels=2)})
