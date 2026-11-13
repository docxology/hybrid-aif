"""The keywords that name state factors, and the noise arguments that take two types.

A mask or a choice over state factors alone is spelled ``states`` / ``state`` /
``subsystem_states`` / ``num_states``; the conditioning functions take ``noise`` (a
lower-triangular Cholesky factor or a ``DiagonalNoise``) and a ``Conditioner`` reads it with
``noise_at``. The keywords the same functions do not take are refused.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import Gaussian, composition
from hybrid_aif.composition.merge import subsystem_states
from hybrid_aif.envs import Env, GenerativeEnv
from hybrid_aif.inference import DiagonalNoise, condition
from hybrid_aif.inference.linear_gaussian import condition_on_expected
from hybrid_aif.objectives import hybrid as _hybrid
from hybrid_aif.objectives.hybrid import Conditioner
from hybrid_aif.reference.envs.mountain_car import MountainCar
from hybrid_aif.system.core import Subsystem
from hybrid_aif.system.mirror import mirror
from tests.test_action_nodes import pusher
from tests.test_merge import merged
from tests.test_structure_learning import _growable_system

KEY = jax.random.PRNGKey(5)
PUSH = {"push": jnp.asarray(1, jnp.int32)}


# ------------------------------------------------------------------ masks over state factors


def test_step_and_step_actuated_take_states():
    system, params = pusher(noise=0.0)
    env, ep = system.as_env(params)
    _, state = env.reset(KEY, ep)
    _, kept = env.step(KEY, state, PUSH, ep, states=())
    assert int(kept["x"]) == int(state["x"])
    _, kept, realised = env.step_actuated(KEY, state, PUSH, ep, states=())
    assert int(kept["x"]) == int(state["x"]) and int(realised["push"]) == 1
    _, moved = env.step(KEY, state, PUSH, ep, states=("x",))
    _, full = env.step(KEY, state, PUSH, ep)
    assert int(moved["x"]) == int(full["x"])


def test_step_and_step_actuated_refuse_factors():
    system, params = pusher()
    env, ep = system.as_env(params)
    _, state = env.reset(KEY, ep)
    with pytest.raises(TypeError, match="factors"):
        env.step(KEY, state, PUSH, ep, factors=())
    with pytest.raises(TypeError, match="factors"):
        env.step_actuated(KEY, state, PUSH, ep, factors=())


def test_sample_step_takes_states_and_refuses_factors():
    system, params = pusher()
    world, wp = mirror(system, params)
    _, state = world.sample_initial(KEY, wp)
    inputs = {"u_push": jnp.asarray(2, jnp.int32)}
    _, kept = world.sample_step(KEY, wp, state, inputs, states=())
    assert int(kept["x"]) == int(state["x"])
    with pytest.raises(TypeError, match="factors"):
        world.sample_step(KEY, wp, state, inputs, factors=())


# ------------------------------------------------------------ the growable state factor


def test_grow_and_reduce_take_state():
    system, params = _growable_system()
    window = jnp.asarray(np.random.default_rng(0).normal(size=(6, 2)), jnp.float32)
    by_keyword = system.grow(params, {"y": window}, state="z")
    by_position = system.grow(params, {"y": window}, "z")
    for a, b in zip(jax.tree.leaves(by_keyword), jax.tree.leaves(by_position), strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    reduced = system.reduce(by_keyword, params, state="z")
    every = system.reduce(by_keyword, params)
    for a, b in zip(jax.tree.leaves(reduced), jax.tree.leaves(every), strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    with pytest.raises(ValueError, match="not a growable state factor"):
        system.reduce(params, params, state="w")


def test_grow_and_reduce_refuse_factor():
    system, params = _growable_system()
    with pytest.raises(TypeError, match="factor"):
        system.grow(params, {"y": jnp.zeros((3, 2))}, factor="z")
    with pytest.raises(TypeError, match="factor"):
        system.reduce(params, params, factor="z")


# ------------------------------------------------------------------- subsystem records


def test_a_subsystem_record_names_its_state_factors_states():
    composite, _ = merged(policy_len=1)
    meta = dict(composite.subsystems)
    assert meta["agent"].states == (0, 1) and meta["world"].states == (2, 3)
    assert set(meta["agent"].states) <= set(subsystem_states(composite, "agent"))
    assert composition.subsystem_states is subsystem_states
    assert not hasattr(composition, "subsystem_factors")
    assert "factors" not in Subsystem._fields
    assert not hasattr(meta["agent"], "factors")
    with pytest.raises(TypeError, match="factors"):
        Subsystem(factors=(0,), modalities=())
    with pytest.raises(ValueError, match="factors"):
        meta["agent"]._replace(factors=(0,))


# ------------------------------------------------------------ the number of state factors


def test_environments_count_their_state_factors_as_num_states():
    system, params = pusher()
    env, _ = system.as_env(params)
    assert isinstance(env, GenerativeEnv)
    assert env.num_states == env.inner.num_states == 1
    assert MountainCar().num_states == 1
    for e in (env, env.inner, MountainCar()):
        assert not hasattr(e, "num_factors")
    assert "num_states" in Env.__abstractmethods__
    assert "num_factors" not in Env.__abstractmethods__


# ------------------------------------------------------------------- noise arguments


def _belief_and_head():
    g = Gaussian(loc=jnp.asarray([0.3, -0.2]), chol=jnp.asarray([[1.0, 0.0], [0.4, 0.8]]))
    H = jnp.asarray([[1.0, 0.5], [0.0, 1.0], [0.2, 0.3]])
    return g, H, jnp.zeros(3), DiagonalNoise(jnp.asarray([0.5, 0.7, 0.9]))


def test_the_conditioning_functions_take_noise_of_either_type():
    g, H, c, noise = _belief_and_head()
    y = jnp.asarray([0.1, 0.2, -0.3])
    for n in (noise, noise.chol()):
        post, ll = condition(g, H, c, noise=n, y=y)
        want, want_ll = condition(g, H, c, n, y)
        np.testing.assert_array_equal(np.asarray(post.chol), np.asarray(want.chol))
        np.testing.assert_array_equal(np.asarray(ll), np.asarray(want_ll))
        got = condition_on_expected(g, H, noise=n)
        np.testing.assert_array_equal(
            np.asarray(got.chol), np.asarray(condition_on_expected(g, H, n).chol)
        )
    cond = Conditioner(factor=0, column=0, H=H, c=c, noise_at=lambda loc: noise)
    read = _hybrid.condition_expected(g, cond, None)
    given = _hybrid.condition_expected(g, cond, None, noise=noise.chol())
    np.testing.assert_allclose(np.asarray(read.chol), np.asarray(given.chol), atol=1e-5)
    assert cond.noise_at(g.loc) is noise


def test_the_conditioning_functions_refuse_r_chol():
    g, H, c, noise = _belief_and_head()
    with pytest.raises(TypeError, match="R_chol"):
        condition(g, H, c, R_chol=noise, y=jnp.zeros(3))
    with pytest.raises(TypeError, match="R_chol"):
        condition_on_expected(g, H, R_chol=noise)
    cond = Conditioner(factor=0, column=0, H=H, c=c, noise_at=lambda loc: noise)
    with pytest.raises(TypeError, match="R_chol"):
        _hybrid.condition_expected(g, cond, None, R_chol=noise)
    with pytest.raises(TypeError, match="R_chol_at"):
        Conditioner(factor=0, column=0, H=H, c=c, R_chol_at=lambda loc: noise)
    assert "R_chol_at" not in Conditioner._fields
