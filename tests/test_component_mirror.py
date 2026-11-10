"""The mirror per component: ``mirror(system, params, per_component=True)`` and
``with_mirror(..., per_component=True)``.

What is pinned: the layout of the merge of per-component mirrors for the arena (two minds and an
arena of state relays) and for the two robots of ``compose_minds``; the axis surgery on a tabular
and a linear-Gaussian transition; a shared parameter variable staying one; the timing the
emissions give (a transition reads a foreign state at the boundary under ``"simultaneous"``; a
modality whose parents all belong to one other component is observed as the System observes it;
a modality with parents in two components reads the boundary value, one round older under
``"simultaneous"`` and under ``"sequential"`` when its owner steps after the reader in the derived
order, which is pinned here); the derived sequential order, which makes the arena's minds
bitwise to the composite form; a control node driven by an action node inside the System following
its driver when the mirror is sampled alone; that the mirror carries no durations; and what is
refused.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import Categorical, ConditionalCategorical, ConditionalDirichlet
from hybrid_aif.composition import Identify, couple, merge, plan, run, sample, with_mirror
from hybrid_aif.composition.component_mirror import (
    _move_gaussian,
    _move_tabular,
    world_emissions,
)
from hybrid_aif.distributions import ConditionalGaussian
from hybrid_aif.factors import LinearGaussianTransition, Parameter
from hybrid_aif.system import PartParams
from hybrid_aif.system.mirror import mirror
from tests.test_clock_parity import _replay, _robots, states
from tests.test_driver import _arena, arena_minds
from tests.test_merge import OBSERVATION_BINDINGS
from tests.test_particle import SAME_STEP, _shared, merged
from tests.test_run import idle
from tests.test_run_parity import cm

KEY = jax.random.PRNGKey(0)


def arena_steps():
    return {
        "model.a": plan(selection="deterministic"),
        "model.b": plan(selection="deterministic"),
        "model.arena": idle,
        "world.a": sample(),
        "world.b": sample(),
        "world.arena": sample(),
    }


# ------------------------------------------------------------------------- the layout


def test_the_arena_mirror_is_a_merge_of_one_leaf_per_component_with_the_systems_names():
    composite, cp = _arena()
    world, wp = mirror(composite, cp, per_component=True)
    assert [n for n, _ in world.subsystems] == ["a", "b", "arena"]
    assert world.names.states == composite.names.states
    assert world.names.controls == ("a.u_hand", "b.u_hand", "arena.u_a_hand", "arena.u_b_hand")
    # a state factor of another component that a transition reads is a driven control node of the
    # reader (the arena's lock_b reads b.hand); a modality whose parents all belong to one other
    # component is an emission of that component (a.see reads arena.token_a, so it is arena.a_see)
    assert world.names.actions == ("a.a_hand", "b.a_hand", None, None)
    assert world.names.emissions == (
        "a.b_watch",
        "a.hand_read",
        "b.a_watch",
        "b.hand_read",
        "arena.a_see",
        "arena.b_see",
    )
    assert dict(world.observes) == {
        "arena.u_a_hand": "a.hand_read",
        "arena.u_b_hand": "b.hand_read",
    }
    assert world_emissions(composite) == {
        "a.see": "arena.a_see",
        "a.watch": "b.a_watch",
        "b.see": "arena.b_see",
        "b.watch": "a.b_watch",
    }
    reads = dict(world.emissions)["a.hand_read"]
    assert (reads.parents, reads.realise, reads.kernel) == (("a.hand",), "mode", "identity")
    # no table of the arena changed but for the no-op axis the relays left, which a factor that
    # reads control nodes does not carry
    for factor in ("arena.token_a", "arena.lock_b"):
        np.testing.assert_array_equal(
            wp.transition[factor].log_table, cp.transition[factor].log_table[..., 0]
        )
        assert wp.transition[factor].parent_ndims == cp.transition[factor].parent_ndims - 1


def test_the_compose_minds_robots_mirror_to_two_leaves_with_status_as_an_emission_of_each():
    (sa, pa), (sb, pb) = cm.robot(2, 2), cm.robot(2, 2)
    composite, cp = merge({"a": sa, "b": sb}, {"a": pa, "b": pb}, list(cm.LINKS))
    world, wp = mirror(composite, cp, per_component=True)
    assert world.names.controls == ("a.u_move", "a.u_b_me", "b.u_move", "b.u_a_me")
    # status reads both robots' cells, so it stays an emission of its own robot with the other's
    # cell fed through a driven node; the observation of the other robot's cell alone is an
    # emission of that robot
    assert world.names.emissions == (
        "a.me_obs",
        "a.status",
        "a.b_other_obs",
        "a.me_read",
        "b.me_obs",
        "b.status",
        "b.a_other_obs",
        "b.me_read",
    )
    assert dict(world.emissions)["a.status"].parents == ("a.me", "a.u_b_me")
    assert dict(world.observes) == {"a.u_b_me": "b.me_read", "b.u_a_me": "a.me_read"}
    # sampled alone, each robot's status is its own cell and the other's, as the owner left it
    traj = run(world, wp, 6, KEY, {"a": sample(), "b": sample()}, record_carries=True)
    a, b = states(traj, "a.me", "a"), states(traj, "b.me", "b")
    status = np.asarray(traj["a"].emitted["status"])
    table = np.asarray(cm.status_table())
    expected = [int(np.argmax(table[:, a[t + 1], b[t]])) for t in range(6)]
    assert status.tolist() == expected


def test_the_mirror_refuses_a_system_with_no_components_and_nested_ones():
    system, params = cm.robot(2, 2)
    with pytest.raises(ValueError, match="no component to mirror"):
        mirror(system, params, per_component=True)
    coupled, cparams = with_mirror(*_arena())
    with pytest.raises(NotImplementedError, match="leaves of a composed component"):
        mirror(coupled, cparams, per_component=True)


# ------------------------------------------------------------------------ axis surgery


def test_a_tabular_table_moves_a_parent_axis_behind_the_others():
    rng = np.random.default_rng(0)
    table = jnp.asarray(rng.random((2, 3, 4, 5)))  # (child, p0, p1, p2)
    moved = _move_tabular(ConditionalCategorical(log_table=table, parent_ndims=3), [0, 2, 1])
    assert moved.parent_ndims == 3
    np.testing.assert_array_equal(moved.log_table, np.transpose(np.asarray(table), (0, 1, 3, 2)))
    dirichlet = ConditionalDirichlet(concentration=table, parent_ndims=3)
    moved = _move_tabular(dirichlet, [1, 2, 0])
    np.testing.assert_array_equal(
        moved.concentration, np.transpose(np.asarray(table), (0, 2, 3, 1))
    )


def test_a_trailing_no_op_axis_is_dropped_when_the_factor_now_reads_a_control_node():
    table = jnp.ones((2, 3, 4, 1))
    moved = _move_tabular(ConditionalCategorical(log_table=table, parent_ndims=3), [0, 1])
    assert moved.log_table.shape == (2, 3, 4) and moved.parent_ndims == 2


def test_a_linear_gaussian_transition_moves_a_mode_axis():
    rng = np.random.default_rng(1)
    d = 2
    F = jnp.asarray(rng.random((3, 4, d, d)))
    decl = LinearGaussianTransition(
        F=F,
        innovations=ConditionalGaussian(
            loc=jnp.asarray(rng.random((3, 4, d))),
            chol=jnp.asarray(rng.random((3, 4, d, d))),
            parent_ndims=2,
        ),
        G=jnp.asarray(rng.random((3, 4, d, 1))),
    )
    moved = _move_gaussian(decl, [1, 0])
    np.testing.assert_array_equal(moved.F, np.swapaxes(np.asarray(F), 0, 1))
    np.testing.assert_array_equal(
        moved.innovations.loc, np.swapaxes(np.asarray(decl.innovations.loc), 0, 1)
    )
    np.testing.assert_array_equal(
        moved.innovations.chol, np.swapaxes(np.asarray(decl.innovations.chol), 0, 1)
    )
    np.testing.assert_array_equal(moved.G, np.swapaxes(np.asarray(decl.G), 0, 1))
    assert moved.innovations.parent_ndims == 2
    shared = LinearGaussianTransition(
        F=jnp.eye(d),
        innovations=ConditionalGaussian(
            loc=decl.innovations.loc, chol=0.1 * jnp.eye(d), parent_ndims=2
        ),
    )
    again = _move_gaussian(shared, [1, 0])
    assert again.F.shape == (d, d) and again.innovations.chol.shape == (d, d)


# ------------------------------------------------------------- a shared parameter variable


def test_a_parameter_variable_that_two_components_read_stays_one_in_the_merge():
    composite, cp, _, _ = _shared()
    assert set(cp.parameters) == {"rover.field"}
    world, wp = mirror(composite, cp, per_component=True)
    assert set(wp.parameters) == {"rover.field"}

    # one variable, and every reader reads it: the transitions and the emissions that read it are
    # the transitions and the likelihoods that did
    def readers(p, slots):
        items = [getattr(p, slot).items() for slot in slots]
        return {k for kv in items for k, v in kv if isinstance(v, Parameter)}

    assert readers(cp, ("transition", "likelihood")) == readers(wp, ("transition", "emission"))
    assert readers(wp, ("transition", "emission"))
    lowered, lp = mirror(composite, cp, per_component=True, lower=True)
    assert lp.parameters == {} and lp.references == ()


# --------------------------------------------------------------------------------- timing


@pytest.mark.parametrize("order", ["simultaneous", "sequential"])
def test_a_transition_reads_a_foreign_state_as_the_components_of_the_system_do(order):
    composite, cp = _arena()
    system, params = with_mirror(composite, cp, per_component=True)
    levels = np.asarray([1, 2, 0, 2, 1, 0, 0, 1])
    steps = {
        **arena_steps(),
        "model.a": _replay("model.a.a_hand", levels),
        "model.b": _replay("model.b.a_hand", levels[::-1]),
    }
    traj = run(system, params, 8, KEY, steps, tick_order=order, record_carries=True)
    hand_b = states(traj, "world.b.hand", "world.b")
    lock_b = states(traj, "world.arena.lock_b", "world.arena")
    # lock_b copies b's hand: as moved this round under sequential (the arena steps after b),
    # as the round began under simultaneous
    if order == "sequential":
        np.testing.assert_array_equal(lock_b[1:], hand_b[1:])
    else:
        np.testing.assert_array_equal(lock_b[1:], hand_b[:-1])


@pytest.mark.parametrize("order", ["simultaneous", "sequential"])
def test_a_modality_whose_parents_belong_to_one_other_component_is_observed_as_the_system_does(
    order,
):
    """``a.see`` reads ``arena.token_a``, so it is the arena's emission ``a_see``, drawn from the
    factor as the arena's step left it: the state the composite form observes at the next round."""
    composite, cp = _arena()
    system, params = with_mirror(composite, cp, per_component=True)
    traj = run(system, params, 8, KEY, arena_steps(), tick_order=order, record_carries=True)
    token_a = states(traj, "world.arena.token_a", "world.arena")
    see = np.asarray(traj["world.arena"].emitted["a_see"])
    np.testing.assert_array_equal(see, token_a[1:])


@pytest.mark.parametrize("order", ["simultaneous", "sequential"])
def test_a_modality_with_parents_in_two_components_reads_the_boundary_value(order):
    """The known limitation: a robot's ``status`` reads both robots' cells, so it stays an
    emission of the robot with the other's cell fed through a driven node, which is read at the
    boundary under ``"simultaneous"`` (and as it stands under ``"sequential"``, this round's for
    the robot that steps after the other): one round older than the System observes it."""
    (sa, pa), (sb, pb) = cm.robot(2, 2), cm.robot(2, 2)
    composite, cp = merge({"a": sa, "b": sb}, {"a": pa, "b": pb}, list(cm.LINKS))
    world, wp = mirror(composite, cp, per_component=True)
    traj = run(
        world, wp, 6, KEY, {"a": sample(), "b": sample()}, tick_order=order, record_carries=True
    )
    a, b = states(traj, "a.me", "a"), states(traj, "b.me", "b")
    table = np.asarray(cm.status_table())
    seen_a = np.asarray(traj["a"].emitted["status"])
    seen_b = np.asarray(traj["b"].emitted["status"])
    # a steps first, so it reads b's cell entering the round; b reads a's as moved this round
    # under sequential and entering the round under simultaneous
    assert seen_a.tolist() == [int(np.argmax(table[:, a[t + 1], b[t]])) for t in range(6)]
    follows = a[1:7] if order == "sequential" else a[:6]
    assert seen_b.tolist() == [int(np.argmax(table[:, b[t + 1], follows[t]])) for t in range(6)]


def test_under_sequential_each_mind_steps_right_after_the_world_leaf_it_drives():
    """``run`` derives the sequential order: the sources (the model leaves and the arena) in
    listing order and each driven world leaf right after its driver, so ``model.b`` observes a's
    hand as ``a`` moved it this round, as the composite form's mind did."""
    composite, cp = _arena()
    composite_run = run(
        composite, cp, arena_minds(composite, cp), ticks=6, key=KEY, tick_order="sequential"
    )
    system, params = with_mirror(composite, cp, per_component=True)
    assert [n for n, _ in system.subsystems if n.count(".") == 1] == [
        "model.a",
        "model.b",
        "model.arena",
        "world.a",
        "world.b",
        "world.arena",
    ]
    calls: list = []

    def tagged(name, step):
        def traced(system, key, obs, carry, params):
            if obs is not None:
                calls.append(name)
            return step(system, key, obs, carry, params)

        traced.sliced = getattr(step, "sliced", False)
        return traced

    steps = {n: tagged(n, f) for n, f in arena_steps().items()}
    traj = run(system, params, 6, KEY, steps, tick_order="sequential", record_carries=True)
    assert calls[:6] == ["model.a", "world.a", "model.b", "world.b", "model.arena", "world.arena"]
    hand_a = states(traj, "world.a.hand", "world.a")
    np.testing.assert_array_equal(traj["model.b"].received["watch"][:6], hand_a[1:7])
    np.testing.assert_array_equal(
        traj["model.b"].received["watch"][:6], composite_run.states["a.hand"][1:7]
    )
    for m in "ab":
        np.testing.assert_array_equal(
            traj[f"model.{m}"].emitted["a_hand"], composite_run.actions[f"{m}.a_hand"]
        )


def test_a_control_node_driven_inside_the_system_follows_its_driver_when_the_mirror_is_sampled():
    composite, cp = merged(bindings=[SAME_STEP, *OBSERVATION_BINDINGS])
    world, wp = mirror(composite, cp, per_component=True)
    assert dict(world.observes)["world.u_location"] == "agent.a_location"
    assert world.names.actions[world.names.controls.index("world.u_location")] is None
    for order in ("simultaneous", "sequential", "async"):
        traj = run(
            world,
            wp,
            8,
            KEY,
            {"agent": sample(), "world": sample()},
            tick_order=order,
            record_carries=True,
        )
        driver = np.asarray(traj["agent"].emitted["a_location"])
        np.testing.assert_array_equal(traj["world"].received["u_location"][:8], driver)
        np.testing.assert_array_equal(
            states(traj, "world.location", "world"), states(traj, "agent.location", "agent")
        )
        assert len(set(driver.tolist())) > 1


def test_with_mirror_per_component_drives_the_worlds_copy_of_a_driven_node_by_the_models_action():
    composite, cp = merged(bindings=[SAME_STEP, *OBSERVATION_BINDINGS])
    system, params = with_mirror(composite, cp, per_component=True)
    steps = {
        "model.agent": plan(selection="stochastic"),
        "model.world": idle,
        "world.agent": sample(),
        "world.world": sample(),
    }
    for order in ("simultaneous", "sequential"):
        traj = run(system, params, 8, KEY, steps, tick_order=order, record_carries=True)
        chosen = np.asarray(traj["model.agent"].emitted["a_location"])
        np.testing.assert_array_equal(traj["world.agent"].received["u_location"][:8], chosen)
        np.testing.assert_array_equal(traj["world.world"].received["u_location"][:8], chosen)
        assert len(set(chosen.tolist())) > 1


# ----------------------------------------------------------------------------- refusals


def test_a_habit_on_a_component_is_refused_with_the_roadmap_note():
    composite, cp = _arena()
    habit = Categorical.from_probs(jnp.ones(27) / 27)
    parts = {"a": dataclasses.replace(cp.parts.get("a"), policy_prior=habit)} if cp.parts else None
    if parts is None:
        from hybrid_aif.system import PartParams

        parts = {"a": PartParams(policy_prior=habit)}
    with pytest.raises(NotImplementedError, match="habit.*on the roadmap"):
        mirror(composite, dataclasses.replace(cp, parts=parts), per_component=True)


def test_a_param_key_without_lower_is_refused_as_the_whole_mirror_refuses_it():
    composite, cp = _arena()
    with pytest.raises(ValueError, match="param_key.*lower=True"):
        mirror(composite, cp, per_component=True, param_key=KEY)


def test_per_component_with_mirror_is_a_coupling_of_the_system_and_its_merged_mirror():
    composite, cp = _arena()
    system, params = with_mirror(composite, cp, per_component=True)
    leaves = {n for n, _ in system.subsystems}
    assert {"model", "world", "model.a", "model.arena", "world.a", "world.arena"} <= leaves
    assert system.names.actions[:2] == ("model.a.a_hand", "model.b.a_hand")
    assert set(dict(system.observes)) >= {"model.a.see", "model.b.watch", "world.arena.u_a_hand"}


def test_a_linear_gaussian_factor_with_a_foreign_discrete_parent_is_refused():
    from hybrid_aif.composition import Link
    from hybrid_aif.distributions import Gaussian
    from hybrid_aif.factors import LinearGaussianObservation
    from hybrid_aif.system import System, SystemParams

    mode = System(
        B_deps=((0,),),
        controlled=(False,),
        factor_names=("mode",),
        modality_names=(),
        policy_len=1,
        epistemic=False,
        mixture_k=3,
    )
    mode_params = SystemParams(
        transition={
            "mode": ConditionalCategorical(
                log_table=jnp.log(jnp.roll(jnp.eye(3), 1, axis=0) + 1e-9), parent_ndims=1
            )
        },
        initial_state_prior={"mode": Categorical.from_probs(jnp.eye(3)[0])},
    )
    body = System(
        B_deps=((1,), ()),
        A_deps=((0,),),
        controls=(1,),
        action_names=("a_unused",),
        factor_names=("pos", "u_mode"),
        modality_names=("see",),
        policy_len=1,
        epistemic=False,
        mixture_k=3,
    )
    body_params = SystemParams(
        likelihood={
            "see": LinearGaussianObservation(H=jnp.eye(1), c=jnp.zeros(1), noise=0.1 * jnp.eye(1))
        },
        transition={
            "pos": LinearGaussianTransition(
                F=jnp.eye(1),
                innovations=ConditionalGaussian(
                    loc=jnp.asarray([[0.0], [1.0], [2.0]]), chol=0.01 * jnp.eye(1), parent_ndims=1
                ),
            )
        },
        initial_state_prior={"pos": Gaussian(loc=jnp.zeros(1), chol=0.1 * jnp.eye(1))},
    )
    composite, cp = merge(
        {"m": mode, "x": body}, {"m": mode_params, "x": body_params}, [Link("m.mode", "x.u_mode")]
    )
    with pytest.raises(NotImplementedError, match="linear-Gaussian.*within the step.*roadmap"):
        mirror(composite, cp, per_component=True)


def test_the_mirror_carries_no_durations_so_a_model_with_durations_runs_its_world_every_round():
    (sa, pa), (sb, pb) = _robots([4.0, 2.0, 4.0])[0], cm.robot(2, 2)
    composite, cp = merge({"a": sa, "b": sb}, {"a": pa, "b": pb}, list(cm.LINKS))
    world, wp = mirror(composite, cp, per_component=True)
    assert wp.durations == {} and not any(p.durations for p in (wp.parts or {}).values())
    system, params = with_mirror(composite, cp, per_component=True)
    steps = {
        "model.a": plan(selection="stochastic"),
        "model.b": plan(selection="stochastic"),
        "world.a": sample(),
        "world.b": sample(),
    }
    for order in ("simultaneous", "sequential", "async"):
        traj = run(system, params, 12, KEY, steps, tick_order=order)
        decided = np.asarray(traj["model.a"].decided)
        assert not decided.all() and decided[0]
        assert traj["world.a"].decided.all() and traj["world.b"].decided.all()
        assert traj["model.b"].decided.all()


def test_a_sampled_world_that_declares_durations_on_a_driven_node_is_refused_naming_the_cause():
    (sa, pa), (sb, pb) = _robots([4.0, 2.0, 4.0])[0], cm.robot(2, 2)
    system, params = couple(
        {"a": (sa, pa), "b": (sb, pb), "world": mirror(*cm.world_model(), lower=True)},
        [Identify("a.move", "world.u_a"), Identify("b.move", "world.u_b")]
        + [
            Identify(f"world.{p}", f"{who}.{m}")
            for who, wires in (
                ("a", (("a_pos", "me_obs"), ("b_pos", "other_obs"), ("status", "status"))),
                ("b", (("b_pos", "me_obs"), ("a_pos", "other_obs"), ("status", "status"))),
            )
            for p, m in wires
        ],
    )
    world_params = dataclasses.replace(
        params,
        parts={**params.parts, "world": PartParams(durations={"world.u_a": jnp.full(3, 2.0)})},
    )
    steps = {"a": plan(), "b": plan(), "world": sample()}
    with pytest.raises(ValueError, match="'world' declares durations on its control node.*drives"):
        run(system, world_params, 2, KEY, steps)


def test_a_component_with_its_own_d_deps_mirrors_with_one_entry_per_node():
    """A robot's ``other`` is conditioned on ``me`` at the start: the leaf's ``D_deps`` runs over
    every node, the control nodes it gains included."""

    def robot():
        system, params = cm.robot(2, 2)
        system = dataclasses.replace(system, D_deps=((), (0,), ()))
        prior = ConditionalCategorical(log_table=jnp.log(jnp.full((5, 5), 0.2)), parent_ndims=1)
        return system, dataclasses.replace(
            params, initial_state_prior={**params.initial_state_prior, "other": prior}
        )

    (sa, pa), (sb, pb) = robot(), robot()
    composite, cp = merge({"a": sa, "b": sb}, {"a": pa, "b": pb}, list(cm.LINKS))
    world, wp = mirror(composite, cp, per_component=True)
    assert len(world.D_deps) == len(world.B_deps) == 8
    assert world.D_deps[:4] == ((), (0,), (), (2,))
    run(world, wp, 3, KEY, {"a": sample(), "b": sample()})


def test_the_robots_per_component_coupling_round_trips_through_the_description():
    """A control node read only by an emission (the robots' driven ``u_b_me``) is a node the
    document names through the emission."""
    from hybrid_aif import build, describe

    (sa, pa), (sb, pb) = cm.robot(2, 2), cm.robot(2, 2)
    composite, cp = merge({"a": sa, "b": sb}, {"a": pa, "b": pb}, list(cm.LINKS))
    system, params = with_mirror(composite, cp, per_component=True)
    document, arrays = describe(system, params)
    rebuilt, rebuilt_params = build(document, arrays=arrays)
    assert rebuilt.names == system.names and rebuilt.B_deps == system.B_deps
    assert rebuilt.emissions == system.emissions and dict(rebuilt.observes) == dict(system.observes)
    again, _ = describe(rebuilt, rebuilt_params)
    assert again == document


def test_a_per_component_world_stepped_as_one_node_runs_under_every_order():
    from tests.test_action_nodes import pusher

    composite, cp = merge({n: pusher()[0] for n in "abc"}, {n: pusher()[1] for n in "abc"}, [])
    system, params = with_mirror(composite, cp, per_component=True)
    steps = {f"model.{n}": plan(selection="stochastic") for n in "abc"} | {"world": sample()}
    for order in ("simultaneous", "sequential", "async"):
        traj = run(system, params, 4, KEY, steps, tick_order=order)
        assert traj["world"].emitted["a.see"].shape == (4,)
