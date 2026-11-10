"""Coupling through blankets: Systems joined by identifications of their ports, run as one
closed System.

A coupling's components are Systems; an ``Identify`` joins an out-port (an action node or an
emission) to an in-port of another component (a modality, or a control node it then drives). An
action node is realised once per round from its node's actuator and every identification of it
carries that value. A merged composite is run against its own mirror (``with_mirror``). Each run
is held bitwise to the wired or composite form's run of the same case, under that form's key layout:
by ``tests/test_run_parity.py`` where it records the case, else by ``migration_couple.npz``. The
tests of sections 7 and 8 are of the wired form's own contract and go with it.

Design: docs/design/control-states.md C12, C13; docs/design/nesting-systems.md principle 6.
"""

import dataclasses
import pathlib

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import Categorical, ConditionalCategorical
from hybrid_aif.composition import (
    Identify,
    Link,
    MindCarry,
    Wiring,
    agent_step,
    ancestral_step,
    couple,
    merge,
    plan,
    run,
    sample,
    subsystem,
    with_mirror,
)
from hybrid_aif.controls import Box
from hybrid_aif.distributions import ConditionalGaussian, Gaussian
from hybrid_aif.envs import GenerativeEnv
from hybrid_aif.envs._process import DiscreteProcess
from hybrid_aif.factors import LinearGaussianObservation, LinearGaussianTransition
from hybrid_aif.system import System, SystemParams
from hybrid_aif.system.mirror import mirror
from tests.test_action_nodes import KERNEL, pusher
from tests.test_emissions import world as emitting_world
from tests.test_merge import OBSERVATION_BINDINGS, agent_and_world
from tests.test_system import tmaze_system

KEY = jax.random.PRNGKey(0)

#: P(a | u) over the T-maze's five locations: the intended one most often.
KERNEL5 = 0.4 * jnp.eye(5) + 0.12


def watcher():
    """A native System with one state ``x``, one control node (action ``push``), and two
    three-level modalities on ``x``, ``see`` and ``see2``."""
    system = System(
        A_deps=((0,), (0,)),
        B_deps=((0, 1), ()),
        controls=(1,),
        action_names=("push",),
        factor_names=("x", "u_push"),
        modality_names=("see", "see2"),
        policy_len=1,
        epistemic=False,
    )
    table = ConditionalCategorical(log_table=jnp.log(0.8 * jnp.eye(3) + 0.2 / 3), parent_ndims=1)
    _, pp = pusher()
    params = SystemParams(
        likelihood={"see": table, "see2": table},
        transition=pp.transition,
        initial_state_prior={"x": Categorical(logits=jnp.zeros(3))},
    )
    return system, params


def process(system, params):
    """``system`` run forward as a process, its actuator left to the mind."""
    env, ep = system.as_env(params)
    return env, dataclasses.replace(ep, actuator={})


def constant(level):
    """A Step emitting ``level`` on the action node ``push`` every tick."""

    def step(key, obs, carry, params):
        return {"push": jnp.asarray(level, jnp.int32)}, carry, params

    return step


def equal(a, b):
    return bool(jnp.array_equal(a, b))


def ancestral():
    """The habit draw of ``tests/test_run_parity.py`` (imported on use: that module imports this
    one)."""
    from tests.test_run_parity import ancestral as step

    return step()


def _run_old_keys(minds, *args, **kwargs):
    """``run`` under the key layout of the wired and composite forms
    (``tests/test_run_parity.py``)."""
    from tests.test_run_parity import _run_old_keys as patched

    return patched(minds, *args, **kwargs)


# --------------------------------------------------------- 1. a composite against its mirror

FIXTURE = np.load(pathlib.Path(__file__).parent / "data" / "migration_couple.npz")


def same(got, name):
    np.testing.assert_array_equal(np.asarray(got), FIXTURE[name], err_msg=name)


def _two_minds(kernel):
    """The zoo pair as a composite, a view per component, and the composite form's steps; the
    signature test of the composite form (section 8) runs it."""
    (agent, ap), (world, wp) = agent_and_world(policy_len=1)
    composite, cp = merge(
        {"agent": agent, "world": world}, {"agent": ap, "world": wp}, OBSERVATION_BINDINGS
    )
    if kernel:
        #  the agent's view reads the world's node, so a kernel there would be refused
        cp = dataclasses.replace(cp, actuator={"agent.a_location": KERNEL5})
    views = {name: subsystem(composite, cp, name) for name in ("agent", "world")}
    responds = {
        "agent": agent_step(views["agent"][0]),
        "world": ancestral_step(views["world"][0]),
    }
    return composite, cp, views, responds


def _zoo_closed(kernel):
    """The zoo pair coupled to its own mirror: the agent plans, the world component draws its
    node from its habit, the mirror is sampled."""
    (agent, ap), (world, wp) = agent_and_world(policy_len=1)
    composite, cp = merge(
        {"agent": agent, "world": world}, {"agent": ap, "world": wp}, OBSERVATION_BINDINGS
    )
    if kernel:
        cp = dataclasses.replace(cp, actuator={"agent.a_location": KERNEL5})
    system, params = with_mirror(composite, cp)
    steps = {"model.agent": plan(), "model.world": ancestral(), "world": sample()}
    return system, params, steps


def _states(trajectory, factor, node="world"):
    return jnp.stack([jnp.asarray(c[factor]) for c in trajectory[node].carries])


@pytest.mark.parametrize("kernel", [False, True], ids=["delta", "kernel"])
def test_a_composite_against_its_mirror_routes_each_nodes_actions_and_the_worlds_emissions(kernel):
    """Each model leaf realises its own action node, the world steps on the realised values, and
    under ``"simultaneous"`` the agent receives the world's emissions as they stood at the round's
    boundary. The run is held bitwise to the composite form by the zoo case of
    ``tests/test_run_parity.py`` (every order, with and without a kernel)."""
    system, params, steps = _zoo_closed(kernel)
    traj = run(system, params, 12, KEY, steps, record_carries=True)
    agent, world = traj["model.agent"], traj["world"]
    assert list(agent.emitted) == list(agent.intended) == ["a_location"]
    assert list(traj["model.world"].emitted) == ["a_location"]
    for m in ("location_obs", "reward", "cue"):
        emitted = jnp.concatenate([world.start[f"agent.{m}"][None], world.emitted[f"agent.{m}"]])
        assert equal(agent.received[m], emitted), m
        assert bool(jnp.all(agent.present[m]))
    #  the world's copy of each node carries the realised value of the model's
    assert equal(world.received["agent.u_location"][:-1], agent.emitted["a_location"])
    assert equal(world.received["world.u_location"][:-1], traj["model.world"].emitted["a_location"])
    if kernel:  # the kernel moved some draw off the intended level
        assert not equal(agent.emitted["a_location"], agent.intended["a_location"])
    else:
        assert equal(agent.emitted["a_location"], agent.intended["a_location"])


def _three_pushers(order):
    """Three merged pushers, listed in ``order``, a kernel on every action node, coupled to their
    mirror: ``a`` plans, ``b`` and ``c`` draw from the habit."""
    parts = {n: pusher() for n in order}
    composite, cp = merge(
        {n: s for n, (s, _) in parts.items()}, {n: p for n, (_, p) in parts.items()}, []
    )
    cp = dataclasses.replace(cp, actuator={f"{n}.push": KERNEL for n in order})
    system, params = with_mirror(composite, cp)
    steps = {f"model.{n}": plan() if n == "a" else ancestral() for n in order}
    return system, params, {**steps, "world": sample()}


def test_three_merged_pushers_against_their_mirror_realise_each_node_through_its_kernel():
    """Each pusher's node is realised through its own kernel and recorded under its own leaf.
    Held bitwise to the composite form by the pushers case of ``tests/test_run_parity.py``."""
    system, params, steps = _three_pushers(("a", "b", "c"))
    traj = run(system, params, 8, jax.random.PRNGKey(3), steps)
    for name in ("a", "b", "c"):
        leaf = traj[f"model.{name}"]
        assert list(leaf.intended) == list(leaf.emitted) == ["push"]
        assert not equal(leaf.emitted["push"], leaf.intended["push"]), name
        assert equal(traj["world"].received[f"{name}.u_push"][:-1], leaf.emitted["push"])


def test_components_out_of_order_run_and_record_by_node():
    """The model's components listed ``c, a, b``: the record is a tree in that listing, each leaf
    holding its own action node and modality, the world one leaf holding every factor."""
    system, params, steps = _three_pushers(("c", "a", "b"))
    traj = run(system, params, 4, KEY, steps, tick_order="async", record_carries=True)
    assert list(traj.parts) == ["model", "world"]
    assert list(traj["model"].parts) == ["c", "a", "b"]
    for name in ("c", "a", "b"):
        leaf = traj[f"model.{name}"]
        assert list(leaf.intended) == list(leaf.emitted) == ["push"]
        assert leaf.emitted["push"].shape == (4,)
        assert list(leaf.received) == ["see"]
    assert set(traj["world"].carry) == {"world.a.x", "world.b.x", "world.c.x"}


# ------------------------------------------------------------ 2. two minds, two processes


def _two_processes():
    system, params = couple(
        {
            "A": pusher(),
            "B": watcher(),
            "P1": mirror(*pusher(), lower=True),
            "P2": mirror(*pusher(), lower=True),
        },
        [
            Identify("A.push", "P1.u_push"),
            Identify("P1.see", "A.see"),
            Identify("P1.see", "B.see"),
            Identify("B.push", "P2.u_push"),
            Identify("P2.see", "B.see2"),
        ],
    )
    return system, params, {"A": plan(), "B": ancestral(), "P1": sample(), "P2": sample()}


def test_two_minds_on_two_processes_record_by_node_and_match_a_hand_loop():
    system, params, steps = _two_processes()
    ticks = 5
    traj = run(system, params, ticks, KEY, steps, record_carries=True)
    assert list(traj.parts) == ["A", "B", "P1", "P2"]
    assert set(traj["A"].received) == {"see"}
    assert set(traj["B"].received) == {"see", "see2"}
    for name in ("A", "B"):
        node = traj[name]
        assert all(v.shape == (ticks + 1,) for v in node.received.values())
        assert list(node.intended) == list(node.emitted) == ["push"]
        assert node.intended["push"].shape == node.emitted["push"].shape == (ticks,)
    assert isinstance(traj["A"].carry, MindCarry)
    for name in ("P1", "P2"):
        assert list(traj[name].emitted) == ["see"] and traj[name].emitted["see"].shape == (ticks,)

    #  the hand loop: the documented keys (node j of N = 4 steps with key j of the round, reset
    #  with fold_in(k_reset, j)), the processes sampled on the recorded realised actions
    count = 4
    key, k_reset = jax.random.split(KEY)
    views = {p: subsystem(system, params, p) for p in ("P1", "P2")}
    drivers = {"P1": "A", "P2": "B"}
    state = {
        p: views[p][0].sample_initial(jax.random.fold_in(k_reset, j), views[p][1])[1]
        for j, p in ((2, "P1"), (3, "P2"))
    }
    want = {p: [state[p][f"{p}.x"]] for p in state}
    for t in range(ticks):
        key, k_tick = jax.random.split(key)
        keys = jax.random.split(k_tick, 2 * count + 1)
        for j, p in ((2, "P1"), (3, "P2")):
            inputs = {f"{p}.u_push": traj[drivers[p]].emitted["push"][t]}
            _, state[p] = views[p][0].sample_step(keys[j], views[p][1], state[p], inputs)
            want[p].append(state[p][f"{p}.x"])
    for p in ("P1", "P2"):
        assert equal(_states(traj, f"{p}.x", p), jnp.stack(want[p])), p
    #  what A and B received of P1 is P1's emission as it stood at each round's boundary
    p1 = jnp.concatenate([traj["P1"].start["see"][None], traj["P1"].emitted["see"]])
    assert equal(traj["A"].received["see"], p1)
    assert equal(traj["B"].received["see"], p1)


def test_two_minds_on_two_processes_are_the_wired_form_bitwise():
    """The wired form's run of the same participants (``migration_couple.npz``), replayed under
    its key layout: actions, both processes' states and emissions, bit for bit."""
    system, params, steps = _two_processes()
    traj = _run_old_keys(2, system, params, 5, KEY, steps, record_carries=True)
    same(traj["A"].intended["push"], "two_procs/A/intended")
    same(traj["A"].emitted["push"], "two_procs/A/realised")
    same(traj["B"].emitted["push"], "two_procs/B/realised")
    for p in ("P1", "P2"):
        same(_states(traj, f"{p}.x", p), f"two_procs/{p}/x")
        emitted = jnp.concatenate([traj[p].start["see"][None], traj[p].emitted["see"]])
        same(emitted, f"two_procs/{p}/see")


# ------------------------------------------------------------------------- 3. a direct wire


def _direct(tick_order, kernel=None):
    """Two habit-drawing pushers seeing each other's action node, ``A``'s node also driving a
    noiseless process. Held bitwise to the wired form, every order and both kernels, by the
    direct-wire case of ``tests/test_run_parity.py``."""
    system, params = couple(
        {"A": pusher(kernel=kernel), "B": pusher(), "P1": mirror(*pusher(noise=0.0), lower=True)},
        [Identify("A.push", "P1.u_push"), Identify("A.push", "B.see"), Identify("B.push", "A.see")],
    )
    steps = {"A": ancestral(), "B": ancestral(), "P1": sample()}
    return run(system, params, 12, KEY, steps, tick_order=tick_order, record_carries=True)


def test_a_direct_wire_lags_a_round_when_simultaneous():
    """Under ``"simultaneous"`` each pusher reads the other's action node at the round's boundary:
    absent at round 0 (no action realised yet), then the other's realised action of the round
    before."""
    traj = _direct("simultaneous")
    a, b = traj["A"], traj["B"]
    assert not bool(a.present["see"][0]) and not bool(b.present["see"][0])
    assert equal(b.received["see"][1:], a.emitted["push"])
    assert equal(a.received["see"][1:], b.emitted["push"])


def test_a_later_mind_sees_an_earlier_minds_act_this_round_when_sequential():
    """Entry ``t`` of ``received`` is what the node stepped on at round ``t``. ``A`` and ``B`` are
    sources stepping in listing order (the process ``A`` drives steps right after ``A``): ``B``'s
    entry ``t`` holds ``A``'s act of round ``t``, and ``A``'s entry ``t + 1`` holds ``B``'s act of
    round ``t``, absent at round 0."""
    traj = _direct("sequential")
    a, b = traj["A"], traj["B"]
    assert equal(b.received["see"][:-1], a.intended["push"])
    assert not bool(a.present["see"][0])
    assert equal(a.received["see"][1:], b.intended["push"])
    assert not equal(a.received["see"][:-1], b.intended["push"])


def _moves(traj):
    x = _states(traj, "P1.x", "P1")
    return (x[1:] - x[:-1]) % 3


def test_an_action_node_is_drawn_once_and_every_identification_carries_the_draw():
    traj = _direct("simultaneous", kernel=KERNEL)
    a, b = traj["A"], traj["B"]
    #  P1 is x' = x + a (mod 3), noiseless, so its states give the action it stepped on
    assert equal(_moves(traj), a.emitted["push"])
    assert equal(b.received["see"][1:], a.emitted["push"])
    assert not equal(a.emitted["push"], a.intended["push"])  # the kernel moved some draw


def test_under_sequential_a_draw_is_consumed_once_and_carried_to_the_later_mind():
    """A's node, identified with P1's control node and B's modality, is drawn once when A
    chooses: P1 (driven by A, so stepping right after it) moves on that draw, B (after A)
    receives it, and A's ``emitted`` records it."""
    traj = _direct("sequential", kernel=KERNEL)
    a, b = traj["A"], traj["B"]
    assert equal(_moves(traj), a.emitted["push"])
    assert equal(b.received["see"][:-1], a.emitted["push"])
    assert not equal(a.emitted["push"], a.intended["push"])


def test_the_per_component_mirror_moves_each_component_once_on_its_own_draw_under_sequential():
    """Two merged noiseless pushers, a kernel on ``a.push`` only, coupled to the merge of one
    mirror per component: each round ``world.a`` steps right after ``model.a`` on a's one draw
    and ``world.b`` after ``model.b`` on b's choice, neither moving twice, as the composite form
    stepped each component's factors at its turn. Bitwise to the composite form's run
    (``migration_couple.npz``) under its key layout, states included."""
    parts = {n: pusher(noise=0.0) for n in ("a", "b")}
    composite, cp = merge(
        {n: s for n, (s, _) in parts.items()}, {n: p for n, (_, p) in parts.items()}, []
    )
    cp = dataclasses.replace(cp, actuator={"a.push": KERNEL})
    system, params = with_mirror(composite, cp, per_component=True)
    steps = {
        "model.a": ancestral(),
        "model.b": ancestral(),
        "world.a": sample(),
        "world.b": sample(),
    }
    traj = _run_old_keys(
        2, system, params, 12, KEY, steps, tick_order="sequential", record_carries=True
    )
    for n in ("a", "b"):
        leaf = traj[f"model.{n}"]
        x = _states(traj, f"world.{n}.x", f"world.{n}")
        assert equal((x[1:] - x[:-1]) % 3, leaf.emitted["push"])
        same(leaf.intended["push"], f"composite_seq/{n}/intended")
        same(leaf.emitted["push"], f"composite_seq/{n}/realised")
        same(x, f"composite_seq/{n}/x")
    assert not equal(traj["model.a"].emitted["push"], traj["model.a"].intended["push"])


def test_a_simultaneous_kernel_run_is_pinned():
    """Under ``"simultaneous"`` the agent's node, with a kernel, is realised with the last key of
    the round (phase one), ``split(fold_in(k, ACTUATOR_TAG), n)[i]`` over the System's action
    nodes; the world component's node has none, so it is realised as intended. Under the
    composite form's key layout the run is the composite form's, value for value; under the
    round keys of ``run`` the agent's draws move and the rest stands."""
    system, params, steps = _zoo_closed(True)
    traj = _run_old_keys(2, system, params, 12, KEY, steps, record_carries=True)
    assert traj["model.agent"].emitted["a_location"].tolist() == [
        4,
        4,
        3,
        4,
        1,
        1,
        2,
        3,
        4,
        0,
        1,
        2,
    ]
    assert traj["model.world"].emitted["a_location"].tolist() == [
        1,
        2,
        2,
        0,
        1,
        4,
        2,
        4,
        1,
        4,
        0,
        4,
    ]
    location = [0, 0, 0, 0, 0, 0, 4, 2, 4, 1, 4, 0, 4]
    assert _states(traj, "world.world.location").tolist() == location
    traj = run(system, params, 12, KEY, steps, record_carries=True)
    assert traj["model.agent"].emitted["a_location"].tolist() == [
        4,
        0,
        3,
        4,
        3,
        0,
        0,
        1,
        4,
        0,
        2,
        2,
    ]
    assert traj["model.world"].emitted["a_location"].tolist() == [
        1,
        2,
        2,
        0,
        1,
        4,
        2,
        4,
        1,
        4,
        0,
        4,
    ]
    assert _states(traj, "world.world.location").tolist() == location


# ---------------------------------------------------------------------------------- 4. fixed=


def _idle_process():
    return mirror(*pusher(noise=0.0), lower=True)


def test_a_fixed_input_advances_the_process_as_an_emitted_one_would():
    """``fixed=`` clamps the process's control node; a model whose step emits the same constant,
    identified with that node, moves the noiseless process identically. Both are the wired form's
    runs (``migration_couple.npz``) under its key layout."""
    ticks = 6
    system, params = couple({"P": _idle_process()}, fixed={"P.u_push": 1})
    assert list(dict(system.subsystems)) == ["P"] and "P.u_push" not in system.names.controls
    fixed = run(system, params, ticks, KEY, {"P": sample()}, record_carries=True)
    x = _states(fixed, "P.x", "P")
    assert equal(x, (x[0] + jnp.arange(ticks + 1)) % 3)
    patched = _run_old_keys(0, system, params, ticks, KEY, {"P": sample()}, record_carries=True)
    same(_states(patched, "P.x", "P"), "fixed/P/x")
    seen = jnp.concatenate([patched["P"].start["see"][None], patched["P"].emitted["see"]])
    same(seen, "fixed/P/see")

    def constant_step(system, key, obs, carry, params):
        return ({} if obs is None else {"M.push": jnp.asarray(1, jnp.int32)}), carry, params

    system, params = couple(
        {"M": pusher(), "P": _idle_process()},
        [Identify("M.push", "P.u_push"), Identify("P.see", "M.see")],
    )
    steps = {"M": constant_step, "P": sample()}
    emitted = run(system, params, ticks, KEY, steps, record_carries=True)
    y = _states(emitted, "P.x", "P")
    assert equal(y, (y[0] + jnp.arange(ticks + 1)) % 3)
    patched_emitted = _run_old_keys(1, system, params, ticks, KEY, steps, record_carries=True)
    same(_states(patched_emitted, "P.x", "P"), "emitted/P/x")
    assert equal(_states(patched_emitted, "P.x", "P"), _states(patched, "P.x", "P"))


def test_fixed_holds_a_control_node_at_a_level_of_its_range():
    """``fixed=`` is a Clamp: a level in the node's range, or a probability vector; a scalar float
    or a name that is no control node is refused."""
    with pytest.raises(ValueError, match=r"pins level 3, which is not one of its 3 control level"):
        couple({"P": _idle_process()}, fixed={"P.u_push": 3})
    with pytest.raises(ValueError, match=r"is the scalar 1.0: a scalar prior is a control \*level"):
        couple({"P": _idle_process()}, fixed={"P.u_push": 1.0})
    with pytest.raises(ValueError, match=r"unknown control node 'P.see'"):
        couple({"P": _idle_process()}, fixed={"P.see": 0})


# ---------------------------------------------------------------------------- 5. continuous


def _continuous(width):
    """A Gaussian state ``x`` (2-d) moved by a continuous node of ``width`` through ``G u``,
    seen through a 2-d linear-Gaussian modality ``pos``."""
    system = System(
        A_deps=((0,),),
        B_deps=((1,), ()),
        controls=(1,),
        action_names=("u",),
        factor_names=("x", "u_u"),
        modality_names=("pos",),
        policy_len=1,
        planner="gradient",
        epistemic=False,
    )
    transition = LinearGaussianTransition(
        F=jnp.eye(2),
        innovations=ConditionalGaussian(
            loc=jnp.zeros((1, 2)), chol=0.1 * jnp.eye(2)[None], parent_ndims=1
        ),
        G=jnp.full((2, width), 0.5),
    )
    params = SystemParams(
        transition={"x": transition},
        initial_state_prior={"x": Gaussian(loc=jnp.zeros(2), chol=0.1 * jnp.eye(2))},
        likelihood={
            "pos": LinearGaussianObservation(H=jnp.eye(2), c=jnp.zeros(2), noise=0.1 * jnp.eye(2))
        },
        control={"u_u": Box(-2.0 * jnp.ones(width), 2.0 * jnp.ones(width))},
    )
    return system, params


def _push_one(system, key, obs, carry, params):
    return ({} if obs is None else {"M.u": jnp.asarray([1.0], jnp.float32)}), carry, params


def test_a_continuous_action_node_drives_a_continuous_input_of_its_width():
    """A model's continuous action node identified with a world's continuous control node of its
    width; the world's 2-d emission feeds the model. The run is the wired form's
    (``migration_couple.npz``) under its key layout."""
    system, params = couple(
        {"M": _continuous(1), "P": mirror(*_continuous(1), lower=True)},
        [Identify("M.u", "P.u_u"), Identify("P.pos", "M.pos")],
    )
    outs = system.ports(params).outs
    assert outs["M.u"] == ("float", 1) and outs["P.pos"] == ("float", 2)
    steps = {"M": _push_one, "P": sample()}
    traj = run(system, params, 4, KEY, steps, record_carries=True)
    assert traj["M"].intended["u"].shape == (4, 1)
    assert equal(traj["M"].emitted["u"], traj["M"].intended["u"])
    x = _states(traj, "P.x", "P")
    assert x.shape == (5, 2)
    pos = jnp.concatenate([traj["P"].start["pos"][None], traj["P"].emitted["pos"]])
    assert equal(traj["M"].received["pos"], pos)
    #  the drift is G u per round on top of the noise: the mean step is (0.5, 0.5)
    assert bool(jnp.all(jnp.mean(jnp.diff(x, axis=0), axis=0) > 0.25))
    patched = _run_old_keys(1, system, params, 4, KEY, steps, record_carries=True)
    same(_states(patched, "P.x", "P"), "continuous/P/x")
    same(
        jnp.concatenate([patched["P"].start["pos"][None], patched["P"].emitted["pos"]]),
        "continuous/P/pos",
    )
    with pytest.raises(ValueError, match=r"2 dimensions \(float\).*1 dimensions.*types must be"):
        couple(
            {"M": _continuous(2), "P": mirror(*_continuous(1), lower=True)},
            [Identify("M.u", "P.u_u"), Identify("P.pos", "M.pos")],
        )


# ----------------------------------------------------- 6. what a coupling of Systems refuses


def test_an_emission_is_an_out_port_of_a_coupling_and_its_type_must_match_the_in_port():
    """A component's emission is identified with another component's modality like an action
    node; a type mismatch is refused, and so is a binding that is no Identify or Clamp."""
    emitter, eparams = emitting_world()
    agent, aparams = tmaze_system()
    parts = {"W": (emitter, eparams), "A": (agent, aparams)}
    system, _ = couple(parts, [Identify("W.fix", "A.cue")])
    assert dict(system.observes)["A.cue"] == "W.fix"
    with pytest.raises(ValueError, match="port types must be equal"):
        couple(parts, [Identify("W.gauge", "A.cue")])
    with pytest.raises(ValueError, match="is not an Identify or a Clamp"):
        couple(parts, [("W.fix", "A.cue")])


def test_an_unknown_component_or_port_is_refused_naming_the_candidates():
    """The coupling's counterpart of the wired form's unknown participant or port: an end naming
    no component lists the components, one naming no port of its component lists that
    component's out-ports and in-ports."""
    parts = {"M": pusher(), "P": _idle_process()}
    with pytest.raises(ValueError, match=r"'Q.u_push', which names no component.*\['M', 'P'\]"):
        couple(parts, [Identify("M.push", "Q.u_push")])
    with pytest.raises(
        ValueError, match=r"'P.shove', which is no port of component 'P'.*in-ports \['P.u_push'\]"
    ):
        couple(parts, [Identify("M.push", "P.shove")])


def test_a_component_name_with_a_dot_is_refused():
    with pytest.raises(ValueError, match=r"\['M.1'\] contain the namespace separator"):
        couple({"M.1": pusher(), "P": _idle_process()})


# ---------------------------------------------------------- 7. the wired form's refusals


def _parts():
    m_sys, m_par = pusher()
    return {"M": (m_sys, m_par, constant(1))}, {"P": process(*pusher())}


GOOD = [Link("M.push", "P.push"), Link("P.see", "M.see")]


def test_an_unknown_participant_or_port_is_refused_with_the_candidates():
    minds, procs = _parts()
    with pytest.raises(ValueError, match=r"names participant 'Q'.*participants are \['M', 'P'\]"):
        couple(minds=minds, processes=procs, links=[Link("M.push", "Q.push"), GOOD[1]])
    with pytest.raises(ValueError, match=r"'P.shove' is not an in-port.*in-ports are \['P.push'\]"):
        couple(minds=minds, processes=procs, links=[Link("M.push", "P.shove"), GOOD[1]])
    with pytest.raises(ValueError, match=r"'M.see' is not an out-port.*a link's target"):
        couple(minds=minds, processes=procs, links=[Link("M.see", "P.push"), GOOD[1]])
    with pytest.raises(ValueError, match="write a port as 'participant.port'"):
        couple(minds=minds, processes=procs, links=[Link("push", "P.push"), GOOD[1]])


def test_a_type_mismatch_is_refused_with_both_types():
    minds, procs = _parts()
    s_sys, s_par = _five()
    with pytest.raises(ValueError, match=r"joins discrete, 5 levels to discrete, 3 levels"):
        couple(
            minds={**minds, "F": (s_sys, s_par, constant(0))},
            processes=procs,
            links=[*GOOD, Link("F.push", "M.see")],
        )


def _five():
    """A pusher-shaped System over five levels."""
    table = jnp.stack([jnp.roll(jnp.eye(5), u, axis=0) for u in range(5)], axis=-1)
    system = System(
        A_deps=(),
        B_deps=((0, 1), ()),
        controls=(1,),
        action_names=("push",),
        factor_names=("x", "u_push"),
        policy_len=1,
        epistemic=False,
    )
    params = SystemParams(
        transition={"x": ConditionalCategorical(log_table=jnp.log(table), parent_ndims=2)},
        initial_state_prior={"x": Categorical(logits=jnp.zeros(5))},
    )
    return system, params


def test_two_wires_into_one_in_port_are_refused():
    minds, procs = _parts()
    procs["Q"] = process(*pusher())
    with pytest.raises(ValueError, match=r"'M.see' is the target of two links.*takes one link"):
        couple(
            minds=minds,
            processes=procs,
            links=[*GOOD, Link("Q.see", "M.see")],
            fixed={"Q.push": 0},
        )
    with pytest.raises(ValueError, match="both linked .* and in fixed="):
        couple(minds=minds, processes=procs, links=GOOD, fixed={"P.push": 0, "Q.push": 0})


def test_an_unfed_in_port_is_refused_naming_its_fix():
    minds, procs = _parts()
    with pytest.raises(ValueError, match=r"process input 'P.push' is fed by no link.*fixed="):
        couple(minds=minds, processes=procs, links=[GOOD[1]])
    with pytest.raises(ValueError, match=r"mind modality 'M.see' is fed by no link.*drop the"):
        couple(minds=minds, processes=procs, links=[GOOD[0]])


def test_fixed_holds_only_a_process_input_of_its_type():
    minds, procs = _parts()
    with pytest.raises(ValueError, match=r"fixed= names 'M.see', which is not a process input"):
        couple(minds=minds, processes=procs, links=GOOD, fixed={"M.see": 0})
    with pytest.raises(ValueError, match=r"fixed= names 'M.push', which is not a process input"):
        couple(processes=procs, fixed={"M.push": 0})
    with pytest.raises(ValueError, match=r"process input 'P.push' is discrete, 3 levels"):
        couple(processes=procs, fixed={"P.push": 3})
    with pytest.raises(ValueError, match=r"process input 'P.push' is discrete, 3 levels"):
        couple(processes=procs, fixed={"P.push": 1.0})


def test_an_actuator_on_a_process_is_refused_naming_the_mind():
    minds, _ = _parts()
    env, ep = pusher(kernel=KERNEL)[0].as_env(pusher(kernel=KERNEL)[1])
    with pytest.raises(ValueError, match=r"actuator belongs to the mind that emits the action"):
        couple(minds=minds, processes={"P": (env, ep)}, links=GOOD)


def test_a_bare_positional_env_is_refused_naming_wrap():
    minds, _ = _parts()
    bare = DiscreteProcess(A_deps=((0,),), B_deps=((0,),))
    with pytest.raises(TypeError, match=r"GenerativeEnv.wrap"):
        couple(minds=minds, processes={"P": bare}, links=GOOD)
    with pytest.raises(TypeError, match=r"GenerativeEnv.wrap"):
        couple(minds=minds, processes={"P": (bare, None)}, links=GOOD)


def test_a_wire_between_two_processes_and_bad_names_are_refused():
    minds, procs = _parts()
    procs["Q"] = process(*pusher())
    with pytest.raises(ValueError, match=r"from a process's observation to a process's input"):
        couple(minds=minds, processes=procs, links=[*GOOD, Link("P.see", "Q.push")])
    with pytest.raises(ValueError, match=r"a non-empty string without a dot"):
        couple(minds={"M.1": minds["M"]}, processes=procs, links=GOOD)
    with pytest.raises(ValueError, match=r"two participants are named 'M'"):
        couple(minds=minds, processes={"M": procs["P"]}, links=GOOD)


# ----------------------------------------------------------------------- 8. run's forms


def test_run_on_a_wiring_refuses_the_composite_arguments():
    minds, procs = _parts()
    wiring = couple(minds=minds, processes=procs, links=GOOD)
    assert isinstance(wiring, Wiring)
    _, cp = pusher()
    sentence = r"run\(wiring, ticks=\.\.\., key=\.\.\.\) takes ticks and key by keyword"
    with pytest.raises(TypeError, match=sentence):
        run(wiring, ticks=2, key=KEY, params=cp)
    with pytest.raises(TypeError, match=sentence):
        run(wiring, ticks=2, key=KEY, minds={})
    with pytest.raises(TypeError, match=sentence):
        run(wiring, 2, KEY)
    with pytest.raises(TypeError, match=sentence):
        run(wiring, cp, minds, 2, KEY)
    with pytest.raises(TypeError, match="needs both ticks and key"):
        run(wiring, ticks=2)
    with pytest.raises(ValueError, match=r"carries names \['Z'\].*minds of this wiring"):
        run(wiring, ticks=2, key=KEY, carries={"Z": MindCarry()})
    with pytest.raises(ValueError, match=r"states names \['M'\].*processes of this wiring"):
        run(wiring, ticks=2, key=KEY, states={"M": {}})
    record = run(wiring, ticks=2, key=KEY, states={"P": {"x": jnp.asarray(2, jnp.int32)}})
    assert int(record["P"].states["x"][0]) == 2


def test_run_keeps_the_composite_signature_by_position_and_by_keyword():
    composite, cp, views, responds = _two_minds(False)
    by_position = run(composite, cp, responds, 2, KEY)
    by_keyword = run(composite=composite, params=cp, minds=responds, ticks=2, key=KEY)
    for f, rows in by_position.states.items():
        assert equal(by_keyword.states[f], rows)
    with pytest.raises(TypeError, match=r"is missing \['key'\]"):
        run(composite, cp, responds, 2)


# ------------------------------------------------------------ typing an outside process


class _Outside(DiscreteProcess):
    """A positional process whose observation is reshaped by ``emit``."""

    emit: object = None

    def reset(self, key, params, state=None):
        obs, s = super().reset(key, params, state)
        return self.emit(obs), s

    def step(self, key, state, action, params):
        obs, s = super().step(key, state, action, params)
        return self.emit(obs), s


class _OwnParams(eqx.Module):
    """An outside env's own params: the positional containers, not GenerativeEnvParams."""

    likelihood: tuple
    transition: tuple
    initial_state_prior: tuple
    actuator: dict


def _outside(emit):
    """The pusher's process as an outside env with its own params, emitting ``emit(obs)``."""
    system, params = pusher()
    env, ep = process(system, params)
    inner = _Outside(
        A_deps=env.inner.A_deps,
        B_deps=env.inner.B_deps,
        D_deps=env.inner.D_deps,
        B_control_deps=env.inner.B_control_deps,
        sensed=env.inner.sensed,
        emit=emit,
    )
    positional = env._params(ep)
    own = _OwnParams(
        positional.likelihood, positional.transition, positional.initial_state_prior, {}
    )
    return GenerativeEnv.wrap(inner, system, params), own


def test_a_port_with_no_level_count_or_a_shapeless_vector_is_refused():
    minds, _ = _parts()
    system, params = pusher()
    env, ep = process(system, params)
    #  wrapped without params, the inputs have no level count
    bare = GenerativeEnv.wrap(env.inner, system)
    with pytest.raises(ValueError, match=r"GenerativeEnv.wrap\(env, system, params\)"):
        couple(minds=minds, processes={"P": (bare, ep)}, links=GOOD)
    #  with params, they do, and an outside env typed by GenerativeEnvParams couples
    typed = GenerativeEnv.wrap(env.inner, system, params)
    assert couple(minds=minds, processes={"P": (typed, ep)}, links=GOOD).types["P.push"] == (
        "level",
        3,
    )
    #  an outside env's own params: a level with no level count is refused
    outside = _outside(lambda obs: obs)
    with pytest.raises(ValueError, match=r"emits 'see' as a level, and its params give no level"):
        couple(minds=minds, processes={"P": outside}, links=GOOD)
    #  a float observation of no vector shape is refused with its shape
    scalar = _outside(lambda obs: tuple(o.astype(jnp.float32) for o in obs))
    with pytest.raises(ValueError, match=r"emits 'see' with shape \(\); a continuous port"):
        couple(minds=minds, processes={"P": scalar}, links=GOOD)


def test_a_wired_mind_emitting_a_level_out_of_range_is_refused():
    minds, procs = _parts()
    minds["M"] = (*minds["M"][:2], constant(7))
    wiring = couple(minds=minds, processes=procs, links=GOOD)
    with pytest.raises(ValueError, match=r"actions\['push'\] = 7; action node 'push' has 3 levels"):
        run(wiring, ticks=1, key=KEY)
