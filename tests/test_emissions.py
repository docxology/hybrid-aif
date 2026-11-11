"""Emissions: the action node of a state factor, an output a System draws from its state.

An emission is declared in ``System.emissions`` (a name and its parents, state factors and
control nodes), carries a kernel of any likelihood family in ``SystemParams.emission``, and has
its port type read off that kernel. ``System.emit`` realises it from a world's point state or an
agent's belief; ``System.ports`` lists the in-ports and out-ports; ``merge`` namespaces
emissions and ``subsystem`` carries a component's own. Nothing in inference, planning or learning
reads one, and nothing in a model may read or score one. Description and description length
treat an emission's kernel as a likelihood of its family.

Design: docs/design/nesting-and-coupling.md N5; docs/design/engine-consolidation.md.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_aif import (
    Categorical,
    ConditionalCategorical,
    Emission,
    Gaussian,
    LinearGaussianObservation,
    MLPConditional,
    System,
    SystemParams,
)
from hybrid_aif.composition import joint_planner
from hybrid_aif.composition.merge import Link, merge, subsystem
from hybrid_aif.controls import FLOAT, LEVEL
from hybrid_aif.description import build, describe
from hybrid_aif.description.mdl import description_length
from hybrid_aif.system import Ports
from tests.test_hybrid_lg_system import make_params as hill_params
from tests.test_hybrid_lg_system import make_system as hill_system
from tests.test_system import tmaze_system

KEY = jax.random.PRNGKey(5)


def _cc(table, parent_ndims: int) -> ConditionalCategorical:
    return ConditionalCategorical(log_table=jnp.log(jnp.asarray(table)), parent_ndims=parent_ndims)


def _move_table() -> np.ndarray:
    """``x' = clip(x + v - 1)``, child-first ``(x', x, v)``."""
    table = np.zeros((3, 3, 3))
    for v in range(3):
        for x in range(3):
            table[min(max(x + v - 1, 0), 2), x, v] = 1.0
    return table


def _img_table() -> np.ndarray:
    """``P(img | x, terrain)``, child-first ``(4, 3, 2)``."""
    return np.random.default_rng(0).dirichlet(np.ones(4), size=(3, 2)).transpose(2, 0, 1)


def world(**emissions):
    """A world: a position ``x`` moved by the control node ``v``, a ``terrain`` that stays, and
    three outputs, over ``x`` (``fix``), over ``(x, terrain)`` (``img``, realised by its mode)
    and over the control node (``gauge``)."""
    declared = emissions or {
        "fix": ("x",),
        "img": Emission(("x", "terrain"), realise="mode"),
        "gauge": ("v",),
    }
    system = System(
        B_deps=((0, 2), (1,), ()),
        controls=(2,),
        factor_names=("x", "terrain", "v"),
        action_names=("a_v",),
        emissions=declared,
        policy_len=1,
    )
    params = SystemParams(
        transition={"x": _cc(_move_table(), 2), "terrain": _cc(np.eye(2), 1)},
        initial_state_prior={
            "x": Categorical(logits=jnp.zeros(3)),
            "terrain": Categorical(logits=jnp.zeros(2)),
        },
        emission={
            "fix": _cc(0.8 * np.eye(3) + 0.2 / 3, 1),
            "img": _cc(_img_table(), 2),
            "gauge": _cc(np.array([[0.9, 0.5, 0.1], [0.1, 0.5, 0.9]]), 1),
        },
    )
    return system, params


POINT = {"x": jnp.asarray(1), "terrain": jnp.asarray(0), "v": jnp.asarray(2)}


# ------------------------------------------------------------------------ declaration


def test_a_mapping_and_pairs_of_parents_or_emits_normalise_to_one_declaration():
    by_tuple, _ = world(fix=("x",), img=("x", "terrain"), gauge=("v",))
    by_emit, _ = world(
        fix=Emission(("x",)),
        img=Emission(("x", "terrain"), "sample"),
        gauge=Emission(("v",), "sample"),
    )
    by_pairs, _ = world(
        **{}
    )  # the default declaration, with img realised by its mode, differs only there
    assert by_tuple == by_emit
    assert by_tuple.emissions == (
        ("fix", Emission(("x",), "sample")),
        ("img", Emission(("x", "terrain"), "sample")),
        ("gauge", Emission(("v",), "sample")),
    )
    assert dict(by_pairs.emissions)["img"].realise == "mode"
    assert by_tuple != by_pairs


def test_the_declaration_order_is_kept_and_a_system_hashes():
    system, _ = world(gauge=("v",), fix=("x",))
    assert system.names.emissions == ("gauge", "fix")
    assert hash(system) == hash(world(gauge=("v",), fix=("x",))[0])


def test_an_emission_is_no_action_and_no_node():
    system, params = world()
    assert system.names.actions == ("a_v",)
    assert system.names.emissions == ("fix", "img", "gauge")
    assert system.names.states == ("x", "terrain") and system.names.controls == ("v",)
    assert system.A_deps == () and system.modality_names == ()
    assert len(system.B_deps) == 3  # no node index for an emission


@pytest.mark.parametrize(
    ("declared", "match"),
    [
        ({"x": ("terrain",)}, "also a state factor or control node's name"),
        ({"a_v": ("x",)}, "also an action node's name"),
        ({"e": ()}, "names no parent"),
        ({"e": ("x", "x")}, "twice among its parents"),
        ({"e": ("nowhere",)}, "not a node of this System"),
        ({"e": ("a_v",)}, "an action node; an emission reads the control node 'v'"),
        ({"e": ("x",), "f": ("e",)}, "an emission, an output that nothing reads"),
        ({"e": Emission(("x",), "median")}, "realise must be one of"),
        ({"e": 3}, "emissions must map"),
    ],
)
def test_a_malformed_declaration_is_refused_with_a_sentence(declared, match):
    with pytest.raises((ValueError, TypeError), match=match):
        world(**declared)


def test_the_kernels_are_checked_against_the_declaration():
    system, params = world()
    with pytest.raises(ValueError, match="one kernel per emission"):
        system._validate(dataclasses.replace(params, emission={"fix": params.emission["fix"]}))
    with pytest.raises(ValueError, match="which are not emissions of this System"):
        system._validate(dataclasses.replace(params, emission={**params.emission, "ghost": 1}))
    bad_axes = dict(params.emission, fix=_cc(np.eye(3)[..., None] * np.ones((3, 3, 2)), 2))
    with pytest.raises(ValueError, match="parent axis/es and the emission names 1"):
        system._validate(dataclasses.replace(params, emission=bad_axes))
    bad_size = dict(params.emission, fix=_cc(np.ones((3, 4)) / 3, 1))
    with pytest.raises(ValueError, match="parent axis 0 has 4 entries"):
        system._validate(dataclasses.replace(params, emission=bad_size))
    not_a_likelihood = dict(params.emission, fix=Gaussian(loc=jnp.zeros(1), chol=jnp.ones(1)))
    with pytest.raises(ValueError, match="no likelihood family"):
        system._validate(dataclasses.replace(params, emission=not_a_likelihood))


# ----------------------------------------------------------------------------- ports


def test_ports_list_modalities_and_control_nodes_in_and_action_nodes_and_emissions_out():
    system, params = world()
    ports = system.ports(params)
    assert isinstance(ports, Ports)
    assert ports.ins == {"v": (LEVEL, 3)}
    assert ports.outs == {
        "a_v": (LEVEL, 3),
        "fix": (LEVEL, 3),
        "img": (LEVEL, 4),
        "gauge": (LEVEL, 2),
    }


def test_a_linear_gaussian_emission_is_a_float_port_and_a_modality_an_in_port():
    system, params = hill_system(), hill_params()
    assert system.ports(params).ins == {"position": (FLOAT, 1), "u_state": (LEVEL, 3)}
    system = dataclasses.replace(system, emissions={"gps": ("state",)})
    params = dataclasses.replace(
        params,
        emission={"gps": LinearGaussianObservation(H=jnp.eye(2), c=jnp.zeros(2), noise=jnp.eye(2))},
    )
    assert system.ports(params).outs == {"a_state": (LEVEL, 3), "gps": (FLOAT, 2)}


def test_a_control_node_only_an_emission_reads_is_sized_by_its_kernel():
    system = System(
        B_deps=((), ()),
        controls=(1,),
        factor_names=("x", "u"),
        action_names=("a",),
        emissions={"gauge": ("u",)},
        policy_len=1,
    )
    params = SystemParams(
        transition={"x": _cc(np.ones((2, 1)) / 2, 1)},
        initial_state_prior={"x": Categorical(logits=jnp.zeros(2))},
        emission={"gauge": _cc(np.ones((2, 4)) / 2, 1)},
    )
    assert system.ports(params).ins["u"] == (LEVEL, 4)


# ----------------------------------------------------------------------------- emit


def test_emit_samples_from_a_point_state_as_the_forward_process_does():
    system, params = world()
    out = system.emit(KEY, params, POINT)
    assert list(out) == ["fix", "img", "gauge"]
    positional = system._p(params)
    process, pp, order = system._emission_observer(
        system._kernelled(positional), system._emission_decls(positional)
    )
    drawn = process._observe(KEY, tuple(POINT[p] for p in order), pp)
    #  `img` is realised by its mode, the others by the draw the process makes
    assert int(out["fix"]) == int(drawn[0]) and int(out["gauge"]) == int(drawn[2])
    again = system.emit(KEY, params, POINT)
    assert all(int(out[k]) == int(again[k]) for k in out)
    other = [int(system.emit(k, params, POINT)["fix"]) for k in jax.random.split(KEY, 60)]
    assert len(set(other)) > 1  # a draw, not a table lookup


def test_emit_mode_and_stats_from_a_point_state():
    system, params = world(fix=Emission(("x",), "stats"), img=Emission(("x", "terrain"), "mode"))
    params = dataclasses.replace(
        params, emission={k: v for k, v in params.emission.items() if k != "gauge"}
    )
    out = system.emit(KEY, params, POINT)
    img = np.exp(np.asarray(params.emission["img"].log_table))[:, 1, 0]
    np.testing.assert_allclose(
        np.asarray(out["fix"]),
        np.exp(np.asarray(params.emission["fix"].log_table))[:, 1],
        atol=1e-6,
    )
    assert int(out["img"]) == int(np.argmax(img))


def test_emit_from_a_belief_draws_the_parents_then_the_kernel():
    system, params = world(fix=("x",), img=("x", "terrain"))
    params = dataclasses.replace(
        params, emission={k: v for k, v in params.emission.items() if k != "gauge"}
    )
    sure = {
        "x": Categorical.from_probs(jax.nn.one_hot(1, 3)),
        "terrain": Categorical.from_probs(jax.nn.one_hot(0, 2)),
    }
    point = {"x": jnp.asarray(1), "terrain": jnp.asarray(0)}
    from_belief = system.emit(KEY, params, sure)
    from_point = system.emit(KEY, params, point)
    assert all(int(from_belief[k]) == int(from_point[k]) for k in from_point)
    #  an uncertain belief draws the predictive: its marginal is the contracted table
    belief = {"x": Categorical.from_probs(jnp.asarray([0.2, 0.5, 0.3])), "terrain": sure["terrain"]}
    keys = jax.random.split(KEY, 4000)
    draws = jax.vmap(lambda k: system.emit(k, params, belief)["fix"])(keys)
    table = np.exp(np.asarray(params.emission["fix"].log_table))
    want = table @ np.asarray([0.2, 0.5, 0.3])
    got = np.bincount(np.asarray(draws), minlength=3) / 4000
    np.testing.assert_allclose(got, want, atol=0.03)


def test_emit_mode_and_stats_from_a_belief_are_the_predictives():
    system, params = world(fix=Emission(("x",), "stats"), img=Emission(("x", "terrain"), "mode"))
    params = dataclasses.replace(
        params, emission={k: v for k, v in params.emission.items() if k != "gauge"}
    )
    belief = {
        "x": Categorical.from_probs(jnp.asarray([0.2, 0.5, 0.3])),
        "terrain": Categorical.from_probs(jnp.asarray([0.6, 0.4])),
    }
    out = system.emit(KEY, params, belief)
    fix = np.exp(np.asarray(params.emission["fix"].log_table))
    np.testing.assert_allclose(np.asarray(out["fix"]), fix @ np.asarray([0.2, 0.5, 0.3]), atol=1e-6)
    img = np.exp(np.asarray(params.emission["img"].log_table))
    marginal = np.einsum("oxt,x,t->o", img, [0.2, 0.5, 0.3], [0.6, 0.4])
    assert int(out["img"]) == int(np.argmax(marginal))


def test_a_linear_gaussian_emission_realises_its_mean_and_covariance():
    system = dataclasses.replace(hill_system(), emissions={"gps": Emission(("state",), "stats")})
    H = jnp.asarray([[1.0, 0.5]])
    R = jnp.asarray([[0.3]])
    params = dataclasses.replace(
        hill_params(),
        emission={"gps": LinearGaussianObservation(H=H, c=jnp.asarray([0.2]), noise=R)},
    )
    belief = Gaussian(loc=jnp.asarray([0.4, -0.2]), chol=jnp.asarray([[0.3, 0.0], [0.1, 0.2]]))
    mean, cov = system.emit(KEY, params, {"state": belief})["gps"]
    np.testing.assert_allclose(np.asarray(mean), np.asarray(H @ belief.loc + 0.2), atol=1e-6)
    want = H @ belief.cov @ H.T + R @ R.T
    np.testing.assert_allclose(np.asarray(cov), np.asarray(want), atol=1e-6)
    point = system.emit(KEY, params, {"state": jnp.asarray([0.4, -0.2])})["gps"]
    np.testing.assert_allclose(np.asarray(point[1]), np.asarray(R @ R.T), atol=1e-6)
    sampled = dataclasses.replace(system, emissions={"gps": ("state",)})
    draw = jax.vmap(lambda k: sampled.emit(k, params, {"state": belief})["gps"])(
        jax.random.split(KEY, 6000)
    )
    np.testing.assert_allclose(np.asarray(draw.mean(0)), np.asarray(mean), atol=0.03)
    np.testing.assert_allclose(float(draw.var()), float(cov[0, 0]), rtol=0.1)


def test_a_neural_kernel_is_realised_by_a_draw_only():
    system = dataclasses.replace(hill_system(), emissions={"net": Emission(("state",), "mode")})
    net = MLPConditional.from_shapes((2,), 1, key=KEY, width=4, depth=1)
    params = dataclasses.replace(hill_params(), emission={"net": net})
    with pytest.raises(NotImplementedError, match="neural kernel has no closed-form"):
        system.emit(KEY, params, {"state": jnp.zeros(2)})
    drawn = dataclasses.replace(system, emissions={"net": ("state",)}).emit(
        KEY, params, {"state": jnp.zeros(2)}
    )
    assert drawn["net"].shape == (1,)


def test_emit_names_the_parents_it_was_not_given():
    system, params = world()
    with pytest.raises(ValueError, match=r"no entry for \['terrain'\]"):
        system.emit(KEY, params, {"x": jnp.asarray(0), "v": jnp.asarray(0)})


def test_a_system_with_no_emission_emits_nothing():
    system, params = tmaze_system()
    assert system.emit(KEY, params, {}) == {}


# --------------------------------------------- nothing in the model reads or scores one


def test_no_read_target_or_selector_may_name_an_emission():
    agent, agent_params = tmaze_system()
    both = dataclasses.replace(agent, emissions={"seen": ("location",)})
    for field, value in (
        ("C_deps", (("seen", ("context",)),)),
        ("C_deps", (("reward", ("seen",)),)),
        ("readings", (("seen", "u_location"),)),
        ("branch_on", ("seen",)),
    ):
        with pytest.raises(ValueError, match="an emission: an emission is an output"):
            dataclasses.replace(both, **{field: value})
    emission = {"seen": _cc(np.eye(5), 1)}
    carrying = dataclasses.replace(agent_params, emission=emission)
    for field, key in (("obs_targets", "seen"), ("state_targets", "seen")):
        targeted = dataclasses.replace(carrying, **{field: {key: Categorical(logits=jnp.zeros(3))}})
        with pytest.raises(ValueError, match="an emission: an emission is an output"):
            both._validate(targeted)
    with pytest.raises(ValueError, match="an emission: an emission is an output"):
        dataclasses.replace(both, epistemic=("obs:seen",)).planner_context(
            carrying, both.initial_priors(carrying)
        )


def test_inference_and_planning_are_bitwise_unchanged_by_adding_an_emission():
    for system, params, emission in (
        (*tmaze_system(), {"seen": _cc(np.eye(5), 1)}),
        (
            hill_system(),
            hill_params(),
            {"gps": LinearGaussianObservation(H=jnp.eye(2), c=jnp.zeros(2), noise=jnp.eye(2))},
        ),
    ):
        name = next(iter(emission))
        parent = system.names.states[0]
        with_emission = dataclasses.replace(system, emissions={name: (parent,)})
        with_params = dataclasses.replace(params, emission=emission)
        assert with_emission._is_hybrid(with_params) == system._is_hybrid(params)
        obs = {
            m: (jnp.asarray(0) if kind == LEVEL else jnp.asarray([0.05]))
            for m, (kind, _) in system._modality_types(params).items()
        }
        for a, b in ((system, params), (with_emission, with_params)):
            a.compiled().infer_states(b, obs)
        b0, _ = system.infer_states(params, obs)
        b1, _ = with_emission.infer_states(with_params, obs)
        for x, y in zip(jax.tree_util.tree_leaves(b0), jax.tree_util.tree_leaves(b1), strict=True):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
        q0, t0 = system.infer_policies(params, b0)
        q1, t1 = with_emission.infer_policies(with_params, b1)
        for x, y in zip(
            jax.tree_util.tree_leaves((q0, t0)), jax.tree_util.tree_leaves((q1, t1)), strict=True
        ):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
        a0 = system.act(KEY, params, b0)[0]
        a1 = with_emission.act(KEY, with_params, b1)[0]
        assert {k: np.asarray(v).tolist() for k, v in a0.items()} == {
            k: np.asarray(v).tolist() for k, v in a1.items()
        }
        if not system._is_hybrid(params):
            np.testing.assert_array_equal(
                np.asarray(system.policies(params)), np.asarray(with_emission.policies(with_params))
            )
        assert with_emission.names.actions == system.names.actions


def test_a_linear_gaussian_emission_on_a_tabular_agent_keeps_the_compiled_path():
    system, params = tmaze_system()
    system = dataclasses.replace(system, emissions={"gps": ("location",)})
    params = dataclasses.replace(
        params,
        emission={
            "gps": LinearGaussianObservation(H=jnp.zeros((1, 0)), c=jnp.zeros(1), noise=jnp.eye(1))
        },
    )
    assert not system._is_hybrid(params)


# ------------------------------------------------------------ merge and subsystem


def _two_worlds():
    system, params = world()
    return merge({"a": system, "b": system}, {"a": params, "b": params})


def test_merge_namespaces_emissions_and_their_kernels():
    cs, cp = _two_worlds()
    assert cs.names.emissions == ("a.fix", "a.img", "a.gauge", "b.fix", "b.img", "b.gauge")
    emitted = dict(cs.emissions)
    assert emitted["a.img"] == Emission(("a.x", "a.terrain"), "mode")
    assert emitted["b.gauge"] == Emission(("b.v",), "sample")
    system, params = world()
    assert set(cp.emission) == set(cs.names.emissions)
    for name in ("fix", "img", "gauge"):
        np.testing.assert_array_equal(
            np.asarray(cp.emission[f"b.{name}"].log_table),
            np.asarray(params.emission[name].log_table),
        )
    assert dict(cs.subsystems)["a"].emissions == ("a.fix", "a.img", "a.gauge")
    assert dict(cs.subsystems)["b"].emissions == ("b.fix", "b.img", "b.gauge")
    assert cs.ports(cp).outs.keys() >= {"a.fix", "b.img", "a.a_v"}


def test_a_control_parent_follows_a_same_step_identification_and_a_removed_one_is_refused():
    system, params = world()
    cs, cp = joint_planner(
        *merge(
            {"a": system, "b": system},
            {"a": params, "b": params},
            [Link("a.a_v", "b.v")],
        )
    )
    assert dict(cs.emissions)["b.gauge"].parents == ("a.v",)
    from hybrid_aif.composition.merge import Clamp

    with pytest.raises(ValueError, match="which a binding removed .terminated by a Clamp"):
        merge(
            {"a": system, "b": system},
            {"a": params, "b": params},
            [Clamp("b.v", 1)],
        )


def test_a_shared_parameter_variable_read_by_emissions_is_namespaced_and_identified():
    from hybrid_aif import Parameter
    from hybrid_aif.composition.merge import Identify
    from hybrid_aif.distributions import ConditionalDirichlet

    system, params = world()
    params = dataclasses.replace(
        params,
        emission=dict(params.emission, fix=Parameter("fixtable")),
        parameters={
            "fixtable": ConditionalDirichlet(
                concentration=jnp.asarray(0.8 * np.eye(3) + 0.2 / 3) * 10, parent_ndims=1
            )
        },
    )
    cs, cp = merge(
        {"a": system, "b": system},
        {"a": params, "b": params},
        [Identify("a.fixtable", "b.fixtable")],
    )
    assert set(cp.parameters) == {"a.fixtable"}
    assert cp.emission["a.fix"] == Parameter("a.fixtable")
    assert cp.emission["b.fix"] == Parameter("a.fixtable")
    out = cs.emit(
        KEY,
        cp,
        {
            **{f"{c}.x": POINT["x"] for c in "ab"},
            **{f"{c}.terrain": POINT["terrain"] for c in "ab"},
            **{f"{c}.v": POINT["v"] for c in "ab"},
        },
    )
    assert set(out) == set(cs.names.emissions)


def test_a_view_carries_its_own_emissions_and_refuses_a_partners_control_node():
    cs, cp = _two_worlds()
    view, vp = subsystem(cs, cp, "a")
    assert view.names.emissions == ("a.fix", "a.img", "a.gauge")
    assert set(vp.emission) == set(view.names.emissions)
    assert dict(view.subsystems)["a"].emissions == view.names.emissions
    assert view.ports(vp).outs.keys() >= {"a.fix", "a.img", "a.gauge"}
    system, params = world()
    shared, sp = joint_planner(
        *merge(
            {"a": system, "b": system},
            {"a": params, "b": params},
            [Link("a.a_v", "b.v")],
        )
    )
    mine, _ = subsystem(shared, sp, "a")  # `a` chooses the node `b`'s emission reads
    assert "a.v" in mine.names.controls
    with pytest.raises(NotImplementedError, match="another component chooses"):
        subsystem(shared, sp, "b")


# -------------------------------------------------------- description and its length


def _round_trip(system, params):
    d = describe(system, params)
    return d, build(d.document, arrays=d.arrays)


def test_an_emission_is_described_as_a_kernel_under_the_actuator_label_and_built_back():
    system, params = world()
    d, (built, built_params) = _round_trip(system, params)
    assert d.document["factors"]["img"]["actuator"]["options"] == {"realise": "mode"}
    assert d.document["factors"]["img"]["actuator"]["parents"] == ["x", "terrain"]
    assert "options" not in d.document["factors"]["fix"]["actuator"]
    assert built == system
    for name in system.names.emissions:
        np.testing.assert_array_equal(
            np.asarray(built_params.emission[name].log_table),
            np.asarray(params.emission[name].log_table),
        )
    assert d.document["relations"]["controls"] == {"v": "a_v"}


def test_an_actuator_on_an_action_node_is_still_the_record_it_was():
    from tests.test_action_nodes import pusher

    kernel = jnp.asarray([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])
    system, params = pusher(kernel=kernel)
    d, (built, built_params) = _round_trip(system, params)
    assert d.document["factors"]["push"]["actuator"]["parents"] == ["u_push"]
    assert built == system and built.emissions == ()
    np.testing.assert_allclose(
        np.asarray(built_params.actuator["push"]), np.asarray(kernel), atol=1e-6
    )


def test_a_linear_gaussian_and_a_reference_emission_survive_the_round_trip():
    from hybrid_aif import Parameter
    from hybrid_aif.distributions import ConditionalDirichlet

    system = dataclasses.replace(hill_system(), emissions={"gps": Emission(("state",), "mode")})
    params = dataclasses.replace(
        hill_params(),
        emission={"gps": LinearGaussianObservation(H=jnp.eye(2), c=jnp.zeros(2), noise=jnp.eye(2))},
    )
    d, (built, built_params) = _round_trip(system, params)
    assert built == system
    np.testing.assert_array_equal(
        np.asarray(built_params.emission["gps"].H), np.asarray(params.emission["gps"].H)
    )
    tabular, tparams = world()
    tparams = dataclasses.replace(
        tparams,
        emission=dict(tparams.emission, fix=Parameter("fixtable")),
        parameters={
            "fixtable": ConditionalDirichlet(
                concentration=jnp.asarray(0.8 * np.eye(3) + 0.2 / 3) * 10, parent_ndims=1
            )
        },
    )
    d, (built, built_params) = _round_trip(tabular, tparams)
    assert built_params.emission["fix"] == Parameter("fixtable")
    assert "parameter" in d.document["factors"]["fix"]["actuator"]


def test_a_merged_world_is_described_with_its_components_emissions():
    cs, cp = _two_worlds()
    d, (built, built_params) = _round_trip(cs, cp)
    shown = tuple((n, sub._replace(recognition=())) for n, sub in cs.subsystems)
    assert built == dataclasses.replace(cs, subsystems=shown)
    assert dict(built.subsystems)["b"].emissions == ("b.fix", "b.img", "b.gauge")


def test_a_document_refuses_what_reads_an_emission_and_a_misplaced_kernel():
    system, params = world()
    d = describe(system, params)
    document = jax.tree_util.tree_map(lambda x: x, d.document)
    reads = {
        **document,
        "factors": {
            **document["factors"],
            "x": {
                **document["factors"]["x"],
                "transition": {
                    **document["factors"]["x"]["transition"],
                    "parents": ["x", "v", "fix"],
                },
            },
        },
    }
    from hybrid_aif.description import DescriptionError

    with pytest.raises(DescriptionError, match="an emission: an emission is an output"):
        build(reads, arrays=d.arrays)
    late = {
        **document,
        "factors": {
            **document["factors"],
            "fix": {"actuator": {**document["factors"]["fix"]["actuator"], "parents": ["x[t-1]"]}},
        },
    }
    with pytest.raises(DescriptionError, match="reads its parents at the child's step"):
        build(late, arrays=d.arrays)


def test_an_emissions_kernel_is_priced_as_the_likelihood_of_its_family():
    emitter, params = world()
    emitter = dataclasses.replace(emitter, emissions={"fix": ("x",)})
    params = dataclasses.replace(params, emission={"fix": params.emission["fix"]})
    reader = System(
        A_deps=((0,),),
        B_deps=((0, 2), (1,), ()),
        controls=(2,),
        factor_names=("x", "terrain", "v"),
        action_names=("a_v",),
        modality_names=("fix",),
        policy_len=1,
    )
    read_params = SystemParams(
        likelihood={"fix": params.emission["fix"]},
        transition=params.transition,
        initial_state_prior=params.initial_state_prior,
    )
    a = description_length(emitter, 200.0, params=params)
    b = description_length(reader, 200.0, params=read_params)
    ea, lb = a.per_address["emission:fix"], b.per_address["likelihood:fix"]
    assert ea == lb
    assert a.total == pytest.approx(
        b.total - b.per_address["graph"].graph + a.per_address["graph"].graph, rel=1e-9
    )
    unscored = {(addr.split(":")[1], link) for addr, link in a.unscored if addr == "emission:fix"}
    assert unscored == {
        (addr.split(":")[1], link) for addr, link in b.unscored if addr == "likelihood:fix"
    }


def _log_star(n):
    from hybrid_aif.description.mdl import log_star

    return log_star(n)


def test_the_graph_code_counts_the_emissions_prefix_free():
    emitter, params = world(fix=("x",))
    params = dataclasses.replace(params, emission={"fix": params.emission["fix"]})
    three, tparams = world()
    one = description_length(emitter, 200.0, params=params).per_address["graph"].graph
    many = description_length(three, 200.0, params=tparams).per_address["graph"].graph
    assert many - one == pytest.approx(_log_star(4) - _log_star(2))
