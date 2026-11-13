"""The node-name keywords of a System: ``state_names``, ``control_names`` and ``modality_names``,
and the derived ``node_names`` the index-based fields index into."""

import dataclasses

import pytest

from hybrid_aif import System


def _native(**kw):
    return System(
        A_deps=((0,), (1,)),
        B_deps=((0, 2), (1, 3), (), ()),
        controls=(2, 3),
        action_names=("a_go", "a_look"),
        policy_len=1,
        **kw,
    )


def test_the_three_keywords_are_read_back_as_attributes():
    s = _native(
        state_names=("pos", "ctx"),
        control_names=("go", "look"),
        modality_names=("pos_obs", "ctx_obs"),
    )
    assert s.state_names == ("pos", "ctx")
    assert s.control_names == ("go", "look")
    assert s.modality_names == ("pos_obs", "ctx_obs")
    assert s.names.states == s.state_names and s.names.controls == s.control_names


def test_node_names_are_the_state_factors_then_the_control_nodes():
    s = _native(state_names=("pos", "ctx"), control_names=("go", "look"))
    assert s.node_names == ("pos", "ctx", "go", "look")
    assert [s.node_names[c] for c in s.controls] == ["go", "look"]
    assert [s.node_names[g] for g in s.B_deps[0]] == ["pos", "go"]


def test_the_combined_keyword_is_refused_with_the_split():
    with pytest.raises(TypeError) as info:
        System(B_deps=((0,),), factor_names=("x",), policy_len=1)
    message = str(info.value)
    assert "factor_names" in message
    for keyword in ("state_names=", "control_names=", "modality_names="):
        assert keyword in message
    assert "len(B_deps) - len(controls)" in message
    with pytest.raises(TypeError, match="state_names="):
        dataclasses.replace(System(B_deps=((0,),), policy_len=1), factor_names=("x",))


def test_the_combined_name_is_no_attribute():
    s = System(B_deps=((0,),), policy_len=1)
    with pytest.raises(AttributeError):
        s.factor_names  # noqa: B018
    assert "factor_names" not in System.__dataclass_fields__


def test_generated_names_under_the_sugar():
    s = System(B_deps=((), ()), controlled=(True, False), policy_len=1, A_deps=((0,),))
    assert s.state_names == ("factor_0", "factor_1")
    assert s.control_names == ("u_factor_0",)
    assert s.node_names == ("factor_0", "factor_1", "u_factor_0")
    assert s.action_names == ("a_factor_0",)
    assert s.modality_names == ("modality_0",)
    named = System(
        B_deps=((), (), ()),
        controlled=(False, True, True),
        state_names=("a", "b", "c"),
        policy_len=1,
    )
    assert named.control_names == ("u_b", "u_c")
    assert named.node_names == ("a", "b", "c", "u_b", "u_c")
    plain = System(B_deps=((0,), (1,)), policy_len=1)
    assert plain.state_names == ("factor_0", "factor_1") and plain.control_names == ()


def test_control_names_under_the_sugar_must_be_the_generated_ones():
    kw = dict(B_deps=((), ()), controlled=(True, False), state_names=("a", "b"), policy_len=1)
    assert System(**kw, control_names=("u_a",)) == System(**kw)
    with pytest.raises(ValueError, match=r"the sugar names each controlled state factor"):
        System(**kw, control_names=("go",))


def test_a_native_control_node_needs_its_name():
    with pytest.raises(ValueError, match=r"control_names=None on a System with control nodes"):
        _native(state_names=("pos", "ctx"))
    #  the state factors' names are generated beside declared control names
    s = _native(control_names=("go", "look"))
    assert s.state_names == ("factor_0", "factor_1")


def test_name_counts_are_checked_against_the_declared_nodes():
    with pytest.raises(ValueError, match=r"state_names has 1 name\(s\) for 2 state factor\(s\)"):
        _native(state_names=("pos",), control_names=("go", "look"))
    with pytest.raises(ValueError, match=r"control_names has 1 name\(s\) for 2 control node\(s\)"):
        _native(state_names=("pos", "ctx"), control_names=("go",))
    with pytest.raises(ValueError, match=r"control_names has 1 name\(s\) for 0 control node\(s\)"):
        System(B_deps=((0,),), control_names=("go",), policy_len=1)
    with pytest.raises(ValueError, match=r"repeat \['go'\]"):
        _native(state_names=("go", "ctx"), control_names=("go", "look"))


def test_replace_keeps_the_names_and_equality():
    s = _native(state_names=("pos", "ctx"), control_names=("go", "look"))
    again = dataclasses.replace(s, gamma=4.0)
    assert again.node_names == s.node_names
    assert dataclasses.replace(again, gamma=s.gamma) == s
    assert hash(dataclasses.replace(s)) == hash(s)
    renamed = dataclasses.replace(s, control_names=("move", "peek"))
    assert renamed.node_names == ("pos", "ctx", "move", "peek") and renamed != s
