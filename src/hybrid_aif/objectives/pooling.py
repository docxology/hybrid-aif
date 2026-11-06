"""Pooled readings: a slow table's one reading counted as several fast ones.

A :class:`Pooled` lists, for each declared factor or modality, the control node whose level at a
plan step selects how many fast readings that step's one reading pools, and the count per level.
:func:`scale_pooled` multiplies the table's novelty leaf (``infogain`` under ``transition:<f>`` or
``likelihood:<m>``) by the step's count; no other leaf and no target reads it.

Design: docs/design/step-durations.md T10.
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from hybrid_aif.controls import ControlLayout, is_belief, node_entry
from hybrid_aif.interval import Interval
from hybrid_aif.objectives.base import INFOGAIN, TermTree


class PooledEntry(NamedTuple):
    """The novelty leaf ``key`` is multiplied by ``counts[level]``, the level being that of
    control node ``node`` (a column of the control table) at the step."""

    key: str
    node: int
    counts: Array


class Pooled(NamedTuple):
    entries: tuple[PooledEntry, ...]


def count_at(entry: PooledEntry, action: object, layout: ControlLayout | None) -> Array:
    """The step's count: the entry at the node's clamped level, or under a probability vector
    over the levels the expected count (the novelty itself is computed on the mixed table)."""
    index = node_entry(action, entry.node, layout)
    if is_belief(index):
        return jnp.sum(index * entry.counts, axis=-1)
    return entry.counts[jnp.asarray(index, jnp.int32)]


def scale_pooled(
    tree: TermTree, pooled: Pooled | None, action: object, layout: ControlLayout | None
) -> TermTree:
    """``tree`` with each declared table's novelty leaf multiplied by the step's count."""
    if pooled is None or INFOGAIN not in tree:
        return tree
    leaves = dict(tree[INFOGAIN])
    for entry in pooled.entries:
        if entry.key in leaves:
            iv = leaves[entry.key]
            n = count_at(entry, action, layout)
            leaves[entry.key] = Interval(iv.lower * n, iv.upper * n, iv.derived, iv.approximate)
    return {**tree, INFOGAIN: leaves}
