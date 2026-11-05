"""Retrace diagnostics: a watcher for ``eqx.filter_jit`` functions and a nudge for eager loops.

:func:`watch` wraps an ``eqx.filter_jit`` function, counts its traces, and on the second and
later trace emits a :class:`RetraceWarning` that names what changed since the previous trace:
the first argument path whose pytree structure, shape, dtype or weak type differs, a static
(non-array) argument whose value changed, or a static field of an argument that is an
``equinox.Module`` (a ``System``'s ``gamma``, ``planner``, ...). A trace is a full recompile, so a
repeat trace with nothing intended to change is the cost the warning reports.
``jax_explain_cache_misses`` names the argument under ``jax.jit`` but its paths are opaque under
``eqx.filter_jit``.

:meth:`hybrid_aif.system.System.compiled` returns a compiled System with the watcher on by
default. :func:`note_eager_call` backs the nudge that fires once per System object after one of
its faces has run eagerly more than ``EAGER_CALL_LIMIT`` times.

Design: docs/design/recompilation.md R6, R7.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import threading
import warnings
import weakref
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax

try:  # a private spelling; without it every call counts as eager
    from jax._src.core import trace_state_clean as _trace_state_clean
except ImportError:  # pragma: no cover

    def _trace_state_clean() -> bool:
        return True


_TRACE_EVENT = "/jax/core/compile/jaxpr_trace_duration"
_trace_events = 0
_monitored = False


def _on_duration(event: str, duration: float, **kwargs: Any) -> None:
    global _trace_events
    if event == _TRACE_EVENT:
        _trace_events += 1


try:  # a jaxpr trace is announced to jax.monitoring listeners; without it, read the cache
    jax.monitoring.register_event_duration_secs_listener(_on_duration)
    _monitored = True
except Exception:  # noqa: BLE001  pragma: no cover
    pass


__all__ = ["EAGER_CALL_LIMIT", "RetraceWarning", "Watched", "note_eager_call", "watch"]

#: An eager face called more often than this on one System object triggers the nudge.
EAGER_CALL_LIMIT = 10


class RetraceWarning(RuntimeWarning):
    """A compiled function was traced again, or an eager loop retraces every call."""


# ---------------------------------------------------------------------------- the watcher


def _describe(leaf: Any) -> tuple:
    if eqx.is_array(leaf):
        weak = bool(getattr(leaf, "weak_type", False))
        return ("array", tuple(leaf.shape), str(leaf.dtype), weak)
    return ("static", leaf)


def _leaf_text(desc: tuple) -> str:
    if desc[0] == "array":
        _, shape, dtype, weak = desc
        return f"{dtype}{'(weak)' if weak else ''}{list(shape)}"
    return f"static {desc[1]!r}"


def _same(a: Any, b: Any) -> bool:
    if a is b:
        return True
    try:
        return bool(a == b)
    except Exception:  # noqa: BLE001  comparison of arrays or odd objects: not the same
        return False


def _snapshot(args: dict[str, Any]) -> dict[str, tuple]:
    """Per argument: its treedef, its leaf descriptions by path, and a Module's static fields."""
    out = {}
    for name, value in args.items():
        flat, treedef = jax.tree_util.tree_flatten_with_path(value)
        leaves = {name + jax.tree_util.keystr(path): _describe(leaf) for path, leaf in flat}
        statics = {}
        if isinstance(value, eqx.Module):
            statics = {
                f.name: getattr(value, f.name)
                for f in dataclasses.fields(value)
                if f.metadata.get("static")
            }
        out[name] = (treedef, leaves, statics, value is None)
    return out


def _explain(before: dict[str, tuple], after: dict[str, tuple]) -> str:
    """The first difference between two snapshots, as a sentence fragment."""
    for name in after:
        if name not in before:
            return f"argument {name!r} is new"
        b_tree, b_leaves, b_statics, b_none = before[name]
        a_tree, a_leaves, a_statics, a_none = after[name]
        for field, value in a_statics.items():
            if field in b_statics and not _same(b_statics[field], value):
                what = "its identity" if callable(value) else f"{b_statics[field]!r} to {value!r}"
                return (
                    f"static field {field!r} of {name!r} changed ({what}); each value of a "
                    "static field is a different compiled program"
                )
        if b_none != a_none:
            now = "None" if a_none else "a value"
            was = "None" if b_none else "a value"
            return f"{name!r} was {was} and is now {now}; the pytree structure differs"
        if b_tree != a_tree:
            new = [p for p in a_leaves if p not in b_leaves]
            gone = [p for p in b_leaves if p not in a_leaves]
            if new:
                return f"the pytree structure of {name!r} differs: {new[0]} is new"
            if gone:
                return f"the pytree structure of {name!r} differs: {gone[0]} is gone"
            return f"the pytree structure of {name!r} differs"
        for path, desc in a_leaves.items():
            old = b_leaves.get(path)
            if old is None or old == desc:
                continue
            if old[0] == "static" and desc[0] == "static":
                if _same(old[1], desc[1]):
                    continue
                return (
                    f"{path} is a static (non-array) value that changed from {old[1]!r} to "
                    f"{desc[1]!r}; pass it as an array to trace it"
                )
            return f"{path} changed from {_leaf_text(old)} to {_leaf_text(desc)}"
    return (
        "no difference from the previous trace was found in the arguments: the compilation "
        "cache was cleared (jax.clear_caches()) or the program evicted, or a static argument "
        "or field changed in a way this check does not name (a callable such as a planner "
        "compares by identity)"
    )


class Watched:
    """A wrapped ``eqx.filter_jit`` function that counts and explains its traces.

    ``traces`` is the number of calls that traced so far; ``fn`` the wrapped function. A trace is
    a jaxpr trace announced to ``jax.monitoring`` during the call (a call that only dispatches
    to a cached program, even with a new argument signature such as NumPy for JAX arrays, is not
    one), or, with ``clock`` given, a rise of that counter. Without ``jax.monitoring`` the
    function's compilation-cache size stands in. ``store``, a dict shared between watchers,
    holds the last traced signature per ``key``, so watchers of one compiled program share what
    they compare against.
    """

    def __init__(
        self,
        fn: Callable,
        *,
        name: str | None = None,
        argnames: tuple[str, ...] | None = None,
        warn: bool = True,
        clock: Callable[[], int] | None = None,
        store: dict | None = None,
        key: Any = None,
    ):
        self.fn = fn
        self.name = name or getattr(fn, "__name__", type(fn).__name__)
        self.warn = warn
        self.traces = 0
        self._argnames = argnames
        self._clock = clock
        self._store = {} if store is None else store
        self._key = key
        self._seen: set = set()

    def _cache_size(self) -> int | None:
        cached = getattr(self.fn, "_cached", None)
        size = getattr(cached, "_cache_size", None)
        return None if size is None else size()

    def _bind(self, args: tuple, kwargs: dict) -> dict[str, Any]:
        names = self._argnames
        if names is None:
            try:
                bound = inspect.signature(self.fn).bind_partial(*args, **kwargs)
                return dict(bound.arguments)
            except (TypeError, ValueError):
                names = ()
        named = {n: a for n, a in zip(names, args, strict=False)}
        named.update({f"arg{i}": a for i, a in enumerate(args) if i >= len(names)})
        named.update(kwargs)
        return named

    def __call__(self, *args, **kwargs):
        if self._clock is not None:
            mark = self._clock
            before = mark()
        elif _monitored:
            mark = lambda: _trace_events  # noqa: E731
            before = mark()
        else:
            mark = self._cache_size
            before = mark()
        out = self.fn(*args, **kwargs)
        snap = None
        if before is None:  # nothing countable: a new call signature stands in for a trace
            snap = _snapshot(self._bind(args, kwargs))
            sig = repr(
                [
                    (n, str(s[0]), tuple(sorted((p, repr(d)) for p, d in s[1].items())))
                    for n, s in snap.items()
                ]
            )
            traced = sig not in self._seen
            self._seen.add(sig)
        else:
            traced = mark() > before
        if traced:
            self.traces += 1
            snap = snap or _snapshot(self._bind(args, kwargs))
            previous = self._store.get(self._key)
            if previous is not None and self.warn:
                warnings.warn(
                    f"{self.name} was traced again: {_explain(previous, snap)}. "
                    "Every trace is a full recompile.",
                    RetraceWarning,
                    stacklevel=2,
                )
            self._store[self._key] = snap
        return out


def watch(
    fn: Callable,
    *,
    name: str | None = None,
    argnames: tuple[str, ...] | None = None,
    warn: bool = True,
    clock: Callable[[], int] | None = None,
    store: dict | None = None,
    key: Any = None,
) -> Watched:
    """Wrap an ``eqx.filter_jit`` function so a repeat trace names what changed.

    The wrapper takes the arguments of ``fn``. On the second and later trace it emits a
    :class:`RetraceWarning` naming the first argument path whose structure, shape, dtype or weak
    type differs from the previous trace, or the static value or static field that changed.
    ``Watched.traces`` counts the traces. ``argnames`` names positional arguments when ``fn``'s
    signature cannot be read; ``warn=False`` counts without warning. ``clock``, a counter that
    rises once per trace of ``fn``, replaces the ``jax.monitoring`` detection, and ``store`` with
    ``key`` shares the last traced signature between watchers of one program.
    """
    return Watched(fn, name=name, argnames=argnames, warn=warn, clock=clock, store=store, key=key)


# ----------------------------------------------------------------------- the eager nudge

_state = threading.local()
_calls: dict[int, tuple[weakref.ref, dict[str, int], list[bool]]] = {}


def note_eager_call(system: Any, face: str) -> None:
    """Count one eager call of ``face`` on ``system``; warn once per System past the limit."""
    key = id(system)
    entry = _calls.get(key)
    if entry is None or entry[0]() is not system:
        try:
            ref = weakref.ref(system, lambda _, k=key: _calls.pop(k, None))
        except TypeError:  # pragma: no cover  an object that cannot be weakly referenced
            return
        entry = (ref, {}, [False])
        _calls[key] = entry
    _, counts, warned = entry
    counts[face] = counts.get(face, 0) + 1
    if counts[face] > EAGER_CALL_LIMIT and not warned[0]:
        warned[0] = True
        warnings.warn(
            f"System.{face} has run eagerly more than {EAGER_CALL_LIMIT} times on this System: "
            "an eager call re-traces and dispatches operation by operation every time, which a "
            "per-tick loop pays on each tick and which can be orders of magnitude slower than "
            "a compiled call. System.compiled() returns the same faces compiled once.",
            RetraceWarning,
            stacklevel=3,
        )


def eager_face(fn: Callable) -> Callable:
    """Decorate a System face so an outermost eager call is counted by :func:`note_eager_call`."""
    face = fn.__name__

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        depth = getattr(_state, "depth", 0)
        if depth == 0 and _trace_state_clean():
            note_eager_call(self, face)
        _state.depth = depth + 1
        try:
            return fn(self, *args, **kwargs)
        finally:
            _state.depth = depth

    return wrapper
