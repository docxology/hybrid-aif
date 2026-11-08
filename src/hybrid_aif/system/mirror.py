"""The mirror of a System: its generative model as a world that acts.

:func:`mirror` takes a System that perceives (modalities in, control nodes it chooses) and
returns the System that generated what it perceives: each modality becomes an **emission** of the
same name over the same parents, with the likelihood as its kernel, so the world's outputs are the
agent's inputs; the state factors, their transitions and priors are kept; the control nodes stay,
now the world's inputs, driven by what an agent sends or free and drawn from their habit. It is
the transform behind :meth:`System.as_env <hybrid_aif.system.System.as_env>`, which runs the
mirror forward and keys the environment's faces by the original System's names.

What is kept, dropped and refused:

- **kept**: the state factors, ``B_deps``, ``D_deps``, ``dims`` and ``cells``, the transitions and
  initial priors, the control nodes with their action nodes, ``params.control``,
  ``params.durations`` and ``params.policy_prior`` (a free node's habit), and the parameter
  variables (``params.parameters`` and the references to them) exactly as they are;
- **converted**: every modality becomes an emission, an action-node parent becoming the control
  node that action node realises; the kernels are the likelihoods, a reference to a parameter
  variable staying a reference;
- **dropped**: everything that belongs to the planner, the learner or the posterior: targets,
  ``C_deps``, ``H_deps``, ``epistemic``, ``terms``, ``scoring``, ``branch_on``, ``planning_belief``,
  ``feedback``, ``readings``, ``lagged`` reads (a read is a datum for inference, and a world
  samples such a modality from its likelihood), ``reduction_*``, ``recognition``,
  ``params.actuator`` (the receiving side is exact: an input is the level taken; an environment
  built by ``as_env`` applies the actuator at its own face) and ``subsystems``;
- **lowered on request**: ``lower=True`` replaces every learnable declaration and parameter
  variable with its posterior mean, or with one draw under ``param_key``, the draws taken in the
  order likelihood, transition, initial prior (the emissions in the likelihood's place), one per
  parameter variable shared by its readers. The mirror itself keeps them, since a world that is
  described or priced is the model;
- **refused**: a System that declares emissions (an agent's own outputs are not the world's), a
  System with cells a tabular factor reads (the forward process reads a table's parent as a
  level), and an amortized System (running its declarations forward is imagination, not a
  world).

Key use of the world sampled forward (``as_env`` and the System's own sampling steps), by the
forward process of :mod:`hybrid_aif.envs._process`. ``reset(key)``: ``keys = split(key, F + 1)``;
state factor ``f`` draws with ``keys[1 + f]`` in factor order, a conditional prior reading its
parents' draws; emission ``e`` draws with ``split(keys[0], E)[e]``, from the drawn state and the
null level of every control node an emission reads. ``step(key, state, action)``: when
``params.actuator`` is not empty the env face first draws action node ``k`` with
``split(fold_in(key, ACTUATOR_TAG), A)[k]``; then ``k_obs, k_state = split(key)``, state factor
``f`` draws with ``split(k_state, F)[f]`` (a factor outside ``factors=`` keeps its value, the split
unchanged; a hybrid linear-Gaussian factor reads the discrete parents drawn this step) and
emission ``e`` with ``split(k_obs, E)[e]`` from the stepped state and the realised level of every
control node an emission reads. A control node that is free (no input drives it) is drawn
before all of these with ``split(fold_in(key, CONTROL_TAG), K)[k]``, so the process's own key
stream is the same as when every node is driven.
"""

import jax
import jax.numpy as jnp
from jax import Array

from hybrid_aif.controls import FLOAT
from hybrid_aif.distributions import Categorical
from hybrid_aif.factors import Parameter
from hybrid_aif.planning.enumerate import enumerate_policies
from hybrid_aif.reads import Emit
from hybrid_aif.system import keyed, validation
from hybrid_aif.system.core import System, SystemParams

__all__ = ["mirror"]


def mirror(
    system: System,
    params: SystemParams,
    *,
    lower: bool = False,
    param_key: Array | None = None,
    face: str = "mirror",
) -> tuple[System, SystemParams]:
    """The world ``system`` models, as a System that acts: ``(mirror System, its params)``.

    Every modality becomes an emission of the same name (parents: the modality's, an action-node
    parent being the control node it realises; kernel: the likelihood; realised by ``sample``),
    and the rest of the generative model is kept (see the module docstring for what is kept,
    dropped and refused). ``lower=True`` replaces every learnable declaration and parameter
    variable with its posterior mean, or with one draw per slot under ``param_key``;
    ``param_key`` without ``lower`` is refused. ``face`` names the face a refusal speaks for.
    """
    if param_key is not None and not lower:
        raise ValueError(
            f"{face}: param_key draws the learnable declarations' parameters, so it needs "
            "lower=True; the mirror itself keeps them"
        )
    keyed_params = params
    params = system._params_in(params, face)
    if system.emissions:
        raise ValueError(
            f"{face} on a System that declares emissions {list(system._emission_names)}: an "
            "emission is an output of this System, and the mirror of a model is the world that "
            "generated what the model perceives, whose outputs are the modalities; mirror the "
            "System before it declares its own outputs, or run it forward as it is"
        )
    if system.cells:
        system._validate(params, face)
    if system._cell_parents():
        raise NotImplementedError(
            f"{face} on a System with cells: the forward process reads a table's parents as "
            "discrete levels and would mis-read the cell a continuous factor falls in; "
            "simulate the world with your own process, or drop cells= for the simulation"
        )
    if system._is_amortized(params):
        system._amortized_structure(params, face)
        raise NotImplementedError(
            f"{face} on an amortized (neural state-space) System: running these declarations "
            "forward as an environment means sampling the code prior and decoding its heads, "
            "which is imagination, not an environment (nothing in it emits a reward the "
            "agent has not predicted, so it cannot score the agent, only agree with it, and "
            "every as_env consumer, from test oracles to the composition driver's ground "
            "truth, assumes scoring). The rollout face for it is "
            "hybrid_aif.train.imagine.imagine over predict_beliefs (design: "
            "amortization.md A6/A11); predict_beliefs itself is the open-loop "
            "prediction, run against a concrete environment of the caller's. "
            "The one coherent unfence, recorded for whoever wants "
            "it: a FROZEN world model stipulated as a synthetic "
            "environment is just a declared generative process, and could be admitted "
            "here on that stipulation; the fence holds while the "
            "model is learning."
        )
    lowered = system._lower_learnable(params, param_key) if lower else params
    if system._is_hybrid(params):
        system._validate(params, face)
    else:
        validation._validate_table_arity(system, params, system._facs)
        if not system._table.default_wiring:
            system._validate(params, face)
    table = system._table
    n_states, n_nodes = table.num_states, len(system.B_deps)
    habit = _first_step_habit(system, keyed_params, lowered.policy_prior, face)

    def parent(g: int) -> str:
        return table.state_names[g] if g < n_states else table.control_names[g - n_nodes]

    emissions = tuple(
        (m, Emit(tuple(parent(g) for g in deps)))
        for m, deps in zip(system.modality_names, system.A_deps, strict=True)
    )
    world = System(
        B_deps=system.B_deps,
        A_deps=(),
        D_deps=system.D_deps,
        controls=system.controls,
        factor_names=system.factor_names,
        action_names=system.action_names,
        modality_names=(),
        emissions=emissions,
        dims=system.dims,
        cells=system.cells,
        mixture_k=system.mixture_k,
        mixture_schedule=system.mixture_schedule,
        policy_len=1,
        epistemic=False,
    )
    kernels = dict(zip(system.modality_names, lowered.likelihood, strict=True))
    parameters, references = lowered.parameters, ()
    if lower:
        parameters = {}
    else:
        read = {i: home for slot, i, home in lowered.references if slot == "likelihood"}
        for i, m in enumerate(system.modality_names):
            if i in read:
                kernels[m] = Parameter(read[i])
        references = tuple(r for r in lowered.references if r[0] == "transition")
    mirrored = SystemParams(
        transition=lowered.transition,
        initial_state_prior=lowered.initial_state_prior,
        likelihood=(),
        emission=kernels,
        control=lowered.control,
        durations=lowered.durations,
        policy_prior=habit,
        parameters=parameters,
        references=references,
    )
    return world, keyed.params_keyed(world._names(), mirrored)


def _first_step_habit(system: System, params: SystemParams, habit, face: str):
    """The habit a free control node of the world draws from: ``policy_prior`` over the
    original System's policy set, as a log-space distribution over the level tuples of the
    mirror's own enumerated set (one step, every combination of levels, in
    :func:`~hybrid_aif.planning.enumerate.enumerate_policies` order), each tuple holding the
    mass of the first steps equal to it. ``None`` stays ``None``; a habit that cannot be so
    converted is refused."""
    if habit is None:
        return None
    if not (isinstance(habit, Categorical) or hasattr(habit, "shape")):
        raise NotImplementedError(
            f"{face}: policy_prior is a policy module, which reads beliefs a world does not hold, "
            "so a free control node of the mirror could not draw from it; declare the habit as a "
            "distribution over the enumerated policies, or drop it"
        )
    if (
        system._is_amortized(system._params_in(params, face))
        or FLOAT in system._space(system._params_in(params, face)).kinds
    ):
        raise NotImplementedError(
            f"{face}: policy_prior is a distribution over an enumerated policy set, and this "
            "System has continuous control nodes (or is amortized), which have none, so the "
            "habit cannot be converted to the first-step level tuples a free node draws; "
            "drop the habit"
        )
    logits = habit.logits if isinstance(habit, Categorical) else habit
    first = system.policies(params)[:, 0, :]
    if jnp.shape(logits) != (first.shape[0],):
        raise ValueError(
            f"{face}: policy_prior has shape {tuple(jnp.shape(logits))} and the System enumerates "
            f"{first.shape[0]} policies; declare one entry per policy"
        )
    levels = system._control_levels(system._params_in(params, face))
    tuples = enumerate_policies(levels, 1, (True,) * len(levels))[:, 0, :]
    same = jnp.all(first[:, None, :] == tuples[None, :, :], axis=-1)
    mass = jax.nn.softmax(logits) @ same.astype(jnp.float32)
    return jnp.log(jnp.maximum(mass, 1e-30))
