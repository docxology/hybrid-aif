"""A frozen two-tier survey agent for the library's own tests: the rover (tiered, no camera) and
its scheduler with the declarations and tables the recorded runs of ``tests/data/clock_parity.npz``
used, every table read from ``tests/data/survey_frozen_tables.npz``.

The showcase example is free to change its model and tuning; the tests that replay runs recorded
with this agent (``tests/test_clock_parity.py``, ``tests/test_run_nested.py``) build it from here.

    rover_system, rover_params, grid = rover()
    mission_system, mission_params = scheduler(grid, policy_len=rover_system.policy_len)
"""

import pathlib
import types

import jax.numpy as jnp
import numpy as np

from hybrid_aif import (
    Categorical,
    ConditionalCategorical,
    ConditionalDirichlet,
    ConditionalGaussian,
    Grid,
    LinearGaussianObservation,
    Parameter,
    System,
    SystemParams,
)
from hybrid_aif.controls import Box
from hybrid_aif.distributions import MixtureGaussian
from hybrid_aif.factors import GridNoiseMap, LinearGaussianTransition

TABLES = np.load(pathlib.Path(__file__).parent / "data" / "survey_frozen_tables.npz")
GRID_SHAPE = 4
N_TERRAIN = 3
N_BANDS = 4
N_SURVEY = 4
N_OPTIONS = N_SURVEY + 1
SECTOR_SD = 1.0
CHARGE_SD = 100.0
FIELD_STRENGTH = 0.1
MIXTURE_K = 3
SURVEY_TICKS = 12
CONSTANTS = types.SimpleNamespace(SURVEY_TICKS=SURVEY_TICKS)


def _field():
    (x0, x1), (y0, y1) = (tuple(float(v) for v in r) for r in TABLES["field"])
    return (x0, x1), (y0, y1)


def field_grid():
    """The rover's ``Grid`` over the position, ``GRID_SHAPE`` cells per axis."""
    (x0, x1), (y0, y1) = _field()
    return Grid(shape=(GRID_SHAPE, GRID_SHAPE), low=(x0, y0), high=(x1, y1), axes=(0, 1))


def _fix_noise_map():
    (x0, x1), (y0, y1) = _field()
    return GridNoiseMap(jnp.asarray(TABLES["fix_noise_values"]), (x0, y0), (x1, y1), axes=(0, 1))


def _option_position_target():
    loc = np.zeros((N_OPTIONS, 4), np.float32)
    loc[:N_SURVEY, :2] = TABLES["sector_centres"]
    chol = np.zeros((N_OPTIONS, 4, 4), np.float32)
    for k in range(N_OPTIONS):
        sd = [SECTOR_SD, SECTOR_SD, 2.0, 2.0] if k < N_SURVEY else [CHARGE_SD] * 4
        chol[k] = np.diag(sd)
    return ConditionalGaussian(loc=jnp.asarray(loc), chol=jnp.asarray(chol), parent_ndims=1)


def _cc(table, parent_ndims):
    return ConditionalCategorical(
        log_table=jnp.log(jnp.clip(jnp.asarray(table, jnp.float32), 1e-6)),
        parent_ndims=parent_ndims,
    )


def rover():
    """``(System, SystemParams, Grid)``: the tiered rover without a camera. Factors terrain,
    ``x`` (position and velocity, switched by the terrain), energy, the velocity command ``v``
    and the option control ``u_opt``; modalities ``fix`` and ``gauge``."""
    grid = field_grid()
    n_cells = grid.n_cells
    system = System(
        A_deps=((1,), (2,)),
        B_deps=((1,), (0, 3), (2, 0, 1), (), ()),
        controls=(3, 4),
        action_names=("a_v", "a_opt"),
        factor_names=("terrain", "x", "energy", "v", "u_opt"),
        modality_names=("fix", "gauge"),
        cells={"x": grid},
        C_deps={"x": ("u_opt",)},
        policy_len=3,
        mixture_k=MIXTURE_K,
        mixture_schedule="mode_aligned",
        planning_belief="prior",
        epistemic=("transition:terrain", "obs:fix"),
        terms=(),
    )
    start = [float(v) for v in TABLES["start"]]
    x0 = MixtureGaussian(
        log_weights=jnp.log(jnp.ones(MIXTURE_K) / MIXTURE_K),
        means=jnp.broadcast_to(jnp.asarray([*start, 0.0, 0.0], jnp.float32), (MIXTURE_K, 4)),
        chols=jnp.broadcast_to(0.3 * jnp.eye(4), (MIXTURE_K, 4, 4)),
    )
    uniform = jnp.full((N_TERRAIN, n_cells), 1.0 / N_TERRAIN)
    energy0 = jnp.log(jnp.asarray([1e-6, 1e-6, 1e-6, 1.0]))
    params = SystemParams(
        parameters={"field": ConditionalDirichlet.from_table(uniform, 1, strength=FIELD_STRENGTH)},
        likelihood={
            "fix": LinearGaussianObservation(
                H=jnp.eye(4)[:2], c=jnp.zeros(2), noise=_fix_noise_map()
            ),
            "gauge": ConditionalCategorical(
                log_table=jnp.log(jnp.asarray(TABLES["gauge"], jnp.float32)), parent_ndims=1
            ),
        },
        transition={
            "terrain": Parameter("field"),
            "x": LinearGaussianTransition(
                F=jnp.asarray(TABLES["F"], jnp.float32),
                G=jnp.asarray(TABLES["G"], jnp.float32),
                innovations=ConditionalGaussian(
                    loc=jnp.asarray(TABLES["loc"], jnp.float32),
                    chol=jnp.asarray(TABLES["chol"], jnp.float32),
                    parent_ndims=1,
                ),
            ),
            "energy": ConditionalCategorical(
                log_table=jnp.log(jnp.clip(jnp.asarray(TABLES["rover_energy"]), 1e-6)),
                parent_ndims=3,
            ),
        },
        initial_state_prior={
            "terrain": Categorical(logits=jnp.zeros(N_TERRAIN)),
            "x": x0,
            "energy": Categorical(logits=energy0),
        },
        control={"v": Box(-jnp.ones(2), jnp.ones(2))},
        state_targets={
            "energy": Categorical(logits=jnp.asarray(TABLES["energy_target"], jnp.float32)),
            "x": _option_position_target(),
        },
        obs_targets={},
    )
    return system, params, grid


def scheduler(grid, policy_len=2):
    """``(System, SystemParams)``: the scheduler over the rover's ``grid``, its option control
    ``u_option`` (four surveys and charge) declaring ``SURVEY_TICKS`` rounds for every level.
    Factors visit, energy and map (which reads the parameter variable ``field``); modalities
    ``cell_obs`` and ``gauge``."""
    n_cells = grid.n_cells
    system = System(
        A_deps=((0,), (1,)),
        B_deps=((0, 3), (1, 0, 3), (0,), ()),
        controls=(3,),
        action_names=("option",),
        factor_names=("visit", "energy", "map", "u_option"),
        modality_names=("cell_obs", "gauge"),
        policy_len=policy_len,
        planner="enumeration",
        mixture_k=MIXTURE_K,
        mixture_schedule="mode_aligned",
        planning_belief="prior",
        epistemic=("transition:map",),
        terms=(),
    )
    uniform = jnp.full((N_TERRAIN, n_cells), 1.0 / N_TERRAIN)
    centres = np.asarray(TABLES["centres"])
    start = int(np.argmin(np.linalg.norm(centres - np.asarray(TABLES["start"])[None], axis=1)))
    params = SystemParams(
        parameters={"field": ConditionalDirichlet.from_table(uniform, 1, strength=FIELD_STRENGTH)},
        likelihood={
            "cell_obs": _cc(TABLES["cell_obs"], 1),
            "gauge": _cc(TABLES["gauge"], 1),
        },
        transition={
            "visit": _cc(TABLES["visit"], 2),
            "energy": _cc(TABLES["scheduler_energy"], 3),
            "map": Parameter("field"),
        },
        initial_state_prior={
            "visit": Categorical(logits=jnp.log(jnp.clip(jnp.eye(n_cells)[start], 1e-6))),
            "energy": Categorical(logits=jnp.log(jnp.clip(jnp.eye(N_BANDS)[-1], 1e-6))),
            "map": Categorical(logits=jnp.zeros(N_TERRAIN)),
        },
        durations={"u_option": jnp.full(N_OPTIONS, float(SURVEY_TICKS))},
        state_targets={
            "energy": Categorical(logits=jnp.asarray(TABLES["energy_target"], jnp.float32))
        },
    )
    return system, params
