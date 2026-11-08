"""The survey rover's slow tier: a scheduler that picks which sector of the field to survey, or
to charge, every dozen fast ticks.

Factors, in declaration order: ``visit`` (the cell the rover is at, one level per cell of the
rover's grid, row-major over the grid shape, x first), ``energy`` (the four bands) and ``map`` (the
terrain at the visited cell, three levels). The control node ``u_option`` has five levels: survey
sector 1 to 4 (the field's quadrants, in ``survey_world.zone_of`` order) and charge. Two
modalities: ``cell_obs`` (the cell the rover's position belief puts it in: in the merged
composite, a sibling-belief read of the rover's position through the cell indicator, realised by
``mode`` and taken at each decision) and ``gauge`` (the energy band). The ``map`` factor reads the
parameter variable ``field``, which an ``Identify`` identifies with the rover's; the
scheduler has no observation of ``map`` and never counts into the field, so its novelty term
(``transition:map``, the expected information about the field from the cells its plan visits) is
honoured by the rover counting on the fast clock. The ``map`` is read where the rover stood one
slow step earlier, so the planner looks two options ahead (``policy_len=2``) and enumerates the
option sequences.

Tables, all built here from the world's maps:

- ``visit[next, cell, option]``: under survey sector k uniform over the cells whose centre lies in
  sector k, whatever the previous cell; under charge the identity (the rover stays where it is),
  so charging promises no information that the cell's counts do not already hold. Its size is
  cells x cells x 5 entries, a known table priced by the description length.
- ``energy[next, band, cell, option]``: the expected change in energy units over one option, as
  a distribution over whole-band shifts that has the fractional part as the probability of the
  larger shift (the band moves by ``floor`` or ``ceil`` of the change), clipped at the empty and
  full bands. Under survey sector k the change is the travel from the visited cell's centre to
  the sector's centre (a straight path sampled at ``PATH_SAMPLES`` points, at cruise speed
  ``CRUISE``, the net of the world's charge and drain at firm ground, so a longer path costs
  more) plus ``SURVEY_TICKS`` fast ticks of surveying (the mean net rate over the sector's cells
  at cruise speed). Under charge it is ``SURVEY_TICKS`` fast ticks at rest, the world's charge at
  the field-wide mean illumination (the scheduler does not know where light is) less the drain
  at rest. The distance enters only through this energy cost.
- ``cell_obs`` is right with probability ``CELL_ACC``, the rest spread uniformly over the other
  cells. This table says how the rover's cell belief relates to the scheduler's own ``visit``
  variable: the scheduler's belief about where the rover is, a different variable at a different
  level, so the reading is an ordinary likelihood on the modality. ``gauge`` is the rover
  model's confusion.
- The initial beliefs: the start cell and the full band, uniform terrain.

The objective scores the novelty of the field and an energy target with the same logits as the
rover's.
"""

import jax
import jax.numpy as jnp
import numpy as np
import survey_model as sm
import survey_world as wd

from hybrid_aif import (
    Categorical,
    ConditionalCategorical,
    ConditionalDirichlet,
    Parameter,
    System,
    SystemParams,
)

SURVEY_TICKS = 12  # fast ticks an option lasts
CRUISE = 0.8  # the speed at which travel and survey are costed
PATH_SAMPLES = 24
CELL_ACC = 0.95  # the cell reading is right with this probability
OPTION_NAMES = [
    "survey sector 1",
    "survey sector 2",
    "survey sector 3",
    "survey sector 4",
    "charge",
]
SECTOR_NAMES = ["sector 1", "sector 2", "sector 3", "sector 4"]
CHARGE = sm.N_SURVEY  # the charge level
# readings of the map per option: a survey pools one reading of a distinct cell per fast tick, a
# charge leaves the rover on one cell, so its twelve ticks read it once
READINGS = jnp.asarray([SURVEY_TICKS] * sm.N_SURVEY + [1], jnp.float32)


def cell_zones(centres):
    """The sector of each cell, by its centre, ``(cells,)``."""
    return np.asarray(jax.vmap(wd.zone_of)(jnp.asarray(centres)))


def shift_distribution(change, n_bands=sm.N_BANDS):
    """``(n_bands, n_bands)`` table ``[next, band]`` for an expected change in energy units: the
    band moves by ``floor(change)`` or one more with the fractional part as probability, clipped
    to the bands."""
    low = int(np.floor(change))
    frac = float(change - low)
    table = np.zeros((n_bands, n_bands))
    for band in range(n_bands):
        for shift, p in ((low, 1.0 - frac), (low + 1, frac)):
            table[int(np.clip(band + shift, 0, n_bands - 1)), band] += p
    return table


def _net_rate(points, speed):
    """The net energy per tick (charge less drain, firm ground) at positions ``(..., 2)``, the
    mean over the points."""
    pts = jnp.asarray(points, jnp.float32).reshape(-1, 2)
    drain, charge = jax.vmap(lambda xy: wd.energy_rates(xy, 0, speed))(pts)
    return float(jnp.mean(charge - drain))


def option_changes(centres):
    """Expected energy change over one option, ``(cell, option)``, in energy units."""
    zones = cell_zones(centres)
    targets = sm.sector_centres()
    sector_rate = [_net_rate(centres[zones == k], CRUISE) for k in range(sm.N_SURVEY)]
    mean_light = float(np.mean(wd.LIGHT))
    rest = SURVEY_TICKS * (wd.CHARGE_RATE * mean_light - wd.BASE_DRAIN)
    out = np.zeros((centres.shape[0], sm.N_OPTIONS))
    ramp = np.linspace(0.0, 1.0, PATH_SAMPLES)[:, None]
    for c, here in enumerate(centres):
        for k in range(sm.N_SURVEY):
            path = here + ramp * (targets[k] - here)
            travel = float(np.linalg.norm(targets[k] - here)) / CRUISE
            out[c, k] = travel * _net_rate(path, CRUISE) + SURVEY_TICKS * sector_rate[k]
        out[c, CHARGE] = rest
    return out


def visit_table(centres):
    """``visit[next, cell, option]``."""
    zones = cell_zones(centres)
    n = centres.shape[0]
    table = np.zeros((n, n, sm.N_OPTIONS))
    for k in range(sm.N_SURVEY):
        table[zones == k, :, k] = 1.0 / np.sum(zones == k)
    table[:, :, CHARGE] = np.eye(n)
    return table


def energy_table(centres):
    """``energy[next, band, cell, option]``."""
    change = option_changes(centres)
    n = centres.shape[0]
    table = np.zeros((sm.N_BANDS, sm.N_BANDS, n, sm.N_OPTIONS))
    for c in range(n):
        for k in range(sm.N_OPTIONS):
            table[:, :, c, k] = shift_distribution(change[c, k])
    return table


def cell_obs_table(n):
    """``cell_obs[reading, cell]``: right with ``CELL_ACC``, the rest uniform."""
    return CELL_ACC * np.eye(n) + (1 - CELL_ACC) / (n - 1) * (1 - np.eye(n))


def _cc(table, parent_ndims):
    return ConditionalCategorical(
        log_table=jnp.log(jnp.clip(jnp.asarray(table, jnp.float32), 1e-6)),
        parent_ndims=parent_ndims,
    )


def scheduler(
    grid,
    centres,
    energy_target=sm.ENERGY_TARGET,
    field_strength=sm.FIELD_STRENGTH,
    epistemic=("transition:map",),
    policy_len=2,
    durations=False,
    readings=False,
):
    """``(System, SystemParams)`` for the scheduler over ``grid`` (the rover's ``Grid``) with the
    cell centres ``centres``. Its parameter variable ``field`` has the shape and prior of the
    rover's, so that ``Identify("rover.field", "mission.field")`` identifies them. With
    ``durations`` the control node declares a round of ``SURVEY_TICKS`` for every level; with
    ``readings`` the ``map`` factor's novelty at each plan step is multiplied by the number of
    readings of the option selected at that step (``SURVEY_TICKS`` for a survey, one for charge,
    since a stationary rover reads one cell), a slow step pooling the readings the fast tier
    takes in it."""
    n_cells = grid.n_cells
    # factors: visit 0, energy 1, map 2, u_option 3 (the option, a control node)
    system = System(
        A_deps=((0,), (1,)),
        B_deps=((0, 3), (1, 0, 3), (0,), ()),
        controls=(3,),
        action_names=("option",),
        factor_names=("visit", "energy", "map", "u_option"),
        modality_names=("cell_obs", "gauge"),
        policy_len=policy_len,
        planner="enumeration",
        mixture_k=sm.MIXTURE_K,
        mixture_schedule="mode_aligned",
        planning_belief="prior",
        epistemic=tuple(epistemic) if epistemic else False,
        terms=(),
        **({"readings": (("map", "u_option"),)} if readings else {}),
    )
    uniform = jnp.full((sm.N_TERRAIN, n_cells), 1.0 / sm.N_TERRAIN)
    start = int(np.argmin(np.linalg.norm(np.asarray(centres) - np.asarray(wd.START)[None], axis=1)))
    params = SystemParams(
        parameters={"field": ConditionalDirichlet.from_table(uniform, 1, strength=field_strength)},
        likelihood={"cell_obs": _cc(cell_obs_table(n_cells), 1), "gauge": _cc(sm.gauge_table(), 1)},
        transition={
            "visit": _cc(visit_table(centres), 2),
            "energy": _cc(energy_table(centres), 3),
            "map": Parameter("field"),
        },
        initial_state_prior={
            "visit": Categorical(logits=jnp.log(jnp.clip(jnp.eye(n_cells)[start], 1e-6))),
            "energy": Categorical(logits=jnp.log(jnp.clip(jnp.eye(sm.N_BANDS)[-1], 1e-6))),
            "map": Categorical(logits=jnp.zeros(sm.N_TERRAIN)),
        },
        durations={"u_option": jnp.full(sm.N_OPTIONS, float(SURVEY_TICKS))}
        if (durations or readings)
        else {},
        **({"readings": {"map": READINGS}} if readings else {}),
        state_targets={"energy": Categorical(logits=jnp.asarray(energy_target, jnp.float32))},
    )
    return system, params
