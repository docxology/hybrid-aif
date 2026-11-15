"""The survey rover's slow tier: a scheduler that picks which sector of the field to survey, or
to go and charge in the sun, every ``SURVEY_TICKS`` fast ticks.

State factors, in declaration order: ``visit`` (the cell the rover is at, one level per cell of the
rover's grid, row-major over the grid shape, x first), ``energy`` (the
``N_BANDS`` bands of half a unit, band 0 the empty band) and ``map`` (the
terrain at the visited cell, three levels). The control node ``u_option`` has five levels: survey
sector 1 to 4 (the field's quadrants, in ``survey_world.zone_of`` order) and charge (drive to the
charging place, ``survey_model.charge_site``, the cell of highest solar yield, and park there). Two
modalities: ``cell_obs`` (the cell the rover's position belief puts it in: in the merged
composite, a sibling-belief read of the rover's position through the cell indicator, realised by
``mode`` and taken at each decision) and ``gauge`` (the band meter's reading of the energy band).
The ``map`` state factor reads the parameter variable ``field``, which an ``Identify`` identifies
with the rover's; the scheduler has no observation of ``map`` and never counts into the field, so
its novelty term (``transition:map``, the expected information about the field from the cells its
plan visits) is honoured by the rover counting on the fast clock. The ``energy`` state factor reads
the parameter variable ``energy_table``, a Dirichlet over its transition that the scheduler learns
from its own band readings, decision to decision (``plan(learn=learn_rates(...))``). The planner
enumerates the sequences of ``HORIZON`` options.

The tables:

- ``visit[next, cell, band, option]``: the cells an option's readings fall in, declared from the
  field's geometry and the rover's measured motion. In the empty band the cell itself, whatever the
  option (a stalled rover stays where it is and reads only its own cell). Otherwise, under survey
  sector k: from a cell in sector k, the sector's cells weighted by a Gaussian of sd
  ``SURVEY_SPREAD`` about its centre (the rover's sector target pulls it there and its novelty
  spreads it); from a cell outside, the cells along the straight path to the sector's centre for
  the share of the option the travel takes at ``CRUISE`` and the sector's weights for the rest.
  Under charge: likewise, the path to the charging place for the share of the option the trip
  takes at ``CHARGE_SPEED`` and the charging place's cell for the rest. A far sector leaves fewer
  of the option's readings in the sector.
- ``energy[next, band, option]``: the parameter variable ``energy_table``, one Dirichlet per band
  and option over the next band (``energy_prior``), ``PRIOR_COUNT`` pseudo-counts per case, derived
  from the rover's declared energy rule over one option (``option_shifts``): a survey drives the
  option at ``CRUISE`` on flat ground at the field's median solar yield, a charge parks the option
  on the charging place's yield (the trip ignored); each level shift becomes band probabilities
  under a uniform position within a band (``shift_probabilities``). From the empty band, under
  every option, half on the same band and half on one higher (a stalled rover cannot drive, it can
  only charge where it stands, at a yield it does not know). Every other entry holds
  ``PRIOR_FLOOR``. The scheduler's learning counts each decision's band posterior against the band
  posterior of the decision before, under the option taken then.
- ``cell_obs`` is right with probability ``CELL_ACC``, the rest spread over the other cells in
  proportion to a Gaussian of sd ``CELL_ERROR_SD`` (one cell side) in the distance between cell
  centres: the reading is the mode of the rover's position belief, so its errors are local. This
  table says how the rover's cell belief relates to the scheduler's own ``visit`` variable: the
  scheduler's belief about where the rover is, a different variable at a different level, so the
  reading is an ordinary likelihood on the modality. ``gauge`` is the band meter's
  confusion.
- The initial beliefs: the start cell and the full band, uniform terrain.

The objective scores the novelty of the field (each step's novelty multiplied by the readings the
option pools) and, once at the end of the plan, a terminal target on the energy band
(``terminal_logits``): equal over every band but the empty one, the empty band lower by
``terminal_penalty(HORIZON)``, twice the most novelty a plan of the horizon could earn at the start
of a mission, so that ending a plan stalled is never worth any amount of mapping. Within the plan a
stall costs what the tables say: in the empty band no option moves the rover, so it reads only its
own cell.
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

SURVEY_TICKS = 24  # fast ticks an option lasts
HORIZON = 4  # options the scheduler looks ahead: 625 sequences (a decision under 1 s, measured)
#  measured on earlier two-tier runs (46 survey and 20 charge options outside the empty band): the
#  travel speed toward a sector, the spread of its positions about the sector's centre and the
#  distinct cells it crosses; the travel speed to the charging place and the cells a charge crosses
CRUISE = wd.CRUISE_SPEED  # the rover's measured travel speed (0.17)
SURVEY_SPREAD = 0.66  # the measured spread (sd per axis) of a survey's positions about the centre
CHARGE_SPEED = 0.15  # the measured travel speed to the charging place
PATH_SAMPLES = 24
CELL_ACC = 0.95  # the cell reading is right with this probability
PRIOR_COUNT = 3.0  # pseudo-counts per band and option of the energy table's prior: three readings
PRIOR_FLOOR = 1e-3  # pseudo-counts of every next band the prior gives no share
OPTION_NAMES = [f"survey sector {k + 1}" for k in range(sm.N_SURVEY)] + ["charge"]
SECTOR_NAMES = [f"sector {k + 1}" for k in range(sm.N_SURVEY)]
CHARGE = sm.CHARGE  # the charge level
CELL_SIDE = (wd.FIELD[0][1] - wd.FIELD[0][0]) / sm.GRID_SHAPE
CELL_ERROR_SD = CELL_SIDE  # the reach of the cell reading's errors: one cell side
# readings of the map per option: an option pools one reading per distinct cell it crosses, the
# median count measured over the options of the closed-loop runs (see the build record)
SURVEY_READINGS = 11
CHARGE_READINGS = 6
READINGS = jnp.asarray([SURVEY_READINGS] * sm.N_SURVEY + [CHARGE_READINGS], jnp.float32)


def cell_zones(centres):
    """The sector of each cell, by its centre, ``(cells,)``."""
    return np.asarray(jax.vmap(wd.zone_of)(jnp.asarray(centres)))


def _path_cells(centres, here, there):
    """The cells within half a cell of the straight path from ``here`` to ``there``, a weight per
    cell (zero off the path), ``(cells,)``."""
    ramp = np.linspace(0.0, 1.0, 4 * PATH_SAMPLES)[:, None]
    path = here + ramp * (there - here)
    near = np.abs(centres[:, None, :] - path[None, :, :]).max(-1) <= CELL_SIDE / 2
    return near.any(axis=1).astype(float)


def visit_table(centres):
    """``visit[next, cell, band, option]``."""
    zones = cell_zones(centres)
    targets = sm.sector_centres()
    n = centres.shape[0]
    moving = np.zeros((n, n, sm.N_OPTIONS))
    for k in range(sm.N_SURVEY):
        inside = zones == k
        w = np.exp(-0.5 * np.sum((centres - targets[k]) ** 2, axis=1) / SURVEY_SPREAD**2) * inside
        w = w / w.sum()
        for c in range(n):
            if inside[c]:
                moving[:, c, k] = w
                continue
            travel = float(np.linalg.norm(targets[k] - centres[c])) / CRUISE
            share = min(travel / SURVEY_TICKS, 1.0)
            path = _path_cells(centres, centres[c], targets[k]) * ~inside
            path = path / path.sum() if path.sum() > 0 else w
            moving[:, c, k] = share * path + (1.0 - share) * w
    site_xy = sm.charge_site()
    site = int(np.argmin(np.linalg.norm(centres - site_xy[None], axis=1)))
    for c in range(n):
        travel = float(np.linalg.norm(site_xy - centres[c])) / CHARGE_SPEED
        share = min(travel / SURVEY_TICKS, 1.0)
        path = _path_cells(centres, centres[c], site_xy)
        path[site] = 0.0
        path = path / path.sum() if path.sum() > 0 else np.eye(n)[site]
        moving[:, c, CHARGE] = share * path + (1.0 - share) * np.eye(n)[site]
    table = np.broadcast_to(moving[:, :, None, :], (n, n, sm.N_BANDS, sm.N_OPTIONS)).copy()
    table[:, :, 0, :] = np.eye(n)[:, :, None]  # the stall: the rover stays on its cell
    return table


def learn_rates(prefix=""):
    """The scheduler's learning, ``System.learn`` keywords for ``plan(learn=...)``: the energy
    table at rate 1 from its band beliefs, every other slot at 0 (the field is counted by the
    rover). ``prefix`` is the scheduler's name prefix in the view its step plans on
    (``"agent.mission."`` in the coupled two-tier System)."""
    return {"lr": 0.0, "lr_by_slot": {"*:*": 0.0, f"parameter:{prefix}energy_table": 1.0}}


def option_shifts():
    """The expected change of the energy level over one option, ``(N_OPTIONS,)``, from the rover's
    declared energy rule (``survey_world.net_rate``) times ``SURVEY_TICKS``: a survey drives at
    ``CRUISE`` on flat ground at the field's median solar yield; a charge parks on the charging
    place's yield (``survey_model.cell_yields`` at ``survey_model.charge_site``)."""
    survey = SURVEY_TICKS * wd.net_rate(CRUISE, 0.0, float(np.median(wd.SUN)))
    charge = SURVEY_TICKS * wd.net_rate(0.0, 0.0, float(np.max(sm.cell_yields())))
    return np.asarray([survey] * sm.N_SURVEY + [charge], np.float64)


def shift_probabilities(delta, band, n=sm.N_BANDS, width=wd.BAND_WIDTH):
    """The next band's probabilities, ``(n,)``, from ``band`` under a level shift ``delta``, the
    level uniform within its band of ``width``: with ``k`` whole bands in ``|delta| / width`` and
    the remainder ``r``, ``1 - r`` on ``k`` steps and ``r`` on ``k + 1`` steps in the shift's
    direction (for ``|delta|`` under a band: ``|delta| / width`` one step, the rest the same band);
    a share that would leave the range folded onto the end band."""
    steps = abs(float(delta)) / width
    k = int(np.floor(steps))
    r = steps - k
    sign = 1 if delta >= 0 else -1
    out = np.zeros(n)
    for move, share in ((k, 1.0 - r), (k + 1, r)):
        out[min(max(band + sign * move, 0), n - 1)] += share
    return out


def energy_prior(count=PRIOR_COUNT, floor=PRIOR_FLOOR):
    """The energy table's prior counts ``[next, band, option]``: per band and option ``count``
    pseudo-counts over the next band. From a band but the empty one, the option's declared level
    shift (``option_shifts``) as band probabilities (``shift_probabilities``); from the empty band,
    under every option, half each on the same and one higher. ``floor`` on every entry."""
    n = sm.N_BANDS
    shifts = option_shifts()
    table = np.full((n, n, sm.N_OPTIONS), float(floor))
    for k in range(sm.N_OPTIONS):
        table[0, 0, k] += count / 2
        table[1, 0, k] += count / 2
        for band in range(1, n):
            table[:, band, k] += count * shift_probabilities(shifts[k], band)
    return table


def novelty_bound(horizon=HORIZON, field_strength=sm.FIELD_STRENGTH, n_cells=sm.GRID_SHAPE**2):
    """The most novelty a plan of ``horizon`` options could earn at the start of a mission: the
    largest weight of the field's novelty under its uniform prior (the most one reading of one cell
    can be worth, ``ConditionalDirichlet.novelty``; the novelty term is an expectation of these
    weights), times the most readings an option pools, times the horizon."""
    uniform = jnp.full((sm.N_TERRAIN, n_cells), 1.0 / sm.N_TERRAIN)
    field = ConditionalDirichlet.from_table(uniform, 1, strength=field_strength)
    return float(horizon * float(jnp.max(READINGS)) * float(jnp.max(field.novelty())))


def terminal_penalty(horizon=HORIZON, field_strength=sm.FIELD_STRENGTH):
    """The terminal target's penalty on the empty band: twice ``novelty_bound``."""
    return 2.0 * novelty_bound(horizon, field_strength)


def terminal_logits(penalty):
    """The terminal target's log-preference over the bands: the empty band ``-penalty``, the
    others zero."""
    return np.asarray([-float(penalty)] + [0.0] * (sm.N_BANDS - 1), np.float32)


def expected_change(table):
    """The expected change of band over one option, ``(band, option)``, of a table
    ``[next, band, option]`` (counts or probabilities)."""
    t = np.asarray(table, np.float64)
    t = t / t.sum(axis=0, keepdims=True)
    bands = np.arange(t.shape[0], dtype=float)
    return np.einsum("n,nbk->bk", bands, t) - bands[:, None]


def cell_obs_table(centres):
    """``cell_obs[reading, cell]`` over the cells with centres ``centres``: right with
    ``CELL_ACC``, the rest spread over the other cells in proportion to a Gaussian of sd
    ``CELL_ERROR_SD`` in the distance between the cells' centres."""
    c = np.asarray(centres, np.float64)
    d2 = np.sum((c[:, None, :] - c[None, :, :]) ** 2, axis=-1)
    near = np.exp(-0.5 * d2 / CELL_ERROR_SD**2) * (1 - np.eye(len(c)))
    return CELL_ACC * np.eye(len(c)) + (1 - CELL_ACC) * near / near.sum(axis=0, keepdims=True)


def _cc(table, parent_ndims):
    return ConditionalCategorical(
        log_table=jnp.log(jnp.clip(jnp.asarray(table, jnp.float32), 1e-6)),
        parent_ndims=parent_ndims,
    )


def scheduler(
    grid,
    centres,
    terminal=None,
    field_strength=sm.FIELD_STRENGTH,
    epistemic=("transition:map",),
    policy_len=HORIZON,
    durations=False,
    readings=False,
    energy_counts=None,
):
    """``(System, SystemParams)`` for the scheduler over ``grid`` (the rover's ``Grid``) with the
    cell centres ``centres``. Its parameter variable ``field`` has the shape and prior of the
    rover's, so that ``Identify("rover.field", "mission.field")`` identifies them; its parameter
    variable ``energy_table`` holds the energy transition's counts, ``energy_counts`` (``[next,
    band, option]``) or the prior ``energy_prior``. ``terminal`` is the terminal target's penalty
    on the empty band (``terminal_logits``), ``terminal_penalty(policy_len)`` by default. With
    ``durations`` the control node declares a round of ``SURVEY_TICKS`` for every level; with
    ``readings`` the ``map`` state factor's novelty at each plan step is multiplied by the number
    of readings of the option selected at that step (``SURVEY_READINGS`` for a survey,
    ``CHARGE_READINGS`` for charge), a slow step pooling the readings the fast tier takes in it."""
    n_cells = grid.n_cells
    penalty = terminal_penalty(policy_len, field_strength) if terminal is None else terminal
    counts = energy_prior() if energy_counts is None else energy_counts
    # state factors: visit 0, energy 1, map 2; u_option 3 (the option, a control node)
    system = System(
        A_deps=((0,), (1,)),
        B_deps=((0, 1, 3), (1, 3), (0,), ()),
        controls=(3,),
        action_names=("option",),
        state_names=("visit", "energy", "map"),
        control_names=("u_option",),
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
        parameters={
            "field": ConditionalDirichlet.from_table(uniform, 1, strength=field_strength),
            "energy_table": ConditionalDirichlet(
                concentration=jnp.asarray(counts, jnp.float32), parent_ndims=2
            ),
        },
        likelihood={"cell_obs": _cc(cell_obs_table(centres), 1), "gauge": _cc(sm.gauge_table(), 1)},
        transition={
            "visit": _cc(visit_table(centres), 3),
            "energy": Parameter("energy_table"),
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
        terminal_targets={"energy": Categorical(logits=jnp.asarray(terminal_logits(penalty)))},
    )
    return system, params
