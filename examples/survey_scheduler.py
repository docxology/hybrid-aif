"""The survey rover's slow tier: a scheduler that picks which sector of the field to survey, or
to go and charge in the sun, every ``SURVEY_TICKS`` fast ticks.

Factors, in declaration order: ``visit`` (the cell the rover is at, one level per cell of the
rover's grid, row-major over the grid shape, x first), ``energy`` (the four bands) and ``map`` (the
terrain at the visited cell, three levels). The control node ``u_option`` has five levels: survey
sector 1 to 4 (the field's quadrants, in ``survey_world.zone_of`` order) and charge (drive to the
charging place, ``survey_model.charge_site``, the cell of highest solar yield, and park there). Two
modalities: ``cell_obs`` (the cell the rover's position belief puts it in: in the merged
composite, a sibling-belief read of the rover's position through the cell indicator, realised by
``mode`` and taken at each decision) and ``gauge`` (the band meter's reading of the energy band).
The ``map`` factor reads the parameter variable ``field``, which an ``Identify`` identifies with
the rover's; the scheduler has no observation of ``map`` and never counts into the field, so its
novelty term (``transition:map``, the expected information about the field from the cells its plan
visits) is honoured by the rover counting on the fast clock. The ``map`` is read where the rover
stood one slow step earlier, so the planner looks two options ahead (``policy_len=2``) and
enumerates the option sequences.

Tables, all built here from the world's maps and from what the rover was measured to do under each
option (closed-loop runs of the two-tier agent; see the build record), so that the scheduler's
model of an option describes the rover's behaviour under it. The empty band is the stall at this
tier's resolution: the drive fails inside it (``survey_world.STALL_LEVEL``), so in it no option
moves the rover.

- ``visit[next, cell, band, option]``: the cells an option's readings fall in. In the empty band
  the cell itself, whatever the option (a stalled rover stays where it is and reads only its own
  cell). Otherwise, under survey sector k: from a cell in sector k, the sector's cells weighted by
  a Gaussian of sd ``SURVEY_SPREAD`` about its centre (the rover's sector target pulls it there and
  its novelty spreads it); from a cell outside, the cells along the straight path to the sector's
  centre for the share of the option the travel takes at ``CRUISE`` and the sector's weights for
  the rest. Under charge: likewise, the path to the charging place for the share of the option
  the trip takes at ``CHARGE_SPEED`` and the charging place's cell for the rest.
- ``energy[next, band, cell, option]``: the expected change in energy units over one option, as
  a distribution over whole-band shifts that has the fractional part as the probability of the
  larger shift (the band moves by ``floor`` or ``ceil`` of the change), clipped at the empty and
  full bands. Under survey sector k: the travel from the cell's centre to the sector's centre (a
  straight path sampled at ``PATH_SAMPLES`` points, at the measured speed ``CRUISE``, the net of
  the world's charge and drain along it) plus the option's remaining ticks at the measured net
  rate of surveying ``SURVEY_RATE``; under charge: the travel to the charging place at
  ``CHARGE_SPEED`` plus the remaining ticks parked there at the measured rate ``SITE_RATE``. A
  travel longer than the option is costed in full. In the empty band, under every option, the
  option's ticks parked on the cell (the stall): on ground of poor solar yield the empty band
  holds, a dead end.
- ``cell_obs`` is right with probability ``CELL_ACC``, the rest spread uniformly over the other
  cells. This table says how the rover's cell belief relates to the scheduler's own ``visit``
  variable: the scheduler's belief about where the rover is, a different variable at a different
  level, so the reading is an ordinary likelihood on the modality. ``gauge`` is the band meter's
  confusion.
- The initial beliefs: the start cell and the full band, uniform terrain.

The objective scores the novelty of the field and a target over the energy bands
(``survey_model.ENERGY_TARGET``).
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
#  measured on the two-tier runs of the build record (46 survey and 20 charge options outside the
#  empty band): the travel speed toward a sector, the net energy per tick inside it, the spread
#  of its positions about the sector's centre and the distinct cells it crosses; the travel speed
#  to the charging place, the net energy per tick parked there and the cells a charge crosses
CRUISE = wd.CRUISE_SPEED  # the rover's measured travel speed (0.17)
SURVEY_RATE = -0.013  # the measured net energy per tick while surveying
SURVEY_SPREAD = 0.66  # the measured spread (sd per axis) of a survey's positions about the centre
CHARGE_SPEED = 0.15  # the measured travel speed to the charging place
SITE_RATE = 0.043  # the measured net energy per tick parked at the charging place
PATH_SAMPLES = 24
CELL_ACC = 0.95  # the cell reading is right with this probability
OPTION_NAMES = [f"survey sector {k + 1}" for k in range(sm.N_SURVEY)] + ["charge"]
SECTOR_NAMES = [f"sector {k + 1}" for k in range(sm.N_SURVEY)]
CHARGE = sm.CHARGE  # the charge level
CELL_SIDE = (wd.FIELD[0][1] - wd.FIELD[0][0]) / sm.GRID_SHAPE
# readings of the map per option: an option pools one reading per distinct cell it crosses, the
# median count measured over the options of the closed-loop runs (see the build record)
SURVEY_READINGS = 11
CHARGE_READINGS = 6
READINGS = jnp.asarray([SURVEY_READINGS] * sm.N_SURVEY + [CHARGE_READINGS], jnp.float32)


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
    """The world's net energy per tick (charge less drain) at a speed, the mean over positions
    ``(..., 2)``."""
    pts = jnp.asarray(points, jnp.float32).reshape(-1, 2)
    drain, charge = jax.vmap(lambda xy: wd.energy_rates(xy, speed))(pts)
    return float(jnp.mean(charge - drain))


def _travel(here, there, speed=CRUISE):
    """``(ticks, energy)`` of a straight trip at ``speed`` from ``here`` to ``there``."""
    ramp = np.linspace(0.0, 1.0, PATH_SAMPLES)[:, None]
    ticks = float(np.linalg.norm(there - here)) / speed
    return ticks, ticks * _net_rate(here + ramp * (there - here), speed)


def option_changes(centres):
    """Expected energy change over one option, ``(cell, option)``, in energy units, outside the
    empty band."""
    targets = sm.sector_centres()
    site = sm.charge_site()
    out = np.zeros((centres.shape[0], sm.N_OPTIONS))
    for c, here in enumerate(centres):
        for k in range(sm.N_SURVEY):
            ticks, energy = _travel(here, targets[k])
            out[c, k] = energy + max(SURVEY_TICKS - ticks, 0.0) * SURVEY_RATE
        ticks, energy = _travel(here, site, CHARGE_SPEED)
        out[c, CHARGE] = energy + max(SURVEY_TICKS - ticks, 0.0) * SITE_RATE
    return out


def stalled_changes(centres):
    """Expected energy change over one option in the empty band, ``(cell,)``: the option's ticks
    parked on the cell."""
    return np.asarray([SURVEY_TICKS * _net_rate(c[None], 0.0) for c in centres])


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


def energy_table(centres):
    """``energy[next, band, cell, option]``."""
    change = option_changes(centres)
    stall = stalled_changes(centres)
    n = centres.shape[0]
    table = np.zeros((sm.N_BANDS, sm.N_BANDS, n, sm.N_OPTIONS))
    for c in range(n):
        for k in range(sm.N_OPTIONS):
            table[:, :, c, k] = shift_distribution(change[c, k])
            table[:, 0, c, k] = shift_distribution(stall[c])[:, 0]
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
    readings of the option selected at that step (``SURVEY_READINGS`` for a survey,
    ``CHARGE_READINGS`` for charge), a slow step pooling the readings the fast tier takes in it."""
    n_cells = grid.n_cells
    # factors: visit 0, energy 1, map 2, u_option 3 (the option, a control node)
    system = System(
        A_deps=((0,), (1,)),
        B_deps=((0, 1, 3), (1, 0, 3), (0,), ()),
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
        parameters={"field": ConditionalDirichlet.from_table(uniform, 1, strength=field_strength)},
        likelihood={"cell_obs": _cc(cell_obs_table(n_cells), 1), "gauge": _cc(sm.gauge_table(), 1)},
        transition={
            "visit": _cc(visit_table(centres), 3),
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
