"""The survey rover's model: a terrain field it learns by driving, an energy it must keep up, and
a preference for bright images that depends on how much energy it has left.

Four factors in declaration order: ``terrain`` (three classes), which reads the parameter
variable ``field`` (one Dirichlet entry per cell of a grid over the position, learned by counting
the terrain seen, under the rate key ``parameter:field``);
``x`` (the vehicle's state: position, heading as a unit vector and forward speed, moved by the
world's own ``VehicleStep`` through a ``NonlinearGaussianTransition``, the same on every terrain;
the model knows neither the field's wall clip nor the empty band's stall, which the world applies);
``energy`` (four bands, a known
table of drain and charge per terrain and cell); and the velocity command ``v``. The modalities:
a satellite position fix whose noise follows a map over the field, a noisy gauge of the energy
band, and the camera, in one of two forms: the image itself, through a frozen neural decoder over
position and terrain (``camera=True``), or the terrain class a frozen recogniser reads off the
image, through the recogniser's measured confusion table (``look``), which informs the terrain
and not the position. The
energy target prefers the full band, shallowly below it (a survey of a far sector must be
affordable at the full band). The field's walls are a preference over the position fix
(``wall_target``): flat inside the field and falling off past its edges, so a plan that would
leave the field is scored down. The image target is a Gaussian over pixels conditional on the
energy band: a strong preference for bright pictures when the energy is low, almost none when it
is full, so that mapping is not coloured by a standing taste for light.

    system, params, grid, centres = rover(decoder)

With ``tiered`` the model has a fifth node, the option control ``u_opt`` (survey sector 1 to 4, or
charge) that the scheduler supplies by a ``Link`` from its option action node: the rover plans no
decision for it, and its factors read what is supplied as they read any control node. A position
target sits at the sector's centre (no pull under charge) and the image preference is conditional on
the energy band and the option (under charge the strongest scale at every band).

The objective has no uncertainty term on the position: the fix's information gain is the rover's
localisation drive, and the field's novelty its mapping drive.
"""

import jax
import jax.numpy as jnp
import numpy as np
import survey_world as wd

from hybrid_aif import (
    Categorical,
    ConditionalCategorical,
    ConditionalDirichlet,
    ConditionalGaussian,
    Grid,
    LinearGaussianObservation,
    Parameter,
)
from hybrid_aif.controls import Box
from hybrid_aif.distributions import MixtureGaussian
from hybrid_aif.factors import GridNoiseMap, NonlinearGaussianTransition
from hybrid_aif.system import System, SystemParams

N_TERRAIN = 3
N_BANDS = 4
N_SURVEY = 4  # survey options: one per sector
N_OPTIONS = N_SURVEY + 1  # and charge
SECTOR_SD = 1.0  # position sd of the survey target
CHARGE_SD = 100.0  # the survey target's sd under charge: no effective pull on the position
PARK_SD = 0.01  # under charge the speed target sits at zero with this sd: the rover parks
GRID_SHAPE = 16  # field cells per axis
NOISE_NODES = 25  # fix-noise map nodes per axis
POLICY_LEN = 12  # ticks the rover plans ahead: 9.6 map cells at the top speed of 0.3
START_SD = (0.3, 0.3, 0.05, 0.05, 0.02)  # the initial belief's sd: position, heading pair, speed
PROPAGATION = "first_order"  # how a belief is pushed through the vehicle's motion
WALL_SPACING = 0.4  # the wall preference's component grid, field units
WALL_SD = 0.2  # each component's sd: the preference falls off over this distance past a wall
FIELD_STRENGTH = 0.1  # pseudo-counts per cell of the uniform field prior
MIXTURE_K = 1  # one component: the motion does not switch, the camera conditions one Gaussian
# the posterior chain does not reach a mixture-carried factor, so "posterior" would change nothing
PLANNING_BELIEF = "prior"
# log-preference per band, full is best; the ladder is shallow below full because a survey of a
# far sector must be affordable at the full band (a steep ladder, (-12, -6, -2, 0), priced every
# survey above its novelty and left the scheduler charging)
ENERGY_TARGET = (-12.0, -4.0, -0.3, 0.0)
IMAGE_SCALES = (0.3, 0.5, 2.0, 40.0)  # target pixel sd per energy band (empty .. full)
BRIGHT = 0.8  # the preferred pixel value
TERRAIN_NAMES = list(wd.TERRAIN_NAMES)
ENERGY_NAMES = ["empty", "low", "medium", "full"]


def field_grid(shape=None):
    """The grid over the position and the centre of each cell, ``(cells, 2)``, row-major."""
    shape = shape or GRID_SHAPE
    (x0, x1), (y0, y1) = wd.FIELD
    grid = Grid(shape=(shape, shape), low=(x0, y0), high=(x1, y1), axes=(0, 1))
    xs = x0 + (np.arange(shape) + 0.5) * (x1 - x0) / shape
    ys = y0 + (np.arange(shape) + 0.5) * (y1 - y0) / shape
    centres = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
    return grid, centres.astype(np.float32)


def fix_noise_map():
    """A GridNoiseMap of the fix's noise over the field: values ``(G, G, 2, 2)``."""
    xs = np.linspace(*wd.FIELD[0], NOISE_NODES)
    ys = np.linspace(*wd.FIELD[1], NOISE_NODES)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    pos = jnp.stack([jnp.asarray(px), jnp.asarray(py)], axis=-1).reshape(-1, 2)
    sd = np.asarray(jax.vmap(wd.sd_fix)(pos)).reshape(NOISE_NODES, NOISE_NODES)
    values = np.zeros((NOISE_NODES, NOISE_NODES, 2, 2), np.float32)
    values[..., 0, 0] = sd
    values[..., 1, 1] = sd
    low = (wd.FIELD[0][0], wd.FIELD[1][0])
    high = (wd.FIELD[0][1], wd.FIELD[1][1])
    return GridNoiseMap(jnp.asarray(values), low, high, axes=(0, 1))


def _band_table(d):
    """The band transition ``(next, band, ...)`` for an expected change per tick ``d`` (any
    shape): the band moves up with probability ``clip(d, 0, 0.5)`` when ``d > 0`` and down with
    ``clip(-d, 0, 0.5)`` when ``d < 0``; the empty and full bands absorb at the ends."""
    up = np.clip(d, 0.0, 0.5)
    down = np.clip(-d, 0.0, 0.5)
    table = np.zeros((N_BANDS, N_BANDS, *d.shape), np.float32)
    for b in range(N_BANDS):
        go_up = up if b < N_BANDS - 1 else np.zeros_like(up)
        go_down = down if b > 0 else np.zeros_like(down)
        table[b, b] = 1.0 - go_up - go_down
        if b < N_BANDS - 1:
            table[b + 1, b] = go_up
        if b > 0:
            table[b - 1, b] = go_down
    return table


def energy_table(centres, tiered=False):
    """The band transition ``(next, band, terrain, cell)`` from the world's own energy rates at
    the cell's centre and a terrain: driving at the measured survey speed (``SURVEY_SPEED``, no
    charge: a moving rover does not charge). With ``tiered`` it is also indexed by the option,
    ``(next, band, terrain, cell, option)``: under a survey option the rover drives, under charge
    it parks (``option_position_target`` holds it still) and charges at the cell's solar yield."""
    cells = jnp.asarray(centres)
    d = np.zeros((N_TERRAIN, centres.shape[0], 2))
    for m in range(N_TERRAIN):
        for j, speed in enumerate((wd.SURVEY_SPEED, 0.0)):
            drain, charge = jax.vmap(lambda xy, m=m, v=speed: wd.energy_rates(xy, m, v))(cells)
            d[m, :, j] = np.asarray(charge - drain)
    if not tiered:
        return _band_table(d[..., 0])
    per_option = np.concatenate([np.repeat(d[..., :1], N_SURVEY, axis=-1), d[..., 1:]], axis=-1)
    return _band_table(per_option)


def wall_target():
    """The wall preference over the position fix: an equal-weight mixture of Gaussians of sd
    ``WALL_SD`` on a grid of spacing ``WALL_SPACING`` filling the field (the first centre half a
    spacing inside each wall). Its log density is flat to a few hundredths of a nat more than
    ``2 WALL_SD`` inside the walls and falls off past them like a Gaussian tail: about ``log 2``
    lower at a wall and about 3.8 nats lower ``2 WALL_SD`` beyond one. A mixture preference is
    scored as an interval, from the closed-form mixture cross-entropy bounds."""
    (x0, x1), (y0, y1) = wd.FIELD
    nx = int(round((x1 - x0) / WALL_SPACING))
    ny = int(round((y1 - y0) / WALL_SPACING))
    xs = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    ys = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
    centres = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
    k = centres.shape[0]
    return MixtureGaussian(
        log_weights=jnp.full(k, -np.log(k), jnp.float32),
        means=jnp.asarray(centres, jnp.float32),
        chols=jnp.broadcast_to(WALL_SD * jnp.eye(2), (k, 2, 2)),
    )


def gauge_table():
    """The gauge's confusion ``(reading, band)``: the true band with the world's accuracy."""
    acc = wd.GAUGE_ACC
    return acc * np.eye(N_BANDS) + (1 - acc) / (N_BANDS - 1) * (1 - np.eye(N_BANDS))


def image_target(tiered=False):
    """The image preference: one Gaussian per energy band over the pixels, the same bright
    reference picture with a pixel sd per band. The covariance is diagonal, stored as the
    per-pixel standard deviation ``(4, p)``. With ``tiered`` the Gaussian is also indexed by the
    option, ``(4, 5, p)``: a survey option keeps the per-band scales, charge uses the strongest
    scale at every band."""
    p = wd.IMG_PIXELS
    scales = jnp.asarray(IMAGE_SCALES, jnp.float32)
    if not tiered:
        loc = jnp.full((N_BANDS, p), BRIGHT, jnp.float32)
        sd = jnp.broadcast_to(scales[:, None], (N_BANDS, p))
        return ConditionalGaussian(loc=loc, chol=sd, parent_ndims=1)
    per_option = jnp.concatenate(
        [jnp.broadcast_to(scales[:, None], (N_BANDS, N_SURVEY)), jnp.full((N_BANDS, 1), scales[0])],
        axis=1,
    )
    loc = jnp.full((N_BANDS, N_OPTIONS, p), BRIGHT, jnp.float32)
    sd = jnp.broadcast_to(per_option[:, :, None], (N_BANDS, N_OPTIONS, p))
    return ConditionalGaussian(loc=loc, chol=sd, parent_ndims=2)


def sector_centres():
    """The centre of each survey sector (the field's quadrants), ``(4, 2)``, in zone order."""
    (x0, x1), (y0, y1) = wd.FIELD
    sx, sy = wd.ZONE_SPLIT
    xs = ((x0 + sx) / 2, (sx + x1) / 2)
    ys = ((y0 + sy) / 2, (sy + y1) / 2)
    return np.asarray([(xs[z % 2], ys[z // 2]) for z in range(N_SURVEY)], np.float32)


def option_position_target():
    """The target over the vehicle's state conditional on the option, ``(5, 5)`` locations and
    ``(5, 5, 5)`` factors: a Gaussian at the survey sector's centre (position sd ``SECTOR_SD``,
    heading pair and speed sd 2) under a survey option; under charge no pull on the position or
    heading and the speed held at zero (sd ``PARK_SD``), so the rover parks to charge."""
    loc = np.zeros((N_OPTIONS, 5), np.float32)
    loc[:N_SURVEY, :2] = sector_centres()
    chol = np.zeros((N_OPTIONS, 5, 5), np.float32)
    for k in range(N_OPTIONS):
        sd = (
            [SECTOR_SD, SECTOR_SD, 2.0, 2.0, 2.0]
            if k < N_SURVEY
            else [CHARGE_SD] * 4 + [PARK_SD]  # charge: anywhere, but parked
        )
        chol[k] = np.diag(sd)
    return ConditionalGaussian(loc=jnp.asarray(loc), chol=jnp.asarray(chol), parent_ndims=1)


def rover(
    decoder,
    grid_shape=GRID_SHAPE,
    energy_target=ENERGY_TARGET,
    field_strength=FIELD_STRENGTH,
    epistemic=("transition:terrain", "obs:fix"),
    image_preference=True,
    policy_len=POLICY_LEN,
    camera=True,
    planning_belief=PLANNING_BELIEF,
    tiered=False,
    wall=True,
    look=None,
    propagation=PROPAGATION,
):
    """``(System, SystemParams, Grid, centres)`` for the survey rover. ``decoder`` is the
    camera's likelihood; ``epistemic`` names the information-gain terms the planner scores (the
    field's novelty and the fix by default; ``False`` for none); ``image_preference=False``
    leaves the image target out; ``camera=False`` leaves the image out of the model;
    ``planning_belief`` is the System's setting of that name; ``tiered`` adds the option control
    node ``u_opt`` that the scheduler supplies by a ``Link`` from its option action node (the
    rover plans no decision for it, and its factors read what is supplied as they read any
    control node), with the position target and the image preference conditional on it;
    ``wall=False`` leaves the wall preference (``wall_target``) out; ``look`` (a confusion table
    ``[reading, terrain]``) adds the modality ``look``, the recogniser's reading of the terrain,
    with that table as its likelihood."""
    grid, centres = field_grid(grid_shape)
    n_cells = grid.n_cells
    # factors: terrain 0, x 1, energy 2, v 3 (float control), u_opt 4 (the option, tiered only)
    modalities = ["fix"] + (["img"] if camera else []) + (["look"] if look is not None else [])
    modalities += ["gauge"]
    A_deps = [(1,)] + ([(1, 0)] if camera else []) + ([(0,)] if look is not None else [])
    A_deps += [(2,)]
    conditional = camera and image_preference
    c_deps = {}
    if tiered:
        c_deps["x"] = ("u_opt",)
    if conditional:
        c_deps["img"] = ("energy", "u_opt") if tiered else ("energy",)
    system = System(
        A_deps=tuple(A_deps),
        B_deps=((1,), (3,), (2, 0, 1, 4) if tiered else (2, 0, 1), ()) + (((),) if tiered else ()),
        controls=(3, 4) if tiered else (3,),
        action_names=("a_v", "a_opt") if tiered else ("a_v",),
        factor_names=("terrain", "x", "energy", "v") + (("u_opt",) if tiered else ()),
        modality_names=tuple(modalities),
        cells={"x": grid},
        C_deps=c_deps or (),
        policy_len=policy_len,
        mixture_k=MIXTURE_K,
        mixture_schedule="mode_aligned",
        planning_belief=planning_belief,
        epistemic=tuple(epistemic) if epistemic else False,
        terms=(),  # no uncertainty term: the fix's information gain is the localisation drive
    )
    #  a mixture: the camera's likelihood switches with the terrain, so the belief over x splits
    #  by the terrain it is conditioned on (``mixture_schedule`` keeps one component per mode)
    x0 = MixtureGaussian(
        log_weights=jnp.log(jnp.ones(MIXTURE_K) / MIXTURE_K),
        means=jnp.broadcast_to(
            jnp.asarray([*wd.START, 0.0, 1.0, 0.0], jnp.float32), (MIXTURE_K, 5)
        ),
        chols=jnp.broadcast_to(jnp.diag(jnp.asarray(START_SD)), (MIXTURE_K, 5, 5)),
    )
    uniform = jnp.full((N_TERRAIN, n_cells), 1.0 / N_TERRAIN)
    likelihood = {
        "fix": LinearGaussianObservation(H=jnp.eye(5)[:2], c=jnp.zeros(2), noise=fix_noise_map()),
        "gauge": ConditionalCategorical(
            log_table=jnp.log(jnp.asarray(gauge_table(), jnp.float32)), parent_ndims=1
        ),
    }
    if camera:
        likelihood["img"] = decoder
    if look is not None:
        table = np.clip(np.asarray(look, np.float64), 1e-4, None)  # no reading impossible
        table = table / table.sum(axis=0, keepdims=True)
        likelihood["look"] = ConditionalCategorical(
            log_table=jnp.log(jnp.asarray(table, jnp.float32)), parent_ndims=1
        )
    energy0 = jnp.log(jnp.asarray([1e-6, 1e-6, 1e-6, 1.0]))
    params = SystemParams(
        parameters={"field": ConditionalDirichlet.from_table(uniform, 1, strength=field_strength)},
        likelihood=likelihood,
        transition={
            "terrain": Parameter("field"),
            "x": NonlinearGaussianTransition(
                f=wd.VehicleStep(),
                innovations=ConditionalGaussian(
                    loc=jnp.zeros((1, 5), jnp.float32),
                    chol=jnp.diag(jnp.asarray(wd.STATE_SD, jnp.float32))[None],
                    parent_ndims=1,
                ),
                propagation=propagation,
            ),
            "energy": ConditionalCategorical(
                log_table=jnp.log(jnp.clip(jnp.asarray(energy_table(centres, tiered)), 1e-6)),
                parent_ndims=4 if tiered else 3,
            ),
        },
        initial_state_prior={
            "terrain": Categorical(logits=jnp.zeros(N_TERRAIN)),
            "x": x0,
            "energy": Categorical(logits=energy0),
        },
        control={"v": Box(-jnp.ones(2), jnp.ones(2))},
        state_targets={
            "energy": Categorical(logits=jnp.asarray(energy_target, jnp.float32)),
            **({"x": option_position_target()} if tiered else {}),
        },
        obs_targets={
            **({"fix": wall_target()} if wall else {}),
            **({"img": image_target(tiered)} if conditional else {}),
        },
    )
    return system, params, grid, centres
