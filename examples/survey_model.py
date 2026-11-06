"""The survey rover's model: a terrain field it learns by driving, an energy it must keep up, and
a preference for bright images that depends on how much energy it has left.

Four factors in declaration order: ``terrain`` (three classes), which reads the parameter
variable ``field`` (one Dirichlet entry per cell of a grid over the position, learned by counting
the terrain seen, under the rate key ``parameter:field``);
``x`` (position and velocity, a mixture switched by the terrain); ``energy`` (four bands, a known
table of drain and charge per terrain and cell); and the velocity command ``v``. Three
modalities: a satellite position fix whose noise follows a map over the field, the forward camera
(a frozen neural decoder over position and terrain) and a noisy gauge of the energy band. The
energy target prefers the full band, shallowly below it (a survey of a far sector must be
affordable at the full band), and there is no position target: the true field has
walls that the model does not know about. The image target is a Gaussian over pixels
conditional on the energy band: a strong preference for bright pictures when the energy is low,
almost none when it is full, so that mapping is not coloured by a standing taste for light.

    system, params, grid, centres = rover(decoder)

With ``tiered`` the model has a fifth node, the option control ``u_opt`` (survey sector 1 to 4, or
charge) that a scheduler drives: a position target at the sector's centre (no pull under charge)
and the image preference conditional on the energy band and the option (under charge the
strongest scale at every band).

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
from hybrid_aif.factors import GridNoiseMap, LinearGaussianTransition
from hybrid_aif.system import System, SystemParams

N_TERRAIN = 3
N_BANDS = 4
N_SURVEY = 4  # survey options: one per sector
N_OPTIONS = N_SURVEY + 1  # and charge
SECTOR_SD = 1.0  # position sd of the survey target
CHARGE_SD = 100.0  # the survey target's sd under charge: no effective pull
GRID_SHAPE = 16  # field cells per axis
NOISE_NODES = 25  # fix-noise map nodes per axis
CRUISE_SPEED = 0.8  # the speed at which the energy table is built
FIELD_STRENGTH = 0.1  # pseudo-counts per cell of the uniform field prior
MIXTURE_K = 3
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


def energy_table(centres):
    """The band transition ``(next, band, terrain, cell)``: with the expected change per tick
    ``d = charge - drain`` at the cell's centre, a terrain and the cruise speed, the band moves
    up with probability ``clip(d, 0, 0.5)`` when ``d > 0`` and down with ``clip(-d, 0, 0.5)``
    when ``d < 0``; the empty and full bands absorb at the ends."""
    n = centres.shape[0]
    cells = jnp.asarray(centres)
    d = np.zeros((N_TERRAIN, n))
    for m in range(N_TERRAIN):
        drain, charge = jax.vmap(lambda xy, m=m: wd.energy_rates(xy, m, CRUISE_SPEED))(cells)
        d[m] = np.asarray(charge - drain)
    up = np.clip(d, 0.0, 0.5)
    down = np.clip(-d, 0.0, 0.5)
    table = np.zeros((N_BANDS, N_BANDS, N_TERRAIN, n), np.float32)
    for b in range(N_BANDS):
        go_up = up if b < N_BANDS - 1 else np.zeros_like(up)
        go_down = down if b > 0 else np.zeros_like(down)
        table[b, b] = 1.0 - go_up - go_down
        if b < N_BANDS - 1:
            table[b + 1, b] = go_up
        if b > 0:
            table[b - 1, b] = go_down
    return table


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
    """The position target conditional on the option, ``(5, 4)`` locations and ``(5, 4, 4)``
    factors: a Gaussian at the survey sector's centre (position sd ``SECTOR_SD``, velocity sd 2)
    under a survey option, and a very wide one (no effective pull) under charge."""
    loc = np.zeros((N_OPTIONS, 4), np.float32)
    loc[:N_SURVEY, :2] = sector_centres()
    chol = np.zeros((N_OPTIONS, 4, 4), np.float32)
    for k in range(N_OPTIONS):
        sd = [SECTOR_SD, SECTOR_SD, 2.0, 2.0] if k < N_SURVEY else [CHARGE_SD] * 4
        chol[k] = np.diag(sd)
    return ConditionalGaussian(loc=jnp.asarray(loc), chol=jnp.asarray(chol), parent_ndims=1)


def rover(
    decoder,
    grid_shape=GRID_SHAPE,
    energy_target=ENERGY_TARGET,
    field_strength=FIELD_STRENGTH,
    epistemic=("transition:terrain", "obs:fix"),
    image_preference=True,
    policy_len=3,
    camera=True,
    planning_belief=PLANNING_BELIEF,
    tiered=False,
):
    """``(System, SystemParams, Grid, centres)`` for the survey rover. ``decoder`` is the
    camera's likelihood; ``epistemic`` names the information-gain terms the planner scores (the
    field's novelty and the fix by default; ``False`` for none); ``image_preference=False``
    leaves the image target out; ``camera=False`` leaves the image out of the model;
    ``planning_belief`` is the System's setting of that name; ``tiered`` adds the option control
    node ``u_opt`` that a scheduler drives, with the position target and the image preference
    conditional on it."""
    grid, centres = field_grid(grid_shape)
    n_cells = grid.n_cells
    # factors: terrain 0, x 1, energy 2, v 3 (float control), u_opt 4 (the option, tiered only)
    modalities = ["fix"] + (["img"] if camera else []) + ["gauge"]
    A_deps = [(1,)] + ([(1, 0)] if camera else []) + [(2,)]
    conditional = camera and image_preference
    c_deps = {}
    if tiered:
        c_deps["x"] = ("u_opt",)
    if conditional:
        c_deps["img"] = ("energy", "u_opt") if tiered else ("energy",)
    system = System(
        A_deps=tuple(A_deps),
        B_deps=((1,), (0, 3), (2, 0, 1), ()) + (((),) if tiered else ()),
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
    F, G, loc, chol = (np.stack([wd.dynamics(m)[i] for m in range(N_TERRAIN)]) for i in range(4))
    x0 = MixtureGaussian(
        log_weights=jnp.log(jnp.ones(MIXTURE_K) / MIXTURE_K),
        means=jnp.broadcast_to(jnp.asarray([*wd.START, 0.0, 0.0], jnp.float32), (MIXTURE_K, 4)),
        chols=jnp.broadcast_to(0.3 * jnp.eye(4), (MIXTURE_K, 4, 4)),
    )
    uniform = jnp.full((N_TERRAIN, n_cells), 1.0 / N_TERRAIN)
    likelihood = {
        "fix": LinearGaussianObservation(H=jnp.eye(4)[:2], c=jnp.zeros(2), noise=fix_noise_map()),
        "gauge": ConditionalCategorical(
            log_table=jnp.log(jnp.asarray(gauge_table(), jnp.float32)), parent_ndims=1
        ),
    }
    if camera:
        likelihood["img"] = decoder
    energy0 = jnp.log(jnp.asarray([1e-6, 1e-6, 1e-6, 1.0]))
    params = SystemParams(
        parameters={"field": ConditionalDirichlet.from_table(uniform, 1, strength=field_strength)},
        likelihood=likelihood,
        transition={
            "terrain": Parameter("field"),
            "x": LinearGaussianTransition(
                F=jnp.asarray(F, jnp.float32),
                G=jnp.asarray(G, jnp.float32),
                innovations=ConditionalGaussian(
                    loc=jnp.asarray(loc, jnp.float32),
                    chol=jnp.asarray(chol, jnp.float32),
                    parent_ndims=1,
                ),
            ),
            "energy": ConditionalCategorical(
                log_table=jnp.log(jnp.clip(jnp.asarray(energy_table(centres)), 1e-6)),
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
            "energy": Categorical(logits=jnp.asarray(energy_target, jnp.float32)),
            **({"x": option_position_target()} if tiered else {}),
        },
        obs_targets={"img": image_target(tiered)} if conditional else {},
    )
    return system, params, grid, centres
