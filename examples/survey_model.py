"""The survey rover's model: a terrain field it learns by driving, an energy it must keep up, and
a preference for bright images that depends on how much energy it has left.

Four factors in declaration order: ``terrain`` (three classes), the learnable field read through a
grid over the position (one Dirichlet entry per cell, learned by counting the terrain seen);
``x`` (position and velocity, a mixture switched by the terrain); ``energy`` (four bands, a known
table of drain and charge per terrain and cell); and the velocity command ``v``. Three
modalities: a satellite position fix whose noise follows a map over the field, the forward camera
(a frozen neural decoder over position and terrain) and a noisy gauge of the energy band. The
energy target prefers the full band steeply, and there is no position target: the true field has
walls that the model does not know about. The image target is a Gaussian over pixels
conditional on the energy band: a strong preference for bright pictures when the energy is low,
almost none when it is full, so that mapping is not coloured by a standing taste for light.

    system, params, grid, centres = rover(decoder)

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
)
from hybrid_aif.controls import Box
from hybrid_aif.distributions import MixtureGaussian
from hybrid_aif.factors import GridNoiseMap, LinearGaussianTransition
from hybrid_aif.system import System, SystemParams

N_TERRAIN = 3
N_BANDS = 4
GRID_SHAPE = 16  # field cells per axis
NOISE_NODES = 25  # fix-noise map nodes per axis
CRUISE_SPEED = 0.8  # the speed at which the energy table is built
FIELD_STRENGTH = 0.1  # pseudo-counts per cell of the uniform field prior
MIXTURE_K = 3
# the posterior chain does not reach a mixture-carried factor, so "posterior" would change nothing
PLANNING_BELIEF = "prior"
ENERGY_TARGET = (-12.0, -6.0, -2.0, 0.0)  # log-preference per band, full is best
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


def image_target():
    """The image preference: one Gaussian per energy band over the pixels, the same bright
    reference picture with a pixel sd per band. The covariance is diagonal, stored as the
    per-pixel standard deviation ``(4, p)``."""
    p = wd.IMG_PIXELS
    loc = jnp.full((N_BANDS, p), BRIGHT, jnp.float32)
    sd = jnp.broadcast_to(jnp.asarray(IMAGE_SCALES, jnp.float32)[:, None], (N_BANDS, p))
    return ConditionalGaussian(loc=loc, chol=sd, parent_ndims=1)


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
):
    """``(System, SystemParams, Grid, centres)`` for the survey rover. ``decoder`` is the
    camera's likelihood; ``epistemic`` names the information-gain terms the planner scores (the
    field's novelty and the fix by default; ``False`` for none); ``image_preference=False``
    leaves the image target out; ``camera=False`` leaves the image out of the model;
    ``planning_belief`` is the System's setting of that name."""
    grid, centres = field_grid(grid_shape)
    n_cells = grid.n_cells
    # factors: terrain 0, x 1, energy 2, v 3 (float control); action node: a_v 4
    modalities = ["fix"] + (["img"] if camera else []) + ["gauge"]
    A_deps = [(1,)] + ([(1, 0)] if camera else []) + [(2,)]
    conditional = camera and image_preference
    system = System(
        A_deps=tuple(A_deps),
        B_deps=((1,), (0, 3), (2, 0, 1), ()),
        controls=(3,),
        action_names=("a_v",),
        factor_names=("terrain", "x", "energy", "v"),
        modality_names=tuple(modalities),
        cells={"x": grid},
        C_deps={"img": ("energy",)} if conditional else (),
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
        likelihood=likelihood,
        transition={
            "terrain": ConditionalDirichlet.from_table(uniform, 1, strength=field_strength),
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
        },
        obs_targets={"img": image_target()} if conditional else {},
    )
    return system, params, grid, centres
