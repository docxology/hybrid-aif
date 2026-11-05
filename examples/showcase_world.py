"""The true world of the two-tier rover showcase, a process separate from both minds.

It holds the rover's position and velocity over terrain patches, its energy, three rocks and the
readings the two minds receive: the mission's gauge, zone, rock checks and payoff, and the
rover's position fix and camera image. It is a hand-written ``GenerativeEnv`` wired to the
minds with ``couple``. The physical constants, the true dynamics and the image renderer live
here; the minds' models read them as their own.
"""

import jax
import jax.numpy as jnp
import numpy as np
import showcase_mission as mm

from hybrid_aif import ConditionalCategorical, LinearGaussianObservation
from hybrid_aif.controls import FLOAT, LEVEL
from hybrid_aif.envs.base import Env
from hybrid_aif.envs.generative import GenerativeEnv, GenerativeEnvParams
from hybrid_aif.system import keyed

FIELD = ((-1.0, 5.0), (-2.0, 4.0))
RIDGE = (2.0, 1.0)  # where the position fix is sharp
START = mm.ZONES["start"]
DT = 1.0
GAIN = np.asarray([0.8, 0.05])  # velocity gain per sensor level (short, long exposure)
KEEP = np.asarray([0.2, 0.05])  # velocity persistence per sensor level
TERRAIN_SD = np.asarray([0.04, 0.12, 0.12])
DRIFT = {1: (2, -0.25), 2: (3, 0.25)}  # sand pushes along -x, ice along +y
FIX_SD = ((0.5, 0.2), (0.15, 0.04))  # (open field, at the ridge) per sensor level

ZONE_RADIUS = 0.8
PATCHES = {1: ((1.0, 1.5), 0.9), 2: ((3.7, 1.6), 0.9)}  # terrain level -> (centre, radius)
ENERGY_PER_UNIT = 0.15  # bands lost per unit of distance moved
REFILL_PER_TICK = 1.0  # bands regained per tick at the dock
ENERGY_MAX = 3.99
REGROW = 0.02  # per tick, a sampled rock returns to good and, separately, to bad
P_GOOD = 0.55
SCAN_ACC = 0.9
GAUGE_ACC = 0.88
ZONE_CENTRES = np.asarray([mm.ZONES[z] for z in mm.ZONE_NAMES], np.float32)
ZONE_IDX = {z: i for i, z in enumerate(mm.ZONE_NAMES)}

IMG_SIDE = 32  # the camera image is IMG_SIDE x IMG_SIDE pixels of terrain around the rover
IMG_PIXELS = IMG_SIDE * IMG_SIDE
PIXEL_SD = 0.1
PATCH_HALF = 1.0  # the patch spans +-PATCH_HALF field units around the rover, on each axis
TERRAIN_BRIGHTNESS = (0.15, 0.40, 0.65)
STRIPE_AMPLITUDE = (0.0, 0.18, 0.18)
STRIPE_FREQUENCY = (0.0, 1.5, 3.0)  # spatial frequency per field unit, along the stripe normal
STRIPE_ANGLE = (0.0, 0.3, -0.9)  # radians, the stripe normal's angle from the x axis
MOTTLE = 0.08  # amplitude of the faint mottle on firm ground

FACTORS = ("terrain", "x", "energy", "rock_1", "rock_2", "rock_3", "pay_mem", "clock")
MODALITIES = (*mm.MISSION_MODALITIES, "fix", "img")
ACTIONS = ("a_v", "a_sensor", "option")


def fix_sd(level, px, py):
    """The fix's standard deviation at a position under a sensor level: coarse in the open,
    sharp near the ridge, and sharper under the long exposure."""
    base, near = FIX_SD[level]
    d2 = (px - RIDGE[0]) ** 2 + (py - RIDGE[1]) ** 2
    return base + (near - base) * jnp.exp(-d2 / (2 * 0.9**2))


def dynamics(terrain, sensor):
    """(F, G, loc, chol) of the true position/velocity process at a terrain and sensor level."""
    F = np.eye(4)
    F[0, 2] = F[1, 3] = DT
    F[2, 2] = F[3, 3] = KEEP[sensor]
    G = np.zeros((4, 2))
    G[2, 0] = G[3, 1] = GAIN[sensor]
    loc = np.zeros(4)
    if terrain in DRIFT:
        loc[DRIFT[terrain][0]] = DRIFT[terrain][1]
    return F, G, loc, TERRAIN_SD[terrain] * np.eye(4)


_OFFSETS = np.linspace(-PATCH_HALF, PATCH_HALF, IMG_SIDE)
# pixel p is cell (p // IMG_SIDE, p % IMG_SIDE): row 0 is the largest y offset, columns run
# along x, so the patch reads as a top-down picture with +x to the right and +y up.
PATCH_DX = np.tile(_OFFSETS, IMG_SIDE).astype(np.float32)
PATCH_DY = np.repeat(_OFFSETS[::-1], IMG_SIDE).astype(np.float32)


def render(x, terrain):
    """The noiseless camera image (IMG_PIXELS values) at a state ``x`` (position first) and a
    terrain level: a top-down patch of terrain centred on the rover. Brightness and a stripe
    texture depend on the terrain; the stripe phase depends on the world position of each pixel.
    Firm ground carries a faint mottle instead of stripes. The patch holds no mark of the rover."""
    wx = x[0] + jnp.asarray(PATCH_DX)
    wy = x[1] + jnp.asarray(PATCH_DY)
    angle = jnp.asarray(STRIPE_ANGLE, jnp.float32)[terrain]
    phase = jnp.asarray(STRIPE_FREQUENCY, jnp.float32)[terrain] * (
        jnp.cos(angle) * wx + jnp.sin(angle) * wy
    )
    stripes = jnp.asarray(STRIPE_AMPLITUDE, jnp.float32)[terrain] * jnp.sin(phase)
    mottle = jnp.where(terrain == 0, MOTTLE, 0.0) * jnp.sin(3 * wx) * jnp.sin(3 * wy)
    return jnp.asarray(TERRAIN_BRIGHTNESS, jnp.float32)[terrain] + stripes + mottle


def to_uint8(image):
    """An image as the 0..255 integers the viewer shows (values clipped to [0, 1])."""
    return np.asarray(np.round(np.clip(np.asarray(image), 0.0, 1.0) * 255), np.uint8)


def zone_of(pos):
    d = jnp.linalg.norm(jnp.asarray(ZONE_CENTRES) - pos[None, :], axis=1)
    j = jnp.argmin(d)
    return jnp.where(d[j] <= ZONE_RADIUS, j, 0).astype(jnp.int32)


def terrain_of(pos):
    t = jnp.asarray(0, jnp.int32)
    for level, (centre, radius) in PATCHES.items():
        inside = jnp.linalg.norm(pos - jnp.asarray(centre)) <= radius
        t = jnp.where(inside, level, t)
    return t


def band_of(energy):
    return jnp.clip(jnp.floor(energy), 0, 3).astype(jnp.int32)


class _Shell(Env):
    """The positional shell GenerativeEnv requires; the keyed faces are overridden."""

    @property
    def num_factors(self):
        return len(FACTORS)

    @property
    def num_modalities(self):
        return len(MODALITIES)

    def reset(self, key, params):  # pragma: no cover
        raise NotImplementedError

    def step(self, key, state, action, params):  # pragma: no cover
        raise NotImplementedError


class RoverWorld(GenerativeEnv):
    """Keyed faces over the hand-written process; ``params`` only types the out-ports."""

    def _observe(self, key, state, level):
        keys = jax.random.split(key, 6)
        pos = state["x"][:2]
        zone = zone_of(pos)
        band = band_of(state["energy"])
        eye = jnp.eye(4)
        gauge_p = GAUGE_ACC * eye[band] + (1 - GAUGE_ACC) / 3 * (1 - eye[band])
        obs = {
            "gauge": jax.random.categorical(keys[0], jnp.log(gauge_p)).astype(jnp.int32),
            "zone_obs": zone,
            "payoff": state["pay_mem"].astype(jnp.int32),
        }
        for k in range(3):
            rock = state[f"rock_{k + 1}"]
            informative = (zone == 1 + k) & (rock < 2)
            p_good = jnp.where(informative, jnp.where(rock == 0, SCAN_ACC, 1 - SCAN_ACC), 0.5)
            obs[f"check_{k + 1}"] = (jax.random.uniform(keys[1 + k]) >= p_good).astype(jnp.int32)
        sd = jnp.where(level == 0, fix_sd(0, *pos), fix_sd(1, *pos))
        obs["fix"] = pos + sd * jax.random.normal(keys[4], (2,))
        image = render(state["x"], state["terrain"])
        obs["img"] = image + PIXEL_SD * jax.random.normal(keys[5], (IMG_PIXELS,))
        return obs

    def reset(self, key, params, state=None):
        k_rocks, k_obs = jax.random.split(key)
        rocks = (jax.random.uniform(k_rocks, (3,)) >= P_GOOD).astype(jnp.int32)  # 0 good, 1 bad
        full = {
            "terrain": jnp.asarray(0, jnp.int32),
            "x": jnp.asarray([*START, 0.0, 0.0], jnp.float32),
            "energy": jnp.asarray(ENERGY_MAX, jnp.float32),
            "rock_1": rocks[0],
            "rock_2": rocks[1],
            "rock_3": rocks[2],
            "pay_mem": jnp.asarray(0, jnp.int32),
            "clock": jnp.asarray(0, jnp.int32),
        }
        full.update({k: jnp.asarray(v, full[k].dtype) for k, v in (state or {}).items()})
        return self._observe(k_obs, full, jnp.asarray(0)), full

    def step(self, key, state, action, params, factors=None):
        k_move, k_regrow, k_obs = jax.random.split(key, 3)
        v, level, option = action["a_v"], action["a_sensor"], action["option"]
        terrain = terrain_of(state["x"][:2])
        table = [[dynamics(m, s) for s in range(2)] for m in range(3)]
        F = jnp.asarray([[c[0] for c in row] for row in table], jnp.float32)[terrain, level]
        G = jnp.asarray([[c[1] for c in row] for row in table], jnp.float32)[terrain, level]
        loc = jnp.asarray([[c[2] for c in row] for row in table], jnp.float32)[terrain, level]
        sd = jnp.asarray(TERRAIN_SD, jnp.float32)[terrain]
        x = F @ state["x"] + G @ v + loc + sd * jax.random.normal(k_move, (4,))
        moved = jnp.linalg.norm(x[:2] - state["x"][:2])
        zone = zone_of(x[:2])
        energy = state["energy"] - ENERGY_PER_UNIT * moved
        energy = jnp.where(zone == ZONE_IDX["dock"], energy + REFILL_PER_TICK, energy)
        energy = jnp.clip(energy, 0.0, ENERGY_MAX)
        dur = jnp.asarray(mm.DURATIONS, jnp.int32)[option]
        fresh = state["clock"] == 0  # the first tick of an option instance clears the payoff
        pay = jnp.where(fresh, 0, state["pay_mem"])
        new = {"terrain": terrain_of(x[:2]), "x": x, "energy": energy}
        kr = jax.random.split(k_regrow, 3)
        for k in range(3):
            rock = state[f"rock_{k + 1}"]
            sampling = (option == 3 + k) & (zone == 1 + k) & (rock < 2)
            pay = jnp.where(sampling, jnp.where(rock == 0, 1, 2), pay)
            u = jax.random.uniform(kr[k])
            regrown = jnp.where(u < REGROW, 0, jnp.where(u < 2 * REGROW, 1, 2))
            new[f"rock_{k + 1}"] = jnp.where(
                sampling, 2, jnp.where(rock == 2, regrown, rock)
            ).astype(jnp.int32)
        new["pay_mem"] = pay.astype(jnp.int32)
        new["clock"] = ((state["clock"] + 1) % dur).astype(jnp.int32)
        return self._observe(k_obs, new, level), new

    def step_actuated(self, key, state, action, params, factors=None):
        obs, new = self.step(key, state, action, params)
        return obs, new, dict(action)


def _cc(table, parent_ndims):
    return ConditionalCategorical(
        log_table=jnp.log(jnp.clip(jnp.asarray(table, jnp.float32), 1e-12)),
        parent_ndims=parent_ndims,
    )


def world():
    """``(GenerativeEnv, env_params)`` for ``couple``."""
    n_level = {"gauge": 4, "zone_obs": 5, "check_1": 2, "check_2": 2, "check_3": 2, "payoff": 3}
    like = {m: _cc(np.eye(n), 1) for m, n in n_level.items()}
    like["fix"] = LinearGaussianObservation(H=jnp.eye(4)[:2], c=jnp.zeros(2), noise=jnp.eye(2))
    like["img"] = LinearGaussianObservation(
        H=jnp.zeros((IMG_PIXELS, 4)), c=jnp.zeros(IMG_PIXELS), noise=jnp.eye(IMG_PIXELS)
    )
    names = keyed.FaceNames(
        FACTORS,
        MODALITIES,
        None,
        None,
        controls=ACTIONS,
        kinds=(FLOAT, LEVEL, LEVEL),
        sizes=(2, 2, len(mm.OPTION_NAMES)),
        actions=ACTIONS,
    )
    env = RoverWorld(_Shell(), names, factor_actions=False)
    params = GenerativeEnvParams(
        likelihood=like, transition={}, initial_state_prior={}, actuator={}
    )
    return env, params
