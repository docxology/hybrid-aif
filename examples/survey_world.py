"""The true world of the survey rover showcase, a process separate from the rover's mind.

A single rover drives over a walled field of smooth hills (the position is clipped to the field
and the velocity into a wall is lost). It holds the rover's position and velocity, the terrain
class under it (firm, rough or ice, a function of position), its energy (a rover with an empty
band cannot drive, though it still charges in the light), and the readings the rover receives: a
satellite position fix whose noise grows where the terrain closes the sky, a forward camera image
and a noisy gauge of the energy band (the option of a slow tier never reaches the world). The
field's height map, sky openness, sunlight, slope and fix noise are tabulated at import. The
world is a hand-written ``GenerativeEnv`` wired to a mind with ``couple``; the physical
constants, the true dynamics, the energy rates and the image renderer live here, and the rover's
model reads them as its own.

    python examples/survey_world.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.ndimage import map_coordinates

from hybrid_aif import ConditionalCategorical, LinearGaussianObservation
from hybrid_aif.controls import FLOAT
from hybrid_aif.envs.base import Env
from hybrid_aif.envs.generative import GenerativeEnv, GenerativeEnvParams
from hybrid_aif.system import keyed

FIELD = ((-1.0, 5.0), (-2.0, 4.0))
START = (-0.2, -0.6)
DT = 1.0
GRID_N = 121

TERRAIN_NAMES = ("firm", "rough", "ice")
GAIN = np.asarray([0.8, 0.3, 0.25])  # velocity gain per terrain
KEEP = np.asarray([0.2, 0.1, 0.85])  # velocity persistence per terrain
TERRAIN_SD = np.asarray([0.04, 0.08, 0.12])  # process sd on velocity per terrain
ICE_PATCH = ((1.4, 2.1), 0.9)  # (centre, radius) in the low basin
ROUGH_PATCHES = (((3.6, 0.2), 0.8), ((-0.2, 2.6), 0.6))

SUN_AZIMUTH = np.deg2rad(200.0)
SUN_ELEVATION = np.deg2rad(35.0)
N_AZIMUTH = 16
HORIZON_RANGE = 4.0
HORIZON_STEP = 0.1
AMBIENT = 0.1

ENERGY_MAX = 3.99
BASE_DRAIN = 0.02  # energy per tick at rest
MOVE_DRAIN = 0.136  # energy per tick per unit of speed
SLOPE_DRAIN = 0.2  # extra energy per tick per unit of slope and of speed
ROUGH_DRAIN = 0.16  # extra energy per tick per unit of speed on rough ground
CHARGE_RATE = 0.23  # energy per tick per unit of light
GAUGE_ACC = 0.88
ZONE_SPLIT = (2.0, 1.0)  # the field's quadrants split at these x and y
ZONES = 4

IMG_SIDE = 48
IMG_PIXELS = IMG_SIDE * IMG_SIDE
PIXEL_SD = 0.04
SKY_ROWS = IMG_SIDE // 4

FACTORS = ("terrain", "x", "energy")
MODALITIES = ("fix", "img", "gauge")
ACTIONS = ("a_v",)


def _bump(x, y, cx, cy, s, a):
    return a * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s**2))


def _height_fn(x, y):
    return (
        _bump(x, y, 2.0, 1.0, 1.1, 1.0)
        + _bump(x, y, 4.2, 3.2, 0.9, 0.7)
        + _bump(x, y, -0.3, -1.2, 0.8, 0.5)
        - _bump(x, y, 1.0, 3.0, 0.9, 0.5)
    )


XS = np.linspace(FIELD[0][0], FIELD[0][1], GRID_N)
YS = np.linspace(FIELD[1][0], FIELD[1][1], GRID_N)
_GX, _GY = np.meshgrid(XS, YS, indexing="ij")  # arrays are indexed [ix, iy]
_H0 = _height_fn(_GX, _GY)
HEIGHT = _H0 - _H0.min()
_DX = XS[1] - XS[0]
_DY = YS[1] - YS[0]


def _bilinear_np(table, x, y):
    """Bilinear lookup of a grid table at points (numpy), clamped to the field."""
    fx = np.clip((x - XS[0]) / _DX, 0, GRID_N - 1)
    fy = np.clip((y - YS[0]) / _DY, 0, GRID_N - 1)
    ix = np.minimum(fx.astype(int), GRID_N - 2)
    iy = np.minimum(fy.astype(int), GRID_N - 2)
    tx, ty = fx - ix, fy - iy
    return (
        table[ix, iy] * (1 - tx) * (1 - ty)
        + table[ix + 1, iy] * tx * (1 - ty)
        + table[ix, iy + 1] * (1 - tx) * ty
        + table[ix + 1, iy + 1] * tx * ty
    )


def _openness():
    """Mean over azimuths of cos(horizon elevation)^2, looking out to HORIZON_RANGE."""
    dists = np.arange(HORIZON_STEP, HORIZON_RANGE + 1e-9, HORIZON_STEP)
    total = np.zeros_like(HEIGHT)
    for az in np.linspace(0, 2 * np.pi, N_AZIMUTH, endpoint=False):
        best = np.zeros_like(HEIGHT)
        for d in dists:
            rx = _GX + d * np.cos(az)
            ry = _GY + d * np.sin(az)
            inside = (
                (rx >= FIELD[0][0])
                & (rx <= FIELD[0][1])
                & (ry >= FIELD[1][0])
                & (ry <= FIELD[1][1])
            )
            elev = np.arctan2(_bilinear_np(HEIGHT, rx, ry) - HEIGHT, d)
            best = np.where(inside, np.maximum(best, elev), best)
        total += np.cos(best) ** 2
    return total / N_AZIMUTH


OPEN = _openness()
_HX, _HY = np.gradient(HEIGHT, _DX, _DY)
SLOPE = np.hypot(_HX, _HY)
_NORM = np.sqrt(_HX**2 + _HY**2 + 1.0)
_SUN = np.asarray(
    [
        np.cos(SUN_ELEVATION) * np.cos(SUN_AZIMUTH),
        np.cos(SUN_ELEVATION) * np.sin(SUN_AZIMUTH),
        np.sin(SUN_ELEVATION),
    ]
)
_LAMBERT = np.maximum(0.0, (-_HX * _SUN[0] - _HY * _SUN[1] + _SUN[2]) / _NORM)
LIGHT = AMBIENT + (1 - AMBIENT) * _LAMBERT * OPEN**1.5
SD_FIX = np.clip(0.04 + 1.5 * (1 - OPEN), 0.04, 0.6)


def _terrain_grid():
    t = np.zeros(HEIGHT.shape, np.int32)
    for (cx, cy), r in ROUGH_PATCHES:
        t[np.hypot(_GX - cx, _GY - cy) <= r] = 1
    (cx, cy), r = ICE_PATCH
    t[np.hypot(_GX - cx, _GY - cy) <= r] = 2
    return t


TERRAIN_GRID = _terrain_grid()


def _lookup(table):
    """A jittable bilinear lookup of ``table`` at a position ``xy`` (clamped to the field)."""
    t = jnp.asarray(table, jnp.float32)

    def f(xy):
        fx = jnp.clip((xy[0] - XS[0]) / _DX, 0, GRID_N - 1)
        fy = jnp.clip((xy[1] - YS[0]) / _DY, 0, GRID_N - 1)
        return map_coordinates(t, [fx, fy], order=1, mode="nearest")

    return f


height = _lookup(HEIGHT)
openness = _lookup(OPEN)
light = _lookup(LIGHT)
slope = _lookup(SLOPE)
sd_fix = _lookup(SD_FIX)


def maps():
    """The tabulated field maps, each indexed ``[ix, iy]``:
    ``(xs, ys, height, openness, light, slope, sd_fix, terrain)``."""
    return XS, YS, HEIGHT, OPEN, LIGHT, SLOPE, SD_FIX, TERRAIN_GRID


def terrain_of(xy):
    """The terrain class (0 firm, 1 rough, 2 ice) at a position."""
    t = jnp.asarray(0, jnp.int32)
    for centre, radius in ROUGH_PATCHES:
        t = jnp.where(jnp.linalg.norm(xy - jnp.asarray(centre)) <= radius, 1, t)
    centre, radius = ICE_PATCH
    return jnp.where(jnp.linalg.norm(xy - jnp.asarray(centre)) <= radius, 2, t).astype(jnp.int32)


def dynamics(terrain):
    """(F, G, loc, chol) of the true position/velocity process on a terrain class."""
    F = np.eye(4)
    F[0, 2] = F[1, 3] = DT
    F[2, 2] = F[3, 3] = KEEP[terrain]
    G = np.zeros((4, 2))
    G[2, 0] = G[3, 1] = GAIN[terrain]
    return F, G, np.zeros(4), TERRAIN_SD[terrain] * np.eye(4)


def zone_of(xy):
    """The quadrant (survey sector) of a position: 0 low x and low y, 1 high x and low y, 2 low x
    and high y, 3 high x and high y, split at ``ZONE_SPLIT``."""
    return ((xy[0] >= ZONE_SPLIT[0]) + 2 * (xy[1] >= ZONE_SPLIT[1])).astype(jnp.int32)


def band_of(energy):
    return jnp.clip(jnp.floor(energy), 0, 3).astype(jnp.int32)


def energy_rates(xy, terrain, speed):
    """``(drain, charge)`` of energy per tick at a position, terrain class and speed."""
    drain = (
        BASE_DRAIN + (MOVE_DRAIN + SLOPE_DRAIN * slope(xy) + ROUGH_DRAIN * (terrain == 1)) * speed
    )
    return drain, CHARGE_RATE * light(xy)


def _bumps(rng, n, wx_range, wy_range, r_range):
    cx = rng.uniform(*wx_range, n)
    cy = rng.uniform(*wy_range, n)
    r = rng.uniform(*r_range, n)
    return cx, cy, r


def _ground_coordinates():
    """World coordinates (wx, wy) of the ground pixels, row-major over all IMG_SIDE rows; sky
    rows hold the horizon's. The ground rows span the perspective ``depth = 1 / (0.12 + v)``,
    with v from 0 at the horizon to 1 at the bottom row."""
    cols = (np.arange(IMG_SIDE) + 0.5) / IMG_SIDE
    ground_rows = IMG_SIDE - SKY_ROWS
    v = np.maximum((np.arange(IMG_SIDE) - SKY_ROWS + 0.5) / ground_rows, 0.0)
    depth = 1.0 / (0.12 + v)
    wx = (cols[None, :] - 0.5) * depth[:, None] * 2
    wy = np.broadcast_to(depth[:, None], (IMG_SIDE, IMG_SIDE))
    return wx, wy


def _textures():
    """Fixed ground textures per class, (3, IMG_SIDE, IMG_SIDE); they do not depend on position."""
    rng = np.random.default_rng(7)
    wx, wy = _ground_coordinates()

    def total(cx, cy, r):
        d2 = (wx[..., None] - cx) ** 2 + (wy[..., None] - cy) ** 2
        return np.exp(-d2 / (2 * r**2)).sum(-1)

    gravel = total(*_bumps(rng, 500, (-3.0, 3.0), (0.8, 8.0), (0.05, 0.09)))
    firm = 0.5 + 0.1 * (gravel - gravel.mean()) / gravel.std()
    boulders = total(*_bumps(rng, 60, (-2.5, 2.5), (1.0, 7.0), (0.10, 0.22)))
    rough = np.clip(0.55 - 0.45 * boulders, 0.1, 0.7)
    glare = 0.12 * np.exp(-((wx - 0.6) ** 2 + (wy - 2.5) ** 2) / (2 * 1.2**2))
    ice = 0.85 + glare
    return np.stack([firm, rough, ice]).astype(np.float32)


TEXTURES = _textures()


def render(x, terrain):
    """The noiseless camera image (IMG_PIXELS values in [0, 1], row-major, row 0 at the top) at
    a state ``x`` (position first) and a terrain class: the class's fixed ground texture scaled
    by the illumination at the rover's position, under a sky band whose brightness follows the
    same illumination."""
    lit = light(x[:2])
    ground = jnp.asarray(TEXTURES)[terrain] * lit
    sky = 0.7 * (0.5 + 0.5 * lit)
    rows = jnp.arange(IMG_SIDE)[:, None]
    img = jnp.where(rows < SKY_ROWS, sky, ground)
    return jnp.clip(img, 0.0, 1.0).reshape(-1)


def to_uint8(image):
    """An image as the 0..255 integers the viewer shows (values clipped to [0, 1])."""
    return np.asarray(np.round(np.clip(np.asarray(image), 0.0, 1.0) * 255), np.uint8)


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


class SurveyWorld(GenerativeEnv):
    """Keyed faces over the hand-written process; ``params`` only types the out-ports."""

    def _observe(self, key, state):
        k_gauge, k_fix, k_img = jax.random.split(key, 3)
        pos = state["x"][:2]
        band = band_of(state["energy"])
        eye = jnp.eye(4)
        gauge_p = GAUGE_ACC * eye[band] + (1 - GAUGE_ACC) / 3 * (1 - eye[band])
        image = render(state["x"], state["terrain"])
        return {
            "gauge": jax.random.categorical(k_gauge, jnp.log(gauge_p)).astype(jnp.int32),
            "fix": pos + sd_fix(pos) * jax.random.normal(k_fix, (2,)),
            "img": image + PIXEL_SD * jax.random.normal(k_img, (IMG_PIXELS,)),
        }

    def reset(self, key, params, state=None):
        full = {
            "terrain": terrain_of(jnp.asarray(START, jnp.float32)),
            "x": jnp.asarray([*START, 0.0, 0.0], jnp.float32),
            "energy": jnp.asarray(ENERGY_MAX, jnp.float32),
        }
        full.update({k: jnp.asarray(v, full[k].dtype) for k, v in (state or {}).items()})
        return self._observe(key, full), full

    def step(self, key, state, action, params, factors=None):
        k_move, k_obs = jax.random.split(key)
        a = jnp.clip(action["a_v"], -1.0, 1.0)
        terrain = terrain_of(state["x"][:2])
        table = [dynamics(m) for m in range(3)]
        F = jnp.asarray([c[0] for c in table], jnp.float32)[terrain]
        G = jnp.asarray([c[1] for c in table], jnp.float32)[terrain]
        sd = jnp.asarray(TERRAIN_SD, jnp.float32)[terrain]
        stalled = band_of(state["energy"]) == 0  # an empty rover cannot drive
        G = jnp.where(stalled, 0.0, G)
        x = F @ state["x"] + G @ a + sd * jax.random.normal(k_move, (4,))
        lo = jnp.asarray([FIELD[0][0], FIELD[1][0]], jnp.float32)
        hi = jnp.asarray([FIELD[0][1], FIELD[1][1]], jnp.float32)
        hit = (x[:2] < lo) | (x[:2] > hi)  # the field has walls: clip, and stop that velocity
        x = jnp.concatenate([jnp.clip(x[:2], lo, hi), jnp.where(hit, 0.0, x[2:])])
        speed = jnp.linalg.norm(state["x"][2:])
        drain, charge = energy_rates(state["x"][:2], terrain, speed)
        energy = jnp.clip(state["energy"] - drain + charge, 0.0, ENERGY_MAX)
        new = {"terrain": terrain_of(x[:2]), "x": x, "energy": energy}
        return self._observe(k_obs, new), new

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
    like = {"gauge": _cc(np.eye(4), 1)}
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
        kinds=(FLOAT,),
        sizes=(2,),
        actions=ACTIONS,
    )
    env = SurveyWorld(_Shell(), names, factor_actions=False)
    params = GenerativeEnvParams(
        likelihood=like, transition={}, initial_state_prior={}, actuator={}
    )
    return env, params


def world_summary():
    """Print the ranges of the tabulated maps and the terrain areas."""
    for name, arr in (
        ("height", HEIGHT),
        ("openness", OPEN),
        ("light", LIGHT),
        ("slope", SLOPE),
        ("sd_fix", SD_FIX),
    ):
        print(f"{name:9s} min {arr.min():.3f}  mean {arr.mean():.3f}  max {arr.max():.3f}")
    for k, n in enumerate(TERRAIN_NAMES):
        print(f"{n:9s} {np.mean(TERRAIN_GRID == k):.3f} of the field")


def calibration():
    """The energy calibration table: net energy per tick in the three reference situations."""
    flat = SLOPE < 0.05
    firm = TERRAIN_GRID == 0
    ref = flat & firm & (LIGHT > np.percentile(LIGHT, 40)) & (LIGHT < np.percentile(LIGHT, 60))
    lines = []
    v = 0.8
    mean_light = float(LIGHT.mean())
    drain = BASE_DRAIN + MOVE_DRAIN * v
    lines.append(f"mean light {mean_light:.3f}; median light {np.median(LIGHT):.3f}")
    lines.append(
        f"cruise: speed {v}, firm, flat, mean light: drain {drain:.3f} charge "
        f"{CHARGE_RATE * mean_light:.3f} net {CHARGE_RATE * mean_light - drain:+.3f}/tick"
    )
    if ref.any():
        lines.append(f"  (flat firm cells at median light: {int(ref.sum())})")
    s = float(np.percentile(SLOPE, 98))
    d = BASE_DRAIN + (MOVE_DRAIN + SLOPE_DRAIN * s) * v
    c = CHARGE_RATE * mean_light
    lines.append(
        f"climb: speed {v}, slope {s:.2f} (98th pct): drain {d:.3f} net {c - d:+.3f}/tick "
        f"-> a band in {1 / max(d - c, 1e-9):.1f} ticks (at mean light)"
    )
    d = BASE_DRAIN + (MOVE_DRAIN + ROUGH_DRAIN) * v
    lines.append(
        f"rough: speed {v}, flat: drain {d:.3f} net {c - d:+.3f}/tick "
        f"-> a band in {1 / max(d - c, 1e-9):.1f} ticks (at mean light)"
    )
    lmax = float(np.percentile(LIGHT, 99))
    g = CHARGE_RATE * lmax - BASE_DRAIN
    lines.append(
        f"sun: standing in light {lmax:.2f} (99th pct): net {g:+.3f}/tick -> a band in "
        f"{1 / g:.1f} ticks"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    world_summary()
    print(calibration())
