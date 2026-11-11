"""The true world of the survey rover showcase, a black box coupled to the rover's mind.

A single rover drives over a walled field of smooth hills (the position is clipped to the field
and a rover that hits a wall stops). It holds the rover's position, heading and forward speed, the
terrain class under it (firm, rough or ice, a function of position), its energy (a rover with an
empty band cannot drive, though it still charges in the light), and the readings the rover
receives: a satellite position fix whose noise grows where the terrain closes the sky, a camera
image and a noisy gauge of the energy band (the option of a slow tier never reaches the world).
The field's height map, sky openness, sunlight, slope and fix noise are tabulated at import.

The rover is a wheeled vehicle with a heading of its own (``VehicleStep``): its state is the
position, the heading as a unit vector and the forward speed; the command accelerates it along the
heading and turns it (by at most ``TURN_MAX`` a tick, in place too). The speed keeps most of itself
per tick (``KEEP``), so the top speed ``GAIN / (1 - KEEP)`` is reached over tens of ticks and stays
under one map cell per tick. The camera is mounted forward and looks along the heading; it renders
the scene from the field's geometry: the ground along each pixel's ray, ray-marched over the height
map, lit by the sun, textured in world coordinates, under a sky; see ``render``.

    python examples/survey_world.py
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.ndimage import map_coordinates

from hybrid_aif.composition.black_box import black_box
from hybrid_aif.controls import FLOAT, LEVEL
from hybrid_aif.description import register_transition

FIELD = ((-1.0, 5.0), (-2.0, 4.0))
START = (-0.2, -0.6)
DT = 1.0
GRID_N = 121

TERRAIN_NAMES = ("firm", "rough", "ice")
GAIN = 0.045  # forward speed added per tick per unit of the acceleration command
KEEP = 0.85  # forward speed kept per tick
TURN_MAX = np.deg2rad(20.0)  # the heading turns at most this far per tick (in place too)
STATE_SD = np.asarray([0.005, 0.005, 0.01, 0.01, 0.004])  # process sd per state coordinate
ICE_PATCH = ((1.4, 2.1), 0.9)  # (centre, radius) in the low basin
ROUGH_PATCHES = (((3.6, 0.2), 0.8), ((-0.2, 2.6), 0.6))

SUN_AZIMUTH = np.deg2rad(200.0)
SUN_ELEVATION = np.deg2rad(35.0)
N_AZIMUTH = 16
HORIZON_RANGE = 4.0
HORIZON_STEP = 0.1
AMBIENT = 0.1

ENERGY_MAX = 3.99
SURVEY_SPEED = 0.15  # the rover's typical speed while surveying, measured (see the build record)
#  the drains are per tick and per unit of speed, set from the vehicle's measured motion (see the
#  build record): driving on firm flat ground at the rover's typical survey speed costs about one
#  band over a 24-tick option, rough ground and slopes cost more
BASE_DRAIN = 0.005  # energy per tick at rest
MOVE_DRAIN = 0.233  # energy per tick per unit of speed
SLOPE_DRAIN = 0.233  # extra energy per tick per unit of slope and of speed
ROUGH_DRAIN = 0.233  # extra energy per tick per unit of speed on rough ground
#  solar charging: only a parked rover charges (fully at rest, falling to nothing at PARK_SPEED),
#  and only where the ground is lit above LIGHT_FLOOR, in proportion to the excess, so sunny
#  ground is a charging place and shaded ground is not
CHARGE_RATE = 0.14  # energy per tick, parked in full sun
PARK_SPEED = 0.03
LIGHT_FLOOR = 0.45
GAUGE_ACC = 0.88
ZONE_SPLIT = (2.0, 1.0)  # the field's quadrants split at these x and y
ZONES = 4

IMG_SIDE = 32
IMG_PIXELS = IMG_SIDE * IMG_SIDE
PIXEL_SD = 0.04
FOCAL = (IMG_SIDE / 2) / np.tan(np.deg2rad(35.0))  # pixels: a 70 degree field of view
CAMERA_HEIGHT = 0.12  # the lens above the ground under the rover
CAMERA_TILT = np.deg2rad(8.0)  # the lens points this far below the rover's forward axis
CAMERA_HEADING = np.pi / 2  # the comparison camera's fixed direction (``observe(fixed=True)``)
NEAR, REACH, MARCH = 0.15, 5.0, 96  # the ray march: nearest and farthest distance, samples
SUBCOLUMNS = 3  # rays per pixel across its width (the pixel's footprint, a midpoint rule)
WALL_EDGE = 0.05  # the ground's albedo ramps to WALL_ALBEDO over this width across a wall
HAZE = 2.5  # distance over which the ground fades into the haze at the horizon
WALL_ALBEDO = 0.15  # the ground beyond the field's walls is dark
EXPOSURE = 1.4  # pixel value per unit of albedo times sunlight
ALBEDO = np.asarray([0.55, 0.40, 0.82])  # mean ground albedo per terrain class
TEXTURE_AMP = np.asarray([0.15, 0.22, 0.07])  # texture amplitude per class
TEXTURE_WAVES = 6  # plane waves summed per class
TEXTURE_WAVELENGTH = (0.9, 1.8)  # field units: about two and a half to five map cells
SKY_ZENITH, SKY_HORIZON, SUN_GLOW = 0.55, 0.8, 0.12


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


_H_MIN = float(_H0.min())


def smooth_height(points):
    """The ground's height at world points ``(..., 2)``: the height map's own analytic form
    (``HEIGHT`` is it tabulated), smooth in the point, held beyond the walls at the value the
    form takes there."""
    x, y = points[..., 0], points[..., 1]

    def bump(cx, cy, s, a):
        return a * jnp.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s**2))

    return (
        bump(2.0, 1.0, 1.1, 1.0)
        + bump(4.2, 3.2, 0.9, 0.7)
        + bump(-0.3, -1.2, 0.8, 0.5)
        - bump(1.0, 3.0, 0.9, 0.5)
        - _H_MIN
    )


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


class VehicleStep(eqx.Module):
    """The rover's motion, a wheeled vehicle with a heading of its own: the mean of the next state
    ``(x, y, cos theta, sin theta, s)`` (position, the heading as a unit vector, the forward speed)
    from the state and the command ``(acceleration, turn)`` in ``[-1, 1]^2``::

        x += DT * s * cos theta,   y += DT * s * sin theta
        theta += DT * TURN_MAX * turn          (a rotation of the unit vector; in place too)
        s = KEEP * s + GAIN * acceleration     (negative s reverses)

    The heading is carried as ``(cos theta, sin theta)``, renormalised before it is used, so
    there is no wrap-around and the rotation is smooth in the state and the command. Called as a
    ``NonlinearGaussianTransition``'s mean map ``f(x, controls)``; the world's ``move`` calls it
    too, so the model and the world share one declaration of the motion. The same on every
    terrain."""

    def __call__(self, x, controls):
        a = jnp.clip(controls[0], -1.0, 1.0)
        norm = jnp.sqrt(x[2] ** 2 + x[3] ** 2) + 1e-6
        c, s = x[2] / norm, x[3] / norm
        w = DT * TURN_MAX * a[1]
        return jnp.stack(
            [
                x[0] + DT * x[4] * c,
                x[1] + DT * x[4] * s,
                c * jnp.cos(w) - s * jnp.sin(w),
                s * jnp.cos(w) + c * jnp.sin(w),
                KEEP * x[4] + GAIN * a[0],
            ]
        )


#  the motion's class is named for the model's description (``describe`` writes code by name);
#  a second import of this module in one interpreter finds the name taken and keeps the first
try:
    register_transition("survey.vehicle", VehicleStep)
except ValueError:
    pass


def top_speed():
    """The steady forward speed under a full acceleration command, ``GAIN / (1 - KEEP)``."""
    return float(GAIN / (1.0 - KEEP))


def heading_of(x):
    """The heading (radians from +x) of a state ``(x, y, cos theta, sin theta, s)``."""
    return jnp.arctan2(x[3], x[2])


def velocity_of(x):
    """The velocity ``s * (cos theta, sin theta)`` of a state."""
    norm = jnp.sqrt(x[2] ** 2 + x[3] ** 2) + 1e-6
    return x[4] * jnp.stack([x[2], x[3]]) / norm


def zone_of(xy):
    """The quadrant (survey sector) of a position: 0 low x and low y, 1 high x and low y, 2 low x
    and high y, 3 high x and high y, split at ``ZONE_SPLIT``."""
    return ((xy[0] >= ZONE_SPLIT[0]) + 2 * (xy[1] >= ZONE_SPLIT[1])).astype(jnp.int32)


def band_of(energy):
    return jnp.clip(jnp.floor(energy), 0, 3).astype(jnp.int32)


def solar_yield(xy):
    """The share of ``CHARGE_RATE`` a parked rover collects at a position: the sunlight there above
    ``LIGHT_FLOOR``, scaled so full light (1) gives all of it."""
    return jnp.clip((light(xy) - LIGHT_FLOOR) / (1.0 - LIGHT_FLOOR), 0.0, 1.0)


def parked(speed):
    """How parked a rover is at a speed: one at rest, falling linearly to zero at ``PARK_SPEED``."""
    return jnp.clip(1.0 - jnp.abs(speed) / PARK_SPEED, 0.0, 1.0)


def energy_rates(xy, terrain, speed):
    """``(drain, charge)`` of energy per tick at a position, terrain class and speed: the drain
    grows with speed, slope and rough ground; the charge is the solar yield there for a parked
    rover (``parked``)."""
    drain = (
        BASE_DRAIN + (MOVE_DRAIN + SLOPE_DRAIN * slope(xy) + ROUGH_DRAIN * (terrain == 1)) * speed
    )
    return drain, CHARGE_RATE * solar_yield(xy) * parked(speed)


def _waves():
    """Per class, ``TEXTURE_WAVES`` plane waves ``(kx, ky, phase)``, ``(3, TEXTURE_WAVES, 3)``,
    from a fixed seed, with wavelengths drawn from ``TEXTURE_WAVELENGTH``."""
    rng = np.random.default_rng(7)
    out = np.zeros((len(TERRAIN_NAMES), TEXTURE_WAVES, 3))
    for c in range(len(TERRAIN_NAMES)):
        k = 2 * np.pi / rng.uniform(*TEXTURE_WAVELENGTH, TEXTURE_WAVES)
        angle = rng.uniform(0, 2 * np.pi, TEXTURE_WAVES)
        phase = rng.uniform(0, 2 * np.pi, TEXTURE_WAVES)
        out[c] = np.stack([k * np.cos(angle), k * np.sin(angle), phase], 1)
    return out.astype(np.float32)


WAVES = _waves()


def texture(points, terrain):
    """The ground albedo of a terrain class at world points ``(..., 2)``: the class's mean albedo
    plus its texture, a sum of plane waves over world coordinates scaled to unit sd and then to
    ``TEXTURE_AMP``. A function of the world point alone, so the ground keeps its pattern as the
    rover moves and turns."""
    w = jnp.asarray(WAVES)[terrain]  # (TEXTURE_WAVES, 3)
    phase = points[..., 0, None] * w[:, 0] + points[..., 1, None] * w[:, 1] + w[:, 2]
    wave = jnp.sum(jnp.cos(phase), axis=-1) / np.sqrt(TEXTURE_WAVES / 2)
    return jnp.asarray(ALBEDO, jnp.float32)[terrain] + jnp.asarray(TEXTURE_AMP)[terrain] * wave


def _outside(points):
    """How far world points ``(..., 2)`` lie beyond the field's walls, as a share from 0 (more than
    ``WALL_EDGE / 2`` inside) to 1 (more than ``WALL_EDGE / 2`` beyond): a linear ramp across the
    wall, so the ground's albedo has no step."""
    gap = jnp.maximum(
        jnp.maximum(FIELD[0][0] - points[..., 0], points[..., 0] - FIELD[0][1]),
        jnp.maximum(FIELD[1][0] - points[..., 1], points[..., 1] - FIELD[1][1]),
    )
    return jnp.clip(gap / WALL_EDGE + 0.5, 0.0, 1.0)


_COLS = np.arange(IMG_SIDE) + 0.5 - IMG_SIDE / 2  # pixel offsets from the centre, left to right
_ALPHA = np.arctan(_COLS / FOCAL)  # each column's angle to the right of the heading
_SUB = (np.arange(SUBCOLUMNS) + 0.5) / SUBCOLUMNS - 0.5  # sub-column offsets within a pixel
_SUBCOLS = (_COLS[:, None] + _SUB[None, :]).reshape(-1)  # every ray's column offset
_SUBALPHA = np.arctan(_SUBCOLS / FOCAL)
_ROWS = np.arange(IMG_SIDE) + 0.5  # pixel row centres, top to bottom
_DISTANCES = NEAR * (REACH / NEAR) ** (np.arange(MARCH) / (MARCH - 1))


def _pose(x, heading):
    """The camera's pitch and roll at a position ``x[:2]`` facing ``heading``: the body pitches
    with the slope of the ground along the heading and rolls with the slope across it, and the
    lens points ``CAMERA_TILT`` below the body; and the eye's height."""
    pos = x[:2]
    gx, gy = jax.grad(lambda p: smooth_height(p))(pos)
    pitch = jnp.arctan(gx * jnp.cos(heading) + gy * jnp.sin(heading)) - CAMERA_TILT
    roll = jnp.arctan(-gx * jnp.sin(heading) + gy * jnp.cos(heading))  # ground rising left
    return pitch, roll, smooth_height(pos) + CAMERA_HEIGHT


def _march(x, heading, offsets, alphas):
    """Per ray (a column offset ``offsets`` at angle ``alphas`` to the right of the heading): the
    ground points at the march distances, the row each projects to, the topmost row reached so
    far (nearer ground hides farther ground), and the azimuth."""
    pos = x[:2]
    pitch, roll, eye = _pose(x, heading)
    az = heading - jnp.asarray(alphas, jnp.float32)
    dirs = jnp.stack([jnp.cos(az), jnp.sin(az)], axis=-1)
    dist = jnp.asarray(_DISTANCES, jnp.float32)
    pts = pos[None, None, :] + dist[None, :, None] * dirs[:, None, :]  # (C, K, 2)
    fwd = dist[None, :] * jnp.cos(jnp.asarray(alphas, jnp.float32))[:, None]
    elev = jnp.arctan((smooth_height(pts) - eye) / fwd)
    cols = jnp.asarray(offsets, jnp.float32)[:, None]
    row = IMG_SIDE / 2 - FOCAL * jnp.tan(elev - pitch) - cols * jnp.tan(roll)
    return pts, jax.lax.cummin(row, axis=1), az, pitch


def ground_view(x, heading):
    """The geometry of the forward view at each pixel's centre, at a position ``x[:2]`` facing
    ``heading`` (radians from +x): per pixel (row-major ``(IMG_SIDE, IMG_SIDE)``) the world point
    the ray through the pixel's centre meets on the ground, the distance to it along the ground
    and whether that ray meets no ground (sky); plus each column's azimuth and the camera's
    pitch. The march is ``render``'s; the picture itself integrates over each pixel."""
    pts, top, az, pitch = _march(x, heading, _COLS, _ALPHA)
    r = jnp.asarray(_ROWS, jnp.float32)
    k = jnp.sum(top[:, None, :] > r[None, :, None], axis=-1)  # (C, R)
    sky = k >= MARCH
    k_hi = jnp.clip(k, 0, MARCH - 1)
    k_lo = jnp.clip(k - 1, 0, MARCH - 1)
    top_hi = jnp.take_along_axis(top, k_hi, axis=1)
    top_lo = jnp.take_along_axis(top, k_lo, axis=1)
    frac = jnp.where(k > 0, jnp.clip((top_lo - r) / jnp.maximum(top_lo - top_hi, 1e-6), 0, 1), 1.0)
    dist = jnp.asarray(_DISTANCES, jnp.float32)
    d = dist[k_lo] + frac * (dist[k_hi] - dist[k_lo])
    dirs = jnp.stack([jnp.cos(az), jnp.sin(az)], axis=-1)
    points = x[None, None, :2] + d[..., None] * dirs[:, None, :]
    return (
        jnp.swapaxes(points, 0, 1),
        jnp.swapaxes(d, 0, 1),
        jnp.swapaxes(sky, 0, 1),
        az,
        pitch,
    )


def _sky_value(rows, az, pitch):
    """The sky's brightness at image rows ``rows`` (any shape broadcast with ``az``): brighter
    toward the horizon and toward the sun's azimuth."""
    elev = jnp.arctan((IMG_SIDE / 2 - rows) / FOCAL) + pitch
    up = jnp.clip(elev / (np.pi / 4), 0.0, 1.0)
    return (
        SKY_HORIZON
        + (SKY_ZENITH - SKY_HORIZON) * up
        + SUN_GLOW * jnp.maximum(jnp.cos(az - SUN_AZIMUTH), 0.0)
    )


def render(x, terrain, heading):
    """The noiseless camera image (IMG_PIXELS values in [0, 1], row-major, row 0 at the top) at a
    position ``x[:2]``, a terrain class and a heading (radians from +x).

    Each pixel's value is the light it collects over its footprint, a smooth function of the
    position and the heading:

    - across the pixel's width, the mean of ``SUBCOLUMNS`` rays (a midpoint rule);
    - down each ray, the exact integral over the pixel's row interval of the ray's profile: each
      ray marches ``MARCH`` distances from ``NEAR`` to ``REACH`` (spaced geometrically) over the
      analytic height field (``smooth_height``) and projects each sample to a row; the running
      topmost row hides farther ground behind nearer, and between two visible samples the
      ground's value varies linearly in the row. Rows below the nearest sample show the nearest
      sample's ground; rows above the topmost show the sky. A pixel straddling the skyline or an
      occluding ridge takes each part in proportion to the rows it covers, so the picture has no
      threshold.

    A ground sample shows the class's ``texture`` at its world point, blended toward the dark
    ``WALL_ALBEDO`` across a wall (``_outside``), times the sunlight there (``light``) and
    ``EXPOSURE``, fading over ``HAZE`` into the sky's brightness at the horizon. The whole visible
    ground carries the texture of the class under the rover. The sky brightens toward the horizon
    and toward the sun's azimuth."""
    pts, top, az, pitch = _march(x, heading, _SUBCOLS, _SUBALPHA)  # (C', K, 2), (C', K)
    outside = _outside(pts)
    albedo = (1 - outside) * texture(pts, terrain) + outside * WALL_ALBEDO
    lit = light(jnp.moveaxis(pts, -1, 0))
    fade = jnp.exp(-jnp.asarray(_DISTANCES, jnp.float32) / HAZE)[None, :]
    value = EXPOSURE * albedo * lit * fade + SKY_HORIZON * (1 - fade)  # (C', K)
    lo_row = jnp.arange(IMG_SIDE, dtype=jnp.float32)[None, :, None]  # each pixel's [r, r + 1]
    hi_row = lo_row + 1.0
    # the visible segments between consecutive samples, from row top[k] up to row top[k + 1]
    t0, t1 = top[:, None, :-1], top[:, None, 1:]  # (C', 1, K - 1), t1 <= t0
    v0, v1 = value[:, None, :-1], value[:, None, 1:]
    a = jnp.maximum(t1, lo_row)
    b = jnp.minimum(t0, hi_row)
    seg = jnp.maximum(b - a, 0.0)
    mid = 0.5 * (a + b)
    span = jnp.maximum(t0 - t1, 1e-9)
    share = jnp.clip((t0 - mid) / span, 0.0, 1.0)
    ground = jnp.sum(seg * (v0 + share * (v1 - v0)), axis=-1)  # (C', R)
    lo_row, hi_row = lo_row[..., 0], hi_row[..., 0]
    near = jnp.maximum(hi_row - jnp.maximum(top[:, :1], lo_row), 0.0)  # below the nearest
    ground = ground + near * value[:, :1]
    far = top[:, -1:]
    sky_cover = jnp.maximum(jnp.minimum(far, hi_row) - lo_row, 0.0)  # above the topmost
    sky_mid = 0.5 * (lo_row + jnp.minimum(far, hi_row))
    sky = sky_cover * _sky_value(sky_mid, az[:, None], pitch)
    img = (ground + sky).reshape(IMG_SIDE, SUBCOLUMNS, IMG_SIDE).mean(axis=1)  # (C, R)
    return jnp.clip(img.T, 0.0, 1.0).reshape(-1)


def to_uint8(image):
    """An image as the 0..255 integers the viewer shows (values clipped to [0, 1])."""
    return np.asarray(np.round(np.clip(np.asarray(image), 0.0, 1.0) * 255), np.uint8)


def gauge_table():
    """The gauge's confusion ``(reading, band)``: the true band with probability ``GAUGE_ACC``,
    the rest spread over the other bands."""
    eye = np.eye(4)
    return GAUGE_ACC * eye + (1 - GAUGE_ACC) / 3 * (1 - eye)


def initial_state(given=None):
    """The world's state at the start: the rover at ``START``, at rest, facing +y (``x`` is
    ``(x, y, cos theta, sin theta, s)``), on the terrain there, with a full battery; ``given`` (a
    partial state keyed by name) overrides entries."""
    full = {
        "terrain": terrain_of(jnp.asarray(START, jnp.float32)),
        "x": jnp.asarray([*START, 0.0, 1.0, 0.0], jnp.float32),
        "energy": jnp.asarray(ENERGY_MAX, jnp.float32),
    }
    full.update({k: jnp.asarray(v, full[k].dtype) for k, v in (given or {}).items()})
    return full


def observe(key, state, fixed=False):
    """The readings at a state: the gauge (the band, right with ``GAUGE_ACC``), the position fix
    (noise ``sd_fix`` at the position) and the camera image: the render along the rover's heading
    (the camera is mounted forward), or along ``CAMERA_HEADING`` with ``fixed`` (a comparison
    camera that does not turn with the rover), plus pixel noise."""
    k_gauge, k_fix, k_img = jax.random.split(key, 3)
    pos = state["x"][:2]
    band = band_of(state["energy"])
    gauge_p = jnp.asarray(gauge_table(), jnp.float32)[:, band]  # the readings given the band
    view = CAMERA_HEADING if fixed else heading_of(state["x"])
    image = render(state["x"], state["terrain"], view)
    return {
        "gauge": jax.random.categorical(k_gauge, jnp.log(gauge_p)).astype(jnp.int32),
        "fix": pos + sd_fix(pos) * jax.random.normal(k_fix, (2,)),
        "img": image + PIXEL_SD * jax.random.normal(k_img, (IMG_PIXELS,)),
    }


def move(key, state, command):
    """One tick of the true dynamics from ``state`` under the ``command`` (acceleration, turn):
    the vehicle's step (``VehicleStep``, the model's own mean map) plus noise of sd ``STATE_SD``,
    the heading renormalised to a unit vector, and where the world differs from the model: an
    empty band cannot drive or turn (the command is zeroed), and the field has walls (the
    position is clipped to them and a rover that hits one stops). Then the energy's drain and
    charge at the old position and speed."""
    a = jnp.clip(command, -1.0, 1.0)
    terrain = terrain_of(state["x"][:2])
    stalled = band_of(state["energy"]) == 0  # an empty rover cannot drive
    a = jnp.where(stalled, 0.0, a)
    x = VehicleStep()(state["x"], (a,)) + jnp.asarray(STATE_SD, jnp.float32) * jax.random.normal(
        key, (5,)
    )
    norm = jnp.sqrt(x[2] ** 2 + x[3] ** 2)
    lo = jnp.asarray([FIELD[0][0], FIELD[1][0]], jnp.float32)
    hi = jnp.asarray([FIELD[0][1], FIELD[1][1]], jnp.float32)
    hit = jnp.any((x[:2] < lo) | (x[:2] > hi))  # the field has walls: clip, and stop
    x = jnp.concatenate([jnp.clip(x[:2], lo, hi), x[2:4] / norm, jnp.where(hit, 0.0, x[4:])])
    speed = jnp.abs(state["x"][4])
    drain, charge = energy_rates(state["x"][:2], terrain, speed)
    energy = jnp.clip(state["energy"] - drain + charge, 0.0, ENERGY_MAX)
    return {"terrain": terrain_of(x[:2]), "x": x, "energy": energy}


def world_system(look=False):
    """``(System, SystemParams)``: the world's declaration, for ``couple``: a black box, its
    ports and their types alone.

    The in-port ``v`` is the velocity command, a float vector of width 2 that the rover's action
    node drives once the two are coupled; the out-ports are the readings, ``fix`` (a float vector
    of width 2, the position fix), ``img`` (a float vector of ``IMG_PIXELS``, the camera image)
    and ``gauge`` (four levels, the energy band's reading), and with ``look`` the terrain class a
    recogniser reads off the image (three levels; ``world_stepper``). What lies between them is
    the hand-written step; the world states no factor and no table, so its description length is
    reported unscored."""
    outs = {"fix": (FLOAT, 2), "img": (FLOAT, IMG_PIXELS), "gauge": (LEVEL, 4)}
    if look:
        outs["look"] = (LEVEL, len(TERRAIN_NAMES))
    return black_box(ins={"v": (FLOAT, 2)}, outs=outs)


def world_stepper(recognise=None, fixed=False):
    """The world's step for ``run``: the carry is the true state (keyed by name), ``obs`` holds
    the velocity command under the name of ``v`` in ``system``, and the readings are returned
    under the names of the emissions. The first call (``obs is None``) starts the world, from
    ``initial_state`` with the carry as its overrides; every other call moves it one tick
    (``move``) and reads it (``observe``). With ``recognise`` (an image to a terrain class) the
    camera's pipeline also reports that class, the ``look`` out-port; ``fixed`` points the camera
    along ``CAMERA_HEADING`` (the comparison camera). It never learns, so
    ``params`` come back as given."""

    def step(system, key, obs, carry, params):
        names = system.names
        name = {full.rpartition(".")[2]: full for full in (*names.emissions, *names.controls)}
        if obs is None:
            state = initial_state(carry)
            readings = observe(key, state, fixed)
        else:
            k_move, k_obs = jax.random.split(key)
            state = move(k_move, carry, obs[name["v"]])
            readings = observe(k_obs, state, fixed)
        if recognise is not None:
            readings["look"] = jnp.asarray(recognise(readings["img"]), jnp.int32)
        return {name[k]: value for k, value in readings.items() if k in name}, state, params

    return step


world_step = world_stepper()


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


def calibration(speed=SURVEY_SPEED):
    """The energy calibration table: energy per tick and per 24-tick option driving at ``speed``
    on flat firm ground, up a steep slope and on rough ground, and parked in the sun, at median
    light and in shade."""
    yield_ = np.clip((LIGHT - LIGHT_FLOOR) / (1.0 - LIGHT_FLOOR), 0.0, 1.0)
    lines = [
        f"top forward speed {top_speed():.3f}; survey speed {speed}",
        f"solar yield: share of the field above the light floor {np.mean(yield_ > 0):.2f}; yield "
        f"median {np.median(yield_):.2f}, 90th pct {np.percentile(yield_, 90):.2f}, max "
        f"{yield_.max():.2f}",
    ]
    s = float(np.percentile(SLOPE, 98))
    for name, d in (
        ("drive, firm flat", BASE_DRAIN + MOVE_DRAIN * speed),
        (f"drive, slope {s:.2f} (98th pct)", BASE_DRAIN + (MOVE_DRAIN + SLOPE_DRAIN * s) * speed),
        ("drive, rough flat", BASE_DRAIN + (MOVE_DRAIN + ROUGH_DRAIN) * speed),
    ):
        lines.append(f"{name}: {-d:+.3f}/tick, {-24 * d:+.2f} per option")
    for name, y in (
        ("parked, 90th pct sun", np.percentile(yield_, 90)),
        ("parked, median light", np.median(yield_)),
        ("parked, shade", 0.0),
    ):
        net = CHARGE_RATE * y - BASE_DRAIN
        lines.append(f"{name}: {net:+.3f}/tick, {24 * net:+.2f} per option")
    return "\n".join(lines)


if __name__ == "__main__":
    world_summary()
    print(calibration())
