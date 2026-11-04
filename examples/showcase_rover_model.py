"""The fast rover tier of the two-tier rover showcase.

A terrain-switched 4-D position and velocity under a velocity command, three sensing modalities
(a position fix whose noise is a levelled map over position and the sensor node, a terrain cue,
and a camera image whose likelihood is a frozen neural decoder) and a position target that is
conditional on the mission's option node. An optional recognition entry replaces the engine's
inversion of the image with a frozen encoder's measurement of the position.
"""

import jax.numpy as jnp
import numpy as np
import showcase_mission as mm
import showcase_world as wd

from hybrid_aif import (
    Categorical,
    ConditionalCategorical,
    ConditionalGaussian,
    LinearGaussianObservation,
)
from hybrid_aif.controls import Box
from hybrid_aif.distributions import MixtureGaussian
from hybrid_aif.factors import GridNoiseMap, LinearGaussianTransition, Recognition
from hybrid_aif.system import System, SystemParams

N_TERRAIN = 3
N_SENSOR = 2
GRID = 25  # noise-map nodes per axis
SENSOR_NAMES = ["short exposure", "long exposure"]
TERRAIN_NAMES = ["firm", "sand", "ice"]


def _cc(table, parent_ndims):
    return ConditionalCategorical(
        log_table=jnp.log(jnp.clip(jnp.asarray(table, jnp.float32), 1e-12)),
        parent_ndims=parent_ndims,
    )


def fix_noise_map():
    """A GridNoiseMap with a leading sensor-level axis: values (2, G, G, 2, 2)."""
    xs = np.linspace(*wd.FIELD[0], GRID)
    ys = np.linspace(*wd.FIELD[1], GRID)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    values = np.zeros((N_SENSOR, GRID, GRID, 2, 2), np.float32)
    for s in range(N_SENSOR):
        sd = np.asarray(wd.fix_sd(s, jnp.asarray(px), jnp.asarray(py)))
        values[s, :, :, 0, 0] = sd
        values[s, :, :, 1, 1] = sd
    low = (wd.FIELD[0][0], wd.FIELD[1][0])
    high = (wd.FIELD[0][1], wd.FIELD[1][1])
    return GridNoiseMap(jnp.asarray(values), low, high, axes=(0, 1))


def rover(decoder, encoder=None, precision=4.0, epistemic=True, policy_len=3, camera=True):
    """The rover System and params. ``decoder`` is the camera's likelihood; with ``encoder``, a
    recognition entry measures the position from the image. ``camera=False`` leaves the image
    out of the model."""
    # factors: terrain 0, x 1, v 2 (float control), sensor 3, u_opt 4 (the mission's option);
    # action nodes: a_v 5, a_sensor 6, a_opt 7
    modalities = ["fix", "cue"] + (["img"] if camera else [])
    A_deps = [(1, 6), (0,)] + ([(1, 0)] if camera else [])
    system = System(
        A_deps=tuple(A_deps),
        B_deps=((0,), (0, 2, 3), (), (), ()),
        controls=(2, 3, 4),
        action_names=("a_v", "a_sensor", "a_opt"),
        factor_names=("terrain", "x", "v", "sensor", "u_opt"),
        modality_names=tuple(modalities),
        C_deps={"x": ("u_opt",)},
        policy_len=policy_len,
        mixture_k=mm.MIXTURE_K,
        mixture_schedule="mode_aligned",
        planning_belief="prior",
        epistemic=tuple(f"obs:{m}" for m in modalities) if epistemic else False,
        terms=("uncertainty:x",),
        recognition=(Recognition("x", ("obs:img",)),) if camera and encoder is not None else (),
    )
    M, S = N_TERRAIN, N_SENSOR
    F = np.zeros((M, S, 4, 4))
    G = np.zeros((M, S, 4, 2))
    loc = np.zeros((M, S, 4))
    chol = np.zeros((M, S, 4, 4))
    for m in range(M):
        for s in range(S):
            F[m, s], G[m, s], loc[m, s], chol[m, s] = wd.dynamics(m, s)
    stick = np.log(0.92 * np.eye(M) + 0.04)[:, :, None]
    x0 = MixtureGaussian(
        log_weights=jnp.log(jnp.ones(M) / M),
        means=jnp.broadcast_to(jnp.asarray([*wd.START, 0.0, 0.0]), (M, 4)),
        chols=jnp.broadcast_to(0.5 * jnp.eye(4), (M, 4, 4)),
    )
    n_opt = len(mm.OPTION_NAMES)
    sd_pos = 1.0 / np.sqrt(precision)
    tgt_loc = np.zeros((n_opt, 4))
    tgt_chol = np.zeros((n_opt, 4, 4))
    for o in range(n_opt):
        tgt_loc[o, :2] = mm.ZONES[mm.ZONE_NAMES[mm.OPTION_ZONE[o]]]
        tgt_chol[o] = np.diag([sd_pos, sd_pos, 2.0, 2.0])
    likelihood = {
        "fix": LinearGaussianObservation(H=jnp.eye(4)[:2], c=jnp.zeros(2), noise=fix_noise_map()),
        "cue": _cc(wd.CUE, 1),
    }
    if camera:
        likelihood["img"] = decoder
    params = SystemParams(
        likelihood=likelihood,
        transition={
            "terrain": ConditionalCategorical(
                log_table=jnp.asarray(stick, jnp.float32), parent_ndims=2
            ),
            "x": LinearGaussianTransition(
                F=jnp.asarray(F, jnp.float32),
                G=jnp.asarray(G, jnp.float32),
                innovations=ConditionalGaussian(
                    loc=jnp.asarray(loc, jnp.float32),
                    chol=jnp.asarray(chol, jnp.float32),
                    parent_ndims=2,
                ),
            ),
        },
        initial_state_prior={"terrain": Categorical(logits=jnp.zeros(M)), "x": x0},
        control={"v": Box(-jnp.ones(2), jnp.ones(2))},
        state_targets={
            "x": ConditionalGaussian(
                loc=jnp.asarray(tgt_loc, jnp.float32),
                chol=jnp.asarray(tgt_chol, jnp.float32),
                parent_ndims=1,
            )
        },
        recognition={"x": encoder} if camera and encoder is not None else {},
    )
    return system, params
