"""The slow mission tier of the two-tier rover showcase.

Energy (four bands), three rocks of five internal levels each, the zone the rover is in, and a
seven-level option node (visit a rock zone, sample a rock, charge at the dock). The option node
is the mission's control: it is held for the option's duration, sets the rover's position target
(a per-step target conditional on the option) and is sized by its per-level durations.
"""

import jax.numpy as jnp
import numpy as np

from hybrid_aif import Categorical, ConditionalCategorical
from hybrid_aif.system import System, SystemParams

ZONES = {
    "start": (0.0, 0.0),
    "rock_1": (4.0, 0.0),
    "rock_2": (3.5, 3.0),
    "rock_3": (0.0, 3.0),
    "dock": (0.5, -1.2),
}
ZONE_NAMES = list(ZONES)
OPTION_NAMES = ["visit_1", "visit_2", "visit_3", "sample_1", "sample_2", "sample_3", "charge"]
OPTION_ZONE = [1, 2, 3, 1, 2, 3, 4]  # the zone each option lands in
NOPT = 7
DURATIONS = (8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0)  # rounds of the fast clock per option
ENERGY_NAMES = ["empty", "low", "mid", "full"]
# A modality reads the state after the transition, so a payoff over (rock, option) can tell a
# sampled good rock from a sampled bad one only if the rock carries that distinction.
ROCK_LEVELS = ["good", "bad", "sampled_good", "sampled_bad", "depleted"]
ROCK_NAMES = ["good", "bad", "sampled"]  # the reported three: levels 2-4 collapse to sampled
ROCK_COLLAPSE = (0, 1, 2, 2, 2)
PAYOFF_NAMES = ["nothing", "reward", "penalty"]
BAND_PER_ROUND = 0.12  # expected energy bands lost per round of travel
MIXTURE_K = 3  # one component per terrain mode of the switching position
MISSION_MODALITIES = ("gauge", "zone_obs", "check_1", "check_2", "check_3", "payoff")


def _cc(table, parent_ndims):
    return ConditionalCategorical(
        log_table=jnp.log(jnp.clip(jnp.asarray(table, jnp.float32), 1e-12)),
        parent_ndims=parent_ndims,
    )


def energy_table():
    """table[next, prev, option]: bands lost with the option's duration; charge refills."""
    t = np.zeros((4, 4, NOPT))
    for o in range(NOPT):
        if o == 6:
            for prev in range(4):
                t[3, prev, o] = 0.9
                t[prev, prev, o] += 0.1
            continue
        x = BAND_PER_ROUND * DURATIONS[o]
        lo, frac = int(np.floor(x)), x - np.floor(x)
        for prev in range(4):
            t[max(prev - lo, 0), prev, o] += 1 - frac
            t[max(prev - lo - 1, 0), prev, o] += frac
    return t


def rock_table(k, regrow=0.16, swap=0.02):
    """table[next, prev, option] for rock k. Sampling it (option 3 + k) moves a good rock to
    sampled_good and a bad one to sampled_bad; any sampled rock decays to depleted a round later,
    and a depleted rock slowly returns to good or bad."""
    t = np.zeros((5, 5, NOPT))
    for o in range(NOPT):
        for prev in range(5):
            if o == 3 + k:
                nxt = {0: 2, 1: 3}.get(prev, 4)
                t[nxt, prev, o] = 1.0
            elif prev in (2, 3):
                t[4, prev, o] = 1.0
            elif prev == 4:
                t[4, prev, o] = 1 - regrow
                t[0, prev, o] = regrow / 2
                t[1, prev, o] = regrow / 2
            else:
                t[prev, prev, o] = 1 - swap
                t[1 - prev, prev, o] = swap
    return t


def zone_table(stay=0.92):
    t = np.zeros((5, NOPT))
    for o in range(NOPT):
        t[OPTION_ZONE[o], o] = stay
        t[0, o] = 1 - stay
    return t


def check_table(k, accuracy=0.9):
    """table[reading, rock level, zone]: informative about a good or bad rock k in rock k's own
    zone only; flat elsewhere."""
    t = np.full((2, 5, 5), 0.5)
    t[:, 0, 1 + k] = [accuracy, 1 - accuracy]
    t[:, 1, 1 + k] = [1 - accuracy, accuracy]
    return t


def payoff_table():
    """table[payoff, rock_1, rock_2, rock_3, option]: the reading after the option. Sampling rock
    k and finding it in the sampled_good (sampled_bad) state pays the reward (penalty)."""
    t = np.zeros((3, 5, 5, 5, NOPT))
    t[0] = 1.0
    for k in range(3):
        for idx in np.ndindex(5, 5, 5):
            lvl = idx[k]
            if lvl in (2, 3):
                t[(slice(None), *idx, 3 + k)] = [0.0, 1.0, 0.0] if lvl == 2 else [0.0, 0.0, 1.0]
    return t


def mission(sharp="steep", epistemic=True, p_good=0.55, reward=2.0, penalty=2.0, policy_len=3):
    """The mission System and its params."""
    energy_pref = {
        "steep": [-8.0, -1.5, 0.0, 0.0],
        "mild": [-2.0, -0.5, 0.0, 0.0],
    }[sharp]
    names = ("energy", "rock_1", "rock_2", "rock_3", "zone", "u_option")
    system = System(
        A_deps=((0,), (4,), (1, 4), (2, 4), (3, 4), (1, 2, 3, 6)),
        B_deps=((0, 5), (1, 5), (2, 5), (3, 5), (5,), ()),
        controls=(5,),
        action_names=("option",),
        factor_names=names,
        modality_names=MISSION_MODALITIES,
        policy_len=policy_len,
        mixture_k=MIXTURE_K,
        mixture_schedule="mode_aligned",
        planning_belief="prior",
        epistemic=("obs:check_1", "obs:check_2", "obs:check_3") if epistemic else False,
    )
    gauge = 0.88 * np.eye(4) + 0.04
    params = SystemParams(
        likelihood={
            "gauge": _cc(gauge, 1),
            "zone_obs": _cc(np.eye(5), 1),
            "check_1": _cc(check_table(0), 2),
            "check_2": _cc(check_table(1), 2),
            "check_3": _cc(check_table(2), 2),
            "payoff": _cc(payoff_table(), 4),
        },
        transition={
            "energy": _cc(energy_table(), 2),
            "rock_1": _cc(rock_table(0), 2),
            "rock_2": _cc(rock_table(1), 2),
            "rock_3": _cc(rock_table(2), 2),
            "zone": _cc(zone_table(), 1),
        },
        initial_state_prior={
            "energy": Categorical.from_probs(jnp.asarray([0.0, 0.0, 0.05, 0.95]) + 1e-6),
            **{
                f"rock_{k + 1}": Categorical.from_probs(
                    jnp.asarray([p_good, 1 - p_good, 0, 0, 0]) + 1e-6
                )
                for k in range(3)
            },
            "zone": Categorical.from_probs(jnp.asarray([1.0, 0, 0, 0, 0]) + 1e-6),
        },
        state_targets={"energy": Categorical(logits=jnp.asarray(energy_pref))},
        obs_targets={"payoff": Categorical(logits=jnp.asarray([0.0, reward, -penalty]))},
        durations={"u_option": jnp.asarray(DURATIONS)},
    )
    return system, params
