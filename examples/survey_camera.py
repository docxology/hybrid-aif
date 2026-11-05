"""The survey rover's camera: a frozen decoder and a frozen terrain recogniser, trained on
synthetic images.

The decoder is an ``MLPConditional`` from the rover's state vector (position and velocity) and a
one-hot terrain class to the pixels of ``survey_world``'s forward image, trained with the
camera's pixel noise. As a likelihood, it is what the engine inverts and plans through: the
ground texture marks the terrain and the brightness is the illumination at the position, a smooth
field the decoder learns. The recogniser is an ``MLPCategorical`` from the image to the three
terrain classes, trained by cross-entropy on states drawn uniformly over the field and a uniform
terrain, so its output is a measurement that carries no prior of its own. Both are trained with
optax from a fixed seed and cached as files, so a re-run loads them instead of training.

    python examples/survey_camera.py --cache DIR
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import survey_world as wd

from hybrid_aif.distributions import MLPCategorical, MLPConditional

VELOCITY_SD = 0.5  # training velocities are N(0, VELOCITY_SD^2)
DECODER_SHAPE = dict(width=256, depth=3)
ENCODER_SHAPE = dict(width=128, depth=2)
DECODER_STEPS = 12000
ENCODER_STEPS = 6000
N_TERRAIN = 3
BATCH = 256
RMSE_LIMIT = 0.01  # the decoder's held-out error against the clean image must stay below this


def sample_states(key, n):
    """``n`` states (position uniform over the field, velocity normal) and terrains (uniform)."""
    k1, k2, k3 = jax.random.split(key, 3)
    lo = jnp.asarray([wd.FIELD[0][0], wd.FIELD[1][0]])
    hi = jnp.asarray([wd.FIELD[0][1], wd.FIELD[1][1]])
    pos = lo + (hi - lo) * jax.random.uniform(k1, (n, 2))
    vel = VELOCITY_SD * jax.random.normal(k2, (n, 2))
    terrain = jax.random.randint(k3, (n,), 0, N_TERRAIN)
    return jnp.concatenate([pos, vel], axis=1), terrain


def sample_images(key, states, terrain):
    """Noisy renderer images of the given states."""
    clean = jax.vmap(wd.render)(states, terrain)
    return clean + wd.PIXEL_SD * jax.random.normal(key, clean.shape)


def _fit(model, loss_fn, steps, key, lr=2e-3):
    """Minimise ``loss_fn(model, key)`` with Adam, cosine-decayed; returns the model and loss."""
    schedule = optax.cosine_decay_schedule(lr, steps)
    opt = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(schedule))
    state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(model, state, k):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, k)
        updates, state = opt.update(grads, state, eqx.filter(model, eqx.is_inexact_array))
        return eqx.apply_updates(model, updates), state, loss

    loss = jnp.nan
    for k in jax.random.split(key, steps):
        model, state, loss = step(model, state, k)
    return model, float(loss)


def train_decoder(key):
    """The decoder, ``(state, terrain one-hot) -> image``, frozen."""
    k_init, k_fit = jax.random.split(key)
    net = MLPConditional.from_shapes(
        (4, N_TERRAIN), wd.IMG_PIXELS, key=k_init, init_scale=wd.PIXEL_SD, **DECODER_SHAPE
    )

    def loss(model, k):
        k1, k2 = jax.random.split(k)
        states, terrain = sample_states(k1, BATCH)
        images = sample_images(k2, states, terrain)
        onehot = jax.nn.one_hot(terrain, N_TERRAIN)
        return jnp.mean(jax.vmap(lambda y, s, t: model.nll(y, (s, t)))(images, states, onehot))

    net, final = _fit(net, loss, DECODER_STEPS, k_fit)
    return dataclasses.replace(net, frozen=True), final


def _new_encoder(key):
    return MLPCategorical.from_shapes((wd.IMG_PIXELS,), 1, N_TERRAIN, key=key, **ENCODER_SHAPE)


def _freeze_encoder(net):
    return dataclasses.replace(net, trunk=dataclasses.replace(net.trunk, frozen=True))


def terrain_probs(encoder, image):
    """The recogniser's probability vector over terrain classes for one image."""
    return encoder.probs((image,))[0]


def train_encoder(key):
    """The recogniser, ``image -> terrain probabilities``, trained by cross-entropy and frozen."""
    k_init, k_fit = jax.random.split(key)
    net = _new_encoder(k_init)

    def loss(model, k):
        k1, k2 = jax.random.split(k)
        states, terrain = sample_states(k1, BATCH)
        images = sample_images(k2, states, terrain)
        onehot = jax.nn.one_hot(terrain, N_TERRAIN)[:, None, :]
        return -jnp.mean(jax.vmap(lambda y, c: model.log_prob(c, (y,)))(images, onehot))

    net, final = _fit(net, loss, ENCODER_STEPS, k_fit)
    return _freeze_encoder(net), final


def evaluate(decoder, encoder, key, n=2000):
    """Held-out numbers: the decoder's reconstruction error (asserted below ``RMSE_LIMIT``) and
    the recogniser's accuracy and mean log probability of the true terrain."""
    k1, k2 = jax.random.split(key)
    states, terrain = sample_states(k1, n)
    images = sample_images(k2, states, terrain)
    onehot = jax.nn.one_hot(terrain, N_TERRAIN)
    clean = jax.vmap(wd.render)(states, terrain)
    mean, scale = jax.vmap(lambda s, t: decoder.diag_moments((s, t)))(states, onehot)
    probs = jax.vmap(lambda y: terrain_probs(encoder, y))(images)
    true_p = jnp.take_along_axis(probs, terrain[:, None], axis=1)[:, 0]
    out = dict(
        decoder_rmse_vs_clean=float(jnp.sqrt(jnp.mean((mean - clean) ** 2))),
        decoder_scale_mean=float(jnp.mean(scale)),
        pixel_noise_sd=wd.PIXEL_SD,
        recogniser_accuracy=float(jnp.mean(jnp.argmax(probs, axis=1) == terrain)),
        recogniser_mean_log_prob_true=float(jnp.mean(jnp.log(true_p))),
    )
    assert out["decoder_rmse_vs_clean"] < RMSE_LIMIT, out
    return out


def _skeletons():
    key = jax.random.PRNGKey(0)
    dec = MLPConditional.from_shapes(
        (4, N_TERRAIN), wd.IMG_PIXELS, key=key, init_scale=wd.PIXEL_SD, **DECODER_SHAPE
    )
    enc = _new_encoder(key)
    return dataclasses.replace(dec, frozen=True), _freeze_encoder(enc)


def load_or_train(cache, seed=0, verbose=True):
    """``(decoder, encoder, report)``: read from ``cache`` when its files exist, else train
    both from ``seed`` and write them."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    files = {n: cache / f"camera_{n}.eqx" for n in ("decoder", "encoder")}
    report_path = cache / "camera_report.json"
    dec_skel, enc_skel = _skeletons()
    if all(f.exists() for f in files.values()) and report_path.exists():
        dec = eqx.tree_deserialise_leaves(files["decoder"], dec_skel)
        enc = eqx.tree_deserialise_leaves(files["encoder"], enc_skel)
        return dec, enc, json.loads(report_path.read_text())
    k_dec, k_enc, k_eval = jax.random.split(jax.random.PRNGKey(seed), 3)
    t0 = time.time()
    dec, dec_loss = train_decoder(k_dec)
    t1 = time.time()
    enc, enc_loss = train_encoder(k_enc)
    t2 = time.time()
    report = evaluate(dec, enc, k_eval)
    report.update(
        decoder_final_nll=dec_loss,
        encoder_final_nll=enc_loss,
        decoder_train_seconds=t1 - t0,
        encoder_train_seconds=t2 - t1,
        image_side=wd.IMG_SIDE,
        decoder_shape=DECODER_SHAPE,
        encoder_shape=ENCODER_SHAPE,
    )
    eqx.tree_serialise_leaves(files["decoder"], dec)
    eqx.tree_serialise_leaves(files["encoder"], enc)
    report_path.write_text(json.dumps(report, indent=1))
    if verbose:
        print(json.dumps(report, indent=1), flush=True)
    return dec, enc, report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    load_or_train(args.cache, args.seed)
