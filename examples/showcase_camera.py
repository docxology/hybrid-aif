"""The rover's camera: a frozen decoder and a frozen encoder, trained on synthetic images.

The decoder is an ``MLPConditional`` from the rover's state vector (position and velocity) and a
one-hot terrain to the image's pixels, trained to the renderer of ``showcase_world`` with the
camera's pixel noise. As a likelihood, it is what the engine inverts and plans through. The
encoder is an ``MLPConditional`` from the image to the state vector's mean and scale, trained as
a likelihood: on states drawn uniformly over the field (and a wide uniform velocity range), so its
output is a measurement that carries no prior of its own. Both are trained with optax from a
fixed seed and cached as files, so a re-run loads them instead of training.

    python examples/showcase_camera.py --cache DIR
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
import showcase_world as wd

from hybrid_aif.distributions import MLPConditional

VELOCITY_RANGE = 4.0  # training velocities are uniform on +-VELOCITY_RANGE
DECODER_SHAPE = dict(width=96, depth=2)
ENCODER_SHAPE = dict(width=128, depth=2)
DECODER_STEPS = 12000
ENCODER_STEPS = 12000
BATCH = 256


def sample_states(key, n):
    """``n`` states (position uniform over the field, wide uniform velocity) and terrains."""
    k1, k2, k3 = jax.random.split(key, 3)
    lo = jnp.asarray([wd.FIELD[0][0], wd.FIELD[1][0]])
    hi = jnp.asarray([wd.FIELD[0][1], wd.FIELD[1][1]])
    pos = lo + (hi - lo) * jax.random.uniform(k1, (n, 2))
    vel = VELOCITY_RANGE * (2 * jax.random.uniform(k2, (n, 2)) - 1)
    terrain = jax.random.randint(k3, (n,), 0, 3)
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
        (4, 3), wd.IMG_PIXELS, key=k_init, init_scale=wd.PIXEL_SD, **DECODER_SHAPE
    )

    def loss(model, k):
        k1, k2 = jax.random.split(k)
        states, terrain = sample_states(k1, BATCH)
        images = sample_images(k2, states, terrain)
        onehot = jax.nn.one_hot(terrain, 3)
        return jnp.mean(jax.vmap(lambda y, s, t: model.nll(y, (s, t)))(images, states, onehot))

    net, final = _fit(net, loss, DECODER_STEPS, k_fit)
    return dataclasses.replace(net, frozen=True), final


def train_encoder(key):
    """The encoder, ``image -> state (mean, scale)``, trained as a likelihood and frozen."""
    k_init, k_fit = jax.random.split(key)
    net = MLPConditional.from_shapes(
        (wd.IMG_PIXELS,), 4, key=k_init, init_scale=0.5, **ENCODER_SHAPE
    )

    def loss(model, k):
        k1, k2 = jax.random.split(k)
        states, terrain = sample_states(k1, BATCH)
        images = sample_images(k2, states, terrain)
        return jnp.mean(jax.vmap(lambda y, s: model.nll(s, (y,)))(images, states))

    net, final = _fit(net, loss, ENCODER_STEPS, k_fit)
    return dataclasses.replace(net, frozen=True), final


def evaluate(decoder, encoder, key, n=4000):
    """Held-out numbers: the decoder's reconstruction error and the encoder's calibration."""
    k1, k2 = jax.random.split(key)
    states, terrain = sample_states(k1, n)
    images = sample_images(k2, states, terrain)
    onehot = jax.nn.one_hot(terrain, 3)
    clean = jax.vmap(wd.render)(states, terrain)
    mean, scale = jax.vmap(lambda s, t: decoder.diag_moments((s, t)))(states, onehot)
    enc_mean, enc_scale = jax.vmap(lambda y: encoder.diag_moments((y,)))(images)
    err = enc_mean - states
    out = dict(
        decoder_rmse_vs_clean=float(jnp.sqrt(jnp.mean((mean - clean) ** 2))),
        decoder_scale_mean=float(jnp.mean(scale)),
        pixel_noise_sd=wd.PIXEL_SD,
        encoder_position_rmse=[float(jnp.sqrt(jnp.mean(err[:, i] ** 2))) for i in (0, 1)],
        encoder_position_scale_mean=[float(jnp.mean(enc_scale[:, i])) for i in (0, 1)],
        encoder_velocity_rmse=[float(jnp.sqrt(jnp.mean(err[:, i] ** 2))) for i in (2, 3)],
        encoder_velocity_scale_mean=[float(jnp.mean(enc_scale[:, i])) for i in (2, 3)],
        encoder_z_sd=[float(jnp.std(err[:, i] / enc_scale[:, i])) for i in range(4)],
    )
    return out


def _skeletons():
    key = jax.random.PRNGKey(0)
    dec = MLPConditional.from_shapes(
        (4, 3), wd.IMG_PIXELS, key=key, init_scale=wd.PIXEL_SD, **DECODER_SHAPE
    )
    enc = MLPConditional.from_shapes((wd.IMG_PIXELS,), 4, key=key, init_scale=0.5, **ENCODER_SHAPE)
    return dataclasses.replace(dec, frozen=True), dataclasses.replace(enc, frozen=True)


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
