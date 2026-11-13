"""The survey rover's camera: a frozen decoder and a frozen terrain recogniser, trained on
synthetic images.

The camera is mounted forward on the rover and looks along its heading, so a picture is a function
of the position, the heading and the terrain: the skyline and the ground's shading set by the
height map and the sun, and the ground textured in world coordinates by the class under the rover.
The decoder is an ``MLPConditional`` from the rover's state ``(x, y, cos theta, sin theta, s)`` and
a one-hot terrain class to the pixels; it is trained with the speed at zero and its first-layer
weights on the speed zeroed after, so it reads the position, the heading (as the unit vector, smooth
in every direction and at rest) and the terrain. ``FIXED_VIEW`` trains the comparison camera
instead, fixed along the world's ``CAMERA_HEADING`` (experiment B, not the showcase).

The decoder's noise is calibrated (``calibrate``): it is fit to the decoder's residuals on noisy
images (sensor noise and the decoder's own error together) and inflated by ``sqrt(N / N_eff)``,
where ``N_eff`` is the effective number of independent pixels in those residuals
(``effective_pixels``, bias-corrected for the sample correlation's own noise), times
``EMPIRICAL_TEMPER`` (one: no factor beyond the measurement).

The image's evidence is then tempered for the decoder's own error (``TEMPERED``): the decoder's
error against the clean picture is not pixel noise but a coherent shift of the position the image
points to, which the diagonal noise does not carry. The rule, measured on held-out states
(``tempering``): at each state the shift ``b`` that the decoder's error alone causes in a single
image's position estimate, in units of that estimate's own covariance, ``m^2 = b' I b`` (``I`` the
image's information about the position); with pixel noise alone a single image's squared error has
mean 2 (two coordinates), so the image's precision is scaled by ``alpha = 2 / (2 + E[m^2])``. The
declared likelihood is the tempered one: its noise sd is the calibrated sd over ``sqrt(alpha)``
(``temper`` refits the scale half of the head to it). For the showcase's camera E[m^2] is 4.68 and
alpha 0.300, so the image's noise sd is 1.83 times the calibrated one. This is a declared stand-in
for a structured decoder-error term in the likelihood. The recogniser is an
``MLPCategorical`` from the image to the three terrain classes, trained by cross-entropy on states
drawn over the field and a uniform terrain, so its output is a measurement that carries no prior
of its own; ``evaluate`` measures its confusion table (the reading given the true class) on
held-out images, the likelihood of a model that observes the recogniser's reading. Both are trained
with optax from a fixed seed on a fixed set of rendered images (fresh pixel noise every batch) and
cached as files, so a re-run loads them instead of training.

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
import numpy as np
import optax
import survey_world as wd

from hybrid_aif.distributions import MLPCategorical, MLPConditional

#  the large field (``survey_world.FIELD_SIZE``) holds nine times the scene, so its decoder is
#  wider and trained on more images for longer (see the build record for the measurements)
LARGE = wd.FIELD_SIZE == "large"
DECODER_SHAPE = dict(width=1024 if LARGE else 512, depth=4, activation="silu")
ENCODER_SHAPE = dict(width=256, depth=2)
DECODER_STEPS = 400000 if LARGE else 80000
ENCODER_STEPS = 20000
CALIBRATION_STEPS = 6000  # steps of the noise calibration (the scale head alone)
CALIBRATION_IMAGES = 16000  # held-out images the effective pixel count is estimated from
#  the sample sizes at which the effective pixel count is also reported, to show it has converged
CONVERGENCE_IMAGES = (2000, 4000, 8000, 16000)
#  an empirical factor on the calibrated noise, beyond the measured correlation; 1 means none
EMPIRICAL_TEMPER = 1.0
#  the comparison camera (experiment B): fixed along the world's CAMERA_HEADING, not the heading
FIXED_VIEW = False
#  whether the image's evidence is tempered for the decoder's error (``tempering``, ``temper``)
TEMPERED = True
TEMPER_STATES = 2000  # held-out states the tempering rule is measured on
TEMPER_KEY = 12345  # their key, used by no training draw
TEMPER_FIT_STATES = 20000  # states the tempered noise head is fitted on
_ACTIVATIONS = {"silu": jax.nn.silu, "gelu": jax.nn.gelu, "tanh": jnp.tanh, "relu": jax.nn.relu}
DATASET = 400000 if LARGE else 80000  # rendered training images
N_TERRAIN = 3
BATCH = 256
LEARNING_RATE = 1e-3
# the decoder's held-out error against the clean image must stay below this: a quarter above the
# camera's pixel noise (the texture's finest waves and the sharpest skylines are left to the noise)
RMSE_LIMIT = 1.25 * wd.PIXEL_SD
#  the state's spread over the training set, folded into the decoder's first layer so that every
#  input enters at unit scale: position, heading pair, speed, then the terrain's one-hot
#  (on the large field the position enters at 0.6, a third of its sd over the field: a network
#  reading it at unit scale resolves the scene's detail far less well, by the measurements of the
#  build record)
INPUT_LOC = np.asarray([np.mean(wd.FIELD[0]), np.mean(wd.FIELD[1]), 0, 0, 0, 0, 0, 0])
_POSITION_SCALE = 0.6 if LARGE else 1.7
INPUT_SCALE = np.asarray([_POSITION_SCALE, _POSITION_SCALE, 0.7, 0.7, 0.2, 1.0, 1.0, 1.0])


def sample_states(key, n):
    """``n`` states ``(x, y, cos theta, sin theta, s)`` and terrains (uniform): the position
    uniform over the field, the heading uniform, the speed zero (the picture does not depend on
    it)."""
    k1, k2, k3 = jax.random.split(key, 3)
    lo = jnp.asarray([wd.FIELD[0][0], wd.FIELD[1][0]])
    hi = jnp.asarray([wd.FIELD[0][1], wd.FIELD[1][1]])
    pos = lo + (hi - lo) * jax.random.uniform(k1, (n, 2))
    theta = jax.random.uniform(k3, (n,), minval=-jnp.pi, maxval=jnp.pi)
    heading = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=1)
    terrain = jax.random.randint(k2, (n,), 0, N_TERRAIN)
    return jnp.concatenate([pos, heading, jnp.zeros((n, 1))], axis=1), terrain


def view_of(x):
    """The direction the camera looks at a state: the rover's heading, or the fixed
    ``CAMERA_HEADING`` with ``FIXED_VIEW``."""
    return wd.CAMERA_HEADING if FIXED_VIEW else wd.heading_of(x)


def render_many(states, terrain, chunk=4096):
    """The clean camera images of many states (along ``view_of``), rendered in chunks."""
    f = jax.jit(jax.vmap(lambda x, t: wd.render(x, t, view_of(x))))
    parts = [f(states[i : i + chunk], terrain[i : i + chunk]) for i in range(0, len(states), chunk)]
    return jnp.concatenate(parts)


def dataset(key, n=DATASET):
    """``(states, terrain, clean images)``: ``n`` sampled states and their rendered images."""
    states, terrain = sample_states(key, n)
    return states, terrain, render_many(states, terrain)


def _fit(model, loss_fn, steps, key, lr=LEARNING_RATE):
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


def _standardised(net):
    """``net`` with its first layer reading standardised inputs: ``(input - INPUT_LOC) /
    INPUT_SCALE`` folded into the layer's weights and bias."""
    first = net.layers[0]
    scale = jnp.asarray(1.0 / INPUT_SCALE, first.weight.dtype)
    weight = first.weight * scale[None, :]
    bias = first.bias - weight @ jnp.asarray(INPUT_LOC, first.weight.dtype)
    first = eqx.tree_at(lambda m: (m.weight, m.bias), first, (weight, bias))
    return eqx.tree_at(lambda m: m.layers[0], net, first)


def _new_decoder(key):
    net = MLPConditional.from_shapes(
        (5, N_TERRAIN), wd.IMG_PIXELS, key=key, init_scale=wd.PIXEL_SD, **DECODER_SHAPE
    )
    return _standardised(net)


def _picture_inputs(net):
    """``net`` with its first layer's weights zeroed on the inputs the picture does not depend on:
    the speed, and with ``FIXED_VIEW`` the heading too."""
    first = net.layers[0]
    cols = slice(2, 5) if FIXED_VIEW else slice(4, 5)
    return eqx.tree_at(lambda m: m.layers[0].weight, net, first.weight.at[:, cols].set(0.0))


def _batch(data, key):
    """A training batch from ``data``: states, one-hot terrains and noisy images."""
    states, terrain, clean = data
    k1, k2 = jax.random.split(key)
    idx = jax.random.randint(k1, (BATCH,), 0, states.shape[0])
    noisy = clean[idx] + wd.PIXEL_SD * jax.random.normal(k2, (BATCH, clean.shape[1]))
    return states[idx], terrain[idx], noisy


def train_decoder(key, data):
    """The decoder, ``(state, terrain one-hot) -> image``, frozen."""
    k_init, k_fit = jax.random.split(key)
    net = _new_decoder(k_init)

    def loss(model, k):
        states, terrain, images = _batch(data, k)
        onehot = jax.nn.one_hot(terrain, N_TERRAIN)
        return jnp.mean(jax.vmap(lambda y, s, t: model.nll(y, (s, t)))(images, states, onehot))

    net, final = _fit(net, loss, DECODER_STEPS, k_fit)
    return dataclasses.replace(_picture_inputs(net), frozen=True), final


def _new_encoder(key):
    return MLPCategorical.from_shapes((wd.IMG_PIXELS,), 1, N_TERRAIN, key=key, **ENCODER_SHAPE)


def _freeze_encoder(net):
    return dataclasses.replace(net, trunk=dataclasses.replace(net.trunk, frozen=True))


def terrain_probs(encoder, image):
    """The recogniser's probability vector over terrain classes for one image."""
    return encoder.probs((image,))[0]


def train_encoder(key, data):
    """The recogniser, ``image -> terrain probabilities``, trained by cross-entropy and frozen."""
    k_init, k_fit = jax.random.split(key)
    net = _new_encoder(k_init)

    def loss(model, k):
        _, terrain, images = _batch(data, k)
        onehot = jax.nn.one_hot(terrain, N_TERRAIN)[:, None, :]
        return -jnp.mean(jax.vmap(lambda y, c: model.log_prob(c, (y,)))(images, onehot))

    net, final = _fit(net, loss, ENCODER_STEPS, k_fit)
    return _freeze_encoder(net), final


def _held_out(key, n):
    """Fresh held-out states, terrains, clean and noisy images, one-hot terrains."""
    k1, k2 = jax.random.split(key)
    states, terrain = sample_states(k1, n)
    clean = render_many(states, terrain)
    images = clean + wd.PIXEL_SD * jax.random.normal(k2, clean.shape)
    return states, terrain, clean, images, jax.nn.one_hot(terrain, N_TERRAIN)


def effective_pixels(decoder, key, n=CALIBRATION_IMAGES, sizes=CONVERGENCE_IMAGES):
    """``(N_eff, N, by_size)``: the effective number of independent pixels in the decoder's
    residuals on ``n`` held-out noisy images. The residuals are standardised by the decoder's own
    scale and their correlation matrix ``R`` over the ``N`` pixels is estimated; ``N_eff = N**2 /
    sum(R**2)`` is the participation ratio of ``R``'s eigenvalues (``N`` for independent pixels, 1
    for pixels that move together). A sample correlation from ``m`` images has ``E[r**2]`` of about
    ``1 / (m - 1)`` on every off-diagonal entry even for independent pixels, so the off-diagonal
    sum is reduced by ``N (N - 1) / (m - 1)`` (floored at zero) before the ratio is taken.
    ``by_size`` gives the raw and the corrected estimate on the first ``m`` images for each ``m``
    in ``sizes``, to show the corrected one has converged."""
    states, _, _, images, onehot = _held_out(key, n)
    mean, scale = jax.vmap(lambda s, t: decoder.diag_moments((s, t)))(states, onehot)
    z_all = np.asarray((images - mean) / scale, np.float64)
    pixels = z_all.shape[1]

    def estimate(z):
        m = z.shape[0]
        corr = np.corrcoef(z - z.mean(axis=0), rowvar=False)
        off = np.sum(corr**2) - pixels
        corrected = max(off - pixels * (pixels - 1) / (m - 1), 0.0)
        return float(pixels**2 / (pixels + off)), float(pixels**2 / (pixels + corrected))

    by_size = {}
    for m in sorted(set(sizes) | {n}):
        if m <= n:
            raw, corrected = estimate(z_all[:m])
            by_size[str(m)] = dict(raw=raw, corrected=corrected)
    return by_size[str(n)]["corrected"], pixels, by_size


def calibrate(decoder, data, key, inflation, steps=CALIBRATION_STEPS):
    """The decoder with its noise fit to its residuals on the training images (noisy: sensor
    noise and the decoder's error together) inflated by ``inflation``: the scale half of the head
    layer is trained alone, on the Gaussian likelihood of ``inflation * (y - mean)`` with the mean
    held, so the scale learns ``inflation`` times the predictive error, as a function of the
    position and the terrain."""
    net = dataclasses.replace(decoder, frozen=False)
    spec = jax.tree_util.tree_map(lambda _: False, net)
    spec = eqx.tree_at(
        lambda m: (m.layers[-1].weight, m.layers[-1].bias), spec, replace=(True, True)
    )

    def loss(model, k):
        states, terrain, images = _batch(data, k)
        onehot = jax.nn.one_hot(terrain, N_TERRAIN)
        mean, scale = jax.vmap(lambda s, t: model.diag_moments((s, t)))(states, onehot)
        r = inflation * (images - jax.lax.stop_gradient(mean))
        return jnp.mean(jnp.sum(0.5 * (r / scale) ** 2 + jnp.log(scale), axis=1))

    trainable, fixed = eqx.partition(net, spec)

    def part_loss(t, k):
        return loss(eqx.combine(t, fixed), k)

    trainable, final = _fit(trainable, part_loss, steps, key)
    return dataclasses.replace(eqx.combine(trainable, fixed), frozen=True), final


def shift_sizes(decoder, states, terrain):
    """Per state, ``m^2 = b' I b``: ``I = J' W J`` the image's information about the position
    (``J`` the decoder mean's derivative in ``x, y``, ``W`` its noise precision per pixel) and
    ``b = I^-1 J' W e`` the shift of the single-image position estimate that the decoder's error
    ``e`` (its mean less the clean render) alone causes, in units of that estimate's covariance."""

    def one(z, t):
        onehot = jax.nn.one_hot(t, N_TERRAIN)

        def head(xy):
            return decoder.diag_moments((jnp.concatenate([xy, z[2:]]), onehot))

        mean, scale = head(z[:2])
        jac = jax.jacfwd(lambda xy: head(xy)[0])(z[:2])
        clean = wd.render(z, t, view_of(z))
        w = 1.0 / scale**2
        info = jac.T @ (w[:, None] * jac)
        b = jnp.linalg.solve(info, jac.T @ (w * (mean - clean)))
        return b @ info @ b

    return jax.jit(jax.vmap(one))(states, terrain)


def tempering(decoder, n=TEMPER_STATES, key=TEMPER_KEY):
    """The tempering of the image's evidence for the decoder's error, measured on ``n`` held-out
    states drawn as the training states are (``sample_states``, key ``key``): ``alpha = 2 / (2 +
    E[m^2])`` with ``m^2`` from ``shift_sizes``; a dict of alpha and the statistics of ``m^2``."""
    states, terrain = sample_states(jax.random.PRNGKey(key), n)
    m2 = np.asarray(shift_sizes(decoder, states, terrain), np.float64)
    return dict(
        alpha=float(2.0 / (2.0 + m2.mean())),
        rule="alpha = 2 / (2 + E[m^2]), m^2 = b' I b, the decoder error's shift of a single "
        "image's position estimate in units of its covariance, on held-out states",
        states=n,
        key=key,
        mean_m2=float(m2.mean()),
        median_m2=float(np.median(m2)),
        p90_m2=float(np.percentile(m2, 90)),
    )


def _hidden(net, parents):
    """The decoder's last hidden layer at ``parents``, the features its head layer reads."""
    h = jnp.concatenate(parents)
    act = _ACTIVATIONS[net.activation]
    for i, layer in enumerate(net.layers[:-1]):
        h = layer(h)
        if net.norms:
            h = net.norms[i](h)
        h = act(h)
    return h


def temper(decoder, alpha, key, n=TEMPER_FIT_STATES):
    """The decoder with its noise sd divided by ``sqrt(alpha)``, so that its likelihood is the
    tempered one (the image's precision times ``alpha``); the mean is untouched. The head's scale
    is ``min_scale + softplus(raw)``, which no change of its weights scales exactly, so the scale
    half of the head is refitted: per pixel, the raw scale is replaced by an affine map of itself,
    ``a raw + d``, fitted by least squares on ``n`` states (drawn as the training states are) to
    the raw value whose scale is the tempered one, and folded into the head's weights. Returns the
    decoder and the fit's relative error of the scale on those states and on as many fresh ones
    (median, 99th percentile, largest)."""
    k_fit, k_check = jax.random.split(key)
    head = decoder.layers[-1]
    d = decoder.child_dim
    weight, bias = head.weight[d:], head.bias[d:]
    c = 1.0 / np.sqrt(alpha)

    def scale_of(raw):
        return decoder.min_scale + np.logaddexp(0.0, raw)

    states, terrain = sample_states(k_fit, n)
    onehot = jax.nn.one_hot(terrain, N_TERRAIN)
    hidden = jax.jit(jax.vmap(lambda s, t: _hidden(decoder, (s, t))))(states, onehot)
    raw = np.asarray(hidden @ weight.T + bias, np.float64)
    target = scale_of(raw) * c - decoder.min_scale
    want = np.log(np.expm1(target))  # the raw value whose scale is the tempered one
    mx, my = raw.mean(0), want.mean(0)
    a = ((raw - mx) * (want - my)).sum(0) / ((raw - mx) ** 2).sum(0)
    off = my - a * mx
    new_weight = jnp.asarray(a[:, None], weight.dtype) * weight
    new_bias = jnp.asarray(a * np.asarray(bias, np.float64) + off, bias.dtype)
    tempered = eqx.tree_at(
        lambda m: (m.layers[-1].weight, m.layers[-1].bias),
        decoder,
        (head.weight.at[d:].set(new_weight), head.bias.at[d:].set(new_bias)),
    )

    def error(got, want):
        rel = np.abs(np.asarray(got, np.float64) / np.asarray(want, np.float64) - 1.0)
        return dict(
            median=float(np.median(rel)),
            p99=float(np.percentile(rel, 99)),
            max=float(rel.max()),
        )

    states, terrain = sample_states(k_check, n)
    onehot = jax.nn.one_hot(terrain, N_TERRAIN)
    mean0, scale0 = jax.jit(jax.vmap(lambda s, t: decoder.diag_moments((s, t))))(states, onehot)
    mean1, scale1 = jax.jit(jax.vmap(lambda s, t: tempered.diag_moments((s, t))))(states, onehot)
    return tempered, dict(
        factor=float(c),
        fit=error(scale_of(a * raw + off), c * scale_of(raw)),
        fresh=error(scale1, c * scale0),
        fresh_mean_change=float(jnp.max(jnp.abs(mean1 - mean0))),
    )


def likelihood_report(decoder, key, n=2000):
    """How the decoder's likelihood fits held-out noisy images: the mean scale, the residual
    RMSE, the mean squared standardised residual (one when the scale is the predictive error,
    ``1 / inflation**2`` after ``calibrate``) and the mean log likelihood of an image (nats)."""
    states, _, _, images, onehot = _held_out(key, n)
    mean, scale = jax.vmap(lambda s, t: decoder.diag_moments((s, t)))(states, onehot)
    loglik = -jax.vmap(lambda y, s, t: decoder.nll(y, (s, t)))(images, states, onehot)
    return dict(
        scale_mean=float(jnp.mean(scale)),
        residual_rmse=float(jnp.sqrt(jnp.mean((images - mean) ** 2))),
        standardised_square_mean=float(jnp.mean(((images - mean) / scale) ** 2)),
        log_likelihood_mean=float(jnp.mean(loglik)),
    )


def evaluate(decoder, encoder, key, n=4000):
    """Held-out numbers on fresh states: the decoder's reconstruction error against the clean
    image (asserted below ``RMSE_LIMIT``; also over the ground pixels alone and per class), the
    recogniser's accuracy and mean log probability of the true terrain, and its confusion table
    ``[reading, true class]`` (each column sums to one): the reading is the most probable class."""
    states, terrain, clean, images, onehot = _held_out(key, n)
    mean, scale = jax.vmap(lambda s, t: decoder.diag_moments((s, t)))(states, onehot)
    sky = jax.vmap(lambda s: wd.ground_view(s, view_of(s))[2].reshape(-1))(states)
    err = (mean - clean) ** 2
    probs = jax.vmap(lambda y: terrain_probs(encoder, y))(images)
    reading = np.asarray(jnp.argmax(probs, axis=1))
    true = np.asarray(terrain)
    confusion = np.zeros((N_TERRAIN, N_TERRAIN))
    np.add.at(confusion, (reading, true), 1.0)
    confusion = confusion / confusion.sum(axis=0, keepdims=True)
    true_p = jnp.take_along_axis(probs, terrain[:, None], axis=1)[:, 0]
    out = dict(
        decoder_rmse_vs_clean=float(jnp.sqrt(jnp.mean(err))),
        decoder_rmse_ground=float(jnp.sqrt(jnp.sum(err * ~sky) / jnp.sum(~sky))),
        decoder_rmse_by_class=[
            float(jnp.sqrt(jnp.mean(err[terrain == c]))) for c in range(N_TERRAIN)
        ],
        decoder_rmse_p95_image=float(jnp.percentile(jnp.sqrt(jnp.mean(err, axis=1)), 95)),
        decoder_scale_mean=float(jnp.mean(scale)),
        pixel_noise_sd=wd.PIXEL_SD,
        rmse_limit=RMSE_LIMIT,
        clean_image_sd=float(jnp.mean(jnp.std(clean, axis=0))),
        recogniser_accuracy=float(np.mean(reading == true)),
        recogniser_mean_log_prob_true=float(jnp.mean(jnp.log(true_p))),
        recogniser_confusion=confusion.tolist(),
    )
    assert out["decoder_rmse_vs_clean"] < RMSE_LIMIT, out
    return out


def _skeletons():
    key = jax.random.PRNGKey(0)
    dec = _new_decoder(key)
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
        report = json.loads(report_path.read_text())
        if report.get("image_side") != wd.IMG_SIDE:
            raise SystemExit(f"{cache} holds a camera for {report.get('image_side')}-pixel images")
        dec = eqx.tree_deserialise_leaves(files["decoder"], dec_skel)
        enc = eqx.tree_deserialise_leaves(files["encoder"], enc_skel)
        return _tempered(dec, cache, report, report_path), enc, report
    k_data, k_dec, k_enc, k_eval, k_eff, k_cal, k_lik = jax.random.split(
        jax.random.PRNGKey(seed), 7
    )
    t0 = time.time()
    data = dataset(k_data)
    t1 = time.time()
    fitted, dec_loss = train_decoder(k_dec, data)
    n_eff, pixels, by_size = effective_pixels(fitted, k_eff)
    inflation = float(np.sqrt(pixels / n_eff)) * EMPIRICAL_TEMPER
    dec, cal_loss = calibrate(fitted, data, k_cal, inflation)
    t2 = time.time()
    enc, enc_loss = train_encoder(k_enc, data)
    t3 = time.time()
    report = evaluate(dec, enc, k_eval)
    report.update(
        effective_pixels=n_eff,
        effective_pixels_by_sample_size=by_size,
        empirical_temper=EMPIRICAL_TEMPER,
        fixed_view=FIXED_VIEW,
        pixels=pixels,
        noise_inflation=inflation,
        likelihood_fitted=likelihood_report(fitted, k_lik),
        likelihood_calibrated=likelihood_report(dec, k_lik),
        calibration_final_loss=cal_loss,
        decoder_final_nll=dec_loss,
        encoder_final_nll=enc_loss,
        render_seconds=t1 - t0,
        decoder_train_seconds=t2 - t1,
        encoder_train_seconds=t3 - t2,
        dataset=DATASET,
        decoder_steps=DECODER_STEPS,
        image_side=wd.IMG_SIDE,
        decoder_shape=DECODER_SHAPE,
        encoder_shape=ENCODER_SHAPE,
    )
    eqx.tree_serialise_leaves(files["decoder"], dec)
    eqx.tree_serialise_leaves(files["encoder"], enc)
    report_path.write_text(json.dumps(report, indent=1))
    dec = _tempered(dec, cache, report, report_path)
    if verbose:
        print(json.dumps(report, indent=1), flush=True)
    return dec, enc, report


def _tempered(decoder, cache, report, report_path):
    """With ``TEMPERED``, the calibrated ``decoder`` with its evidence tempered (``tempering``,
    ``temper``): the rule's alpha and the refit are measured once and cached beside the camera
    (the report's ``tempering`` entry and ``camera_decoder_tempered.eqx``); else ``decoder``."""
    if not TEMPERED:
        return decoder
    path = Path(cache) / "camera_decoder_tempered.eqx"
    if "tempering" not in report:
        report["tempering"] = tempering(decoder)
    if path.exists() and "fit" in report["tempering"]:
        return eqx.tree_deserialise_leaves(path, decoder)
    tempered, fit = temper(
        decoder, report["tempering"]["alpha"], jax.random.PRNGKey(TEMPER_KEY + 1)
    )
    report["tempering"]["fit"] = fit
    eqx.tree_serialise_leaves(path, tempered)
    report_path.write_text(json.dumps(report, indent=1))
    return tempered


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fixed", action="store_true", help="the comparison camera fixed along +y")
    args = ap.parse_args()
    FIXED_VIEW = args.fixed
    load_or_train(args.cache, args.seed)
