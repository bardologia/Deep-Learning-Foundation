---
type: model
domain: model
status: current
tags:
  - tomosar
  - tomosar/model
aliases:
  - JEPA Profile AE coupling
family: jepa
registry_key: resunet
summary: JEPA backbone regresses a pretrained profile-autoencoder embedding instead of Gaussian params directly.
group: jepa-ae
---

# JEPA Backbone + Profile AE

A joint-embedding predictive variant in which the supervised backbone stops regressing Gaussian parameters directly and instead predicts the latent embedding of a pretrained profile autoencoder. The autoencoder is imported from a finished profile-autoencoder run; it is never trained from scratch here.

This is one configuration of the unified JEPA pipeline (`python -m main.training.train_jepa`): it is selected by pointing `JepaEntryConfig.profile_autoencoder_run` at a profile-autoencoder run and leaving `image_autoencoder_run` empty. The full mechanics live in [[JEPA Profile-Embedding]].

## Data flow

- The backbone receives the raw input stack and its output head is resized to emit the profile-autoencoder embedding dimension, producing a dense per-pixel prediction `z_hat` of shape `D x P x P`.
- The supervision target is built on the fly: the ground-truth Gaussian parameters are denormalised and reconstructed into an elevation profile, normalised with the profile-autoencoder's own profile normaliser, and passed through the profile-autoencoder encoder to give the target embedding `z*`.
- An auxiliary curve-reconstruction branch decodes `z_hat` through the profile-autoencoder decoder back to a profile and compares it with the reconstructed ground-truth profile.

## Objective

- Embedding match between `z_hat` and `z*` (MSE by default; cosine and smooth-L1 are available).
- Curve reconstruction between the decoded prediction and the ground-truth profile (MSE by default; L1, Huber and Charbonnier are available).

Both embeddings are passed through the autoencoder's embedding normalisation before the match.

## Coupling

- The profile autoencoder is either frozen or fine-tuned. Frozen is the default and only the backbone is trained.
- The target branch is produced by a stop-gradient pass by default. An EMA copy of the encoder, or a fully live differentiable target, are available but require a trainable (fine-tuned) autoencoder; a live target additionally requires the curve-reconstruction anchor so the embedding cannot collapse to a constant.

## Related

- The target space is defined by a model from the profile-autoencoder zoo (default [[MLP Autoencoder]]).
- Add a learned image front-end with [[JEPA Image AE + Backbone + Profile AE]].
