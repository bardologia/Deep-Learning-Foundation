---
type: model
domain: model
status: current
tags:
  - tomosar
  - tomosar/model
aliases:
  - full JEPA coupling
  - dual-autoencoder JEPA
family: jepa
registry_key: resunet
summary: Full JEPA coupling with both a pretrained image-autoencoder front-end and a profile-autoencoder embedding target.
group: jepa-ae
---

# JEPA Image AE + Backbone + Profile AE

The full coupling: a pretrained image autoencoder encodes the SAR stack into the backbone, the backbone predicts a profile-autoencoder embedding, and the profile autoencoder supplies both the target embedding and the decoder for curve reconstruction. Both autoencoders are imported from finished runs and are never trained from scratch here.

This is the two-front-end configuration of the unified JEPA pipeline (`python -m main.training.train_jepa`): it is selected by pointing both `JepaEntryConfig.image_autoencoder_run` and `profile_autoencoder_run` at finished runs. The full mechanics live in [[JEPA Profile-Embedding]].

## Data flow

- The image-autoencoder `encode_features` path re-encodes the input stack at the original resolution, so the backbone input channel count becomes the image-autoencoder embedding dimension.
- The backbone's output head is resized to the profile-autoencoder embedding dimension, producing the dense per-pixel prediction `z_hat` of shape `D x P x P`.
- The target embedding `z*` is built by reconstructing the ground-truth profile from the ground-truth Gaussian parameters, normalising it, and encoding it with the profile-autoencoder encoder.
- An auxiliary branch decodes `z_hat` through the profile-autoencoder decoder and compares the result with the ground-truth profile.

## Objective

- Embedding match between `z_hat` and `z*` (MSE by default).
- Curve reconstruction between the decoded prediction and the ground-truth profile (MSE by default).

## Coupling

- Each autoencoder is independently frozen or fine-tuned; frozen is the default for both, leaving the backbone as the only trainable module. Fine-tuned autoencoders join the optimiser as separate parameter groups with their own learning rate and weight decay.
- The target branch is produced by stop-gradient by default; an EMA or live target requires a trainable profile autoencoder, and a live target additionally requires the curve-reconstruction anchor.

## Related

- Drop the image front-end for [[JEPA Backbone + Profile AE]]; drop the embedding target for [[JEPA Image AE + Backbone]].
- Front-end and target spaces come from the [[Conv2D Image Autoencoder]] and [[MLP Autoencoder]] zoos by default.
