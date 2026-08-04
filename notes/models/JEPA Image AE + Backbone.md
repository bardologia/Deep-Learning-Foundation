---
type: model
domain: model
status: current
tags:
  - tomosar
  - tomosar/model
aliases:
  - image-frontend JEPA
  - JEPA param backbone
family: jepa
registry_key: resunet
summary: Pretrained image autoencoder feeds the backbone, which still regresses Gaussian params directly (no embedding target).
group: jepa-ae
---

# JEPA Image AE + Backbone

A variant that places a pretrained image autoencoder in front of the backbone as a learned encoder of the SAR stack. There is no embedding target here: the backbone still regresses the Gaussian parameters and is trained with the ordinary supervised parameter loss.

This is one configuration of the unified JEPA pipeline (`python -m main.training.train_jepa`): it is selected by pointing `JepaEntryConfig.image_autoencoder_run` at an image-autoencoder run and leaving `profile_autoencoder_run` empty. With no profile autoencoder, the run is scored at inference via `JEPA_PARAM_INFERENCE_COMPONENTS` (backbone predicts parameters through the front-end); see [[JEPA Profile-Embedding]].

## Data flow

- The image autoencoder is imported from a finished image-autoencoder run. Its `encode_features` path re-encodes the input stack and returns features at the original spatial resolution, so the backbone's input channel count becomes the image-autoencoder embedding dimension.
- The backbone consumes the encoded stack and its output head emits the full Gaussian parameter tensor of shape `3K x P x P`.

## Objective

- Parameter loss between the predicted and ground-truth Gaussian parameters (L1 by default). No embedding or curve term is involved, because there is no profile autoencoder.

## Coupling

- The image autoencoder is either frozen or fine-tuned. Frozen is the default; with fine-tuning, the front-end is added to the optimiser as its own parameter group with a smaller learning rate and weight decay.
- The input representation and secondaries of the JEPA dataset must match the image-autoencoder run, since the front-end was trained for a specific input channel layout.

## Related

- The front-end is a model from the image-autoencoder zoo (default [[Conv2D Image Autoencoder]]).
- Add an embedding target with [[JEPA Image AE + Backbone + Profile AE]].
