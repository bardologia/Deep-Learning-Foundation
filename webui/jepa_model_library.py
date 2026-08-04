from __future__ import annotations

from model_library_base import ModelNoteLibrary


class JepaModelLibrary(ModelNoteLibrary):

    NOTE_FILES = {
        "jepa_profile" : "JEPA Backbone + Profile AE.md",
        "jepa_image"   : "JEPA Image AE + Backbone.md",
        "jepa_full"    : "JEPA Image AE + Backbone + Profile AE.md",
    }

    def collect(self) -> list[dict]:
        return self._families()

    def _families(self) -> list[dict]:
        return [
            {
                "family" : "JEPA",
                "blurb"  : "Joint-embedding predictive training: a pretrained profile autoencoder defines a target embedding space and/or a pretrained image autoencoder provides a learned front-end, while the backbone is the trainable predictor.",
                "models" : [
                    {
                        "key": "jepa_profile", "name": "Backbone + Profile AE", "skip": "Backbone + Profile AE",
                        "head": "Embedding MSE + curve recon", "params": "backbone + frozen AE", "recommended": False,
                        "when": "Predict in embedding space. The backbone head emits the profile-autoencoder embedding dimension; the frozen profile-AE encoder turns the reconstructed ground-truth profile into the target z*, and its decoder maps the prediction back to a curve for an auxiliary reconstruction loss.",
                    },
                    {
                        "key": "jepa_image", "name": "Image AE + Backbone", "skip": "Image AE + Backbone",
                        "head": "Parameter L1", "params": "frozen AE + backbone", "recommended": False,
                        "when": "Learned front-end. The frozen image-autoencoder encoder re-encodes the SAR stack before the backbone, which still regresses the Gaussian parameters directly against the supervised parameter loss; no embedding target is involved.",
                    },
                    {
                        "key": "jepa_full", "name": "Image AE + Backbone + Profile AE", "skip": "Image AE + Backbone + Profile AE",
                        "head": "Embedding MSE + curve recon", "params": "two frozen AEs + backbone", "recommended": True,
                        "when": "The full coupling. The image-autoencoder encoder feeds the backbone, the backbone predicts the profile-autoencoder embedding, and the profile autoencoder supplies both the target embedding and the decoder for curve reconstruction.",
                    },
                ],
            },
        ]
