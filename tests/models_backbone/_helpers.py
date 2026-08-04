from __future__ import annotations

WINDOW = 64

SMALL_OVERRIDES = {
    "unet"           : {"features": [8, 16], "bottleneck_factor": 1},
    "unet_skip"      : {"features": [8, 16], "bottleneck_factor": 1},
    "resunet"        : {"features": [8, 16], "bottleneck_factor": 1},
    "attention_unet" : {"features": [8, 16], "bottleneck_factor": 1},
    "unetplusplus"   : {"features": [8, 16, 32, 64], "bottleneck_factor": 1},
    "linknet"        : {"features": [16, 32, 64, 128], "initial_kernel_size": 3},
    "swin_unet"      : {"image_size": WINDOW, "embedding_dim": 24, "depths": [2, 2, 2, 2], "num_heads": [1, 2, 4, 8], "window_size": 4},
    "transunet"      : {"image_size": WINDOW, "cnn_features": [8, 16, 32, 64], "transformer_layers": 2, "transformer_heads": 2},
    "unetr"          : {"image_size": WINDOW, "embedding_dim": 64, "transformer_layers": 4, "transformer_heads": 4, "decoder_features": [32, 16, 8, 8]},
    "deeplabv3plus"  : {"features": [16, 32, 64, 128]},
    "segformer"      : {"embedding_dims": [16, 32, 64, 128], "depths": [1, 1, 1, 1], "decoder_channels": 64},
    "convnext_unet"  : {"features": [16, 32, 64, 128], "blocks_per_stage": 1, "bottleneck_factor": 1},
    "dense_unet"     : {"growth_rate": 8, "block_layers": [2, 2, 2], "bottleneck_layers": 2},
    "hrnet"          : {"base_channels": 16, "n_branches": 3, "blocks_per_stage": 1},
    "multires_unet"  : {"features": [16, 32, 64, 128], "bottleneck_factor": 1},
    "fpn"            : {"features": [16, 32, 64, 128], "pyramid_channels": 32, "segmentation_convs": 1},
    "u2net"          : {"features": [16, 32, 64, 128], "rsu_heights": (4, 3, 2)},
    "pixel_mlp"      : {"features": [32, 32]},
    "local_cnn"      : {"features": [8, 16]},
    "nafnet"         : {"width": 8, "enc_blocks": [1, 1], "middle_blocks": 1, "dec_blocks": [1, 1]},
}
