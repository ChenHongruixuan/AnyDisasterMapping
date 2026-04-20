"""
Model and dataset registry for the unified trainer.

Provides lazy-imported factories so heavy dependencies (torch, model libs)
are only loaded when a model or dataset is actually instantiated.
"""

import importlib


# ---------------------------------------------------------------------------
# Lazy import helper
# ---------------------------------------------------------------------------

def lazy_import(module_path: str, class_name: str):
    """Return a callable factory that defers the actual import until invocation.

    Usage::

        factory = lazy_import("src.models.UNet", "UNet")
        model   = factory(in_channels=3, num_classes=5)
    """
    def _factory(**kwargs):
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(**kwargs)
    return _factory


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

model_libs = {
    "unet":             lazy_import("src.models.UNet", "UNet"),
    "deeplabv3plus":    lazy_import("src.models.DeepLabV3Plus", "DeepLabV3Plus"),
    "unetplusplus":     lazy_import("src.models.UNetPlusPlus", "UNetPlusPlus"),
    "segformer":        lazy_import("src.models.SegFormer.SegFormer", "WeTr"),
    "swinupernet":      lazy_import("src.models.SwinUperNet", "SwinUperNet"),
    "convnext":         lazy_import("src.models.convnext", "ConvNeXtUPerNet"),
    "farseg":           lazy_import("src.models.FarSeg", "FarSeg"),
    "farsegpp":         lazy_import("src.models.FarSegPP", "FarSegPP"),
    "changeos":         lazy_import("src.models.ChangeOS", "ChangeOS"),
    "dsifn":            lazy_import("src.models.DSIFN.DSIFN", "DSIFN"),
    "bit":              lazy_import("src.models.BIT.BIT", "BASE_Transformer"),
    "siamcrnn":         lazy_import("src.models.SiamCRNN", "SiamCRNN"),
    "changemamba":      lazy_import("src.models.ChangeMamba.ChangeMamba", "ChangeMamba"),
    "rs3mamba":         lazy_import("src.models.RS3Mamba.RS3Mamba", "RS3Mamba"),
    "dinov2dpt":        lazy_import("src.models.DinoV2DPT", "DinoV2DPT"),
    "dinov3dpt":        lazy_import("src.models.dinov3.DINOV3DPT", "DinoV3DPT"),
    "dinov2dpt_lora":   lazy_import("src.models.DINOV2DPT_LoRA", "DinoV2DPTLoRA"),
    "samdpt":           lazy_import("src.models.SAMDPT.SAMDPT", "SAMDPT"),
    "clipdpt":          lazy_import("src.models.CLIP.CLIPDPT", "CLIPDPT"),
    "sam2fpn":          lazy_import("src.models.sam2.SAM2FPN", "SAM2FPN"),
    "sam2mamba":        lazy_import("src.models.sam2.SAM2Mamba", "SAM2Mamba"),
    "skysense":         lazy_import("src.models.SkySense.SkySenseUPerNet", "SkySenseUPerNet"),
    "satmae":           lazy_import("src.models.SatMAE.SatMAEDPT", "SatMAEDPT"),
    "spectralgpt":      lazy_import("src.models.SpectralGPT.SpectralGPT", "SpectralGPT"),
    "hypersigma":       lazy_import("src.models.HyperSigma.hypersigma", "HyperSigma"),
    "damageformer":     lazy_import("src.models.DamageFormer", "DamageFormer"),
    "mask2former":      lazy_import("src.models.mask2former.Mask2Former", "Mask2Former"),
    "unetformer":       lazy_import("src.models.UNetFormer", "UNetFormer"),
    "changemamba_scd":  lazy_import("src.models.ChangeMamba.ChangeMambaSCD", "ChangeMambaSCD"),
    "fcn8s":            lazy_import("src.models.FCN8s", "FCN8s"),
    # HRNet needs special factory — see below
}


# -- HRNet (requires yacs config object + conv1 replacement) ----------------

def _hrnet_factory(in_channels=6, num_classes=5, cfg_path=None, **kwargs):
    from src.models.HRNet.config import config, update_config
    from src.models.HRNet.seg_hrnet import get_seg_model
    import argparse
    import torch.nn as nn

    args = argparse.Namespace(cfg=cfg_path, opts=[])
    if cfg_path:
        update_config(config, args)

    config.defrost()
    config.DATASET.NUM_CLASSES = num_classes
    config.freeze()

    model = get_seg_model(config)

    if in_channels != 3:
        old = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels, old.out_channels,
            kernel_size=old.kernel_size, stride=old.stride,
            padding=old.padding, bias=old.bias is not None,
        )
        nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")

    return model


model_libs["hrnet"] = _hrnet_factory

# -- Aliases for common alternative names ------------------------------------
model_libs["deeplabv3p"] = model_libs["deeplabv3plus"]
model_libs["satmaedpt"] = model_libs["satmae"]
model_libs["sam2"] = model_libs["sam2fpn"]


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

dataset_libs = {
    "xbd":       lazy_import("src.datasets.infra_xbd", "xBDDataset"),
    "bright":    lazy_import("src.datasets.infra_bright", "BRIGHTDataset"),
    "rescuenet": lazy_import("src.datasets.infra_rescuenet", "RescueNetDataset"),
    "second":    lazy_import("src.datasets.second", "SECONDDataset"),
    "cau_flood": lazy_import("src.datasets.flood_cau", "CAUFloodDataset"),
    "kurosiwo":  lazy_import("src.datasets.flood_kurosiwo", "KuroSiwoDataset"),
    "urbansar":  lazy_import("src.datasets.flood_urbansar", "UrbanSARFloodsDataset"),
    "gvlm":      lazy_import("src.datasets.landslide_gvlm", "GVLMDataset"),
    "l4s":       lazy_import("src.datasets.landslide_l4s", "Landslide4SenseDataset"),
    "hrgldd":    lazy_import("src.datasets.landslide_hrgldd", "HRGLDDDataset"),
    "s2wcd":     lazy_import("src.datasets.wildfire_s2wcd", "S2WCDDataset"),
    "s2_wcd":    lazy_import("src.datasets.wildfire_s2wcd", "S2WCDDataset"),  # alias
    "floga":     lazy_import("src.datasets.wildfire_floga", "FLOGADataset"),
    "satellite_burned_area": lazy_import("src.datasets.wildfire_satba", "SatelliteBurnedAreaDataset"),
    "fire_spread": lazy_import("src.datasets.wildfire_firespread", "FireSpreadDataset"),
}


# ---------------------------------------------------------------------------
# Dual-input model metadata
# ---------------------------------------------------------------------------

DUAL_INPUT_MODELS = {
    "dsifn":        {"forward": "dual", "arg_names": ("t1_input", "t2_input")},
    "bit":          {"forward": "dual", "arg_names": ("x1", "x2")},
    "siamcrnn":     {"forward": "dual", "arg_names": ("pre_data", "post_data"), "dual_head": True},
    "hypersigma":   {"forward": "dual", "arg_names": ("x", "y")},
    "changemamba":  {"forward": "dual", "arg_names": ("pre_data", "post_data"), "dual_head": True},
    "damageformer": {"forward": "dual", "arg_names": ("pre_data", "post_data"), "dual_head": True},
    "sam2mamba":    {"forward": "dual", "arg_names": ("pre_image", "post_image"), "dual_head": True},
    "changeos":       {"forward": "single"},
    "changemamba_scd": {"forward": "dual", "arg_names": ("pre_data", "post_data")},
}


# ---------------------------------------------------------------------------
# Deep-supervision model metadata
# ---------------------------------------------------------------------------

DEEP_SUPERVISION_MODELS = {
    "dsifn":        {"weights": [1.0, 0.4, 0.3, 0.2, 0.1], "eval_agg": "first"},
    "unetplusplus": {"weights": None, "eval_agg": "average"},  # None = equal weights
}
