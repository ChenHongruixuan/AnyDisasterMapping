"""
Standalone model profiling script.

Measures parameter count, FLOPs, GPU peak memory, and inference latency
for any model defined in this framework, driven by the same YAML config
used for training.

Usage:
    # Profile a single model
    python -m src.utils.profiling --config flood/cau/seg_segformer_b3

    # Profile all models for one dataset
    python -m src.utils.profiling --dataset flood/cau

    # Display summary tables from all saved profiling results
    python -m src.utils.profiling
"""

import argparse
import glob
import io
import json
import logging
import os
import platform
import time
import unittest.mock
from os.path import basename, dirname, join, relpath

import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

PROFILE_DIR = "./profiling_results"


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def count_parameters(model):
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_flops(model, forward_fn):
    """Return FLOPs (MACs) via fvcore, with PyTorch native fallback.

    Args:
        model: the model (used by fvcore).
        forward_fn: zero-arg callable that runs one forward pass.

    Returns None if both methods fail.
    """
    # Primary: fvcore
    try:
        from fvcore.nn import FlopCountAnalysis
        fa = FlopCountAnalysis(model, forward_fn.dummy_inputs)
        fa.unsupported_ops_warnings(False)
        fa.uncalled_modules_warnings(False)
        return fa.total()
    except Exception as e:
        log.warning(f"  fvcore FlopCountAnalysis failed: {e}")

    # Fallback: PyTorch native (torch >= 2.1)
    try:
        from torch.utils.flop_counter import FlopCounterMode
        counter = FlopCounterMode(display=False)
        with counter:
            forward_fn()
        return counter.get_total_flops()
    except Exception as e:
        log.warning(f"  PyTorch FlopCounterMode also failed: {e}")

    return None


def measure_memory(forward_fn, device):
    """Return peak GPU memory (bytes) for a single forward pass.

    Returns None if device is CPU.
    """
    if device.type != "cuda":
        return None

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    with torch.no_grad():
        forward_fn()

    return torch.cuda.max_memory_allocated(device)


def measure_latency(forward_fn, device, n_warmup=30, n_runs=300):
    """Return per-sample latency in milliseconds."""
    if device.type == "cuda":
        for _ in range(n_warmup):
            with torch.no_grad():
                forward_fn()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start.record()
        for _ in range(n_runs):
            with torch.no_grad():
                forward_fn()
        end.record()
        torch.cuda.synchronize()

        return start.elapsed_time(end) / n_runs
    else:
        for _ in range(n_warmup):
            with torch.no_grad():
                forward_fn()

        t0 = time.perf_counter()
        for _ in range(n_runs):
            with torch.no_grad():
                forward_fn()
        t1 = time.perf_counter()

        return (t1 - t0) / n_runs * 1000.0


# ---------------------------------------------------------------------------
# Forward function builder
# ---------------------------------------------------------------------------

def _build_forward_fn(model, model_name, config, in_channels, patch_size, device):
    """Build a zero-arg forward callable and attach .dummy_inputs for fvcore.

    Returns (forward_fn, input_shape_list).
    """
    from src.core.registry import DUAL_INPUT_MODELS

    H = W = patch_size
    is_mask2former = config.get("model_has_builtin_loss", False)
    info = DUAL_INPUT_MODELS.get(model_name)
    is_dual = info and info.get("forward") == "dual"

    # SpectralGPT: 4D input (B, T, H, W) instead of (B, C, H, W)
    is_spectralgpt = model_name == "spectralgpt"

    if is_mask2former:
        # Mask2Former: pixel_values kwarg, no labels needed in eval mode
        dummy = torch.randn(1, in_channels, H, W, device=device)
        input_shape = [1, in_channels, H, W]

        def forward_fn():
            return model(pixel_values=dummy)

        forward_fn.dummy_inputs = (dummy,)
        # fvcore passes dummy_inputs positionally; Mask2Former.forward() accepts
        # pixel_values as its first positional param, so this works correctly.

    elif is_dual:
        names = info["arg_names"]
        x1 = torch.randn(1, in_channels, H, W, device=device)
        x2 = torch.randn(1, in_channels, H, W, device=device)
        input_shape = [1, in_channels, H, W]  # per-branch shape

        def forward_fn():
            return model(**{names[0]: x1, names[1]: x2})

        forward_fn.dummy_inputs = (x1, x2)

    elif is_spectralgpt:
        num_frames = in_channels  # for SpectralGPT, in_channels == num_frames
        dummy = torch.randn(1, num_frames, H, W, device=device)
        input_shape = [1, num_frames, H, W]

        def forward_fn():
            return model(dummy)

        forward_fn.dummy_inputs = (dummy,)

    else:
        # Standard single-input model
        dummy = torch.randn(1, in_channels, H, W, device=device)
        input_shape = [1, in_channels, H, W]

        def forward_fn():
            return model(dummy)

        forward_fn.dummy_inputs = (dummy,)

    return forward_fn, input_shape


# ---------------------------------------------------------------------------
# Main profiling function
# ---------------------------------------------------------------------------

def profile_model(model, model_name, config, in_channels, patch_size, device,
                  n_warmup=30, n_runs=300):
    """Run all profiling metrics. Returns a dict."""
    results = {}

    # --- Parameter count ---
    total_params, trainable_params = count_parameters(model)
    results["total_params"] = total_params
    results["total_params_M"] = round(total_params / 1e6, 2)
    results["trainable_params"] = trainable_params
    results["trainable_params_M"] = round(trainable_params / 1e6, 2)
    results["trainable_param_ratio"] = (
        round(trainable_params / total_params, 4) if total_params > 0 else 0.0
    )
    log.info(f"  Params: {results['total_params_M']}M total, "
             f"{results['trainable_params_M']}M trainable "
             f"(ratio: {results['trainable_param_ratio']})")

    # --- Model size on disk ---
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    model_size_bytes = buf.tell()
    results["model_size_bytes"] = model_size_bytes
    results["model_size_MB"] = round(model_size_bytes / (1024 ** 2), 2)
    log.info(f"  Model size: {results['model_size_MB']}MB")

    # --- Build forward function ---
    forward_fn, input_shape = _build_forward_fn(
        model, model_name, config, in_channels, patch_size, device,
    )
    results["input_shape"] = input_shape

    # --- FLOPs ---
    log.info(f"  Measuring FLOPs with input {input_shape}...")
    macs = count_flops(model, forward_fn)
    if macs is not None:
        results["macs"] = macs
        results["macs_G"] = round(macs / 1e9, 2)
        log.info(f"  FLOPs (MACs): {results['macs_G']}G")
    else:
        results["macs"] = None
        results["macs_G"] = None
        log.warning("  FLOPs: measurement failed (model may use untraceable ops)")

    # --- GPU peak memory ---
    log.info("  Measuring peak memory...")
    peak_mem = measure_memory(forward_fn, device)
    if peak_mem is not None:
        results["peak_memory_bytes"] = peak_mem
        results["peak_memory_MB"] = round(peak_mem / (1024 ** 2), 1)

        param_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_bytes = sum(b.nelement() * b.element_size() for b in model.buffers())
        activation_bytes = max(0, peak_mem - param_bytes - buffer_bytes)
        results["activation_memory_bytes"] = activation_bytes
        results["activation_memory_MB"] = round(activation_bytes / (1024 ** 2), 1)
        log.info(f"  Peak memory: {results['peak_memory_MB']}MB "
                 f"(activation: {results['activation_memory_MB']}MB)")
    else:
        results["peak_memory_bytes"] = None
        results["peak_memory_MB"] = None
        results["activation_memory_bytes"] = None
        results["activation_memory_MB"] = None
        log.info("  Peak memory: N/A (CPU mode)")

    # --- Latency ---
    log.info(f"  Measuring latency ({n_warmup} warmup + {n_runs} timed runs)...")
    latency = measure_latency(forward_fn, device, n_warmup, n_runs)
    results["latency_ms"] = round(latency, 2)
    results["throughput_fps"] = round(1000.0 / latency, 2) if latency > 0 else None
    results["latency_n_warmup"] = n_warmup
    results["latency_n_runs"] = n_runs
    log.info(f"  Latency: {results['latency_ms']}ms / sample "
             f"({results['throughput_fps']} FPS)")

    return results


# ---------------------------------------------------------------------------
# Config loading and model instantiation
# ---------------------------------------------------------------------------

def resolve_config_path(config_arg):
    """Resolve config argument to a full YAML file path."""
    if config_arg.endswith(".yaml"):
        path = config_arg
    else:
        path = config_arg + ".yaml"

    if not os.path.isabs(path):
        path = join("./configs", path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    return path


def get_in_channels(config):
    """Extract input channels from config, handling various kwarg names.

    For dual-input CD models (e.g. ChangeMamba, SiamCRNN), the returned value
    is the per-branch channel count (typically 3), NOT the concatenated total.
    """
    from src.core.registry import DUAL_INPUT_MODELS

    kwargs = config.get("model", {}).get("kwargs", {})

    # Check known key names in order of prevalence
    for key in ("in_channels", "in_chans", "input_nc", "in_dim"):
        if key in kwargs:
            return kwargs[key]

    # SpectralGPT uses num_frames as the channel dimension
    if "num_frames" in kwargs:
        return kwargs["num_frames"]

    # Fallback: dual-input models use per-branch channels (default 3),
    # single-input CD models use concatenated channels (default 6)
    model_name = config.get("model", {}).get("name", "")
    info = DUAL_INPUT_MODELS.get(model_name)
    if info and info.get("forward") == "dual":
        return 3

    task = config.get("task", "seg")
    return 6 if task == "cd" else 3


def get_patch_size(config):
    """Extract the inference patch size from config."""
    # 1. inference.sliding_kernel (explicit inference config)
    inf = config.get("inference", {})
    if isinstance(inf, dict) and "sliding_kernel" in inf:
        return inf["sliding_kernel"]

    # 2. model.kwargs.img_size (architectural constraint — SpectralGPT, SkySense, Swin)
    kwargs = config.get("model", {}).get("kwargs", {})
    if "img_size" in kwargs:
        return kwargs["img_size"]

    # 3. augmentation.train.RandomCrop
    aug_train = config.get("augmentation", {}).get("train", {})
    if isinstance(aug_train, dict):
        rc = aug_train.get("RandomCrop", {})
        if isinstance(rc, dict) and "height" in rc:
            return rc["height"]

    # 4. dataset.train.crop_size
    ds_train = config.get("dataset", {}).get("train", {})
    if isinstance(ds_train, dict):
        cs = ds_train.get("crop_size")
        if cs is not None:
            return cs

    # 5. augmentation.train.SmartCrop
    if isinstance(aug_train, dict):
        sc = aug_train.get("SmartCrop", {})
        if isinstance(sc, dict) and "crop_size" in sc:
            return sc["crop_size"]

    # 6. augmentation.train.Resize
    if isinstance(aug_train, dict):
        rs = aug_train.get("Resize", {})
        if isinstance(rs, dict) and "height" in rs:
            return rs["height"]

    log.warning("  Could not determine patch size from config, using default 512")
    return 512


def load_model_from_config(config_path, no_pretrained=True):
    """Load config YAML and instantiate the model.

    Args:
        no_pretrained: If True, strip pretrained weight paths so the model
            is built with random weights (profiling only needs architecture).
    """
    from src.core.registry import model_libs

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name") if isinstance(model_cfg, dict) else config.get("model_name")
    model_kwargs = (
        model_cfg.get("kwargs", {}) if isinstance(model_cfg, dict)
        else config.get("model_kwargs", {})
    )
    # Make a copy to avoid mutating the config
    model_kwargs = dict(model_kwargs)

    # Strip pretrained weight paths
    if no_pretrained:
        _pretrained_keys = [
            "pretrained_weight", "pretrained_weights", "pretrained",
            "pretrained_path", "checkpoint_path", "ckpt_path",
            "pretrained_backbone", "pretrained_backbone_path",
        ]
        for key in _pretrained_keys:
            if key in model_kwargs:
                log.info(f"  Stripping {key}={model_kwargs[key]} (profiling mode)")
                if isinstance(model_kwargs[key], bool):
                    model_kwargs[key] = False
                else:
                    model_kwargs[key] = None

    if model_name not in model_libs:
        raise KeyError(
            f"Model '{model_name}' not found in model_libs. "
            f"Available: {sorted(model_libs.keys())}"
        )

    log.info(f"  Instantiating {model_name}...")

    # Patch torch.load to skip missing pretrained files
    if no_pretrained:
        _real_torch_load = torch.load

        def _fake_load(f, *args, **kwargs):
            if isinstance(f, str) and not os.path.exists(f):
                log.info(f"  Skipping missing pretrained: {f}")
                return {}
            return _real_torch_load(f, *args, **kwargs)

        with unittest.mock.patch("torch.load", _fake_load):
            model = model_libs[model_name](**model_kwargs)
    else:
        model = model_libs[model_name](**model_kwargs)

    model.eval()
    return model, config, model_name


def collect_environment(device):
    """Collect hardware and software environment info for reproducibility."""
    env = {
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "N/A",
        "cudnn_version": (
            str(torch.backends.cudnn.version())
            if torch.backends.cudnn.is_available() else "N/A"
        ),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    if device.type == "cuda":
        idx = device.index or 0
        env["gpu_name"] = torch.cuda.get_device_name(idx)
        props = torch.cuda.get_device_properties(idx)
        env["gpu_memory_GB"] = round(props.total_memory / (1024 ** 3), 1)
        env["gpu_compute_capability"] = f"{props.major}.{props.minor}"
        env["gpu_count"] = torch.cuda.device_count()
    else:
        env["gpu_name"] = "CPU"

    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    env["cpu_name"] = line.split(":")[1].strip()
                    break
    except (FileNotFoundError, PermissionError):
        env["cpu_name"] = platform.processor() or "unknown"

    return env


def get_output_path(config_arg, profile_dir):
    """Derive output path from config argument.

    flood/cau/cd_bit -> <profile_dir>/flood/cau/cd_bit.json
    """
    config_rel = config_arg.replace(".yaml", "")
    return join(profile_dir, config_rel + ".json")


# ---------------------------------------------------------------------------
# Summary table display
# ---------------------------------------------------------------------------

# Canonical model display order matching the benchmark paper table layout:
# General Vision CNN → General Vision Transformer → General Vision FM
# → RS-Specialized → Domain-Specific FM
MODEL_DISPLAY_ORDER = [
    # -- General Vision CNN --
    "unet",
    "dsifn",
    "unetplusplus",
    "deeplabv3plus_resnet50",
    "deeplabv3plus_resnet101",
    "deeplabv3plus_resnet34",
    "hrnet_w18",
    "hrnet_w48",
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    # -- General Vision Transformer --
    "swinupernet_tiny",
    "swinupernet_small",
    "swinupernet_base",
    "segformer_b0",
    "segformer_b1",
    "segformer_b2",
    "segformer_b3",
    "segformer_b4",
    "segformer_b5",
    "mask2former",
    # -- General Vision FM --
    "samdpt",
    "sam2fpn_hiera_s",
    "sam2fpn_hiera_bp",
    "clipdpt_base",
    "clipdpt_large",
    "dinov2dpt_vitb14",
    "dinov2dpt_vitl14",
    "dinov2dpt_vits14",
    "dinov3dpt_vitb16_lvd",
    "dinov3dpt_vitl16_lvd",
    "dinov3dpt_vitl16_sat",
    # -- RS-Specialized --
    "changeos",
    "changemamba",
    "farseg",
    "farsegpp",
    "bit",
    "siamcrnn",
    "unetformer",
    "rs3mamba",
    # -- Domain-Specific FM --
    "satmae",
    "skysense",
    "spectralgpt",
    "hypersigma",
    # -- New additions (not in original paper) --
    "fcn8s",
]

# Build lookup: display_name -> sort index
_MODEL_ORDER_MAP = {name: idx for idx, name in enumerate(MODEL_DISPLAY_ORDER)}


def _model_sort_key(rec):
    """Sort key: paper order first, then alphabetically for unknown models."""
    name = (
        rec.get("display_name")
        or rec.get("config", "").replace(".yaml", "").split("/")[-1]
        or rec.get("model_name", "?")
    )
    idx = _MODEL_ORDER_MAP.get(name)
    if idx is not None:
        return (0, idx, name)
    return (1, 0, name)


def display_summary(profile_dir):
    """Scan profiling_results/ and display prettytable grouped by dataset."""
    from prettytable import PrettyTable

    pattern = join(profile_dir, "**", "*.json")
    json_files = sorted(glob.glob(pattern, recursive=True))

    if not json_files:
        log.info(f"No profiling results found in {profile_dir}/")
        log.info("Run with --config to profile models first.")
        return

    # Load and group by dataset directory
    groups = {}
    for fpath in json_files:
        dataset_key = relpath(dirname(fpath), profile_dir)
        with open(fpath) as f:
            data = json.load(f)
        groups.setdefault(dataset_key, []).append(data)

    # Collect environment info from first result
    first_result = next(iter(next(iter(groups.values()))))
    env = first_result.get("environment", {})

    header_lines = []
    header_lines.append(f"Profiling results from: {os.path.abspath(profile_dir)}")
    if env:
        gpu = env.get("gpu_name", "?")
        gpu_mem = env.get("gpu_memory_GB", "?")
        pt = env.get("pytorch_version", "?")
        cuda = env.get("cuda_version", "?")
        header_lines.append(
            f"Environment: {gpu} ({gpu_mem}GB) | PyTorch {pt} | CUDA {cuda}"
        )
    header_lines.append("")

    def _get_display_name(rec):
        return (
            rec.get("display_name")
            or rec.get("config", "").replace(".yaml", "").split("/")[-1]
            or rec.get("model_name", "?")
        )

    columns = [
        ("Model",             _get_display_name, None),
        ("Input",             "input_shape",
         lambda v: f"{v[1]}x{v[2]}x{v[3]}" if v else "?"),
        ("Params(M) \u2193",  "total_params_M", None),
        ("Size(MB) \u2193",   "model_size_MB", None),
        ("MACs(G) \u2193",    "macs_G",
         lambda v: str(v) if v is not None else "N/A"),
        ("Mem(MB) \u2193",    "peak_memory_MB",
         lambda v: str(v) if v is not None else "N/A"),
        ("Act(MB) \u2193",    "activation_memory_MB",
         lambda v: str(v) if v is not None else "N/A"),
        ("Lat(ms) \u2193",    "latency_ms", None),
        ("FPS \u2191",        "throughput_fps", None),
    ]

    all_output = "\n".join(header_lines)

    for dataset_key in sorted(groups.keys()):
        records = groups[dataset_key]
        records.sort(key=_model_sort_key)

        table = PrettyTable()
        table.field_names = [c[0] for c in columns]
        table.align = "r"
        table.align["Model"] = "l"

        for rec in records:
            row = []
            for col_name, key_or_fn, fmt in columns:
                if callable(key_or_fn):
                    val = key_or_fn(rec)
                else:
                    val = rec.get(key_or_fn, "?")
                if fmt:
                    val = fmt(val)
                row.append(val)
            table.add_row(row)

        section = f"=== {dataset_key} ({len(records)} models) ===\n"
        section += str(table) + "\n"
        log.info(section)
        all_output += section + "\n"

    txt_path = join(profile_dir, "profiling_summary.txt")
    with open(txt_path, "w") as f:
        f.write(all_output)
    log.info(f"Summary saved to: {txt_path}")


# ---------------------------------------------------------------------------
# Batch profiling of all configs under a dataset directory
# ---------------------------------------------------------------------------

def profile_dataset(dataset_filter, profile_dir, device, n_warmup, n_runs,
                    force, with_pretrained):
    """Profile all configs matching a dataset path filter.

    Args:
        dataset_filter: e.g. "flood/cau" -> searches configs/flood/cau/*.yaml
    """
    search_dir = join("./configs", dataset_filter)
    if not os.path.isdir(search_dir):
        log.error(f"Dataset directory not found: {search_dir}")
        return

    yaml_files = sorted(glob.glob(join(search_dir, "*.yaml")))
    if not yaml_files:
        log.error(f"No config files found in {search_dir}")
        return

    total = len(yaml_files)
    passed = 0
    failed = 0
    skipped = 0
    failed_list = []

    log.info("=" * 50)
    log.info(f"  Model Profiling -- {total} configs")
    log.info(f"  Dataset: {dataset_filter}")
    log.info(f"  Device: {device}")
    log.info(f"  Warmup: {n_warmup}, Timed runs: {n_runs}")
    log.info(f"  Output: {profile_dir}/")
    log.info("=" * 50)
    log.info("")

    for idx, cfg_path in enumerate(yaml_files, 1):
        cfg_key = relpath(cfg_path, "./configs").replace(".yaml", "")
        output_path = get_output_path(cfg_key, profile_dir)

        log.info(f"[{idx:3d}/{total}] {cfg_key}")

        if os.path.exists(output_path) and not force:
            log.info(f"  SKIP (already exists)")
            skipped += 1
            continue

        try:
            _profile_single(cfg_key, output_path, device, n_warmup, n_runs,
                            with_pretrained)
            passed += 1
            log.info(f"  OK")
        except Exception as e:
            failed += 1
            failed_list.append(cfg_key)
            log.error(f"  FAIL: {e}")

        # Free GPU memory between models
        if device.type == "cuda":
            torch.cuda.empty_cache()

    log.info("")
    log.info("=" * 50)
    log.info(f"  Done: {passed} profiled, {skipped} skipped, {failed} failed")
    log.info(f"  Total: {total}")
    log.info("=" * 50)

    if failed_list:
        log.info("\nFailed configs:")
        for c in failed_list:
            log.info(f"  {c}")


def _profile_single(config_key, output_path, device, n_warmup, n_runs,
                     with_pretrained):
    """Profile a single model and save results to JSON."""
    config_path = resolve_config_path(config_key)

    model, config, model_name = load_model_from_config(
        config_path, no_pretrained=not with_pretrained,
    )
    in_channels = get_in_channels(config)
    num_classes = config.get("num_classes", 2)
    patch_size = get_patch_size(config)
    log.info(f"  in_channels={in_channels}, num_classes={num_classes}, "
             f"patch_size={patch_size}")

    model.to(device)

    results = profile_model(
        model, model_name, config, in_channels, patch_size, device,
        n_warmup=n_warmup, n_runs=n_runs,
    )

    # Add metadata
    config_rel = config_key.replace(".yaml", "")
    results["config"] = config_key
    results["task"] = config.get("task", "seg")
    results["model_name"] = model_name
    results["display_name"] = config_rel.split("/")[-1]
    results["environment"] = collect_environment(device)

    # Save
    os.makedirs(dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    log.info(f"\nSaved to: {output_path}")
    log.info(json.dumps(results, indent=2))

    # Free model from GPU
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Profile model: params, FLOPs, memory, latency. "
                    "Run with --config to profile one model, "
                    "--dataset to profile all models for a dataset, "
                    "or run without args to display summary.",
    )
    parser.add_argument(
        "--config", default=None,
        help="Config path, e.g. flood/cau/cd_bit. Profile a single model.",
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Dataset path, e.g. flood/cau. Profile all models for this dataset.",
    )
    parser.add_argument("--device", default=None,
                        help="Device (default: cuda if available, else cpu)")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: auto from config)")
    parser.add_argument("--profile_dir", default=PROFILE_DIR,
                        help=f"Directory for profiling results (default: {PROFILE_DIR})")
    parser.add_argument("--n_warmup", type=int, default=30,
                        help="Number of warmup runs for latency")
    parser.add_argument("--n_runs", type=int, default=300,
                        help="Number of timed runs for latency")
    parser.add_argument("--force", action="store_true",
                        help="Re-profile even if output file exists")
    parser.add_argument("--with_pretrained", action="store_true",
                        help="Load pretrained weights (default: skip)")
    args = parser.parse_args()

    # --- Summary mode (no --config, no --dataset) ---
    if args.config is None and args.dataset is None:
        display_summary(args.profile_dir)
        return

    # Resolve device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Dataset batch mode ---
    if args.dataset is not None:
        profile_dataset(
            args.dataset, args.profile_dir, device,
            args.n_warmup, args.n_runs, args.force, args.with_pretrained,
        )
        # Display summary for this dataset
        display_summary(args.profile_dir)
        return

    # --- Single model mode ---
    config_path = resolve_config_path(args.config)
    log.info(f"Config: {config_path}")
    log.info(f"Device: {device}")

    if args.output:
        output_path = args.output
    else:
        output_path = get_output_path(args.config, args.profile_dir)

    if os.path.exists(output_path) and not args.force:
        log.info(f"Profile already exists: {output_path} (use --force to re-run)")
        with open(output_path) as f:
            existing = json.load(f)
        log.info(json.dumps(existing, indent=2))
        return

    _profile_single(args.config, output_path, device,
                     args.n_warmup, args.n_runs, args.with_pretrained)


if __name__ == "__main__":
    main()
