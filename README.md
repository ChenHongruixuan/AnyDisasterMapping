# Any Disaster

Any Disaster is a unified PyTorch training and evaluation framework for remote-sensing tasks across infrastructure damage, flood mapping, landslide segmentation, and wildfire analysis.

## Contents

- [Installation](#installation)
- [Pretrained Weights](#pretrained-weights)
- [Quick Start](#quick-start)
- [Repository Layout](#repository-layout)
- [Dataset Preparation](#dataset-preparation)
- [Architecture and Extension](#architecture-and-extension)

## Installation

```bash
# NOTE: --index-url should match the version of your local CUDA toolkit for compiling ChangeMamba kernels (cu126 is just an example)
pip install torch torchvision xformers --index-url https://download.pytorch.org/whl/cu126
pip install -e .
```

Some models require optional extras:

- ChangeMamba selective scan kernel:
  ```bash
  # run `conda install -c conda-forge gcc=13 gxx=13 -y` if you meet GCC issues
  cd src/models/ChangeMamba/kernels/selective_scan
  pip install . --no-build-isolation
  ```
- Local pretrained checkpoints under `pretrained_weight/` for model families such as SegFormer, HRNet, SAM/SAM2, DINOv3, HyperSigma, SkySense, SpectralGPT, and ChangeMamba. See [Pretrained Weights](#pretrained-weights) below.

## Pretrained Weights

```bash
mkdir -p pretrained_weight

# pretrain-vit-base-e199.pth
wget -O pretrained_weight/pretrain-vit-base-e199.pth \
  https://zenodo.org/records/7338613/files/pretrain-vit-base-e199.pth

# SpectralGPT+.pth
wget -O "pretrained_weight/SpectralGPT+.pth" \
  "https://zenodo.org/records/8412455/files/SpectralGPT+.pth?download=1"

# spec-vit-base-ultra-checkpoint-1599.pth
wget -O pretrained_weight/spec-vit-base-ultra-checkpoint-1599.pth \
  https://huggingface.co/WHU-Sigma/HyperSIGMA/resolve/main/spec-vit-base-ultra-checkpoint-1599.pth

# --- DINOv3 ----------------------------------------------------------------
#   Source: https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/
#   Download and save as:
#     - dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth   (ViT-B/16, LVD-1689M)
#     - dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth   (ViT-L/16, LVD-1689M)
#     - dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth    (ViT-L/16, SAT-493M)

# --- SAM v1 -----------------------------------------------------------------
#   Source: https://github.com/facebookresearch/segment-anything
#   Download and save as:
#     - sam_vit_b_01ec64.pth    (SAM ViT-B)
#     - sam_vit_l_0b3195.pth    (SAM ViT-L)

# --- SAM 2.1 ----------------------------------------------
#   Source: https://github.com/facebookresearch/sam2
#   Download and save as:
#     - sam2.1_hiera_small.pt       (SAM 2.1 Hiera-Small)
#     - sam2.1_hiera_base_plus.pt   (SAM 2.1 Hiera-Base+)

# --- SegFormer MiT encoders --------------------------------------------------
#   Source: https://github.com/NVlabs/SegFormer
#   Download and save as:
#     - mit_b0.pth
#     - mit_b1.pth
#     - mit_b2.pth
#     - mit_b3.pth
#     - mit_b4.pth
#     - mit_b5.pth

# --- HyperSIGMA spatial backbone ---------------------------------------------
#   Source: https://huggingface.co/WHU-Sigma/HyperSIGMA
#   Download the upstream file, rename, and save as:
#     - HSI_spatial_checkpoint-1600.pth

# --- SkySense backbone -------------------------------------------------------
#   Source: 
#     https://github.com/Jack-bo1220/SkySense
#     https://www.notion.so/SkySense-Checkpoints-a7fcff6ce29a4647a08c7fe416910509
#   Select the `hr` (high-resolution RGB / RGBNIR) variant, NOT the `s2`
#   Sentinel-2 variant.
#   Save as:
#     - skysense_model_backbone_hr.pth
#   For commercial use, contact the authors (yansheng.li@whu.edu.cn).

huggingface-cli download UTokyo-Yokoya-Lab/AnyDisaster-Pretrained_Weight \
  vssm_tiny_0230_ckpt_epoch_262.pth --local-dir pretrained_weight --local-dir-use-symlinks False
```

After completing all downloads, `pretrained_weight/` should contain:

```text
pretrained_weight/
├── dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
├── dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
├── dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
├── HSI_spatial_checkpoint-1600.pth
├── mit_b0.pth
├── mit_b1.pth
├── mit_b2.pth
├── mit_b3.pth
├── mit_b4.pth
├── mit_b5.pth
├── pretrain-vit-base-e199.pth
├── sam2.1_hiera_base_plus.pt
├── sam2.1_hiera_small.pt
├── sam_vit_b_01ec64.pth
├── sam_vit_l_0b3195.pth
├── skysense_model_backbone_hr.pth
├── spec-vit-base-ultra-checkpoint-1599.pth
├── SpectralGPT+.pth
└── vssm_tiny_0230_ckpt_epoch_262.pth
```

## Quick Start

Train with a YAML config:

```bash
python train.py --config configs/infra/xbd/unet.yaml
```

Evaluate an experiment directory:

```bash
python test.py --exp_path results/xbd/unet
```

## Repository Layout
``
- `src/core/`: trainer, config loader, registry, augmentation, metrics
- `src/tasks/`: task handlers for segmentation, change detection, and semantic change detection
- `src/datasets/`: dataset adapters and runtime data contracts
- `src/models/`: model wrappers and vendored third-party implementations
- `configs/`: experiment configs grouped by domain and dataset
- `scripts/data_prep/`: dataset preparation guides and helper scripts

## Dataset Preparation

- Infrastructure damage: [scripts/data_prep/infra_damage/README.md](scripts/data_prep/infra_damage/README.md)
- Flood: [scripts/data_prep/flood/README.md](scripts/data_prep/flood/README.md)
- Landslide: [scripts/data_prep/landslide/README.md](scripts/data_prep/landslide/README.md)
- Wildfire: [scripts/data_prep/wildfire/README.md](scripts/data_prep/wildfire/README.md)

## Architecture and Extension

- Architecture overview: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Extension guide: [docs/EXTENSION_GUIDE.md](docs/EXTENSION_GUIDE.md)
