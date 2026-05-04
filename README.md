<div align="center">
<h1 align="center">🌍Any Disaster Mapping</h1>

<h3>Earth Observation for Disaster Mapping: Benchmarks, Methods, Challenges and Future Perspectives</h3>

[Hongruixuan Chen](https://scholar.google.com/citations?user=XOk4Cf0AAAAJ)<sup>1,†</sup>, [Jian Song](https://scholar.google.com/citations?user=fx-27bQAAAAJ)<sup>1,†</sup>, [Weihao Xuan](https://scholar.google.com/citations?user=7e0W-2AAAAAJ)<sup>1,2,†</sup>, [Junjue Wang](https://scholar.google.com/citations?user=H58gKSAAAAAJ)<sup>2,†</sup>, [Heli Qi](https://scholar.google.com/citations?user=CH-rTXsAAAAJ)<sup>1</sup>, [Zeqi Zhou](https://scholar.google.com/citations?user=pqG03s0AAAAJ)<sup>3</sup>, [Pengyu Dai](https://scholar.google.com/citations?user=bxKhSZgAAAAJ)<sup>1,2</sup> <br> [Olivier Dietrich](https://scholar.google.com/citations?user=st6IqcsAAAAJ)<sup>4</sup>, [Erika Gutierrez](https://www.google.com/search?q=Erika+Gutierrez&oq=Erika+Gutierrez&gs_lcrp=EgRlZGdlKgYIABBFGDsyBggAEEUYOzIGCAEQABgeMgYIAhAAGB4yBggDEAAYHjIGCAQQABgeMgYIBRAAGB4yBggGEAAYHjIGCAcQRRg8MgcICBDrBxhA0gEHMjkxajBqNKgCALACAA&sourceid=chrome&ie=UTF-8)<sup>6</sup>, [Lars Bromly](https://www.google.com/search?q=Lars+Bromly&oq=Lars+Bromly&gs_lcrp=EgRlZGdlKgYIABBFGDsyBggAEEUYOzIJCAEQABgNGIAEMggIAhAAGA0YHjIKCAMQABiABBiiBDIHCAQQABjvBTIHCAUQABjvBTIKCAYQABiABBiiBDIGCAcQRRg70gEHMjQ0ajBqN6gCALACAA&sourceid=chrome&ie=UTF-8)<sup>6</sup>, [Edoardo Nemni](https://scholar.google.com/citations?user=eUo-LuAAAAAJ)<sup>7</sup>, [Yafei Ou](https://scholar.google.com/citations?user=2SyrQ1oAAAAJ)<sup>1</sup>, [Jie Zhao](https://scholar.google.com/citations?user=snOLm2MAAAAJ)<sup>8</sup>, [Zhuo Zheng](https://scholar.google.com/citations?user=CREpn_AAAAAJ)<sup>5</sup>, [Yonghao Xu](https://scholar.google.com/citations?user=sQs2ztAAAAAJ)<sup>9</sup> <br> [Ronny Hänsch](https://scholar.google.com/citations?user=mUlpUlUAAAAJ)<sup>10</sup>, [Junshi Xia](https://scholar.google.com/citations?user=n1aKdTkAAAAJ)<sup>1</sup>, [Wenzhe Jiao](https://scholar.google.com/citations?user=1v9ooFUAAAAJ)<sup>11</sup>, [Marco Chini](https://www.google.com/search?q=Marco+Chini&oq=Marco+Chini&gs_lcrp=EgRlZGdlKgYIABBFGDsyBggAEEUYOzIGCAEQRRg7MgYIAhAAGB4yBggDEAAYHjIGCAQQABgeMgYIBRAAGB4yBggGEAAYHjIGCAcQRRg8MgYICBBFGDzSAQcyNzlqMGo3qAIAsAIA&sourceid=chrome&ie=UTF-8)<sup>12</sup>, [Claudio Persello](https://scholar.google.com/citations?user=CI3bxVMAAAAJ)<sup>13</sup>, [Shijian Lu](https://scholar.google.com/citations?user=uYmK-A0AAAAJ)<sup>14</sup>, [Lixin Wang](https://scholar.google.com/citations?user=pQPtUesAAAAJ)<sup>15</sup>, [Zhe Zhu](https://scholar.google.com/citations?user=9ODFYW4AAAAJ)<sup>16</sup> <br>
[Evan Shelhamer](https://scholar.google.com/citations?user=-ltRSM0AAAAJ&hl=zh-CN)<sup>17</sup>, [Jocelyn Chanussot](https://scholar.google.com/citations?user=6owK2OQAAAAJ)<sup>18</sup>, [Konrad Schindler](https://scholar.google.com/citations?user=FZuNgqIAAAAJ)<sup>4</sup>, [Naoto Yokoya](https://scholar.google.com/citations?user=DJ2KOn8AAAAJ)<sup>1,2</sup>

<sup>†</sup>Equal contribution
<br>
<sup>1 </sup>RIKEN AIP, <sup>2 </sup>The University of Tokyo, <sup>3 </sup>Brown University, <sup>4 </sup>ETH Zurich, <sup>5 </sup>United Nations Satellite Centre, 
<br>
<sup>6 </sup>Barcelona School of Economics, <sup>7 </sup>Technical University of Munich, <sup>8 </sup>Stanford University, <sup>9 </sup>Linköping University, <sup>10 </sup>German Aerospace Center (DLR), 
<br>
<sup>11 </sup>Texas A&amp;M University, <sup>12 </sup>Luxembourg Institute of Science and Technology, <sup>13 </sup>University of Twente, <sup>14 </sup>Nanyang Technological University, 
<br>
<sup>15 </sup>Indiana University Indianapolis, <sup>16 </sup>The University of British Columbia, <sup>17 </sup>University of Connecticut, <sup>18 </sup>Université Grenoble Alpes, 

[**Overview**](#overview) | [**Installation**](#installation) | [**Dataset Preparation**](#dataset-preparation) | [**Pretrained Weights**](#pretrained-weights) | [**Quick Start**](#quick-start) | [**Repo Layout**](#repository-layout) 

</div>

<!-- 
## Contents

- [Overview](#overview)
- [Installation](#installation)
- [Pretrained Weights](#pretrained-weights)
- [Quick Start](#quick-start)
- [Repository Layout](#repository-layout)
- [Dataset Preparation](#dataset-preparation)
- [Architecture And Extension](#architecture-and-extension) -->

## 🔭Overview
*Any Disaster Mapping* is the official repository for our review paper in EO-based disaster mapping.

One of our key motivations is that current disaster mapping research is highly fragmented: benchmarks, tasks, and model implementations are often inconsistent across papers, making fair evaluation, reproduction, and further development unnecessarily difficult.

This repo unifies widely used disaster mapping benchmarks and representative deep learning models across major research directions, and provides a consistent training and evaluation pipeline for:
- Infrastructure damage
- Flood mapping
- Landslide segmentation
- Wildfire analysis

It is designed to help researchers:
- Reproduce the results reported in our paper
- Evaluate models under a unified protocol
- Use strong baselines out of the box
- Build and test their own improvements with minimal engineering overhead

## 🛠️Installation

Base environment and optional model-specific extras.

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

## 🧪Dataset Preparation

Dataset preparation guides are organized by disaster domain:

- Infrastructure damage: [scripts/data_prep/infra_damage/README.md](scripts/data_prep/infra_damage/README.md)
- Flood: [scripts/data_prep/flood/README.md](scripts/data_prep/flood/README.md)
- Landslide: [scripts/data_prep/landslide/README.md](scripts/data_prep/landslide/README.md)
- Wildfire: [scripts/data_prep/wildfire/README.md](scripts/data_prep/wildfire/README.md)


## 📦Pretrained Weights

 Recommended local checkpoint layout for the supported model zoo.

Create the local checkpoint directory first:

```bash
mkdir -p pretrained_weight
```

### Direct Downloads

```bash
# pretrain-vit-base-e199.pth
wget -O pretrained_weight/pretrain-vit-base-e199.pth \
  https://zenodo.org/records/7338613/files/pretrain-vit-base-e199.pth

# SpectralGPT+.pth
wget -O "pretrained_weight/SpectralGPT+.pth" \
  "https://zenodo.org/records/8412455/files/SpectralGPT+.pth?download=1"

# spec-vit-base-ultra-checkpoint-1599.pth
wget -O pretrained_weight/spec-vit-base-ultra-checkpoint-1599.pth \
  https://huggingface.co/WHU-Sigma/HyperSIGMA/resolve/main/spec-vit-base-ultra-checkpoint-1599.pth

huggingface-cli download UTokyo-Yokoya-Lab/AnyDisaster-Pretrained_Weight \
  vssm_tiny_0230_ckpt_epoch_262.pth --local-dir pretrained_weight --local-dir-use-symlinks False
```

### Additional Upstream Checkpoints

- DINOv3
  Source: https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/
  Download and save as:
  - `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` (ViT-B/16, LVD-1689M)
  - `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` (ViT-L/16, LVD-1689M)
  - `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth` (ViT-L/16, SAT-493M)

- SAM v1
  Source: https://github.com/facebookresearch/segment-anything
  Download and save as:
  - `sam_vit_b_01ec64.pth` (SAM ViT-B)
  - `sam_vit_l_0b3195.pth` (SAM ViT-L)

- SAM 2.1
  Source: https://github.com/facebookresearch/sam2
  Download and save as:
  - `sam2.1_hiera_small.pt` (SAM 2.1 Hiera-Small)
  - `sam2.1_hiera_base_plus.pt` (SAM 2.1 Hiera-Base+)

- SegFormer MiT encoders
  Source: https://github.com/NVlabs/SegFormer
  Download and save as:
  - `mit_b0.pth`
  - `mit_b1.pth`
  - `mit_b2.pth`
  - `mit_b3.pth`
  - `mit_b4.pth`
  - `mit_b5.pth`

- HyperSIGMA spatial backbone
  Source: https://huggingface.co/WHU-Sigma/HyperSIGMA
  Download the upstream file, rename, and save as:
  - `HSI_spatial_checkpoint-1600.pth`

- SkySense backbone
  Source:
  - https://github.com/Jack-bo1220/SkySense
  - https://www.notion.so/SkySense-Checkpoints-a7fcff6ce29a4647a08c7fe416910509
  Select the `hr` (high-resolution RGB / RGBNIR) variant, not the `s2` Sentinel-2 variant.
  Save as:
  - `skysense_model_backbone_hr.pth`
  For commercial use, contact the authors (`yansheng.li@whu.edu.cn`).

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

## 🚀 Quick Start

Minimal training and evaluation commands.

### Train

Train with a YAML config:

```bash
python train.py --config configs/infra/xbd/unet.yaml
```

### Evaluate

Evaluate an experiment directory:

```bash
python test.py --exp_path results/xbd/unet
```

## 🗂️Repository Layout

 Main directories and what they are responsible for:

- `src/core/`: trainer, config loader, registry, augmentation, metrics
- `src/tasks/`: task handlers for segmentation, change detection, and semantic change detection
- `src/datasets/`: dataset adapters and runtime data contracts
- `src/models/`: model wrappers and vendored third-party implementations
- `configs/`: experiment configs grouped by domain and dataset
- `scripts/data_prep/`: dataset preparation guides and helper scripts
- `docs/`: architecture notes and extension guidance


## 🏗️Architecture And Extension

Internal runtime design and extension entry points.

- Architecture overview: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Extension guide: [docs/EXTENSION_GUIDE.md](docs/EXTENSION_GUIDE.md)

## 📜Reference
If this code repo contributes to your research, please kindly consider citing our paper and give this repo ⭐️ :)
```

```


## 🙋Q & A
For any questions, please feel free to leave it in the issue section.
