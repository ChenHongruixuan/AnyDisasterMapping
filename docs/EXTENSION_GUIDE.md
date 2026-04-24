# Extension Guide

## Contents

- [What To Register](#what-to-register)
- [Dataset Contract](#dataset-contract)
- [Model Contract](#model-contract)
- [Task Contract](#task-contract)
- [Example: `second` + `changemamba_scd`](#example-second--changemamba_scd)
- [Validation](#validation)

## What To Register

- Register datasets and models in `src/core/registry.py`.
- Register new tasks in `src/tasks/__init__.py`.
- Add public YAML configs under `configs/<domain>/...` only when the path is ready to support users.

## Dataset Contract

Return the tuple expected by the task:

- `seg`: `(image, label, sample_id)`
- `cd`: `(pre, post, label, sample_id)` or xBD-style `(pre, post, loc_label, label, sample_id)`
- `scd`: `(pre, post, cd_label, t1_label, t2_label, sample_id)`

Current datasets usually return channel-first NumPy arrays plus integer labels; tensors are created later by the dataloader/task handler path. If the dataset uses albumentations, keep its masks aligned with the task handler’s `augmentation_targets()`.

Some datasets also use `dataset.input`; the trainer passes that block as `input_cfg` when the dataset constructor accepts it.

## Model Contract

- Add the wrapper under `src/models/` and register it in `model_libs`.
- If it needs named dual-input forwarding, add an entry to `DUAL_INPUT_MODELS`.
- If it returns localisation/classification heads separately, set `dual_head: true` there as well.
- If it uses deep supervision, add an entry to `DEEP_SUPERVISION_MODELS`.
- If the model computes its own loss, configs must set `model_has_builtin_loss: true`.

Model kwargs are not fully standardized across this repo. Use the wrapper `__init__` signature as the source of truth.

## Task Contract

A new task usually needs:

- `unpack_batch`
- `run_model`
- `compute_loss`
- `extract_predictions`
- `_create_metrics`
- `augmentation_targets`
- optional `evaluate_sliding`

If an existing task already matches your tuple and metric contract, reuse it instead of creating a new one.

## Example: `second` + `changemamba_scd`

This is a code-aligned example of how the current SCD path fits together:

IMPORTANT: this example is only a wiring example for the current runtime
contracts. It has not been validated here as a reproduction recipe, and it
should not be read as a claim that the repository reproduces the original
reported `second + changemamba_scd` performance.

- `dataset.name: second` maps to `src.datasets.second.SECONDDataset`
- `task: scd` maps to `src.tasks.scd.TaskSCD`
- `model.name: changemamba_scd` maps to `src.models.ChangeMamba.ChangeMambaSCD`
- `DUAL_INPUT_MODELS["changemamba_scd"]` tells the trainer to call the model with `pre_data=` and `post_data=`

`SECONDDataset` returns `(pre, post, cd_label, t1_label, t2_label, id)`, `ChangeMambaSCD` returns `(output_cd, output_t1, output_t2)`, and `TaskSCD` supplies the matching loss and metrics. SECOND semantic labels use `0..6`, so `num_classes` for the semantic heads and SCD metrics is `7`, not `6`.

There is no retained public `configs/second/*.yaml` in the current tree. The snippet below is only a minimal starting point.

```yaml
task: scd
num_classes: 7
ignore_index: 255
model:
  name: changemamba_scd
  kwargs:
    output_cd: 2
    output_clf: 7
    in_chans: 3
    pretrained: pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth
    backbone: vssm_tiny_224_0229flex
dataset:
  name: second
  train:
    dataset_path: data/SECOND/train
    data_list_path: data/SECOND/train.txt
    split: train
  val:
    dataset_path: data/SECOND/test
    data_list_path: data/SECOND/test.txt
    split: val
  test:
    dataset_path: data/SECOND/test
    data_list_path: data/SECOND/test.txt
    split: test
training:
  metric_for_best_model: Sek
  train_batch_size: 8
  eval_batch_size: 1
  num_workers: 4
output_dir: ./results/second/changemamba_scd
```

## Validation

- `python train.py --help`
- `python test.py --help`
- parse the YAML with `Config.from_yaml(...)`
- import the registered dataset and model with the exact registry names you plan to use
- run the smallest relevant regression or smoke check before large experiments
