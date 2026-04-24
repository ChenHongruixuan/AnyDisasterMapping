# Architecture

## Contents

- [Runtime Flow](#runtime-flow)
- [Registries And Tasks](#registries-and-tasks)
- [Config-Driven Build](#config-driven-build)
- [Training And Evaluation](#training-and-evaluation)
- [Practical Notes](#practical-notes)

## Runtime Flow

- `train.py` loads YAML through `Config`, applies CLI overrides, optionally sets `resume`, and calls `Trainer.train()`.
- `test.py` loads `config.yaml` from `--exp_path` or an override config, sets `config.resume` to the requested checkpoint, and calls `Trainer.evaluate()`.
- `Trainer` owns the runtime: model build, optimizer, scheduler, task handler, datasets, dataloaders, training loop, evaluation, and checkpointing.

## Registries And Tasks

- `src/core/registry.py` registers datasets and models through `lazy_import(...)`.
- `DUAL_INPUT_MODELS` describes named dual-input forwards and optional `dual_head` behavior for CD models.
- `DEEP_SUPERVISION_MODELS` describes multi-output models such as `DSIFN` and `UNet++`.
- `src/tasks/__init__.py` registers task handlers for `seg`, `cd`, and `scd`.
- Task handlers define batch unpacking, model invocation, loss composition, prediction extraction, metrics, sliding-window evaluation, and augmentation target syncing.

## Config-Driven Build

- Models come from `model.name` plus `model.kwargs`.
- Datasets come from `dataset.name` plus split-specific blocks under `dataset.train`, `dataset.val`, and `dataset.test`.
- `Trainer` flattens nested dataset `kwargs`, and injects `task` or `dataset.input` only when the dataset constructor accepts them.
- Augmentations are built from `augmentation.train` / `augmentation.val`; synced targets come from the task handler, not the dataset class.

## Training And Evaluation``

- The scheduler is built lazily in `train()` because some schedulers need `len(train_loader)`.
- Training has three main branches:
  - built-in loss via `model_has_builtin_loss: true`
  - deep supervision via `DEEP_SUPERVISION_MODELS`
  - standard task-driven loss via the task handler
- `best.pth` stores weights only. `latest.pth` stores model, optimizer, scheduler, epoch/iteration, best metric, and early-stopping state.
- `Trainer.evaluate()` switches to sliding-window mode when `inference.sliding_kernel` is set.
- `test.py` also supports `--test_splits`, saving per-split metrics and a confusion-matrix-based aggregate.

## Practical Notes

- If the main validation metric is not `mIoU`, set `training.metric_for_best_model` explicitly.
- Model kwargs are not fully standardized across wrappers. Use the wrapper `__init__` signature as the source of truth.
- The shared runtime contract is stable; reported upstream performance is not guaranteed without separate experiment validation.
