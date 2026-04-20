"""Unified Trainer for change-detection (CD) and segmentation (Seg) tasks."""

from __future__ import annotations

import importlib
import inspect
import logging
import os
import random
import sys
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.common.dice_loss import dice_loss_with_logits
from src.common.lovasz_loss import lovasz_softmax
from src.core.config import Config
from src.core.registry import (
    DEEP_SUPERVISION_MODELS,
    DUAL_INPUT_MODELS,
    dataset_libs,
    model_libs,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logger(name: str, log_dir: str | None = None) -> logging.Logger:
    """Dual-stream logger: console (tqdm-compatible via stderr) + file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, "train.log"))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def _set_seed(seed: int) -> None:
    """Set seeds for random, numpy, torch, and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _infer_encoder_param_prefixes(model: torch.nn.Module) -> tuple[str, ...]:
    """Infer encoder parameter prefixes for differential learning rates.

    Prefer an explicit ``encoder_param_prefixes`` attribute when present.
    Otherwise fall back to common backbone attribute names used in this repo.
    """
    logger = logging.getLogger("trainer")

    explicit = getattr(model, "encoder_param_prefixes", None)
    if explicit:
        prefixes = tuple(str(prefix) for prefix in explicit)
        logger.info("Encoder prefixes (explicit): %s", prefixes)
        return prefixes

    candidate_attrs = (
        "encoder",
        "backbone",
        "image_encoder",
        "spat_encoder",
        "spec_encoder",
        "model",
    )
    prefixes = []
    for attr in candidate_attrs:
        module = getattr(model, attr, None)
        if isinstance(module, torch.nn.Module):
            prefixes.append(f"{attr}.")

    if prefixes:
        logger.info("Encoder prefixes (auto-detected): %s", tuple(prefixes))
    else:
        logger.info("Encoder prefixes: none found (single LR group will be used)")
    return tuple(prefixes)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Unified trainer for CD and Seg tasks.

    Supports Mask2Former (built-in loss), deep-supervision models (DSIFN,
    UNet++), dual-head models (ChangeOS, ChangeMamba, ...), and standard
    single-output models through a single code path.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.task: str = config.task  # "cd" or "seg"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        seed = config.get("seed", 42)
        _set_seed(seed)

        # Append a seed suffix to output_dir for non-default seeds.
        if seed != 42:
            out = config.get("output_dir", "./results")
            config._data["output_dir"] = f"{out}_seed{seed}"

        self.logger = _setup_logger("trainer", config.get("output_dir"))

        self.model = self._build_model()
        self.model.to(self.device)

        # Log model parameter counts
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen = total - trainable
        self.logger.info("Model: %s | Total params: %.2fM | Trainable: %.2fM | Frozen: %.2fM",
                         self.model_name, total / 1e6, trainable / 1e6, frozen / 1e6)

        self.optimizer = self._build_optimizer()
        self.scheduler = None          # built lazily in train() after train_loader is known
        self._pending_scheduler_state = None  # deferred from checkpoint
        self._scheduler_step_mode = "iteration"
        self.criterion_clf, self.criterion_loc = self._build_loss()

        from src.tasks import get_task_handler
        self.task_handler = get_task_handler(self.task)
        self.task_handler.init_metrics(config.num_classes)

        t_cfg_init = config.get("training", {})
        self.metric_for_best = (t_cfg_init.get("metric_for_best_model")
                                or config.get("metric_for_best_model", "mIoU"))
        self.metric_mode = (t_cfg_init.get("metric_mode")
                            or config.get("metric_mode", "max"))
        self.best_metric: float = float("-inf") if self.metric_mode == "max" else float("inf")
        self.current_iter: int = 0
        self.current_epoch: int = 0
        self._early_stopping = self._build_early_stopping()

        # Auto-resume: check for latest.pth in output_dir if no explicit --resume
        if config.resume:
            self._load_checkpoint(config.resume)
        else:
            auto_ckpt = os.path.join(config.get("output_dir", ""), "latest.pth")
            if os.path.isfile(auto_ckpt):
                self.logger.info("Auto-resuming from %s", auto_ckpt)
                self._load_checkpoint(auto_ckpt)

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    def _build_model(self) -> torch.nn.Module:
        model_cfg = self.config.get("model", {})
        name = model_cfg.get("name") if isinstance(model_cfg, dict) else self.config.get("model_name")
        kwargs = model_cfg.get("kwargs", {}) if isinstance(model_cfg, dict) else self.config.get("model_kwargs", {})
        self._model_name = name
        factory = model_libs[name]
        return factory(**kwargs)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def _is_dual_head(self) -> bool:
        model = self.model.module if hasattr(self.model, 'module') else self.model
        if hasattr(model, 'has_loc_head'):
            return model.has_loc_head
        info = DUAL_INPUT_MODELS.get(self.model_name, {})
        return info.get("dual_head", False)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        cfg = self.config
        opt_cfg = cfg.get("optimizer", {})
        opt_name = opt_cfg.get("name", "AdamW")
        lr = opt_cfg.get("learning_rate", cfg.get("learning_rate", 1e-4))
        wd = opt_cfg.get("weight_decay", cfg.get("weight_decay", 1e-2))
        trainable = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]

        enc_lr_weight = opt_cfg.get("enc_lr_weight", cfg.get("enc_lr_weight"))
        if enc_lr_weight is not None:
            enc_lr_weight = float(enc_lr_weight)
            if enc_lr_weight <= 0:
                raise ValueError("optimizer.enc_lr_weight must be positive.")

        encoder_prefixes = _infer_encoder_param_prefixes(self.model)
        if enc_lr_weight and encoder_prefixes:
            enc = [p for n, p in trainable if any(n.startswith(px) for px in encoder_prefixes)]
            dec = [p for n, p in trainable if not any(n.startswith(px) for px in encoder_prefixes)]
            if enc and dec:
                params = [{"params": enc, "lr": lr * enc_lr_weight},
                          {"params": dec, "lr": lr}]
                self.logger.info("Optimizer: %s | enc_lr=%.2e (%d params) | dec_lr=%.2e (%d params) | wd=%.2e",
                                 opt_name, lr * enc_lr_weight, len(enc), lr, len(dec), wd)
            else:
                params = [p for _, p in trainable]
                self.logger.info("Optimizer: %s | lr=%.2e | wd=%.2e | enc_lr_weight=%.2f set but no enc/dec split found",
                                 opt_name, lr, wd, enc_lr_weight)
        else:
            params = [p for _, p in trainable]
            self.logger.info("Optimizer: %s | lr=%.2e | wd=%.2e | %d params",
                             opt_name, lr, wd, len(params))

        opt_class = getattr(torch.optim, opt_name, None)
        if opt_class is None:
            raise ValueError(f"Unknown optimizer: {opt_name}. Must be a valid torch.optim class.")
        return opt_class(params, lr=lr, weight_decay=wd)

    def _build_scheduler(self):
        cfg_sched = self.config.get("scheduler", {})
        if not cfg_sched:
            return None
        name = cfg_sched.get("name") if isinstance(cfg_sched, dict) else None
        if not name:
            return None
        kwargs = cfg_sched.get("kwargs", {})
        if not kwargs:
            # Fallback: treat all non-meta keys as scheduler kwargs
            kwargs = {k: v for k, v in (cfg_sched if isinstance(cfg_sched, dict) else {}).items()
                      if k not in ("name", "step_mode")}
        step_mode = cfg_sched.get("step_mode", "iteration")

        # ReduceLROnPlateau must use epoch-level stepping
        if name == "ReduceLROnPlateau" and step_mode != "epoch":
            self.logger.warning("ReduceLROnPlateau requires step_mode='epoch'; overriding.")
            step_mode = "epoch"

        self._scheduler_step_mode = step_mode

        # WarmUpPolyLR: auto-compute total_iters from training config
        if name == "WarmUpPolyLR" and "total_iters" not in kwargs:
            t_cfg = self.config.get("training", {})
            epochs = t_cfg.get("num_epochs", 100)
            steps = len(self.train_loader) if hasattr(self, "train_loader") else 1
            kwargs = dict(kwargs, total_iters=epochs * steps)

        # Try built-in PyTorch schedulers first, then custom ones
        if hasattr(torch.optim.lr_scheduler, name):
            return getattr(torch.optim.lr_scheduler, name)(self.optimizer, **kwargs)
        # Custom schedulers in src/core/lr_scheduler.py
        from src.core import lr_scheduler as _custom_sched
        if hasattr(_custom_sched, name):
            return getattr(_custom_sched, name)(self.optimizer, **kwargs)
        raise ValueError(f"Unknown scheduler: {name}")

    def _build_early_stopping(self):
        cfg_es = self.config.get("early_stopping", {})
        if not cfg_es or not cfg_es.get("enabled", False):
            return None

        return {
            "patience": int(cfg_es.get("patience", 10)),
            "monitor": cfg_es.get("monitor", self.metric_for_best),
            "mode": cfg_es.get("mode", self.metric_mode),
            "min_delta": float(cfg_es.get("min_delta", 0.0)),
            "best": None,
            "num_bad_epochs": 0,
        }

    def _check_early_stopping(self, metrics: dict) -> bool:
        es = self._early_stopping
        if es is None:
            return False

        value = metrics.get(es["monitor"])
        if value is None:
            self.logger.warning("Early stopping monitor '%s' not in metrics; skipping.", es["monitor"])
            return False

        if es["best"] is None:
            es["best"] = value
            return False

        improved = (value > es["best"] + es["min_delta"]) if es["mode"] == "max" \
            else (value < es["best"] - es["min_delta"])

        if improved:
            es["best"] = value
            es["num_bad_epochs"] = 0
        else:
            es["num_bad_epochs"] += 1

        self.logger.info("EarlyStop: %s=%.5f | patience %d/%d",
                         es["monitor"], value, es["num_bad_epochs"], es["patience"])
        return es["num_bad_epochs"] >= es["patience"]

    def _build_loss(self):
        """Return (criterion_clf, criterion_loc) closures.

        Auxiliary loss modes (mutually exclusive -- dice takes priority):
        - use_dice  : CE + dice_weight * Dice
        - use_lovasz: CE + lovasz_weight * Lovasz
        - neither   : pure CE

        Config keys (under ``loss:``):
            use_dice          (bool, default False)
            dice_weight       (float, default 1.0)
            dice_weight_loc   (float, default ``dice_weight``)
            use_lovasz        (bool, default False)
            lovasz_weight     (float, default 0.75)  -- used for clf head
            lovasz_weight_loc (float, default 0.5)   -- used for loc head
        """
        ignore = self.config.get("ignore_index", 255)
        loss_cfg = self.config.get("loss", {})
        if not isinstance(loss_cfg, dict):
            loss_cfg = {}

        use_dice = loss_cfg.get("use_dice", False)
        use_lv = loss_cfg.get("use_lovasz", False)

        # Mutual exclusion: dice takes priority when both are enabled.
        if use_dice and use_lv:
            use_lv = False

        # Dice weights
        d_clf = loss_cfg.get("dice_weight", 1.0)
        d_loc = loss_cfg.get("dice_weight_loc", d_clf)

        # Lovasz weights
        w_clf = loss_cfg.get("lovasz_weight", 0.75)
        w_loc = loss_cfg.get("lovasz_weight_loc", 0.5)

        def _make(lv_w: float, dice_w: float):
            def fn(logits, labels):
                ce = F.cross_entropy(logits, labels, ignore_index=ignore)
                if use_dice:
                    dc = dice_loss_with_logits(logits, labels, ignore_index=ignore)
                    return ce + dice_w * dc
                if use_lv:
                    lv = lovasz_softmax(F.softmax(logits, dim=1), labels, ignore=ignore)
                    return ce + lv_w * lv
                return ce
            return fn
        return _make(w_clf, d_clf), _make(w_loc, d_loc)

    # ------------------------------------------------------------------
    # Dataset / DataLoader
    # ------------------------------------------------------------------

    def _build_dataset(self, split: str):
        from src.core.augmentation import build_transforms

        ds_cfg = self.config.get("dataset", {})
        name = ds_cfg.get("name") if isinstance(ds_cfg, dict) else self.config.get("dataset_name")

        # Get split-specific kwargs from config.dataset[split]
        split_cfg = ds_cfg.get(split, {}) if isinstance(ds_cfg, dict) else {}
        ds_kw: dict[str, Any] = dict(split_cfg)

        # Flatten nested 'kwargs' if present
        if "kwargs" in ds_kw:
            extra = ds_kw.pop("kwargs")
            ds_kw.update(extra)

        # Inject top-level task for flood datasets that need it to
        # determine seg vs cd mode. Only inject if not already set in
        # the split config and if 'task' is not in ds_kw yet.
        # We check by trying to instantiate — if the dataset doesn't
        # accept 'task', we simply don't pass it.
        if "task" not in ds_kw:
            import inspect
            try:
                factory = dataset_libs[name]
                # Resolve actual class through lazy_import closure
                if hasattr(factory, '__closure__') and factory.__closure__:
                    cells = [c.cell_contents for c in factory.__closure__
                             if isinstance(c.cell_contents, str)]
                    if len(cells) >= 2:
                        mod = importlib.import_module(cells[1])
                        cls = getattr(mod, cells[0])
                        if "task" in inspect.signature(cls.__init__).parameters:
                            ds_kw["task"] = self.task
            except Exception:
                pass

        # Pass dataset.input section as input_cfg for datasets that accept it
        # (e.g. flood datasets). Uses the same introspection approach as task
        # injection above to avoid passing it to datasets that don't expect it.
        input_section = ds_cfg.get("input")
        if input_section and "input_cfg" not in ds_kw:
            try:
                factory = dataset_libs[name]
                if hasattr(factory, '__closure__') and factory.__closure__:
                    cells = [c.cell_contents for c in factory.__closure__
                             if isinstance(c.cell_contents, str)]
                    if len(cells) >= 2:
                        mod = importlib.import_module(cells[1])
                        cls = getattr(mod, cells[0])
                        if "input_cfg" in inspect.signature(cls.__init__).parameters:
                            ds_kw["input_cfg"] = input_section
            except Exception:
                pass

        # Build augmentation transforms from config
        aug_cfg = self.config.get("augmentation", {})
        split_aug = aug_cfg.get("train" if split == "train" else "val", {})
        if split_aug:
            ds_kw["transforms"] = build_transforms(split_aug, task=self.task)

        return dataset_libs[name](**ds_kw)

    def _make_loader(self, dataset, *, train: bool) -> DataLoader:
        t_cfg = self.config.get("training", {})
        bs = t_cfg.get("train_batch_size", 8) if train else t_cfg.get("eval_batch_size", 1)
        nw = t_cfg.get("num_workers", 4)
        return DataLoader(
            dataset,
            batch_size=bs, shuffle=train, drop_last=train, pin_memory=True,
            num_workers=nw,
        )

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _unpack_batch(self, batch):
        """Move batch tensors to device. Delegates to task_handler."""
        return self.task_handler.unpack_batch(batch, self.device)

    def _run_model(self, inp, label=None):
        """Run model on unpacked input. Delegates to task_handler."""
        builtin = self.config.get("model_has_builtin_loss", False)
        return self.task_handler.run_model(
            self.model, self.model_name, inp,
            label=label, builtin_loss=builtin,
        )

    # ------------------------------------------------------------------
    # Train step
    # ------------------------------------------------------------------

    def _train_step(self, batch) -> float:
        """Single training iteration. Three output branches:

        1. Mask2Former (config.model_has_builtin_loss): use outputs.loss
        2. Deep supervision (list len > 2): weighted per-scale CE+Lovasz
        3. Standard: single tensor -> criterion_clf; dual-head tuple ->
           criterion_loc(loc) + criterion_clf(clf)
        """
        inp, label, loc_label = self._unpack_batch(batch)
        if not (label != 255).any():
            return 0.0

        # Branch 1: Mask2Former built-in loss
        if self.config.get("model_has_builtin_loss", False):
            outputs = self._run_model(inp, label=label)
            loss = outputs.loss
        else:
            logits = self._run_model(inp)

            # Branch 2: Deep supervision (only for registered models)
            ds_meta = DEEP_SUPERVISION_MODELS.get(self.model_name)
            if (ds_meta is not None
                    and isinstance(logits, (list, Sequence))
                    and not isinstance(logits, torch.Tensor)
                    and len(logits) > 2):
                weights = ds_meta.get("weights") or [1.0] * len(logits)
                loss = sum(w * self.criterion_clf(lg, label)
                           for w, lg in zip(weights, logits) if w > 0)

            # Branch 3: task-specific (dual-head, single, or SCD 3-head)
            else:
                loss = self.task_handler.compute_loss(
                    logits, label, loc_label,
                    self.criterion_clf, self.criterion_loc,
                )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler and self._scheduler_step_mode == "iteration":
            self.scheduler.step()
        return loss.item()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Epoch-based training loop."""
        cfg = self.config
        t_cfg = cfg.get("training", {})
        train_loader = self._make_loader(self._build_dataset("train"), train=True)
        self.train_loader = train_loader

        # Build scheduler here (deferred from __init__) so train_loader length is known
        if self.scheduler is None:
            self.scheduler = self._build_scheduler()
            # Restore scheduler state that was stashed during _load_checkpoint
            if self._pending_scheduler_state is not None and self.scheduler is not None:
                self.scheduler.load_state_dict(self._pending_scheduler_state)
                self.logger.info("Restored scheduler state from checkpoint")
                self._pending_scheduler_state = None

        num_epochs = t_cfg.get("num_epochs", 1)
        max_steps = t_cfg.get("max_steps")  # per-epoch step limit (for smoke tests)
        log_every = t_cfg.get("log_interval", 50)
        eval_every = t_cfg.get("eval_interval")  # None = auto (once per epoch)
        it = self.current_iter

        # If eval_interval not set, default to once per epoch
        if not eval_every:
            eval_every = len(train_loader)

        opt_cfg = cfg.get("optimizer", {})
        lr_display = opt_cfg.get("learning_rate", cfg.get("learning_rate", 0))
        bs_display = t_cfg.get("train_batch_size", 8)
        self.logger.info(
            "Training start | epochs=%d  bs=%d  lr=%.2e  eval_interval=%d",
            num_epochs, bs_display, lr_display, eval_every,
        )

        from tqdm import tqdm

        # Resume: skip completed epochs
        start_epoch = self.current_epoch
        if start_epoch > 0:
            self.logger.info("Resuming from epoch %d (iter %d, skipping %d completed epochs)",
                             start_epoch + 1, it, start_epoch)

        last_completed_epoch = start_epoch  # tracks: next resume starts here
        last_eval_metric = None  # tracks last validation metric for ReduceLROnPlateau
        stop_training = False
        for epoch in range(start_epoch, num_epochs):
            if stop_training:
                break
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
            for batch in pbar:
                loss = self._train_step(batch)
                it += 1
                n_batches += 1
                epoch_loss += loss
                pbar.set_postfix(loss=f"{loss:.4f}", lr=f"{self.optimizer.param_groups[0]['lr']:.2e}")
                if max_steps is not None and n_batches >= max_steps:
                    break

                if it % log_every == 0:
                    self.logger.info("Iter %d | loss: %.4f | lr: %.2e",
                                     it, loss, self.optimizer.param_groups[0]["lr"])

                if it % eval_every == 0:
                    avg_loss = epoch_loss / max(n_batches, 1)
                    metrics = self.evaluate("val")
                    current_val = metrics.get(self.metric_for_best)
                    if current_val is None:
                        self.logger.warning(
                            "metric_for_best_model='%s' not found in validation metrics; falling back to mIoU.",
                            self.metric_for_best,
                        )
                        current_val = metrics.get("mIoU", 0.0)
                    self.logger.info("Epoch %d/%d | iter %d | avg_loss: %.4f | val_%s: %.4f",
                                     epoch + 1, num_epochs, it, avg_loss,
                                     self.metric_for_best, current_val)
                    # End-of-epoch eval: store epoch+1 for exact resume.
                    # Mid-epoch eval: store epoch so resume reruns the current epoch.
                    save_epoch = epoch + 1 if n_batches >= len(train_loader) else epoch
                    self._save_checkpoint(it, save_epoch, tag="latest")
                    improved = (current_val > self.best_metric) if self.metric_mode == "max" \
                        else (current_val < self.best_metric)
                    if improved:
                        self.best_metric = current_val
                        self._save_checkpoint(it, save_epoch, tag="best")
                        self.logger.info("New best %s: %.4f", self.metric_for_best, current_val)

                    last_eval_metric = current_val

                    # Early stopping
                    if self._check_early_stopping(metrics):
                        self.logger.info("Early stopping at epoch %d iter %d", epoch + 1, it)
                        stop_training = True
                        break

                    self.model.train()
            # Epoch fully completed if all batches processed
            if n_batches >= len(train_loader):
                # Epoch-level scheduler step (once per completed epoch)
                if self.scheduler and self._scheduler_step_mode == "epoch":
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        # Use the most recent validation metric
                        if last_eval_metric is not None:
                            self.scheduler.step(last_eval_metric)
                        else:
                            self.scheduler.step(self.best_metric)
                    else:
                        self.scheduler.step()
                last_completed_epoch = epoch + 1
                # Always save latest at end of epoch so mid-epoch crashes
                # don't lose more than one epoch of work.
                self._save_checkpoint(it, last_completed_epoch, tag="latest")

        self.current_iter = it
        self.current_epoch = last_completed_epoch
        self._save_checkpoint(it, last_completed_epoch, tag="latest")
        self.logger.info("Training complete at epoch %d, iter %d",
                         last_completed_epoch, it)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _resolve_logits(self, logits):
        """Collapse deep-supervision lists, dual-head tuples, and Mask2Former
        output objects to a single logits tensor suitable for argmax."""
        # Deep supervision: list with > 2 elements needs model-specific aggregation
        if (isinstance(logits, (list, Sequence))
                and not isinstance(logits, torch.Tensor)
                and len(logits) > 2):
            ds_meta = DEEP_SUPERVISION_MODELS.get(self.model_name, {})
            agg = ds_meta.get("eval_agg", "first")
            if agg == "average":
                return torch.stack(list(logits), dim=0).mean(dim=0)
            return logits[0]
        # Dual-head: take classification branch
        if isinstance(logits, tuple) and len(logits) == 2:
            return logits[1]
        # Mask2Former-style outputs expose a .logits attribute.
        if hasattr(logits, 'logits'):
            return logits.logits
        return logits

    def evaluate(self, split: str = "val") -> dict[str, Any]:
        """Run evaluation on *split* and return metrics dict.

        If config has `inference.sliding_kernel`, uses sliding window inference
        to handle images larger than the model's training crop size.
        """
        self.model.eval()
        self.task_handler.reset_metrics()
        from tqdm import tqdm

        inf_cfg = self.config.get("inference", {})
        sliding_kernel = inf_cfg.get("sliding_kernel") if isinstance(inf_cfg, dict) else None
        sliding_stride = inf_cfg.get("sliding_stride") if isinstance(inf_cfg, dict) else None
        if sliding_kernel and not sliding_stride:
            sliding_stride = sliding_kernel // 2  # default 50% overlap

        eval_dataset = self._build_dataset(split)
        if sliding_kernel:
            # Sliding window processes one image at a time; force batch_size=1
            eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False,
                                     pin_memory=True, num_workers=self.config.get("training", {}).get("num_workers", 4))
        else:
            eval_loader = self._make_loader(eval_dataset, train=False)

        if sliding_kernel:
            sw_batch = inf_cfg.get("sliding_batch_size", -1) if isinstance(inf_cfg, dict) else -1
            self.logger.info("[%s] Sliding window eval: kernel=%d stride=%d batch_size=%s samples=%d",
                             split, sliding_kernel, sliding_stride,
                             "all" if sw_batch <= 0 else sw_batch,
                             len(eval_loader.dataset))
            self._evaluate_sliding(eval_loader, sliding_kernel, sliding_stride)
        else:
            self.logger.info("[%s] Direct eval: samples=%d", split, len(eval_loader.dataset))
            self._evaluate_direct(eval_loader)

        metrics = self.task_handler.compute_metrics()
        self.logger.info(self.task_handler.format_eval_log(split, metrics))
        return metrics

    def _evaluate_direct(self, eval_loader):
        """Standard evaluation: direct model forward on each batch."""
        from tqdm import tqdm
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Eval", leave=False):
                inp, label, loc_label = self._unpack_batch(batch)
                outputs = self._run_model(inp, label=label)

                if self.config.get("model_has_builtin_loss", False):
                    logits = outputs.logits
                else:
                    logits = outputs

                pred_dict = self.task_handler.extract_predictions(
                    logits, label, loc_label,
                    resolve_fn=self._resolve_logits,
                    is_dual_head=self._is_dual_head,
                )
                self.task_handler.update_metrics(**pred_dict)

    def _evaluate_sliding(self, eval_loader, kernel, stride):
        """Sliding window evaluation — delegates to task_handler."""
        inf_cfg = self.config.get("inference", {})
        sw_batch = inf_cfg.get("sliding_batch_size", -1) if isinstance(inf_cfg, dict) else -1

        self.task_handler.evaluate_sliding(
            model=self.model,
            model_name=self.model_name,
            eval_loader=eval_loader,
            kernel=kernel,
            stride=stride,
            device=self.device,
            resolve_fn=self._resolve_logits,
            sw_batch=sw_batch,
            is_dual_head=self._is_dual_head,
        )

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _save_checkpoint(self, iteration: int, epoch: int, tag: str = "") -> None:
        out_dir = self.config.output_dir
        os.makedirs(out_dir, exist_ok=True)

        if tag == "best":
            path = os.path.join(out_dir, "best.pth")
            torch.save(self.model.state_dict(), path)
            self.logger.info("Saved best model -> %s", path)
        else:
            path = os.path.join(out_dir, "latest.pth")
            torch.save({
                "iteration": iteration,
                "epoch": epoch,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "best_metric": self.best_metric,
                "early_stopping": self._early_stopping,
            }, path)
            self.logger.info("Saved latest checkpoint -> %s (epoch %d, iter %d)", path, epoch, iteration)

        cfg_path = os.path.join(out_dir, "config.yaml")
        if not os.path.exists(cfg_path):
            self.config.save_yaml(cfg_path)

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        # SatMAE pos_embed fix: when the checkpoint was trained with a different
        # crop_size, pos_embed shapes mismatch (e.g. 257 vs 197 tokens).
        # We resize the MODEL's pos_embed to match the checkpoint, preserving
        # trained weights exactly rather than interpolating them.
        import math
        model_state = self.model.state_dict()
        for key in list(state_dict.keys()):
            if "pos_embed" in key and key in model_state:
                if state_dict[key].shape != model_state[key].shape:
                    from timm.models.vision_transformer import resample_abs_pos_embed
                    ckpt_embed = state_dict[key]
                    model_embed = model_state[key]
                    num_prefix = 1  # CLS token
                    # Compute checkpoint's grid size
                    ckpt_num_patches = ckpt_embed.shape[1] - num_prefix
                    new_h = int(round(math.sqrt(ckpt_num_patches)))
                    new_w = ckpt_num_patches // new_h
                    # Compute model's current grid size
                    model_num_patches = model_embed.shape[1] - num_prefix
                    old_h = int(round(math.sqrt(model_num_patches)))
                    old_w = model_num_patches // old_h
                    # Resample model's pos_embed to checkpoint's grid
                    resampled = resample_abs_pos_embed(
                        model_embed, new_size=(new_h, new_w),
                        old_size=(old_h, old_w),
                        num_prefix_tokens=num_prefix,
                    )
                    # Update model parameter in-place, then load checkpoint as-is
                    with torch.no_grad():
                        param = dict(self.model.named_parameters()).get(key)
                        if param is not None:
                            param.data = resampled
                        else:
                            # nested attr (e.g. encoder.model.pos_embed)
                            parts = key.split(".")
                            obj = self.model
                            for p in parts[:-1]:
                                obj = getattr(obj, p)
                            setattr(obj, parts[-1], torch.nn.Parameter(resampled))
                    # Also update internal grid size if SatMAEDPT
                    if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, '_pos_grid_size'):
                        self.model.encoder._pos_grid_size = (new_h, new_w)
                    self.logger.info("Resampled model %s: [1,%d,%d] -> [1,%d,%d] to match checkpoint",
                                     key, model_embed.shape[1], model_embed.shape[2],
                                     ckpt_embed.shape[1], ckpt_embed.shape[2])
        self.model.load_state_dict(state_dict, strict=True)
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt and ckpt["scheduler"]:
            if self.scheduler:
                self.scheduler.load_state_dict(ckpt["scheduler"])
            else:
                # Scheduler not yet built (deferred to train()); stash for later
                self._pending_scheduler_state = ckpt["scheduler"]
        if "early_stopping" in ckpt and self._early_stopping is not None:
            self._early_stopping = ckpt["early_stopping"]
        self.current_iter = ckpt.get("iteration", 0)
        self.current_epoch = ckpt.get("epoch", 0)
        default_best = float("-inf") if self.metric_mode == "max" else float("inf")
        self.best_metric = ckpt.get("best_metric", default_best)
        self.logger.info("Resumed from %s  epoch=%d  iter=%d  best=%.4f",
                         path, self.current_epoch, self.current_iter, self.best_metric)
