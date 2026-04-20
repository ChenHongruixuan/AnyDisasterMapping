"""Base TaskHandler — default behavior is Seg (single-image segmentation)."""

import torch


class TaskHandler:
    """Base task handler. Default behavior = Seg task.

    Holds metrics_mgr as internal state after init_metrics() is called.
    """

    def __init__(self):
        self.metrics_mgr = None

    def init_metrics(self, num_classes):
        """Create and store task-specific MetricsManager."""
        self.metrics_mgr = self._create_metrics(num_classes)

    def _create_metrics(self, num_classes):
        """Override in subclasses to return task-specific MetricsManager."""
        from src.core.metrics import MetricsManager
        return MetricsManager(task="seg", num_classes=num_classes)

    def reset_metrics(self):
        self.metrics_mgr.reset()

    def compute_metrics(self):
        return self.metrics_mgr.compute()

    def update_metrics(self, **kwargs):
        self.metrics_mgr.update(**kwargs)

    def best_metric_value(self):
        return self.metrics_mgr.best_metric_value()

    # ------------------------------------------------------------------
    # Task-specific dispatch (7 methods)
    # ------------------------------------------------------------------

    def unpack_batch(self, batch, device):
        """Seg: 3-tuple (image, label, id) → (image, label, None)."""
        image, label, _ = batch
        image = image.to(device, dtype=torch.float32)
        label = label.to(device, dtype=torch.long)
        return image, label, None

    def run_model(self, model, model_name, inp, label=None, builtin_loss=False):
        """Seg: model(image), or dual-input for HyperSigma."""
        from src.core.registry import DUAL_INPUT_MODELS
        if builtin_loss:
            return model(inp, labels=label)
        info = DUAL_INPUT_MODELS.get(model_name)
        if info and info["forward"] == "dual":
            names = info["arg_names"]
            return model(**{names[0]: inp, names[1]: inp})
        return model(inp)

    def compute_loss(self, logits, label, aux_label, criterion_clf, criterion_loc):
        """Seg: criterion_clf(logits, label). Dual-head falls through from base."""
        if isinstance(logits, tuple) and len(logits) == 2:
            loc_logits, clf_logits = logits
            return criterion_loc(loc_logits, aux_label) + criterion_clf(clf_logits, label)
        return criterion_clf(logits, label)

    def extract_predictions(self, logits, label, aux_label, resolve_fn, is_dual_head):
        """Seg: resolve → argmax → {"preds", "labels"}."""
        logits = resolve_fn(logits)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        return {"preds": preds, "labels": label.cpu().numpy()}

    def augmentation_targets(self):
        """Seg: no additional targets."""
        return None

    def evaluate_sliding(self, model, model_name, eval_loader, kernel, stride,
                         device, resolve_fn, sw_batch, is_dual_head):
        """Seg sliding window: uses self.metrics_mgr internally."""
        from tqdm import tqdm
        from src.core.sliding_window import sliding_window_inference
        from src.core.registry import DUAL_INPUT_MODELS

        num_classes = self.metrics_mgr.num_classes
        forward_fn = None
        info = DUAL_INPUT_MODELS.get(model_name)
        if info and info["forward"] == "dual":
            names = info["arg_names"]
            forward_fn = lambda m, b: m(**{names[0]: b, names[1]: b})

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Eval [sliding]", leave=False):
                inp, label, _ = self.unpack_batch(batch, device)
                for i in range(inp.shape[0]):
                    preds = sliding_window_inference(
                        model, inp[i].cpu(), kernel, stride, num_classes,
                        device, forward_fn=forward_fn,
                        resolve_fn=resolve_fn, batch_size=sw_batch,
                    )
                    self.metrics_mgr.update(preds=preds, labels=label[i].cpu().numpy())

    def format_eval_log(self, split, metrics):
        return (f"[{split}] OA={metrics.get('OA', 0)*100:.2f}%  "
                f"mIoU={metrics.get('mIoU', 0)*100:.2f}%")
