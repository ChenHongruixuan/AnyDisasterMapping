"""CD (Change Detection) task handler."""

import torch
import numpy as np
from src.tasks.base import TaskHandler


class TaskCD(TaskHandler):
    """Change Detection task handler."""

    def unpack_batch(self, batch, device):
        if len(batch) == 5:
            pre, post, loc_label, label, _ = batch
        elif len(batch) == 4:
            pre, post, label, _ = batch
            loc_label = (label > 0).long()
            loc_label[label == 255] = 255
        else:
            raise ValueError(f"Unexpected CD batch length {len(batch)}")
        pre = pre.to(device, dtype=torch.float32)
        post = post.to(device, dtype=torch.float32)
        label = label.to(device, dtype=torch.long)
        loc_label = loc_label.to(device, dtype=torch.long)
        return (pre, post), label, loc_label

    def run_model(self, model, model_name, inp, label=None, builtin_loss=False):
        from src.core.registry import DUAL_INPUT_MODELS
        pre, post = inp
        if builtin_loss:
            return model(torch.cat([pre, post], dim=1), labels=label)
        info = DUAL_INPUT_MODELS.get(model_name)
        if info and info["forward"] == "dual":
            names = info["arg_names"]
            return model(**{names[0]: pre, names[1]: post})
        return model(torch.cat([pre, post], dim=1))

    def compute_loss(self, logits, label, aux_label, criterion_clf, criterion_loc):
        if isinstance(logits, tuple) and len(logits) == 2:
            loc_logits, clf_logits = logits
            return criterion_loc(loc_logits, aux_label) + criterion_clf(clf_logits, label)
        return criterion_clf(logits, label)

    def extract_predictions(self, logits, label, aux_label, resolve_fn, is_dual_head):
        if is_dual_head and isinstance(logits, tuple) and len(logits) == 2:
            loc_logits, clf_logits = logits
            preds = torch.argmax(clf_logits, dim=1).cpu().numpy()
            loc_preds = torch.argmax(loc_logits, dim=1).cpu().numpy()
        else:
            logits = resolve_fn(logits)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            loc_preds = (preds > 0).astype(np.int64)
        return {
            "preds": preds,
            "labels": label.cpu().numpy(),
            "loc_preds": loc_preds,
            "loc_labels": aux_label.cpu().numpy(),
        }

    def _create_metrics(self, num_classes):
        from src.core.metrics import MetricsManager
        return MetricsManager(task="cd", num_classes=num_classes)

    def augmentation_targets(self):
        return {"image_post": "image"}

    def evaluate_sliding(self, model, model_name, eval_loader, kernel, stride,
                         device, resolve_fn, sw_batch, is_dual_head):
        """CD sliding window: uses self.metrics_mgr internally."""
        from tqdm import tqdm
        from src.core.sliding_window import sliding_window_inference_cd
        from src.core.registry import DUAL_INPUT_MODELS

        num_classes = self.metrics_mgr.num_classes
        forward_fn = None
        info = DUAL_INPUT_MODELS.get(model_name)
        if info and info["forward"] == "dual":
            names = info["arg_names"]
            forward_fn = lambda m, pre_b, post_b: m(**{names[0]: pre_b, names[1]: post_b})

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Eval [sliding]", leave=False):
                inp, label, loc_label = self.unpack_batch(batch, device)
                pre_batch, post_batch = inp
                for i in range(pre_batch.shape[0]):
                    pre_i = pre_batch[i].cpu()
                    post_i = post_batch[i].cpu()
                    label_i = label[i].cpu().numpy()
                    loc_label_i = loc_label[i].cpu().numpy()

                    result = sliding_window_inference_cd(
                        model, pre_i, post_i, kernel, stride, num_classes,
                        device, forward_fn=forward_fn, batch_size=sw_batch,
                        resolve_fn=resolve_fn, dual_head=is_dual_head,
                    )

                    if isinstance(result, tuple) and is_dual_head:
                        loc_preds, clf_preds = result
                        self.metrics_mgr.update(preds=clf_preds, labels=label_i,
                                                loc_preds=loc_preds, loc_labels=loc_label_i)
                    else:
                        preds = result
                        loc_preds = (preds > 0).astype(np.int64)
                        self.metrics_mgr.update(preds=preds, labels=label_i,
                                                loc_preds=loc_preds, loc_labels=loc_label_i)

    def format_eval_log(self, split, metrics):
        return (f"[{split}] OA={metrics.get('OA', 0)*100:.2f}%  "
                f"mIoU={metrics.get('mIoU', 0)*100:.2f}%  "
                f"loc_f1={metrics.get('loc_f1', 0)*100:.2f}%  "
                f"clf_f1={metrics.get('clf_f1', 0)*100:.2f}%")
