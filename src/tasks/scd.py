"""SCD (Semantic Change Detection) task handler.

Extends TaskCD for dual-temporal input handling.
Dataset returns 6-tuple: (pre, post, cd_label, t1_label, t2_label, id)
Model returns 3-tuple: (output_bcd, output_T1, output_T2)
"""

import torch
import torch.nn.functional as F
from src.tasks.cd import TaskCD


class TaskSCD(TaskCD):
    """Semantic Change Detection task handler."""

    def unpack_batch(self, batch, device):
        pre, post, cd_label, t1_label, t2_label, _ = batch
        pre = pre.to(device, dtype=torch.float32)
        post = post.to(device, dtype=torch.float32)
        cd_label = cd_label.to(device, dtype=torch.long)
        t1_label = t1_label.to(device, dtype=torch.long)
        t2_label = t2_label.to(device, dtype=torch.long)
        return (pre, post), t2_label, {"cd_label": cd_label, "t1_label": t1_label}

    # run_model — inherited from TaskCD (dual-input or channel-concat)

    def compute_loss(self, logits, label, aux_label, criterion_clf, criterion_loc):
        """SCD 3-head loss matching ChangeMamba train_MambaSCD.py exactly."""
        output_cd, output_t1, output_t2 = logits
        cd_label = aux_label["cd_label"]
        t1_label = aux_label["t1_label"]
        t2_label = label

        t1_masked = t1_label.clone()
        t1_masked[t1_masked == 0] = 255
        t2_masked = t2_label.clone()
        t2_masked[t2_masked == 0] = 255

        loss_cd = criterion_clf(output_cd, cd_label)
        loss_t1 = criterion_clf(output_t1, t1_masked)
        loss_t2 = criterion_clf(output_t2, t2_masked)

        sim_mask = (t1_masked == 255).float().unsqueeze(1).expand_as(output_t1)
        sim_loss = F.mse_loss(
            F.softmax(output_t1, dim=1) * sim_mask,
            F.softmax(output_t2, dim=1) * sim_mask,
            reduction='mean',
        )
        return loss_cd + 0.5 * (loss_t1 + loss_t2 + 0.5 * sim_loss)

    def extract_predictions(self, logits, label, aux_label, resolve_fn, is_dual_head):
        """SCD: bypass resolve_fn, directly unpack 3-tuple for SCDMetricsManager."""
        output_cd, output_t1, output_t2 = logits
        cd_preds = torch.argmax(output_cd, dim=1).cpu().numpy()
        t1_preds = torch.argmax(output_t1, dim=1).cpu().numpy() * cd_preds
        t2_preds = torch.argmax(output_t2, dim=1).cpu().numpy() * cd_preds
        return {
            "cd_preds": cd_preds,
            "t1_preds": t1_preds,
            "t2_preds": t2_preds,
            "cd_labels": aux_label["cd_label"].cpu().numpy(),
            "t1_labels": aux_label["t1_label"].cpu().numpy(),
            "t2_labels": label.cpu().numpy(),
        }

    def _create_metrics(self, num_classes):
        from src.core.scd_metrics import SCDMetricsManager
        return SCDMetricsManager(num_classes=num_classes)

    def evaluate_sliding(self, model, model_name, eval_loader, kernel, stride,
                         device, resolve_fn, sw_batch, is_dual_head):
        """SCD sliding window: 3 independent accumulators for cd/t1/t2 heads."""
        from tqdm import tqdm
        from src.core.sliding_window import compute_windows
        from src.core.registry import DUAL_INPUT_MODELS
        import numpy as np

        num_classes_cd = 2
        num_classes_sem = self.metrics_mgr.num_classes

        forward_fn = None
        info = DUAL_INPUT_MODELS.get(model_name)
        if info and info["forward"] == "dual":
            names = info["arg_names"]
            forward_fn = lambda m, pre_b, post_b: m(**{names[0]: pre_b, names[1]: post_b})

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Eval [sliding]", leave=False):
                inp, label, aux_label = self.unpack_batch(batch, device)
                pre_batch, post_batch = inp
                for i in range(pre_batch.shape[0]):
                    pre_i = pre_batch[i].cpu()
                    post_i = post_batch[i].cpu()
                    H, W = pre_i.shape[1], pre_i.shape[2]
                    boxes = compute_windows((H, W), kernel, stride)

                    pre_patches = [pre_i[:, int(b[1]):int(b[3]), int(b[0]):int(b[2])] for b in boxes]
                    post_patches = [post_i[:, int(b[1]):int(b[3]), int(b[0]):int(b[2])] for b in boxes]

                    merged_cd = torch.zeros((1, num_classes_cd, H, W), device=device)
                    merged_t1 = torch.zeros((1, num_classes_sem, H, W), device=device)
                    merged_t2 = torch.zeros((1, num_classes_sem, H, W), device=device)
                    counts = torch.zeros((1, 1, H, W), device=device)

                    bs = len(pre_patches) if sw_batch <= 0 else sw_batch
                    for k in range(0, len(pre_patches), bs):
                        pb = torch.stack(pre_patches[k:k + bs]).to(device)
                        qb = torch.stack(post_patches[k:k + bs]).to(device)
                        if forward_fn is not None:
                            out_cd, out_t1, out_t2 = forward_fn(model, pb, qb)
                        else:
                            out_cd, out_t1, out_t2 = model(torch.cat([pb, qb], dim=1))

                        prob_cd = torch.softmax(out_cd, dim=1)
                        prob_t1 = torch.softmax(out_t1, dim=1)
                        prob_t2 = torch.softmax(out_t2, dim=1)

                        for j, box in enumerate(boxes[k:k + bs]):
                            x0, y0, x1, y1 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                            merged_cd[:, :, y0:y1, x0:x1] += prob_cd[j:j + 1]
                            merged_t1[:, :, y0:y1, x0:x1] += prob_t1[j:j + 1]
                            merged_t2[:, :, y0:y1, x0:x1] += prob_t2[j:j + 1]
                            counts[:, :, y0:y1, x0:x1] += 1

                    cd_preds = (merged_cd / (counts + 1e-8)).argmax(dim=1).squeeze(0).cpu().numpy()
                    t1_preds = (merged_t1 / (counts + 1e-8)).argmax(dim=1).squeeze(0).cpu().numpy() * cd_preds
                    t2_preds = (merged_t2 / (counts + 1e-8)).argmax(dim=1).squeeze(0).cpu().numpy() * cd_preds

                    cd_labels_i = aux_label["cd_label"][i].cpu().numpy()
                    t1_labels_i = aux_label["t1_label"][i].cpu().numpy()
                    t2_labels_i = label[i].cpu().numpy()

                    self.metrics_mgr.update(
                        cd_preds=cd_preds[np.newaxis], t1_preds=t1_preds[np.newaxis],
                        t2_preds=t2_preds[np.newaxis], cd_labels=cd_labels_i[np.newaxis],
                        t1_labels=t1_labels_i[np.newaxis], t2_labels=t2_labels_i[np.newaxis],
                    )

    def augmentation_targets(self):
        return {"image_post": "image", "mask_t1": "mask", "mask_cd": "mask"}

    def format_eval_log(self, split, metrics):
        return (f"[{split}] IoU_mean={metrics.get('IoU_mean', 0)*100:.2f}%  "
                f"Sek={metrics.get('Sek', 0)*100:.2f}%  "
                f"Fscd={metrics.get('Fscd', 0)*100:.2f}%")
