import numpy as np

from src.common.evaluation_metrics import Evaluator


def per_class_metrics(confusion_matrix):
    """Compute precision/recall/F1/IoU per class from a confusion matrix."""
    tp = np.diag(confusion_matrix)
    fp = confusion_matrix.sum(axis=0) - tp
    fn = confusion_matrix.sum(axis=1) - tp
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


class MetricsManager:
    """Manages Evaluator instances based on task type.

    For CD tasks (xBD/BRIGHT): 3 evaluators
        - evaluator_loc   (num_class=2)       binary localization
        - evaluator_clf   (num_class=num_classes) damage classification only
        - evaluator_total (num_class=num_classes) full image

    For Seg tasks (RescueNet): 1 evaluator
        - evaluator_total (num_class=num_classes)
    """

    def __init__(self, task, num_classes):
        self.task = task.lower()
        self.num_classes = num_classes

        self.evaluator_total = Evaluator(num_classes)

        if self.task == "cd":
            self.evaluator_loc = Evaluator(2)
            self.evaluator_clf = Evaluator(num_classes)

    def reset(self):
        """Reset all evaluators."""
        self.evaluator_total.reset()
        if self.task == "cd":
            self.evaluator_loc.reset()
            self.evaluator_clf.reset()

    def update(self, preds, labels, loc_preds=None, loc_labels=None):
        """Update the appropriate evaluators with a batch of predictions.

        Args:
            preds: Full classification predictions.
            labels: Full classification ground-truth labels.
            loc_preds: Binary localization predictions (CD only).
            loc_labels: Binary localization ground-truth labels (CD only).
        """
        self.evaluator_total.add_batch(labels, preds)

        if self.task == "cd":
            # Binary localization evaluator
            if loc_preds is not None and loc_labels is not None:
                self.evaluator_loc.add_batch(loc_labels, loc_preds)

            # Damage classification evaluator: only pixels where ground-truth
            # indicates damage (label > 0)
            damage_mask = labels > 0
            if damage_mask.any():
                self.evaluator_clf.add_batch(labels[damage_mask], preds[damage_mask])

    def compute(self):
        """Compute and return all metrics as a dict.

        Always includes:
            OA, mIoU, per_class_iou, per_class_f1,
            per_class_precision, per_class_recall

        CD tasks additionally include:
            loc_f1, damage_f1_scores
        """
        # Per-class metrics from the total evaluator's confusion matrix
        pcm = per_class_metrics(self.evaluator_total.confusion_matrix)

        metrics = {
            "OA": self.evaluator_total.Pixel_Accuracy(),
            "mIoU": self.evaluator_total.Mean_Intersection_over_Union(),
            "per_class_iou": pcm["iou"],
            "per_class_f1": pcm["f1"],
            "per_class_precision": pcm["precision"],
            "per_class_recall": pcm["recall"],
        }

        if self.task == "cd":
            metrics["loc_f1"] = self.evaluator_loc.Pixel_F1_score()
            per_class_f1 = self.evaluator_clf.Damage_F1_score()
            metrics["clf_f1_per_class"] = per_class_f1
            # Harmonic mean of per-class F1 (same as teammate's clf_f1)
            metrics["clf_f1"] = float(len(per_class_f1) / np.sum(1.0 / (per_class_f1 + 1e-10)))

        return metrics

    def best_metric_value(self):
        """Return the primary scalar metric used for model selection (mIoU)."""
        return self.evaluator_total.Mean_Intersection_over_Union()
