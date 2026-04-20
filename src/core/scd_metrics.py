"""SCD (Semantic Change Detection) metrics.

Implements a 37-class composition matrix approach following the ChangeMamba
evaluation methodology (SCDD_eval_all).
"""

import math
import numpy as np
from scipy import stats


def fast_hist(a, b, n):
    """n x n confusion matrix computation."""
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def cal_kappa(hist):
    """Cohen's Kappa coefficient."""
    if hist.sum() == 0:
        return 0
    po = np.diag(hist).sum() / hist.sum()
    pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
    return 0 if pe == 1 else (po - pe) / (1 - pe)


class SCDMetricsManager:
    """SCD metrics: 37-class composition matrix + Sek/Fscd/IoU_mean.

    Encoding: (t1_class - 1) * 6 + t2_class for changed pixels, 0 for unchanged.
    37 classes = 1 no-change + 6x6 semantic change combinations.

    Interface matches MetricsManager (reset, update, compute, best_metric_value).

    update() signature: update(cd_preds=..., t1_preds=..., t2_preds=...,
                               cd_labels=..., t1_labels=..., t2_labels=...)
    """

    def __init__(self, num_classes=7, num_scd_classes=37):
        self.num_classes = num_classes
        self.num_scd_classes = num_scd_classes
        self.hist = np.zeros((num_scd_classes, num_scd_classes))

    def reset(self):
        self.hist = np.zeros((self.num_scd_classes, self.num_scd_classes))

    def _encode_scd(self, t1, t2, cd):
        """Encode (t1, t2, cd) to 37-class composite label.

        When cd==1 but t1 or t2 is 0 (background), (0-1)*6+t2 produces
        negative values. Clamp to 0 (treat as no-change) to match the
        implicit assumption that changed pixels have non-zero semantics.
        """
        scd = (t1 - 1) * 6 + t2
        scd[cd == 0] = 0
        scd[scd < 0] = 0
        return scd

    def update(self, *, cd_preds, t1_preds, t2_preds, cd_labels, t1_labels, t2_labels):
        """Accumulate 37-class composition matrix from batch predictions."""
        preds_scd = self._encode_scd(t1_preds, t2_preds, cd_preds)
        labels_scd = self._encode_scd(t1_labels, t2_labels, cd_labels)

        for pred, label in zip(preds_scd, labels_scd):
            self.hist += fast_hist(label.flatten(), pred.flatten(), self.num_scd_classes)

    def compute(self):
        """Compute all SCD metrics from accumulated hist matrix."""
        hist = self.hist

        # Binary change/no-change confusion matrix
        c2hist = np.zeros((2, 2))
        c2hist[0][0] = hist[0][0]
        c2hist[0][1] = hist.sum(1)[0] - hist[0][0]
        c2hist[1][0] = hist.sum(0)[0] - hist[0][0]
        c2hist[1][1] = hist[1:, 1:].sum()

        # Kappa (excluding no-change diagonal)
        hist_n0 = hist.copy()
        hist_n0[0][0] = 0
        kappa_n0 = cal_kappa(hist_n0)

        # IoU
        iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist) + 1e-10)
        IoU_fg = iu[1]
        IoU_mean = (iu[0] + iu[1]) / 2

        # Sek
        Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e

        # Fscd
        pixel_sum = hist.sum()
        change_pred_sum = pixel_sum - hist.sum(1)[0].sum()
        change_label_sum = pixel_sum - hist.sum(0)[0].sum()
        SC_TP = np.diag(hist[1:, 1:]).sum()
        SC_Precision = SC_TP / (change_pred_sum + 1e-10)
        SC_Recall = SC_TP / (change_label_sum + 1e-10)
        Fscd = stats.hmean([SC_Precision, SC_Recall]) if (SC_Precision > 0 and SC_Recall > 0) else 0.0

        return {
            "Sek": Sek,
            "Fscd": Fscd,
            "IoU_mean": IoU_mean,
            "kappa_n0": kappa_n0,
            "IoU_fg": IoU_fg,
        }

    def best_metric_value(self):
        """SCD model selection metric = Sek."""
        return self.compute()["Sek"]
