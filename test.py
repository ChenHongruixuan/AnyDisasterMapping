"""
Usage:
    python test.py --exp_path results/xbd/unet
    python test.py --exp_path results/xbd/unet --checkpoint latest.pth
    python test.py --exp_path results/xbd/unet --config configs/infra/xbd/unet.yaml
    python test.py --exp_path results/xbd/unet --force    # re-run even if results exist
    python test.py --exp_path results/urbansar/unet --test_splits data/flood/urbansar/test_jubba.txt data/flood/urbansar/test_nova.txt data/flood/urbansar/test_weihui.txt
"""
import argparse
import json
import os
from src.core.config import Config
from src.core.trainer import Trainer


def _serialize(metrics):
    """Convert metrics dict to JSON-serializable form."""
    out = {}
    for k, v in metrics.items():
        if hasattr(v, 'tolist'):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def _save_and_print(metrics, save_path, label=""):
    """Save metrics to JSON and print scalar values."""
    with open(save_path, "w") as f:
        json.dump(_serialize(metrics), f, indent=2)
    prefix = f"  [{label}]" if label else " "
    print(f"{prefix} Saved to {save_path}")
    for k, v in metrics.items():
        if not hasattr(v, '__len__'):
            print(f"    {k}: {v:.4f}")


def _is_fresh(metrics_path, ckpt_path):
    """Return True if metrics_path exists and is newer than ckpt_path."""
    if not os.path.isfile(metrics_path):
        return False
    if not os.path.isfile(ckpt_path):
        return False
    return os.path.getmtime(metrics_path) >= os.path.getmtime(ckpt_path)


def main():
    parser = argparse.ArgumentParser(description="Unified Disaster Tester")
    parser.add_argument("--exp_path", type=str, required=True,
                        help="Experiment output dir (loads config.yaml + checkpoint from here)")
    parser.add_argument("--config", type=str, default=None,
                        help="Override config (e.g. to use updated YAML with inference settings)")
    parser.add_argument("--checkpoint", type=str, default="best.pth",
                        help="Checkpoint filename (default: best.pth)")
    parser.add_argument("--test_splits", type=str, nargs="+", default=None,
                        help="Multiple test split files to evaluate sequentially. "
                             "Results are saved per-split AND aggregated via confusion "
                             "matrix accumulation across all evaluated splits.")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if test_metrics.json is newer than checkpoint")
    args = parser.parse_args()

    ckpt_path = os.path.join(args.exp_path, args.checkpoint)

    # Load config
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config.from_yaml(os.path.join(args.exp_path, "config.yaml"))

    config.resume = ckpt_path

    if args.test_splits and len(args.test_splits) > 1:
        # Multi-split evaluation with confusion matrix aggregation

        # Skip check: all per-split JSONs + average must be fresh
        if not args.force:
            all_fresh = True
            for sp in args.test_splits:
                sname = os.path.splitext(os.path.basename(sp))[0]
                mp = os.path.join(args.exp_path, f"test_metrics_{sname}.json")
                if not _is_fresh(mp, ckpt_path):
                    all_fresh = False
                    break
            avg_path = os.path.join(args.exp_path, "test_metrics_average.json")
            if all_fresh and _is_fresh(avg_path, ckpt_path):
                print(f"[SKIP] All test results are fresh (newer than {args.checkpoint}). "
                      f"Use --force to re-run.")
                return

        import numpy as np
        trainer = Trainer(config)

        # Check if any split is stale — if so, force re-eval ALL splits
        # because we can't restore confusion matrices from saved scalar JSONs.
        any_stale = args.force
        if not args.force:
            for sp in args.test_splits:
                sname = os.path.splitext(os.path.basename(sp))[0]
                mp = os.path.join(args.exp_path, f"test_metrics_{sname}.json")
                if not _is_fresh(mp, ckpt_path):
                    any_stale = True
                    break
            if any_stale:
                print("[INFO] Some splits are stale — re-evaluating ALL splits for correct aggregation.")

        all_metrics = {}
        cm_total = None

        for split_path in args.test_splits:
            split_name = os.path.splitext(os.path.basename(split_path))[0]

            # Per-split skip: only when ALL splits are fresh (any_stale is False)
            per_split_path = os.path.join(args.exp_path, f"test_metrics_{split_name}.json")
            if not any_stale and _is_fresh(per_split_path, ckpt_path):
                print(f"\n--- Test split: {split_name} [SKIP, fresh] ---")
                with open(per_split_path) as f:
                    all_metrics[split_name] = json.load(f)
                continue

            print(f"\n--- Test split: {split_name} ---")
            ds_cfg = config._data.get("dataset", {})
            test_cfg = ds_cfg.get("test", {})
            test_cfg["data_list_path"] = split_path

            trainer.task_handler.reset_metrics()
            metrics = trainer.evaluate("test")
            all_metrics[split_name] = metrics

            _save_and_print(metrics, per_split_path, label=split_name)

            cm = getattr(trainer.task_handler.metrics_mgr, 'evaluator_total', None)
            if cm is not None and hasattr(cm, 'confusion_matrix'):
                cm_arr = cm.confusion_matrix.copy()
                cm_total = cm_arr if cm_total is None else cm_total + cm_arr

        # Aggregate: recompute metrics from summed confusion matrix
        if cm_total is not None:
            from src.common.evaluation_metrics import Evaluator
            from src.core.metrics import per_class_metrics

            num_classes = cm_total.shape[0]
            agg_eval = Evaluator(num_classes)
            agg_eval.confusion_matrix = cm_total

            pcm = per_class_metrics(cm_total)
            agg_metrics = {
                "OA": agg_eval.Pixel_Accuracy(),
                "mIoU": agg_eval.Mean_Intersection_over_Union(),
                "per_class_iou": pcm["iou"].tolist(),
                "per_class_f1": pcm["f1"].tolist(),
                "per_class_precision": pcm["precision"].tolist(),
                "per_class_recall": pcm["recall"].tolist(),
            }
            print(f"\n--- Test Average (confusion matrix aggregation) ---")
            avg_path = os.path.join(args.exp_path, "test_metrics_average.json")
            _save_and_print(agg_metrics, avg_path, label="average")
    else:
        # Single test split
        metrics_path = os.path.join(args.exp_path, "test_metrics.json")

        # Skip check
        if not args.force and _is_fresh(metrics_path, ckpt_path):
            print(f"[SKIP] {metrics_path} is fresh (newer than {args.checkpoint}). "
                  f"Use --force to re-run.")
            return

        trainer = Trainer(config)

        if args.test_splits:
            ds_cfg = config._data.get("dataset", {})
            test_cfg = ds_cfg.get("test", {})
            test_cfg["data_list_path"] = args.test_splits[0]

        metrics = trainer.evaluate("test")
        _save_and_print(metrics, metrics_path)


if __name__ == "__main__":
    main()
