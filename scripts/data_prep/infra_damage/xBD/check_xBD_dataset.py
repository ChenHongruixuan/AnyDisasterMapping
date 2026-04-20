"""Utility script to sanity-check xBD dataset integrity."""

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np
import imageio.v2 as imageio


def parse_args():
    parser = argparse.ArgumentParser(description='Inspect xBD dataset for common issues')
    parser.add_argument('--dataset_path', type=str, required=True, help='Root directory of the xBD dataset')
    parser.add_argument('--data_list_path', type=str, required=True, help='Text file with scene identifiers (one per line)')
    parser.add_argument('--suffix', type=str, default='.png', help='Image suffix, defaults to .png')
    parser.add_argument('--expected_classes', type=int, default=5, help='Number of semantic classes (including background)')
    parser.add_argument('--max_samples', type=int, default=None, help='Limit the number of items to inspect')
    parser.add_argument('--report_path', type=str, default=None, help='Optional path to write a JSON report')
    return parser.parse_args()


def load_image(path, issues, key):
    if not os.path.exists(path):
        issues[key]['missing'].append(path)
        return None
    try:
        return imageio.imread(path)
    except Exception as exc:  # pragma: no cover - defensive
        issues[key]['read_error'].append({'path': path, 'error': str(exc)})
        return None


def counter_to_serializable(counter):
    """Return a JSON-safe dict from a Counter by stringifying tuple keys."""
    formatted = {}
    for key, value in counter.items():
        if isinstance(key, tuple):
            key_str = 'x'.join(str(dim) for dim in key)
        else:
            key_str = str(key)
        formatted[key_str] = value
    return formatted


def main():
    args = parse_args()

    with open(args.data_list_path, 'r') as f:
        identifiers = [line.strip() for line in f if line.strip()]

    if args.max_samples is not None:
        identifiers = identifiers[: args.max_samples]

    results = {
        'total_items': len(identifiers),
        'checked_items': 0,
        'pre_image_shapes': Counter(),
        'post_image_shapes': Counter(),
        'mask_shapes': Counter(),
        'mask_unique_values': Counter(),
        'mask_min': None,
        'mask_max': None,
        'summary': defaultdict(dict),
        'items_with_invalid_labels': [],
        'items_with_nan': [],
        'items_with_shape_mismatch': [],
    }

    issues = {
        'pre': defaultdict(list),
        'post': defaultdict(list),
        'mask': defaultdict(list),
    }

    expected_values = set(range(args.expected_classes)) | {255}

    for scene_id in identifiers:
        pre_path = os.path.join(args.dataset_path, 'images', scene_id + '_pre_disaster' + args.suffix)
        post_path = os.path.join(args.dataset_path, 'images', scene_id + '_post_disaster' + args.suffix)
        mask_path = os.path.join(args.dataset_path, 'masks', scene_id + '_post_disaster' + args.suffix)

        pre_img = load_image(pre_path, issues, 'pre')
        post_img = load_image(post_path, issues, 'post')
        mask = load_image(mask_path, issues, 'mask')

        if pre_img is None or post_img is None or mask is None:
            continue

        results['checked_items'] += 1

        results['pre_image_shapes'][tuple(pre_img.shape)] += 1
        results['post_image_shapes'][tuple(post_img.shape)] += 1
        results['mask_shapes'][tuple(mask.shape)] += 1

        if pre_img.shape[:2] != post_img.shape[:2] or pre_img.shape[:2] != mask.shape[:2]:
            results['items_with_shape_mismatch'].append({
                'scene_id': scene_id,
                'pre_shape': tuple(pre_img.shape),
                'post_shape': tuple(post_img.shape),
                'mask_shape': tuple(mask.shape),
            })

        if np.isnan(mask).any():
            results['items_with_nan'].append(scene_id)
            mask = np.nan_to_num(mask, nan=255)

        mask_values = np.unique(mask.astype(np.int64))
        for val in mask_values:
            results['mask_unique_values'][int(val)] += 1

        mask_min = int(mask_values.min()) if mask_values.size else None
        mask_max = int(mask_values.max()) if mask_values.size else None
        if results['mask_min'] is None or (mask_min is not None and mask_min < results['mask_min']):
            results['mask_min'] = mask_min
        if results['mask_max'] is None or (mask_max is not None and mask_max > results['mask_max']):
            results['mask_max'] = mask_max

        invalid_values = [int(v) for v in mask_values if v not in expected_values]
        if invalid_values:
            results['items_with_invalid_labels'].append({
                'scene_id': scene_id,
                'invalid_values': invalid_values,
            })

    report = {
        'dataset_path': args.dataset_path,
        'data_list_path': args.data_list_path,
        'suffix': args.suffix,
        'expected_classes': args.expected_classes,
        'summary': {
            'total_items_in_list': len(identifiers),
            'items_checked': results['checked_items'],
            'missing_pre_images': len(issues['pre']['missing']),
            'missing_post_images': len(issues['post']['missing']),
            'missing_masks': len(issues['mask']['missing']),
            'pre_read_errors': len(issues['pre']['read_error']),
            'post_read_errors': len(issues['post']['read_error']),
            'mask_read_errors': len(issues['mask']['read_error']),
            'shape_mismatch_items': len(results['items_with_shape_mismatch']),
            'items_with_nan_in_mask': len(results['items_with_nan']),
            'items_with_invalid_labels': len(results['items_with_invalid_labels']),
            'mask_value_min': results['mask_min'],
            'mask_value_max': results['mask_max'],
            'unique_values_overview': dict(results['mask_unique_values']),
            'pre_shape_counts': counter_to_serializable(results['pre_image_shapes']),
            'post_shape_counts': counter_to_serializable(results['post_image_shapes']),
            'mask_shape_counts': counter_to_serializable(results['mask_shapes']),
        },
        'details': {
            'missing_files': issues,
            'invalid_label_items': results['items_with_invalid_labels'],
            'nan_mask_items': results['items_with_nan'],
            'shape_mismatch_items': results['items_with_shape_mismatch'],
        },
    }

    print(json.dumps(report, indent=2))

    if args.report_path:
        os.makedirs(os.path.dirname(args.report_path), exist_ok=True)
        with open(args.report_path, 'w') as f:
            json.dump(report, f, indent=2)


if __name__ == '__main__':
    main()
