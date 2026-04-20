"""Sliding window inference utilities.

Adapted from the project's earlier sliding-window implementation.
"""

import math
import numpy as np
import torch


def _pair(x):
    return (x, x) if isinstance(x, int) else tuple(x)


def compute_windows(input_size, kernel_size, stride):
    """Compute sliding window coordinates.

    Args:
        input_size: (H, W) of the full image
        kernel_size: window size (int or tuple)
        stride: step size (int or tuple)

    Returns:
        boxes: (N, 4) numpy array of [xmin, ymin, xmax, ymax]
    """
    ih, iw = input_size
    kh, kw = _pair(kernel_size)
    sh, sw = _pair(stride)

    kh = min(kh, ih)
    kw = min(kw, iw)

    num_rows = math.ceil((ih - kh) / sh) if math.ceil((ih - kh) / sh) * sh + kh >= ih \
        else math.ceil((ih - kh) / sh) + 1
    num_cols = math.ceil((iw - kw) / sw) if math.ceil((iw - kw) / sw) * sw + kw >= iw \
        else math.ceil((iw - kw) / sw) + 1

    x, y = np.meshgrid(np.arange(num_cols + 1), np.arange(num_rows + 1))
    xmin = (x * sw).ravel()
    ymin = (y * sh).ravel()

    xmin_offset = np.where(xmin + kw > iw, iw - xmin - kw, 0)
    ymin_offset = np.where(ymin + kh > ih, ih - ymin - kh, 0)

    boxes = np.stack([
        xmin + xmin_offset, ymin + ymin_offset,
        np.minimum(xmin + kw, iw), np.minimum(ymin + kh, ih),
    ], axis=1)

    return boxes


def _forward_batch(model, patches, forward_fn=None, resolve_fn=None):
    """Forward a batch of patches through model and return softmax probs.

    resolve_fn is required — always passed from Trainer._resolve_logits.
    """
    assert resolve_fn is not None, "resolve_fn must be provided"
    with torch.no_grad():
        if forward_fn is not None:
            logits = forward_fn(model, patches)
        else:
            logits = model(patches)
        logits = resolve_fn(logits)
    return torch.softmax(logits, dim=1)


def sliding_window_inference(model, image, kernel_size, stride, num_classes,
                              device, forward_fn=None, resolve_fn=None,
                              batch_size=-1):
    """Run batched sliding window inference on a single image.

    Args:
        model: nn.Module in eval mode
        image: (C, H, W) tensor
        kernel_size: window size (int)
        stride: step size (int)
        num_classes: number of output classes
        device: torch device
        forward_fn: optional callable(model, batch) -> logits.
                    If None, uses model(batch).
        resolve_fn: callable(logits) -> Tensor to collapse
                    deep-supervision / dual-head outputs.
        batch_size: number of patches per forward pass.
                    -1 = all patches at once (fastest, most memory).

    Returns:
        prediction: (H, W) numpy array of class indices
    """
    C, H, W = image.shape
    boxes = compute_windows((H, W), kernel_size, stride)

    # Extract all patches
    patches = [image[:, int(b[1]):int(b[3]), int(b[0]):int(b[2])] for b in boxes]

    merged = torch.zeros((1, num_classes, H, W), device=device)
    counts = torch.zeros((1, 1, H, W), device=device)

    bs = len(patches) if batch_size <= 0 else batch_size

    for i in range(0, len(patches), bs):
        batch = torch.stack(patches[i:i + bs]).to(device)
        probs = _forward_batch(model, batch, forward_fn=forward_fn,
                               resolve_fn=resolve_fn)

        for j, box in enumerate(boxes[i:i + bs]):
            xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            merged[:, :, ymin:ymax, xmin:xmax] += probs[j:j + 1]
            counts[:, :, ymin:ymax, xmin:xmax] += 1

    merged = merged / (counts + 1e-8)
    return merged.argmax(dim=1).squeeze(0).cpu().numpy()


def sliding_window_inference_cd(model, pre, post, kernel_size, stride, num_classes,
                                 device, forward_fn=None, resolve_fn=None,
                                 batch_size=-1, dual_head=False,
                                 loc_num_classes=2):
    """Run batched sliding window inference for CD (dual-image) models.

    Args:
        model: nn.Module in eval mode
        pre: (C_pre, H, W) tensor — pre-disaster image
        post: (C_post, H, W) tensor — post-disaster image (may differ from C_pre)
        kernel_size: window size (int)
        stride: step size (int)
        num_classes: number of output classes (classification head)
        device: torch device
        forward_fn: optional callable(model, pre_batch, post_batch) -> logits.
                    Receives separate pre/post batches for dual-input models.
                    If None, uses model(cat([pre_batch, post_batch], dim=1)).
        resolve_fn: callable(logits) -> Tensor to collapse
                    deep-supervision / dual-head outputs.
                    Ignored when dual_head=True.
        batch_size: number of patches per forward pass.
                    -1 = all patches at once.
        dual_head: if True, model returns (loc_logits, clf_logits) tuple.
                   Each head is accumulated independently and both predictions
                   are returned.
        loc_num_classes: number of classes for the localization head
                         (only used when dual_head=True).

    Returns:
        If dual_head=False:
            prediction: (H, W) numpy array of class indices
        If dual_head=True:
            (loc_preds, clf_preds): tuple of two (H, W) numpy arrays
    """
    H, W = pre.shape[1], pre.shape[2]
    boxes = compute_windows((H, W), kernel_size, stride)

    # Extract all patch pairs separately (pre and post may have different channels)
    pre_patches = []
    post_patches = []
    for b in boxes:
        xmin, ymin, xmax, ymax = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        pre_patches.append(pre[:, ymin:ymax, xmin:xmax])
        post_patches.append(post[:, ymin:ymax, xmin:xmax])

    bs = len(pre_patches) if batch_size <= 0 else batch_size

    if dual_head:
        merged_clf = torch.zeros((1, num_classes, H, W), device=device)
        merged_loc = torch.zeros((1, loc_num_classes, H, W), device=device)
        counts = torch.zeros((1, 1, H, W), device=device)

        for i in range(0, len(pre_patches), bs):
            pre_batch = torch.stack(pre_patches[i:i + bs]).to(device)
            post_batch = torch.stack(post_patches[i:i + bs]).to(device)

            with torch.no_grad():
                if forward_fn is not None:
                    outputs = forward_fn(model, pre_batch, post_batch)
                else:
                    outputs = model(torch.cat([pre_batch, post_batch], dim=1))

                # Dual-head: outputs is (loc_logits, clf_logits)
                loc_logits, clf_logits = outputs

            loc_prob = torch.softmax(loc_logits, dim=1)
            clf_prob = torch.softmax(clf_logits, dim=1)

            for j, box in enumerate(boxes[i:i + bs]):
                xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                merged_loc[:, :, ymin:ymax, xmin:xmax] += loc_prob[j:j + 1]
                merged_clf[:, :, ymin:ymax, xmin:xmax] += clf_prob[j:j + 1]
                counts[:, :, ymin:ymax, xmin:xmax] += 1

        merged_loc = merged_loc / (counts + 1e-8)
        merged_clf = merged_clf / (counts + 1e-8)
        loc_preds = merged_loc.argmax(dim=1).squeeze(0).cpu().numpy()
        clf_preds = merged_clf.argmax(dim=1).squeeze(0).cpu().numpy()
        return (loc_preds, clf_preds)

    # Single-head path (original behavior)
    merged = torch.zeros((1, num_classes, H, W), device=device)
    counts = torch.zeros((1, 1, H, W), device=device)

    for i in range(0, len(pre_patches), bs):
        pre_batch = torch.stack(pre_patches[i:i + bs]).to(device)
        post_batch = torch.stack(post_patches[i:i + bs]).to(device)

        assert resolve_fn is not None, "resolve_fn must be provided"
        with torch.no_grad():
            if forward_fn is not None:
                logits = forward_fn(model, pre_batch, post_batch)
            else:
                logits = model(torch.cat([pre_batch, post_batch], dim=1))
            logits = resolve_fn(logits)
        probs = torch.softmax(logits, dim=1)

        for j, box in enumerate(boxes[i:i + bs]):
            xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            merged[:, :, ymin:ymax, xmin:xmax] += probs[j:j + 1]
            counts[:, :, ymin:ymax, xmin:xmax] += 1

    merged = merged / (counts + 1e-8)
    return merged.argmax(dim=1).squeeze(0).cpu().numpy()
