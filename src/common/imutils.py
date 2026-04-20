import random
import numpy as np
# import cv2

def normalize_img(img, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    """Normalize image by subtracting mean and dividing by std using statistics from ImageNet."""
    img_array = np.asarray(img)
    normalized_img = np.empty_like(img_array, np.float32)

    for i in range(3):  # Loop over color channels
        normalized_img[..., i] = (img_array[..., i] - mean[i]) / std[i]
    
    return normalized_img


def random_fliplr(pre_img, post_img, label):
    if random.random() > 0.5:
        label = np.fliplr(label)
        pre_img = np.fliplr(pre_img)
        post_img = np.fliplr(post_img)

    return pre_img, post_img, label

def random_flipud(pre_img, post_img, label):
    if random.random() > 0.5:
        label = np.flipud(label)
        pre_img = np.flipud(pre_img)
        post_img = np.flipud(post_img)

    return pre_img, post_img, label


# def _resize_array(arr, output_size, interpolation):
#     """Resize 2D or multi-channel arrays using OpenCV."""
#     if arr.ndim not in (2, 3):
#         raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")

#     h, w = arr.shape[:2]
#     new_h, new_w = output_size
#     if h == new_h and w == new_w:
#         return arr.copy()

#     target_size = (new_w, new_h)

#     if arr.ndim == 2:
#         resized = cv2.resize(arr, target_size, interpolation=interpolation)
#     else:
#         resized_channels = [
#             cv2.resize(arr[..., c], target_size, interpolation=interpolation)
#             for c in range(arr.shape[2])
#         ]
#         resized = np.stack(resized_channels, axis=-1)

#     if np.issubdtype(arr.dtype, np.integer):
#         info = np.iinfo(arr.dtype)
#         return np.clip(resized, info.min, info.max).astype(arr.dtype)

#     return resized.astype(arr.dtype, copy=False)


# def random_scale(pre_img, post_img, label=None, scales=(0.75, 1.0, 1.25)):
#     """Randomly scale images (and labels) using the provided scale factors."""
#     if not scales:
#         raise ValueError("`scales` must be a non-empty sequence of scale factors")

#     scale = random.choice(scales)
#     if np.isclose(scale, 1.0):
#         return pre_img, post_img, label

#     h, w = pre_img.shape[:2]
#     new_h = max(1, int(round(h * scale)))
#     new_w = max(1, int(round(w * scale)))
#     output_size = (new_h, new_w)

#     pre_img = _resize_array(pre_img, output_size, interpolation=cv2.INTER_LINEAR)
#     post_img = _resize_array(post_img, output_size, interpolation=cv2.INTER_LINEAR)

#     if label is not None:
#         label = _resize_array(label, output_size, interpolation=cv2.INTER_NEAREST)

#     return pre_img, post_img, label

def random_rot(pre_img, post_img, label):
    k = random.randrange(4)

    pre_img = np.rot90(pre_img, k).copy()
    post_img = np.rot90(post_img, k).copy()
    label = np.rot90(label, k).copy()

    return pre_img, post_img, label


def random_crop(pre_img, post_img, label, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    h, w = label.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    pad_pre_image = np.zeros((H, W, pre_img.shape[-1]), dtype=np.float32)

    pad_post_image = np.zeros((H, W, post_img.shape[-1]), dtype=np.float32)
    # print(pad_post_image.shape, post_img.shape)
    pad_label = np.ones((H, W), dtype=np.float32) * ignore_index

    # pad_pre_image[:, :] = mean_rgb[0]
    pad_pre_image[:, :, 0] = mean_rgb[0]
    pad_pre_image[:, :, 1] = mean_rgb[1]
    pad_pre_image[:, :, 2] = mean_rgb[2]

    pad_post_image[:, :, 0] = mean_rgb[0]
    pad_post_image[:, :, 1] = mean_rgb[1]
    pad_post_image[:, :, 2] = mean_rgb[2]

    H_pad = int(np.random.randint(H - h + 1))
    W_pad = int(np.random.randint(W - w + 1))

    pad_pre_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = pre_img
    pad_post_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = post_img
    pad_label[H_pad:(H_pad + h), W_pad:(W_pad + w)] = label

    def get_random_cropbox(cat_max_ratio=0.75, max_retry=50):

        fallback = None

        for _ in range(max_retry):

            H_start = random.randrange(0, H - crop_size + 1, 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1, 1)
            W_end = W_start + crop_size

            temp_label = pad_label[H_start:H_end, W_start:W_end]
            valid_mask = temp_label != ignore_index
            if not valid_mask.any():
                continue

            valid_labels = temp_label[valid_mask]
            if np.all(valid_labels == 0):
                continue

            fallback = (H_start, H_end, W_start, W_end)

            unique_labels, counts = np.unique(valid_labels, return_counts=True)
            if unique_labels.size > 1:
                ratio = np.max(counts) / np.sum(counts)
                if ratio < cat_max_ratio:
                    return fallback
            else:
                return fallback

        if fallback is not None:
            return fallback

        return 0, crop_size, 0, crop_size

    H_start, H_end, W_start, W_end = get_random_cropbox()
    # print(W_start)
    pre_img = pad_pre_image[H_start:H_end, W_start:W_end, :]
    post_img = pad_post_image[H_start:H_end, W_start:W_end, :]
    label = pad_label[H_start:H_end, W_start:W_end]
   
    return pre_img, post_img, label
