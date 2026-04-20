import argparse
import glob
import os
import shutil
import numpy as np

from os.path import join, dirname

import rasterio
import tifffile
from PIL import Image
from rasterio.warp import reproject, Resampling
from tqdm import tqdm

def clamp_and_scale(sen2_value, a_max=10000):
    scaled_sample = np.clip(sen2_value, a_max=a_max, a_min=0)
    scaled_sample = scaled_sample / a_max
    return scaled_sample

BAND_ORDER = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']


def read_and_resample_band(band_path, ref_profile):
    """Read a band tif; bilinear-resample to 10 m if its resolution differs from the reference.

    Args:
        band_path: path to the band tif file.
        ref_profile: reference profile dict of the 10 m band (height, width, transform, crs).

    Returns:
        np.ndarray: shape (H, W), resampled band data.
    """
    with rasterio.open(band_path) as src:
        if src.res == (10.0, 10.0):
            return src.read(1)
        else:
            dst_data = np.empty(
                (ref_profile['height'], ref_profile['width']),
                dtype=src.dtypes[0]
            )
            reproject(
                source=rasterio.band(src, 1),
                destination=dst_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_profile['transform'],
                dst_crs=ref_profile['crs'],
                resampling=Resampling.bilinear,
            )
            return dst_data


def convert_sen2(src_dir, tgt_dir):
    """Extract the 13-band Sentinel-2 data for every S2-WCD event, resample to 10 m, save as int16 npy.

    Output dirs: <tgt_dir>/s2_wcd/T1_sen2/ and <tgt_dir>/s2_wcd/T2_sen2/
    Output format: (13, 1066, 1066) int16, original scale 0-10000.

    Usage:
        python scripts/data_prep/wildfire/s2_wcd/data_convert.py \\
            --src_dir ./data/wildfire/raw --tgt_dir ./data/wildfire --mode sen2

    Args:
        src_dir: raw data root directory (contains the S2-WCD/ subdirectory).
        tgt_dir: output data root directory (writes to <tgt_dir>/s2_wcd/).
    """
    src_s2wcd_dir = join(src_dir, "S2-WCD")
    tgt_s2wcd_dir = join(tgt_dir, "s2_wcd")
    os.makedirs(join(tgt_s2wcd_dir, "T1_sen2"), exist_ok=True)
    os.makedirs(join(tgt_s2wcd_dir, "T2_sen2"), exist_ok=True)

    event_list = sorted([
        d for d in os.listdir(src_s2wcd_dir)
        if os.path.isdir(join(src_s2wcd_dir, d)) and not d.startswith('.')
    ])

    for event_name in tqdm(event_list):
        src_event_dir = join(src_s2wcd_dir, event_name)

        # Get img_id (the image prefix shared by all files of the event).
        img_ids = set([
            f.split("_")[0]
            for f in os.listdir(join(src_event_dir, "img1_cropped"))
            if f.endswith(".tif")
        ])
        assert len(img_ids) == 1, f"{event_name}: {img_ids}"
        img_id = list(img_ids)[0]

        for t_idx, img_dir in enumerate(["img1_cropped", "img2_cropped"], 1):
            # Read the profile of the reference 10 m band (B02).
            ref_paths = glob.glob(join(src_event_dir, img_dir, f"{img_id}*B02.tif"))
            assert len(ref_paths) == 1, f"{event_name}/{img_dir}/B02: {ref_paths}"
            with rasterio.open(ref_paths[0]) as ref_ds:
                ref_profile = {
                    'height': ref_ds.height,
                    'width': ref_ds.width,
                    'transform': ref_ds.transform,
                    'crs': ref_ds.crs,
                }

            bands = []
            for band_name in BAND_ORDER:
                band_paths = glob.glob(
                    join(src_event_dir, img_dir, f"{img_id}*{band_name}.tif")
                )
                assert len(band_paths) == 1, \
                    f"{event_name}/{img_dir}/{band_name}: {band_paths}"
                band_data = read_and_resample_band(band_paths[0], ref_profile)
                bands.append(band_data)

            # (13, H, W) int16
            sen2_data = np.stack(bands, axis=0).astype(np.int16)
            tgt_subdir = f"T{t_idx}_sen2"
            np.save(join(tgt_s2wcd_dir, tgt_subdir, f"{event_name}.npy"), sen2_data)

        tqdm.write(f"{event_name}: done, shape={sen2_data.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, default="./data")
    parser.add_argument("--tgt_dir", type=str, default=None)
    parser.add_argument("--mode", type=str, default="all",
                        choices=["rgb", "sen2", "all"],
                        help='Conversion mode: "rgb", "sen2" (13-band), or "all" (both, default).')
    args = parser.parse_args()

    if args.tgt_dir is None:
        args.tgt_dir = args.src_dir

    if args.mode in ("rgb", "all"):
        src_s2wcd_dir = join(args.src_dir, "S2-WCD")
        tgt_s2wcd_dir = os.path.join(args.tgt_dir, "s2_wcd")
        os.makedirs(join(tgt_s2wcd_dir, "T1"), exist_ok=True)
        os.makedirs(join(tgt_s2wcd_dir, "T1_vis"), exist_ok=True)
        os.makedirs(join(tgt_s2wcd_dir, "T2"), exist_ok=True)
        os.makedirs(join(tgt_s2wcd_dir, "T2_vis"), exist_ok=True)
        os.makedirs(join(tgt_s2wcd_dir, "GT"), exist_ok=True)

        event_list = sorted([
            d for d in os.listdir(src_s2wcd_dir)
            if os.path.isdir(join(src_s2wcd_dir, d)) and not d.startswith('.')
        ])

        for event_name in tqdm(event_list):
            src_event_dir = join(src_s2wcd_dir, event_name)

            img_ids = set([tif_file.split("_")[0] for tif_file in os.listdir(join(src_event_dir, "img1_cropped"))
                           if tif_file.endswith(".tif")])
            assert len(img_ids) == 1, event_name
            img_id = list(img_ids)[0]
            img_save_id = f"{event_name}"

            # T1
            t1_red_path = glob.glob(join(src_event_dir, "img1_cropped", f"{img_id}*B04.tif"))
            assert len(t1_red_path) == 1
            t1_red_path = t1_red_path[0]
            t1_red_value = tifffile.imread(t1_red_path)
            t1_red_value = clamp_and_scale(t1_red_value)

            t1_green_path = glob.glob(join(src_event_dir, "img1_cropped", f"{img_id}*B03.tif"))
            assert len(t1_green_path) == 1
            t1_green_path = t1_green_path[0]
            t1_green_value = tifffile.imread(t1_green_path)
            t1_green_value = clamp_and_scale(t1_green_value)

            t1_blue_path = glob.glob(join(src_event_dir, "img1_cropped", f"{img_id}*B02.tif"))
            assert len(t1_blue_path) == 1
            t1_blue_path = t1_blue_path[0]
            t1_blue_value = tifffile.imread(t1_blue_path)
            t1_blue_value = clamp_and_scale(t1_blue_value)

            t1_rgb_values = np.stack([t1_red_value, t1_green_value, t1_blue_value], axis=2)
            np.save(join(tgt_s2wcd_dir, "T1", f"{img_save_id}.npy"), t1_rgb_values)

            t1_rgb_vis = (t1_rgb_values * 255).astype(np.uint8)
            Image.fromarray(t1_rgb_vis).save(join(tgt_s2wcd_dir, "T1_vis", f"{img_save_id}.png"))

            # T2
            t2_red_path = glob.glob(join(src_event_dir, "img2_cropped", f"{img_id}*B04.tif"))
            assert len(t2_red_path) == 1
            t2_red_path = t2_red_path[0]
            t2_red_value = tifffile.imread(t2_red_path)
            t2_red_value = clamp_and_scale(t2_red_value)

            t2_green_path = glob.glob(join(src_event_dir, "img2_cropped", f"{img_id}*B03.tif"))
            assert len(t2_green_path) == 1
            t2_green_path = t2_green_path[0]
            t2_green_value = tifffile.imread(t2_green_path)
            t2_green_value = clamp_and_scale(t2_green_value)

            t2_blue_path = glob.glob(join(src_event_dir, "img2_cropped", f"{img_id}*B02.tif"))
            assert len(t2_blue_path) == 1
            t2_blue_path = t2_blue_path[0]
            t2_blue_value = tifffile.imread(t2_blue_path)
            t2_blue_value = clamp_and_scale(t2_blue_value)

            t2_rgb_values = np.stack([t2_red_value, t2_green_value, t2_blue_value], axis=2)
            np.save(join(tgt_s2wcd_dir, "T2", f"{img_save_id}.npy"), t2_rgb_values)

            t2_rgb_vis = (t2_rgb_values * 255).astype(np.uint8)
            Image.fromarray(t2_rgb_vis).save(join(tgt_s2wcd_dir, "T2_vis", f"{img_save_id}.png"))

            # GT
            gt_path = join(src_event_dir, "cm", "cm.tif")
            gt_mask = tifffile.imread(gt_path)
            print(img_save_id, np.unique(gt_mask))
            Image.fromarray(gt_mask).convert("L").save(join(tgt_s2wcd_dir, "GT", f"{img_save_id}.png"))

        # Copy split files bundled with the source code
        split_dir = dirname(__file__)
        for split_file in ["train.txt", "val.txt", "test.txt"]:
            src_path = join(split_dir, split_file)
            tgt_path = join(tgt_s2wcd_dir, split_file)
            if os.path.exists(src_path):
                shutil.copy2(src_path, tgt_path)
                print(f"Copied {split_file} -> {tgt_path}")

    if args.mode in ("sen2", "all"):
        print("\nRunning full-band (sen2) conversion...")
        convert_sen2(args.src_dir, args.tgt_dir)


if __name__ == '__main__':
    main()
