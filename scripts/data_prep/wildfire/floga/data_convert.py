import argparse
import os
import pickle
import shutil
from os.path import join

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def clamp_and_scale(sen2_value, a_max=10000):
    scaled_sample = np.clip(sen2_value, a_max=a_max, a_min=0)
    scaled_sample = scaled_sample / a_max
    return scaled_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, default="./data")
    parser.add_argument("--tgt_dir", type=str, default=None)
    parser.add_argument("--patch_size", type=int, default=256)
    args = parser.parse_args()

    if args.tgt_dir is None:
        args.tgt_dir = args.src_dir

    src_floga_dir = join(args.src_dir, "floga", f"converted_data_{args.patch_size}")
    tgt_floga_dir = join(args.tgt_dir, "floga", f"patch{args.patch_size}")
    os.makedirs(join(tgt_floga_dir, "T1"), exist_ok=True)
    os.makedirs(join(tgt_floga_dir, "T1_vis"), exist_ok=True)
    os.makedirs(join(tgt_floga_dir, "T2"), exist_ok=True)
    os.makedirs(join(tgt_floga_dir, "T2_vis"), exist_ok=True)
    os.makedirs(join(tgt_floga_dir, "GT"), exist_ok=True)
    os.makedirs(join(tgt_floga_dir, "GT_vis"), exist_ok=True)

    total_data = {}
    for split in ["train", "val", "test"]:
        with open(join(src_floga_dir, f"allEvents_60-20-20_r1_{split}.pkl"), "rb") as f:
            split_data = pickle.load(f)
            for key in split_data:
                assert key not in total_data
                total_data[key] = split_data[key]

    split_df = pd.read_csv(join(src_floga_dir, "../data_split.csv"))
    split_df = [split_df.iloc[i] for i in range(len(split_df))]
    split_libs = dict(train=[], val=[], test=[])
    for doc in split_df:
        doc_year = doc['year'].item()
        doc_event_id = doc['event_id'].item()
        split_libs[doc["set"]].append(f"{doc_event_id}_{doc_year}")

    print(split_libs)

    split_data_id_libs = dict(train=[], val=[], test=[])
    for data_id in tqdm(total_data):
        data_event_year_id = "_".join(data_id.split("_")[-2:])
        if data_event_year_id in split_libs['train']:
            data_split = 'train'
        elif data_event_year_id in split_libs['val']:
            data_split = 'val'
        elif data_event_year_id in split_libs['test']:
            data_split = 'test'
        else:
            print("non-found id in split_libs", data_event_year_id, flush=True)
            data_split = 'train'

        split_data_id_libs[data_split].append(data_id)

        pre_sen2_path = total_data[data_id]["S2_before_image"]
        post_sen2_path = total_data[data_id]["S2_after_image"]
        gt_mask_path = total_data[data_id]["label"]

        shutil.copy(pre_sen2_path, join(tgt_floga_dir, "T1", f"{data_id}.npy"))
        shutil.copy(post_sen2_path, join(tgt_floga_dir, "T2", f"{data_id}.npy"))
        shutil.copy(gt_mask_path, join(tgt_floga_dir, "GT", f"{data_id}.npy"))

        pre_sen2_data = np.load(pre_sen2_path)
        pre_rgb_data = pre_sen2_data[[2, 1, 0]].transpose((1, 2, 0)) # (3, 256, 256) -> (256, 256, 3)
        pre_rgb_data = (clamp_and_scale(pre_rgb_data) * 255).astype(np.uint8)
        Image.fromarray(pre_rgb_data).save(join(tgt_floga_dir, "T1_vis", f"{data_id}.png"))

        post_sen2_data = np.load(post_sen2_path)
        post_rgb_data = post_sen2_data[[2, 1, 0]].transpose((1, 2, 0))
        post_rgb_data = (clamp_and_scale(post_rgb_data) * 255).astype(np.uint8)
        Image.fromarray(post_rgb_data).save(join(tgt_floga_dir, "T2_vis", f"{data_id}.png"))

        gt_mask_data = np.load(gt_mask_path)
        print(np.unique(gt_mask_data), flush=True)

        gt_vis = np.zeros(gt_mask_data.shape, dtype=np.uint8)
        gt_vis[gt_mask_data == 2] = 127
        gt_vis[gt_mask_data == 1] = 255
        Image.fromarray(gt_vis).save(join(tgt_floga_dir, "GT_vis", f"{data_id}.png"))

    for split in split_data_id_libs:
        with open(join(tgt_floga_dir, f"{split}.txt"), "w") as f:
            for data_id in split_data_id_libs[split]:
                f.write(f"{data_id}\n")


if __name__ == '__main__':
    main()
