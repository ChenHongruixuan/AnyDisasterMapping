import os

import argparse
import h5py
from pathlib import Path
from tqdm import tqdm

from src.datasets.wildfire_firespread import FireSpreadDataset

# Need to prevent an error with HDF5 files being locked and thereby inaccessible
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="Path to dataset directory", required=True)
    parser.add_argument("--target_dir", type=str, help="Path to directory where the HDF5 files should be stored", required=True)
    args = parser.parse_args()

    # Iterate over all splits to cover all years (train=2018/2019, val=2020, test=2021)
    all_datasets = []
    for split in ["train", "val", "test"]:
        ds = FireSpreadDataset(
            split=split,
            dataset_path=args.data_dir,
            n_leading_observations=1,
            crop_side_length=128,
            load_from_hdf5=False,
            remove_duplicate_features=False,
        )
        all_datasets.append(ds)

    for y in [2018, 2019, 2020, 2021]:
        Path(f"{args.target_dir}/{y}").mkdir(parents=True, exist_ok=True)

    total_fires = sum(
        sum(len(fires) for fires in ds.imgs_per_fire.values())
        for ds in all_datasets
    )

    with tqdm(total=total_fires, desc="Converting fires to HDF5") as pbar:
        for ds in all_datasets:
            for year, fire_name, img_dates, lnglat, imgs in ds.get_generator_for_hdf5():
                h5_path = f"{args.target_dir}/{year}/{fire_name}.hdf5"

                if Path(h5_path).is_file():
                    print(f"File {h5_path} already exists, skipping...")
                    pbar.update(1)
                    continue

                with h5py.File(h5_path, "w") as f:
                    dset = f.create_dataset("data", imgs.shape, data=imgs)
                    dset.attrs["year"] = year
                    dset.attrs["fire_name"] = fire_name
                    dset.attrs["img_dates"] = img_dates
                    dset.attrs["lnglat"] = lnglat

                pbar.update(1)


if __name__ == '__main__':
    main()
