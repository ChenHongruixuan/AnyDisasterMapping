# Data Setup
## Infrastructure Damage

Run the commands below from the repository root. 
For the Python data-prep helpers and `hf` CLI, install dependencies first: `pip install -e .`.


Config-expected runtime paths:
- xBD / xView2: `data/infra_damage/xBD/{train,test,hold}` and `data/infra_damage/xBD/{train_set,val_set,test_set}.txt`
- BRIGHT: `data/infra_damage/bright/{pre-event,post-event,target}` and `data/infra_damage/bright/{train_set,val_set,test_set}.txt`
- RescueNet: `data/infra_damage/rescuenet/{train_crop,val_crop,test_crop}` and `data/infra_damage/rescuenet/{train_crop_tiles,val_crop_tiles,test_crop_tiles}.txt`

### xBD / xView2
```bash
mkdir -p ./data/infra_damage/xBD
# manually download xBD / xView2 challenge archives from https://xview2.org/dataset (registration required)
# if extraction creates temporary top-level dataset folders, move their split folders into ./data/infra_damage/xBD
# arrange the final config layout as:
#   ./data/infra_damage/xBD/train/{images,labels,targets}
#   ./data/infra_damage/xBD/tier3/{images,labels}
#   ./data/infra_damage/xBD/test/{images,labels,targets}
#   ./data/infra_damage/xBD/hold/{images,labels,targets}

# convert json labels into semantic masks for every split
python scripts/data_prep/infra_damage/xBD/mask_polygons.py --input ./data/infra_damage/xBD/train
python scripts/data_prep/infra_damage/xBD/mask_damage_polygons.py --input ./data/infra_damage/xBD/train
python scripts/data_prep/infra_damage/xBD/mask_polygons.py --input ./data/infra_damage/xBD/tier3
python scripts/data_prep/infra_damage/xBD/mask_damage_polygons.py --input ./data/infra_damage/xBD/tier3
python scripts/data_prep/infra_damage/xBD/mask_polygons.py --input ./data/infra_damage/xBD/test
python scripts/data_prep/infra_damage/xBD/mask_damage_polygons.py --input ./data/infra_damage/xBD/test
python scripts/data_prep/infra_damage/xBD/mask_polygons.py --input ./data/infra_damage/xBD/hold
python scripts/data_prep/infra_damage/xBD/mask_damage_polygons.py --input ./data/infra_damage/xBD/hold

# current repo split convention uses train + tier3 as the effective training set
rsync -a --ignore-existing ./data/infra_damage/xBD/tier3/images/ ./data/infra_damage/xBD/train/images/
rsync -a --ignore-existing ./data/infra_damage/xBD/tier3/masks/ ./data/infra_damage/xBD/train/masks/

# place the repo split files at the config-referenced filelist root
cp scripts/data_prep/infra_damage/xBD/train_set.txt ./data/infra_damage/xBD/train_set.txt
cp scripts/data_prep/infra_damage/xBD/val_set.txt ./data/infra_damage/xBD/val_set.txt
cp scripts/data_prep/infra_damage/xBD/test_set.txt ./data/infra_damage/xBD/test_set.txt

# sanity-check the prepared runtime layout
python scripts/data_prep/infra_damage/xBD/check_xBD_dataset.py --dataset_path ./data/infra_damage/xBD/train --data_list_path ./data/infra_damage/xBD/train_set.txt
python scripts/data_prep/infra_damage/xBD/check_xBD_dataset.py --dataset_path ./data/infra_damage/xBD/test --data_list_path ./data/infra_damage/xBD/val_set.txt
python scripts/data_prep/infra_damage/xBD/check_xBD_dataset.py --dataset_path ./data/infra_damage/xBD/hold --data_list_path ./data/infra_damage/xBD/test_set.txt

# optional: regenerate custom split lists from image stems
python scripts/data_prep/infra_damage/xBD/generate_xbd_filelist.py --image_dir ./data/infra_damage/xBD/train/images --output_txt ./data/infra_damage/xBD/train_set.txt
python scripts/data_prep/infra_damage/xBD/generate_xbd_filelist.py --image_dir ./data/infra_damage/xBD/test/images --output_txt ./data/infra_damage/xBD/val_set.txt
python scripts/data_prep/infra_damage/xBD/generate_xbd_filelist.py --image_dir ./data/infra_damage/xBD/hold/images --output_txt ./data/infra_damage/xBD/test_set.txt

# optional: crop the train split into 512x512 tiles
python scripts/data_prep/infra_damage/xBD/crop_xbd.py --input-dir ./data/infra_damage/xBD/train --output-dir ./data/infra_damage/xBD/train_crop_512
python scripts/data_prep/infra_damage/xBD/generate_cropped_tile_list.py --image-dir ./data/infra_damage/xBD/train_crop_512/cropped_images --mask-dir ./data/infra_damage/xBD/train_crop_512/cropped_masks --output ./data/infra_damage/xBD/train_valid_tiles.txt
```

Expected directory structure after setup:
```text
data/
|___infra_damage
|   |___xBD
|   |   |___train
|   |   |   |___images
|   |   |   |   |___*_pre_disaster.png
|   |   |   |   |___*_post_disaster.png
|   |   |   |___masks
|   |   |       |___*_pre_disaster.png
|   |   |       |___*_post_disaster.png
|   |   |___tier3
|   |   |   |___images
|   |   |   |___masks
|   |   |___test
|   |   |   |___images
|   |   |   |___masks
|   |   |___hold
|   |   |   |___images
|   |   |   |___masks
|   |   |___train_crop_512
|   |   |   |___cropped_images
|   |   |   |___cropped_masks
|   |   |___train_set.txt
|   |   |___val_set.txt       # corresponds to ./data/infra_damage/xBD/test
|   |   |___test_set.txt      # corresponds to ./data/infra_damage/xBD/hold
|   |   |___train_valid_tiles.txt
```

### BRIGHT
```bash
mkdir -p ./data/infra_damage/bright/raw
# download BRIGHT from https://huggingface.co/datasets/Kullervo/BRIGHT
# for the current semantic-damage loader and filelists, only these three archives are needed: pre-event.zip post-event.zip target.zip
# this is a minimal setup, not a full mirror of the public BRIGHT repo
# auxiliary files such as post-event-optical, splits, dfc_test.txt, instance annotations, dfc25_* and umim_* are not used by this setup
# the current HF pre-event.zip extracts directly to pre-event/ and includes ukraine-conflict, mexico-hurricane, and myanmar-hurricane files.

hf download Kullervo/BRIGHT pre-event.zip --repo-type dataset --local-dir ./data/infra_damage/bright/raw
hf download Kullervo/BRIGHT post-event.zip --repo-type dataset --local-dir ./data/infra_damage/bright/raw
hf download Kullervo/BRIGHT target.zip --repo-type dataset --local-dir ./data/infra_damage/bright/raw

unzip -o ./data/infra_damage/bright/raw/pre-event.zip -d ./data/infra_damage/bright
unzip -o ./data/infra_damage/bright/raw/post-event.zip -d ./data/infra_damage/bright
unzip -o ./data/infra_damage/bright/raw/target.zip -d ./data/infra_damage/bright

cp scripts/data_prep/infra_damage/bright/train_set.txt ./data/infra_damage/bright/train_set.txt
cp scripts/data_prep/infra_damage/bright/val_set.txt ./data/infra_damage/bright/val_set.txt
cp scripts/data_prep/infra_damage/bright/test_set.txt ./data/infra_damage/bright/test_set.txt
cp scripts/data_prep/infra_damage/bright/enhanced_train_set.txt ./data/infra_damage/bright/enhanced_train_set.txt
```

Expected directory structure after setup:
```text
data/
|___infra_damage
|   |___bright
|   |   |___pre-event
|   |   |   |___*_pre_disaster.tif
|   |   |___post-event
|   |   |   |___*_post_disaster.tif
|   |   |___target
|   |   |   |___*_building_damage.tif
|   |   |___raw
|   |   |   |___pre-event.zip
|   |   |   |___post-event.zip
|   |   |   |___target.zip
|   |   |___train_set.txt
|   |   |___val_set.txt
|   |   |___test_set.txt
|   |   |___enhanced_train_set.txt
```

### RescueNet
```bash
mkdir -p ./data/infra_damage/rescuenet
# manually download RescueNet from https://www.dropbox.com/scl/fo/ntgeyhxe2mzd2wuh7he7x/AFIchlfjVO_7MzPcNc1ZOHE/RescueNet?rlkey=6vxiaqve9gp6vzvzh3t5mz0vv&e=1&st=so10gf5h&subfolder_nav_tracking=1&dl=0
# or from https://www.kaggle.com/datasets/yaroslavchyrko/rescuenet
# if extraction creates a temporary top-level dataset folder, move its split folders into ./data/infra_damage/rescuenet
# downloaded split roots are expected to look like:
#   ./data/infra_damage/rescuenet/train/train-org-img and ./data/infra_damage/rescuenet/train/train-label-img
#   ./data/infra_damage/rescuenet/val/val-org-img and ./data/infra_damage/rescuenet/val/val-label-img
#   ./data/infra_damage/rescuenet/test/test-org-img and ./data/infra_damage/rescuenet/test/test-label-img
# rename them to img/ and label/ so the current scripts and loader can read them

mv ./data/infra_damage/rescuenet/train/train-org-img ./data/infra_damage/rescuenet/train/img
mv ./data/infra_damage/rescuenet/train/train-label-img ./data/infra_damage/rescuenet/train/label
mv ./data/infra_damage/rescuenet/val/val-org-img ./data/infra_damage/rescuenet/val/img
mv ./data/infra_damage/rescuenet/val/val-label-img ./data/infra_damage/rescuenet/val/label
mv ./data/infra_damage/rescuenet/test/test-org-img ./data/infra_damage/rescuenet/test/img
mv ./data/infra_damage/rescuenet/test/test-label-img ./data/infra_damage/rescuenet/test/label

# current configs use the cropped RescueNet roots, so crop each split and rename the outputs to img/ and label/
python scripts/data_prep/infra_damage/rescuenet/crop_rescuenet.py --input-dir ./data/infra_damage/rescuenet/train --output-dir ./data/infra_damage/rescuenet/train_crop
python scripts/data_prep/infra_damage/rescuenet/crop_rescuenet.py --input-dir ./data/infra_damage/rescuenet/val --output-dir ./data/infra_damage/rescuenet/val_crop
python scripts/data_prep/infra_damage/rescuenet/crop_rescuenet.py --input-dir ./data/infra_damage/rescuenet/test --output-dir ./data/infra_damage/rescuenet/test_crop

mv ./data/infra_damage/rescuenet/train_crop/cropped_img ./data/infra_damage/rescuenet/train_crop/img
mv ./data/infra_damage/rescuenet/train_crop/cropped_label ./data/infra_damage/rescuenet/train_crop/label
mv ./data/infra_damage/rescuenet/val_crop/cropped_img ./data/infra_damage/rescuenet/val_crop/img
mv ./data/infra_damage/rescuenet/val_crop/cropped_label ./data/infra_damage/rescuenet/val_crop/label
mv ./data/infra_damage/rescuenet/test_crop/cropped_img ./data/infra_damage/rescuenet/test_crop/img
mv ./data/infra_damage/rescuenet/test_crop/cropped_label ./data/infra_damage/rescuenet/test_crop/label

# place the repo split files at the config-referenced filelist root
cp scripts/data_prep/infra_damage/rescuenet/train_crop_tiles.txt ./data/infra_damage/rescuenet/train_crop_tiles.txt
cp scripts/data_prep/infra_damage/rescuenet/val_crop_tiles.txt ./data/infra_damage/rescuenet/val_crop_tiles.txt
cp scripts/data_prep/infra_damage/rescuenet/test_crop_tiles.txt ./data/infra_damage/rescuenet/test_crop_tiles.txt

# optional: keep the original split lists for reference
cp scripts/data_prep/infra_damage/rescuenet/train_set.txt ./data/infra_damage/rescuenet/train_set.txt
cp scripts/data_prep/infra_damage/rescuenet/val_set.txt ./data/infra_damage/rescuenet/val_set.txt
cp scripts/data_prep/infra_damage/rescuenet/test_set.txt ./data/infra_damage/rescuenet/test_set.txt
```

Expected directory structure after setup:
```text
data/
|___infra_damage
|   |___rescuenet
|   |   |___train
|   |   |   |___img
|   |   |   |   |___*.jpg
|   |   |   |___label
|   |   |       |___*_lab.png
|   |   |___val
|   |   |   |___img
|   |   |   |___label
|   |   |___test
|   |   |   |___img
|   |   |   |___label
|   |   |___train_crop
|   |   |   |___img
|   |   |   |___label
|   |   |___val_crop
|   |   |   |___img
|   |   |   |___label
|   |   |___test_crop
|   |   |   |___img
|   |   |   |___label
|   |   |___train_set.txt
|   |   |___val_set.txt
|   |   |___test_set.txt
|   |   |___train_crop_tiles.txt
|   |   |___val_crop_tiles.txt
|   |   |___test_crop_tiles.txt
```
