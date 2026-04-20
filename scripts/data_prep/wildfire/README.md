# Data Setup
## Wild Fire
### FLOGA
```
mkdir -p ./data/wildfire/floga/raw_h5
# download `S2 20m - MODIS 500m` part to ./data/wildfire/floga/raw_h5
#   FLOGA_dataset_2017_sen2_20_mod_500.h5  (18.5 GB)
#   FLOGA_dataset_2018_sen2_20_mod_500.h5  (13.3 GB)
#   FLOGA_dataset_2019_sen2_20_mod_500.h5  (33.3 GB)
#   FLOGA_dataset_2020_sen2_20_mod_500.h5  (33 GB)
#   FLOGA_dataset_2021_sen2_20_mod_500.h5  (52.1 GB)

wget -O ./data/wildfire/floga/data_split.csv \
    "https://www.dropbox.com/scl/fi/vq3tl8w5ex23lt1k7z89e/data_split.csv?rlkey=v3ph1xvfykhiljkg6rzlsytq2&dl=1"

python scripts/data_prep/wildfire/floga/create_dataset.py \
  --floga_path ./data/wildfire/floga/raw_h5 \
  --out_path ./data/wildfire/floga/converted_data_256 \
  --out_size 256 256 \
  --sample 1 \
  --random_seed 999

python scripts/data_prep/wildfire/floga/data_convert.py \
    --src_dir ./data/wildfire \
    --patch_size 256

data/
|___wildfire
|   |___floga
|       |___converted_data_256
|       |   |___2017
|       |   |   |___sample00000000_1_2017.sen2_20_pre.npy
|       |   |   |___sample00000000_1_2017.sen2_20_post.npy
|       |   |   |___sample00000000_1_2017.label.npy
|       |   |   |___...
|       |   |___2018, 2019, 2020, 2021
|       |   |___allEvents_60-20-20_r1_train.pkl
|       |   |___allEvents_60-20-20_r1_val.pkl
|       |   |___allEvents_60-20-20_r1_test.pkl
|       |___patch256
|           |___T1
|           |   |___*.npy  # original-scaled (0-10000) 9-channel sen2 data
|           |___T2
|           |   |___*.npy  # original-scaled (0-10000) 9-channel sen2 data
|           |___GT
|           |   |___*.npy  # np.int16 mask with values {0,1,2}; 0 = Non-burnt; 1 = Burnt area; 2 = Other events (loader maps 2 -> 255)
|           |___train.txt
|           |___val.txt
|           |___test.txt
```

### S2WCD
```
mkdir -p ./data/wildfire/raw
#   download "Sentinel-2 Wildfire Change Detection (S2-WCD).zip" from
#       https://ieee-dataport.org/documents/sentinel-2-wildfire-change-detection-s2-wcd to ./data/wildfire/raw

cd ./data/wildfire/raw
unzip "Sentinel-2 Wildfire Change Detection (S2-WCD).zip"
mv "Sentinel-2 Wildfire Change Detection (S2-WCD)" "S2-WCD"

cd ../../..
python scripts/data_prep/wildfire/s2_wcd/data_convert.py \
    --src_dir ./data/wildfire/raw \
    --tgt_dir ./data/wildfire

data/wildfire/s2_wcd/
├── T1/*.npy          # (H, W, 3) float32, [0, 1]
├── T1_sen2/*.npy     # (13, H, W) int16, [0, 10000]
├── T2/*.npy
├── T2_sen2/*.npy
├── GT/*.png          # grayscale mask, 0/1
├── train.txt         # 31 event names
├── val.txt
└── test.txt          # 10 event names
```

### WildfireSpreadTS
```
mkdir -p ./data/wildfire/fire_spread
wget -c -O ./data/wildfire/fire_spread/WildfireSpreadTS.zip \
    "https://zenodo.org/records/8006177/files/WildfireSpreadTS.zip?download=1"
    
cd ./data/wildfire/fire_spread
unzip WildfireSpreadTS.zip

ls -d 2018/ 2019/ 2020/ 2021/

cd ../../..
python scripts/data_prep/wildfire/fire_spread/data_convert.py \
    --data_dir ./data/wildfire/fire_spread \
    --target_dir ./data/wildfire/fire_spread

# verify
find ./data/wildfire/fire_spread -name "*.hdf5" | wc -l
# expected: 607

data/
|___wildfire
|   |___fire_spread
|       |___2018
|       |   |___*.hdf5
|       |___2019
|       |   |___*.hdf5
|       |___2020
|       |   |___*.hdf5
|       |___2021
|           |___*.hdf5
```

### SatelliteBurnedArea
```
mkdir -p ./data/wildfire/satellite_burned_area
cd ./data/wildfire/satellite_burned_area

wget "https://zenodo.org/records/6597139/files/satellite_data.csv?download=1" \
    -O satellite_data.csv

for i in 1 2 3 4 5; do
    wget -c "https://zenodo.org/records/6597139/files/Satellite_burned_area_dataset_part${i}.zip?download=1" \
        -O "Satellite_burned_area_dataset_part${i}.zip"
done

for i in 1 2 3 4 5; do
    unzip -o "Satellite_burned_area_dataset_part${i}.zip" -d .
done

for i in 1 2 3 4 5; do
    dir="Satellite_burned_area_dataset_part${i}"
    if [ -d "$dir" ]; then
        mv "$dir"/EMSR* . 2>/dev/null
        rmdir "$dir" 2>/dev/null
    fi
done

rm -f Satellite_burned_area_dataset_part*.zip

# verify
echo "Event folders: $(ls -d EMSR* 2>/dev/null | wc -l)"
# expected: 73

data/
|___wildfire
|   |___satellite_burned_area
|       |___satellite_data.csv     # fold column drives train/val/test split
|       |___EMSR<id>_<aoi>
|       |   |___sentinel2_YYYY-MM-DD.tiff
|       |   |___sentinel1_YYYY-MM-DD.tiff
|       |   |___EMSR<id>_<aoi>_mask.tiff
|       |___...
```