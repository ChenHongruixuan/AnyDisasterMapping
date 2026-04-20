# Data Setup
## Flood

Run the commands below from the repository root. All commands use the default arguments of the scripts in `scripts/data_prep/flood`.
For the Python download helpers, install dependencies first: `pip install -e .`.

The first directory tree shown for each dataset is an intermediate extracted layout. After `reorganize` finishes successfully, those extracted raw folders can be removed to reclaim disk space.

Config-expected runtime paths:
- CAU_Flood: `data/flood/CAU_Flood/{train,validation,test}.txt`
- KuroSiwo: `data/flood/kurosiwo/{train,validation,test}.txt`
- UrbanSARFloods: `data/flood/UrbanSARFloods/{train,validation,test_jubba,test_nova,test_weihui}.txt`

### CAU_Flood
```bash
mkdir -p ./data/flood/CAU_Flood/raw
# manually download train.tar.gz and test.tar.gz from https://pan.baidu.com/s/1i5yxdfwjP-oTyiRmq6FZHQ
# password: rnx6
# put both archives in ./data/flood/CAU_Flood/raw/

tar -xzf ./data/flood/CAU_Flood/raw/train.tar.gz -C ./data/flood/CAU_Flood/raw
tar -xzf ./data/flood/CAU_Flood/raw/test.tar.gz -C ./data/flood/CAU_Flood/raw
python ./scripts/data_prep/flood/reorganize_cau_flood.py

# optional cleanup after successful conversion
rm -rf ./data/flood/CAU_Flood/raw/train ./data/flood/CAU_Flood/raw/test
rm -f ./data/flood/CAU_Flood/raw/train.tar.gz ./data/flood/CAU_Flood/raw/test.tar.gz
```

Expected directory structure after extraction:
```text
data/
|___flood
|   |___CAU_Flood
|       |___raw
|       |   |___train
|       |   |   |___flood_vv
|       |   |   |   |___*.png
|       |   |   |___opt
|       |   |   |   |___*.png
|       |   |   |___vv
|       |   |       |___*.png
|       |   |___test
|       |   |   |___flood_vv
|       |   |   |   |___*.png
|       |   |   |___opt
|       |   |   |   |___*.png
|       |   |   |___vv
|       |   |       |___*.png
|       |   |___train.tar.gz
|       |   |___test.tar.gz
```

After reorganization, `./data/flood/CAU_Flood/raw/train` and `./data/flood/CAU_Flood/raw/test` can be removed if you do not need to rerun the conversion.

Expected directory structure after reorganization:
```text
data/
|___flood
|   |___CAU_Flood
|       |___PRE
|       |   |___*.png
|       |___POST
|       |   |___*.png
|       |___GT
|       |   |___*.png
|       |___train.txt
|       |___validation.txt
|       |___test.txt
```

### KuroSiwo
```bash
mkdir -p ./data/flood/kurosiwo/raw/configs/train ./data/flood/kurosiwo/raw/pickle
bash ./scripts/data_prep/flood/download_kurosiwo.sh
# put data_config.json in ./data/flood/kurosiwo/raw/configs/train/
# put train_pickle and test_pickle in ./data/flood/kurosiwo/raw/pickle/
python ./scripts/data_prep/flood/reorganize_kurosiwo.py

# optional cleanup after successful conversion
rm -rf ./data/flood/kurosiwo/raw/data
```

Expected directory structure before reorganization:
```text
data/flood/kurosiwo/
|___raw
|   |___configs
|   |   |___train
|   |       |___data_config.json
|   |___pickle
|   |   |___train_pickle
|   |   |___test_pickle
|   |___data
|   |   |___catalogue.gpkg
|   |   |___<scene_a>
|   |   |   |___MS1_IVV.tif
|   |   |   |___MS1_IVH.tif
|   |   |   |___SL1_IVV.tif
|   |   |   |___SL1_IVH.tif
|   |   |   |___SL2_IVV.tif
|   |   |   |___SL2_IVH.tif
|   |   |   |___MK0_DEM.tif
|   |   |   |___MK0_SLOPE.tif
|   |   |   |___MK0_MNA.tif
|   |   |   |___MK0_MLU.tif
|   |   |___<scene_b>
|   |       |___...
```

After reorganization, the extracted scene folders under `./data/flood/kurosiwo/raw/data/` can be removed if you do not need to rerun the conversion.

Expected directory structure after reorganization:
```text
data/flood/kurosiwo/
|___pre1_vv
|   |___*.tif
|___pre1_vh
|   |___*.tif
|___pre2_vv
|   |___*.tif
|___pre2_vh
|   |___*.tif
|___post_vv
|   |___*.tif
|___post_vh
|   |___*.tif
|___DEM
|   |___*.tif
|___SLOPE
|   |___*.tif
|___MASK_NODATA
|   |___*.tif
|___GT
|   |___*.tif
|___train.txt
|___validation.txt
|___test.txt
|___raw
|   |___configs
|   |   |___train
|   |       |___data_config.json
|   |___pickle
|   |   |___train_pickle
|   |   |___test_pickle
|   |___data
|   |   |___catalogue.gpkg
|   |   |___<scene_a>
|   |   |   |___MS1_IVV.tif
|   |   |   |___MS1_IVH.tif
|   |   |   |___SL1_IVV.tif
|   |   |   |___SL1_IVH.tif
|   |   |   |___SL2_IVV.tif
|   |   |   |___SL2_IVH.tif
|   |   |   |___MK0_DEM.tif
|   |   |   |___MK0_SLOPE.tif
|   |   |   |___MK0_MNA.tif
|   |   |   |___MK0_MLU.tif
|   |   |___<scene_b>
|   |       |___...
```

### UrbanSARFloods
```bash
mkdir -p ./data/flood/UrbanSARFloods/raw
python ./scripts/data_prep/flood/download_urbansar_floods.py

tar -xzf ./data/flood/UrbanSARFloods/raw/urban_sar_floods.tar.gz -C ./data/flood/UrbanSARFloods/raw
python ./scripts/data_prep/flood/reorganize_urbansar_floods.py

# optional cleanup after successful conversion
rm -rf ./data/flood/UrbanSARFloods/raw/testing_case_orig ./data/flood/UrbanSARFloods/raw/testing_case_256 ./data/flood/UrbanSARFloods/raw/urban_sar_floods
rm -f ./data/flood/UrbanSARFloods/raw/urban_sar_floods.tar.gz
```

Expected directory structure after extraction:
```text
data/
|___flood
|   |___UrbanSARFloods
|       |___raw
|       |   |___testing_case_orig
|       |   |   |___20210727_Weihui
|       |   |   |   |___*_GT.tif
|       |   |   |   |___*_SAR.tif
|       |   |   |___20230609_NovaKakhovka
|       |   |   |   |___*_GT.tif
|       |   |   |   |___*_SAR.tif
|       |   |   |___20231201_Jubba_1
|       |   |   |   |___*_GT.tif
|       |   |   |   |___*_SAR.tif
|       |   |   |___20231201_Jubba_2
|       |   |       |___*_GT.tif
|       |   |       |___*_SAR.tif
|       |   |___testing_case_256
|       |   |   |___...
|       |   |___urban_sar_floods
|       |   |   |___Train_dataset.txt
|       |   |   |___Valid_dataset.txt
|       |   |   |___<subset_a>
|       |   |   |   |___GT
|       |   |   |   |   |___*_GT.tif
|       |   |   |   |___SAR
|       |   |   |       |___*_SAR.tif
|       |   |   |___<subset_b>
|       |   |   |   |___...
|       |   |___urban_sar_floods.tar.gz
```

After reorganization, the extracted folders under `./data/flood/UrbanSARFloods/raw/` can be removed if you do not need to rerun the conversion.

Expected directory structure after reorganization:
```text
data/
|___flood
|   |___UrbanSARFloods
|       |___SAR
|       |   |___*.tif
|       |___GT
|       |   |___*.tif
|       |___train.txt
|       |___validation.txt
|       |___test_weihui.txt
|       |___test_nova.txt
|       |___test_jubba.txt
```
