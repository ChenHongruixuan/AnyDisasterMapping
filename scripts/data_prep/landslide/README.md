# Data Setup
## Landslide

Run the commands below from the repository root. Current landslide configs expect
all prepared datasets under `./data/landslides/`.
For the Python data-prep helpers, install dependencies first: `pip install -e .`.

Config-expected runtime paths:
- Landslide4Sense: `data/landslides/Landslide4Sense/{train,val,test}.txt`
- GVLM: `data/landslides/GVLM_CD/{train,val,test}.txt`
- HRGLDD: `data/landslides/HR_GLDD/{trainX,trainY,valX,valY,testX,testY}.npy`

### Landslide4Sense
```bash
mkdir -p ./data/landslides/Landslide4Sense/raw
# download:
#   https://zenodo.org/records/10463239/files/TrainData.zip?download=1
#   https://zenodo.org/records/10463239/files/ValidData.zip?download=1
#   https://zenodo.org/records/10463239/files/TestData.zip?download=1
# save them as:
#   ./data/landslides/Landslide4Sense/raw/TrainData.zip
#   ./data/landslides/Landslide4Sense/raw/ValidData.zip
#   ./data/landslides/Landslide4Sense/raw/TestData.zip
wget -O ./data/landslides/Landslide4Sense/raw/TrainData.zip https://zenodo.org/records/10463239/files/TrainData.zip
wget -O ./data/landslides/Landslide4Sense/raw/ValidData.zip https://zenodo.org/records/10463239/files/ValidData.zip
wget -O ./data/landslides/Landslide4Sense/raw/TestData.zip https://zenodo.org/records/10463239/files/TestData.zip

unzip -o ./data/landslides/Landslide4Sense/raw/TrainData.zip -d ./data/landslides/Landslide4Sense
unzip -o ./data/landslides/Landslide4Sense/raw/ValidData.zip -d ./data/landslides/Landslide4Sense
unzip -o ./data/landslides/Landslide4Sense/raw/TestData.zip -d ./data/landslides/Landslide4Sense
cp scripts/data_prep/landslide/Landslide4Sense/train.txt ./data/landslides/Landslide4Sense/
cp scripts/data_prep/landslide/Landslide4Sense/val.txt ./data/landslides/Landslide4Sense/
cp scripts/data_prep/landslide/Landslide4Sense/test.txt ./data/landslides/Landslide4Sense/

# optional: reclaim space after verifying setup
# rm -rf ./data/landslides/Landslide4Sense/raw
```

Expected directory structure after setup:
```text
data/
|___landslides
|   |___Landslide4Sense
|   |   |___TrainData
|   |   |   |___img
|   |   |   |   |___image_*.h5
|   |   |   |___mask
|   |   |       |___mask_*.h5
|   |   |___ValidData
|   |   |   |___img
|   |   |   |___mask
|   |   |___TestData
|   |   |   |___img
|   |   |   |___mask
|   |   |___train.txt
|   |   |___val.txt
|   |   |___test.txt
|   |   |___raw
```

### GVLM
```bash
mkdir -p ./data/landslides/GVLM_CD/raw
# manually download GVLM from https://drive.google.com/file/d/1R6U5GmBHVDi9g3XM09jYCnaqWSwEpBj-
# place the raw per-site folders under ./data/landslides/GVLM_CD/raw/GVLM_CD/
# each site folder should contain im1.png, im2.png, and ref.png

python scripts/data_prep/landslide/GVLM/1GVLM_Img_clip.py
cp scripts/data_prep/landslide/GVLM/train.txt ./data/landslides/GVLM_CD/
cp scripts/data_prep/landslide/GVLM/val.txt ./data/landslides/GVLM_CD/
cp scripts/data_prep/landslide/GVLM/test.txt ./data/landslides/GVLM_CD/

# optional: reclaim space after verifying setup
# rm -rf ./data/landslides/GVLM_CD/raw
```

Expected directory structure before clipping:
```text
data/
|___landslides
|   |___GVLM_CD
|   |   |___raw
|   |       |___GVLM_CD
|   |           |___<site_a>
|   |           |   |___im1.png
|   |           |   |___im2.png
|   |           |   |___ref.png
|   |           |___<site_b>
|   |               |___...
```

Expected directory structure after setup:
```text
data/
|___landslides
|   |___GVLM_CD
|   |   |___t1
|   |   |   |___*.jpg
|   |   |___t2
|   |   |   |___*.jpg
|   |   |___label
|   |   |   |___*.png
|   |   |___train.txt
|   |   |___val.txt
|   |   |___test.txt
|   |   |___raw
|   |       |___GVLM_CD
|   |           |___...
```

### HRGLDD
```bash
mkdir -p ./data/landslides/HR_GLDD
# manually download HRGLDD from https://zenodo.org/records/7189381#.Y0a2UHZBxD9
# place the prepared numpy arrays under ./data/landslides/HR_GLDD
wget -O ./data/landslides/HR_GLDD/testX.npy https://zenodo.org/records/7189381/files/testX.npy
wget -O ./data/landslides/HR_GLDD/testY.npy https://zenodo.org/records/7189381/files/testY.npy
wget -O ./data/landslides/HR_GLDD/trainX.npy https://zenodo.org/records/7189381/files/trainX.npy
wget -O ./data/landslides/HR_GLDD/trainY.npy https://zenodo.org/records/7189381/files/trainY.npy
wget -O ./data/landslides/HR_GLDD/valX.npy https://zenodo.org/records/7189381/files/valX.npy
wget -O ./data/landslides/HR_GLDD/valY.npy https://zenodo.org/records/7189381/files/valY.npy

# current configs use trainX.npy, trainY.npy, valX.npy, valY.npy, testX.npy, and testY.npy
```

Expected directory structure after setup:
```text
data/
|___landslides
|   |___HR_GLDD
|   |   |___trainX.npy
|   |   |___trainY.npy
|   |   |___valX.npy
|   |   |___valY.npy
|   |   |___testX.npy
|   |   |___testY.npy
```
