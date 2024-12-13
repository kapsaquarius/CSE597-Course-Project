# Replication of the paper - Composed Image Retrieval using Contrastive Learning and Task-oriented CLIP-based Features

This repository is a revamped implementation of the original [paper](https://arxiv.org/abs/2308.11485)."
Its original code can be found [here](https://github.com/ABaldrati/CLIP4Cir)

## Clone the repo and install packages

1. Clone the repo

```sh
git clone https://github.com/kapsaquarius/CSE597-Course-Project
```

2. Install dependencies

```sh
conda create -n clip4cir -y python=3.8
conda activate clip4cir
conda install -y -c pytorch pytorch=1.11.0 torchvision=0.12.0
conda install -y -c anaconda pandas=1.4.2
pip install comet-ml==3.21.0
pip install git+https://github.com/openai/CLIP.git
```

## Download Data

Download the data from this Google Drive [link](https://drive.google.com/drive/folders/1CYxPeMbxgmMZaBme4TjdkQJOee_YpLgX?usp=drive_link).

Ensure that the folder containing the data looks exactly like this - 

```
project_base_path
└───  fashionIQ_dataset
      └─── captions
            | cap.dress.test.json
            | cap.dress.train.json
            | cap.dress.val.json
            | ...
            
      └───  images
            | B00006M009.jpg
            | B00006M00B.jpg
            | B00006M6IH.jpg
            | ...
            
      └─── image_splits
            | split.dress.test.json
            | split.dress.train.json
            | split.dress.val.json
            | ...

```

## Download Data

To download the fiq_clip_RN50x4_fullft.pt and fiq_comb_RN50x4_fullft.pt from this [link](https://drive.google.com/drive/folders/1BPE33_XSm33Min0OaGW2Sl9rcddZ-WF-?usp=drive_link). Make sure that these models are kept in the root directory of the project.

## If the project is being run locally

### Run training

If the new proposed model is to be run, inside the combiner_train.py file replace

```python
from combiner import Combiner
```
with

```python
from changed_combiner import Combiner
```

The changed_combiner.py file is the newly proposed combiner module with a few architecture changes.

python src/combiner_train.py \
  --dataset FashionIQ \
  --projection-dim 1280 \
  --hidden-dim 2560 \
  --num-epochs 5 \
  --clip-model-name RN50x4 \
  --clip-model-path fiq_clip_RN50x4_fullft.pt \
  --combiner-lr 1e-5 \
  --batch-size 512 \
  --clip-bs 64 \
  --transform targetpad \
  --save-training \
  --save-best \
  --target-ratio 1.25 \
  --validation-frequency 1


```

### Run Inference

```python

python src/validate.py \
    --dataset FashionIQ \
    --combining-function {Value can be combiner or sum here} \
    --combiner-path fiq_comb_RN50x4_fullft.pt \
    --projection-dim 2560 \
    --hidden-dim 5120 \
    --clip-model-name RN50x4 \
    --clip-model-path fiq_clip_RN50x4_fullft.pt \
    --target-ratio 1.25 \
    --transform targetpad

```

## If the project is being run on colab

This github repository contains a main.ipynb file. Run that directly.